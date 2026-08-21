"""diffusers-movie-server gateway(port 8630)。

- 統一管理 API: /api/v1/backends, /api/v1/status, /api/v1/backend/load, /backend/unload
- 統一生成 API(Phase 2): /api/v1/generate, /api/v1/jobs, /api/v1/assets,
  /api/v1/outputs, /api/v1/prompt/enhance
- パススルー: /h3/{path} → 127.0.0.1:8631、/ltx25/{path} → 127.0.0.1:8632
  (method/ヘッダ/ボディ/クエリを素通し、レスポンスはストリーミング転送)
- 出力静的配信: /h3/outputs /ltx25/outputs は gateway が直接配信する
  (StaticFiles マウントをパススルーの catch-all より先に登録するため優先される。
  バックエンド停止後も成果物 URL が生きる)
"""

from __future__ import annotations

import logging
import subprocess
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import assets as assets_mod
import modes
from backends import BACKENDS, ValidationError, catalog
from jobs import NotActiveError, registry
from procman import BusyError, ForeignListenerError, manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("gateway")

app = FastAPI(title="diffusers-movie-server gateway", version="0.1.0")

# パススルー用共有クライアント。生成同期APIの h3 は数分かかるため read は無制限。
_passthrough_client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None))

# ホップバイホップヘッダ(転送しない)
_HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


@app.on_event("startup")
def _startup():
    manager.adopt_orphans()


@app.on_event("shutdown")
async def _shutdown():
    # バックエンドは停止しない(gateway 再起動時に adopt で戻せる設計)
    await _passthrough_client.aclose()


# ---------------------------------------------------------------------------
# 管理 API
# ---------------------------------------------------------------------------

class LoadRequest(BaseModel):
    backend: str
    preset: str | None = None
    overrides: dict[str, str] = Field(default_factory=dict)
    toggles: dict[str, bool] = Field(default_factory=dict)  # 例: {"turbo": true}


@app.get("/api/v1/backends")
def api_backends():
    return {"backends": catalog()}


@app.get("/api/v1/status")
def api_status():
    info = manager.status()
    info["vram"] = _nvidia_smi()
    return info


@app.post("/api/v1/backend/load")
def api_backend_load(req: LoadRequest):
    try:
        return manager.load(req.backend, req.preset, req.overrides, req.toggles)
    except ValidationError as exc:
        raise HTTPException(400, str(exc))
    except BusyError as exc:
        raise HTTPException(409, str(exc))
    except ForeignListenerError as exc:
        raise HTTPException(409, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/v1/backend/unload")
def api_backend_unload():
    try:
        return manager.unload()
    except BusyError as exc:
        raise HTTPException(409, str(exc))


def _nvidia_smi() -> list[dict] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        gpus = []
        for line in out.stdout.strip().splitlines():
            idx, used, total = [part.strip() for part in line.split(",")]
            gpus.append({"index": int(idx), "memory_used_mb": int(used),
                         "memory_total_mb": int(total)})
        return gpus
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


# ---------------------------------------------------------------------------
# 統一生成 API(Phase 2)
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    backend: str
    mode: str
    params: dict[str, Any] = Field(default_factory=dict)
    asset_ids: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
    auto_load: bool = True            # 未起動時に既定/指定プリセットで自動起動
    preset: str | None = None         # auto_load 時のみ使用(省略で既定プリセット)


@app.post("/api/v1/generate", status_code=202)
def api_generate(req: GenerateRequest):
    try:
        return registry.submit(req.backend, req.mode, req.params, req.extra,
                               req.asset_ids, auto_load=req.auto_load,
                               preset=req.preset)
    except (modes.ModeError, assets_mod.AssetError, ValidationError) as exc:
        raise HTTPException(400, str(exc))
    except (BusyError, NotActiveError, ForeignListenerError) as exc:
        raise HTTPException(409, str(exc))
    except RuntimeError as exc:
        raise HTTPException(500, str(exc))


@app.get("/api/v1/jobs/{job_id}")
def api_job_get(job_id: str):
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"ジョブが見つかりません: {job_id!r}"
                                 "(gateway 発行ジョブのみ。永続化は Phase 5 予定)")
    return job


@app.get("/api/v1/jobs")
def api_job_list(limit: int = 50):
    return {"jobs": registry.list(limit)}


@app.post("/api/v1/assets", status_code=201)
async def api_asset_upload(file: UploadFile = File(...)):
    try:
        # UploadFile.file は同期 read 可能な SpooledTemporaryFile
        meta = assets_mod.save_asset(file.filename or "", file.file)
    except assets_mod.AssetError as exc:
        raise HTTPException(415 if "拡張子" in str(exc) else 400, str(exc))
    finally:
        await file.close()
    return {k: meta[k] for k in ("id", "kind", "filename", "size")}


# -- 統合 outputs ------------------------------------------------------------

_OUTPUT_DIRS = {name: BACKENDS[name].dir / "outputs" for name in BACKENDS}
_OUTPUT_SUFFIXES = {".mp4", ".png", ".jpg", ".jpeg", ".webp", ".wav"}


@app.get("/api/v1/outputs")
def api_outputs():
    """両バックエンドの outputs/ を統合列挙(新しい順)。

    注意: OUTPUT_DIR を overrides で変更した ltx25 構成では既定ディレクトリのみを見る
    (Phase 2 の既知の制限)。
    """
    items = []
    for backend, out_dir in _OUTPUT_DIRS.items():
        if not out_dir.is_dir():
            continue
        for path in out_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in _OUTPUT_SUFFIXES:
                continue
            stat = path.stat()
            items.append({
                "backend": backend,
                "filename": path.name,
                "size": stat.st_size,
                "mtime": stat.st_mtime,
                "url": f"/{backend}/outputs/{path.name}",
            })
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return {"items": items}


class OutputDeleteRequest(BaseModel):
    backend: str
    filename: str


@app.post("/api/v1/outputs/delete")
def api_outputs_delete(req: OutputDeleteRequest):
    out_dir = _OUTPUT_DIRS.get(req.backend)
    if out_dir is None:
        raise HTTPException(400, f"未知のバックエンドです: {req.backend!r}")
    # パストラバーサル対策: 区切り文字・.. を拒否し、解決後の親が outputs/ 自身であること
    if "/" in req.filename or "\\" in req.filename or ".." in req.filename:
        raise HTTPException(400, f"不正なファイル名です: {req.filename!r}")
    target = (out_dir / req.filename).resolve()
    if target.parent != out_dir.resolve():
        raise HTTPException(400, f"不正なファイル名です: {req.filename!r}")
    if not target.is_file():
        raise HTTPException(404, f"ファイルが見つかりません: {req.backend}/{req.filename}")
    target.unlink()
    return {"result": "deleted", "backend": req.backend, "filename": req.filename}


# -- プロンプト強化(バックエンド振り分け)------------------------------------

class PromptEnhanceRequest(BaseModel):
    backend: str
    prompt: str
    mode: str | None = None           # h3: brief/storyboard/translate/h3-official、ltx25: t2av/i2v/...
    seconds: float | None = None      # h3 のみ
    task: str | None = None           # h3 h3-official のみ
    lang: str | None = None           # h3 h3-official のみ
    shots: list[str] = Field(default_factory=list)  # ltx25 のみ


@app.post("/api/v1/prompt/enhance")
def api_prompt_enhance(req: PromptEnhanceRequest):
    if req.backend not in BACKENDS:
        raise HTTPException(400, f"未知のバックエンドです: {req.backend!r}")
    if manager.active_backend_name() != req.backend:
        raise HTTPException(
            502,
            f"バックエンド {req.backend} は起動していません。"
            f'POST /api/v1/backend/load {{"backend": "{req.backend}"}} で起動してください')
    base = BACKENDS[req.backend].base_url()
    try:
        with httpx.Client(timeout=180.0) as client:
            if req.backend == "h3":
                data: dict[str, Any] = {"text": req.prompt,
                                        "mode": req.mode or "storyboard"}
                if req.seconds is not None:
                    data["seconds"] = req.seconds
                if req.task is not None:
                    data["task"] = req.task
                if req.lang is not None:
                    data["lang"] = req.lang
                resp = client.post(base + "/api/prompt/enhance", data=data)
            else:
                body: dict[str, Any] = {"prompt": req.prompt,
                                        "mode": req.mode or "t2av"}
                if req.shots:
                    body["shots"] = req.shots
                resp = client.post(base + "/api/prompts/enhance", json=body)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"バックエンド {req.backend} への転送に失敗しました: {exc}")
    try:
        content = resp.json()
    except ValueError:
        content = {"detail": resp.text[:500]}
    return JSONResponse(status_code=resp.status_code, content=content)


# ---------------------------------------------------------------------------
# 出力静的配信(パススルーの catch-all より先に登録するため優先される)
# ---------------------------------------------------------------------------

for _name, _dir in _OUTPUT_DIRS.items():
    _dir.mkdir(parents=True, exist_ok=True)
    app.mount(f"/{_name}/outputs", StaticFiles(directory=str(_dir)),
              name=f"{_name}-outputs")


# ---------------------------------------------------------------------------
# パススルー
# ---------------------------------------------------------------------------

async def _passthrough(prefix: str, path: str, request: Request):
    backend = BACKENDS[prefix]
    if manager.active_backend_name() != prefix:
        return JSONResponse(
            status_code=502,
            content={"detail": (
                f"バックエンド {prefix} は起動していません。"
                f'POST /api/v1/backend/load {{"backend": "{prefix}"}} で起動してください')})

    url = backend.base_url() + "/" + path
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _HOP_BY_HOP}
    upstream_request = _passthrough_client.build_request(
        request.method, url,
        headers=headers,
        params=request.query_params,
        content=request.stream(),
    )
    try:
        upstream = await _passthrough_client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        return JSONResponse(
            status_code=502,
            content={"detail": f"バックエンド {prefix} への転送に失敗しました: {exc}"})

    response_headers = {k: v for k, v in upstream.headers.items()
                        if k.lower() not in _HOP_BY_HOP}

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    return StreamingResponse(body(), status_code=upstream.status_code,
                             headers=response_headers)


@app.api_route("/h3/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def passthrough_h3(path: str, request: Request):
    return await _passthrough("h3", path, request)


@app.api_route("/ltx25/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
async def passthrough_ltx25(path: str, request: Request):
    return await _passthrough("ltx25", path, request)
