"""diffusers-movie-server gateway(port 8630)。

- 統一管理 API: /api/v1/backends, /api/v1/status, /api/v1/backend/load, /backend/unload
- パススルー: /h3/{path} → 127.0.0.1:8631、/ltx25/{path} → 127.0.0.1:8632
  (method/ヘッダ/ボディ/クエリを素通し、レスポンスはストリーミング転送)
"""

from __future__ import annotations

import logging
import subprocess

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backends import BACKENDS, ValidationError, catalog
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
