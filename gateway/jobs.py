"""統一ジョブモデル(gateway 側ジョブレジストリ)。

- メモリ内 dict + threading.Lock のみ(**永続化は Phase 5 送り** — gateway 再起動で
  ジョブ履歴は消える。バックエンド側の成果物/履歴は残る)。
- 統一ステータス: queued | running | completed | failed
- ltx25: バックエンドの非同期ジョブ API(POST /api/jobs)へ委譲し backend_job_id を
  保持。GET 時にバックエンドへ問い合わせて状態を都度変換する
  (queued/running → running、completed → completed、failed 系 → failed)。
- h3: 同期 API を gateway のバックグラウンドスレッドで呼び、実行中は /api/progress を
  中継して progress(0-1)を返す。同時1件(バックエンド自体が排他)。
- Phase 2 ではキューイングしない: バックエンド busy 中の generate は 409。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

import assets as assets_mod
import modes
from backends import BACKENDS
from procman import BusyError, manager

logger = logging.getLogger("gateway.jobs")

TERMINAL = {"completed", "failed"}


class NotActiveError(RuntimeError):
    """対象バックエンド未起動かつ auto_load=false(409 に変換)。"""


def _rewrite_url(backend: str, url: Optional[str]) -> Optional[str]:
    """バックエンドの `/outputs/...` をパススルー URL `/{backend}/outputs/...` へ変換。"""
    if url and url.startswith("/outputs/"):
        return f"/{backend}{url}"
    return url


def backend_busy(name: str) -> bool:
    """バックエンドの busy 判定(procman.backend_busy と同じ規則の名前指定版)。"""
    base = BACKENDS[name].base_url()
    try:
        with httpx.Client(timeout=5.0) as client:
            if name == "h3":
                resp = client.get(base + "/api/status")
                resp.raise_for_status()
                return bool(resp.json().get("busy"))
            resp = client.get(base + "/api/jobs", params={"limit": 10})
            resp.raise_for_status()
            return any(job.get("status") in ("queued", "running")
                       for job in resp.json())
    except httpx.HTTPError as exc:
        logger.warning("busy 判定に失敗(%s): %s — 安全側(busy)扱い", name, exc)
        return True


@dataclass
class UnifiedJob:
    id: str
    backend: str
    mode: str
    status: str = "queued"          # queued | running | completed | failed
    progress: float = 0.0           # 0-1
    error: Optional[str] = None
    result: Optional[dict] = None   # image_url / video_url 等(パススルーURL形式)
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    backend_job_id: Optional[str] = None  # ltx25 のみ

    def public(self) -> dict:
        end = self.finished_at if self.finished_at is not None else time.time()
        return {
            "id": self.id,
            "backend": self.backend,
            "mode": self.mode,
            "status": self.status,
            "progress": round(self.progress, 4),
            "error": self.error,
            "result": self.result,
            "notes": self.notes,
            "created_at": self.created_at,
            "elapsed_s": round(end - self.created_at, 1),
            "backend_job_id": self.backend_job_id,
        }


class JobRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, UnifiedJob] = {}

    # -- submit ------------------------------------------------------------

    def submit(self, backend: str, mode: str, params: dict, extra: dict,
               asset_ids: list[str], auto_load: bool = True,
               preset: Optional[str] = None) -> dict:
        if backend not in BACKENDS:
            raise modes.ModeError(
                f"未知のバックエンドです: {backend!r}(有効: {sorted(BACKENDS)})")
        # モード変換の事前バリデーション(アセット読み込み前に mode/params/extra を検査
        # するため、ダミーではなく実アセットメタで一度組み立てる)
        asset_metas = [assets_mod.get_asset(a) for a in (asset_ids or [])]

        if backend == "h3":
            path, data, files, notes = modes.build_h3_request(
                mode, params or {}, extra or {}, asset_metas)
        else:
            # ltx25 は変換だけ先に検査(backend asset id はまだ無いのでダミーで検査し、
            # 起動確認後に本組み立てする)
            dummy = [{"id": "0" * 32, "kind": m["kind"]} for m in asset_metas]
            modes.build_ltx25_request(mode, params or {}, extra or {}, dummy)

        # バックエンド起動確認(未起動なら auto_load で既定/指定プリセット起動)
        if manager.active_backend_name() != backend:
            if not auto_load:
                raise NotActiveError(
                    f"バックエンド {backend} は起動していません(auto_load=false)。"
                    'POST /api/v1/backend/load で起動してください')
            logger.info("auto_load: %s(preset=%s)を起動します", backend, preset)
            manager.load(backend, preset, {}, {})  # Busy/Validation はそのまま伝播

        # busy → 409(Phase 2 ではゲートウェイ側キューイングはしない)
        if backend_busy(backend):
            raise BusyError(
                f"バックエンド {backend} が生成中(busy)です。"
                "完了を待ってから再試行してください(Phase 2 はキューイング非対応)")

        job = UnifiedJob(id=uuid.uuid4().hex[:16], backend=backend, mode=mode)
        with self._lock:
            self._jobs[job.id] = job

        try:
            if backend == "ltx25":
                self._submit_ltx25(job, mode, params or {}, extra or {}, asset_metas)
            else:
                job.notes.extend(notes)
                self._submit_h3(job, path, data, files)
        except Exception:
            with self._lock:
                self._jobs.pop(job.id, None)
            raise
        return job.public()

    # -- ltx25 -------------------------------------------------------------

    def _submit_ltx25(self, job: UnifiedJob, mode: str, params: dict,
                      extra: dict, asset_metas: list[dict]) -> None:
        base = BACKENDS["ltx25"].base_url()
        backend_assets = []
        with httpx.Client(timeout=httpx.Timeout(connect=10.0, read=120.0,
                                                write=None, pool=None)) as client:
            # アセットをバックエンドへ転送(gateway ID → ltx25 ID)
            for meta in asset_metas:
                with meta["path"].open("rb") as fh:
                    resp = client.post(
                        base + "/api/assets",
                        files={"file": (meta["filename"], fh,
                                        modes._suffix_content_type(meta["suffix"]))})
                if resp.status_code not in (200, 201):
                    raise modes.ModeError(
                        f"ltx25 へのアセット転送に失敗しました({meta['id']}): "
                        f"{resp.status_code} {resp.text[:300]}")
                backend_assets.append(
                    {"id": resp.json()["id"], "kind": meta["kind"]})

            body, notes = modes.build_ltx25_request(mode, params, extra, backend_assets)
            job.notes.extend(notes)
            resp = client.post(base + "/api/jobs", json=body)
            if resp.status_code == 429:
                raise BusyError("ltx25 の生成キューが一杯です(429)")
            if resp.status_code not in (200, 202):
                raise modes.ModeError(
                    f"ltx25 がジョブを受理しませんでした: {resp.status_code} "
                    f"{resp.text[:500]}")
            job.backend_job_id = resp.json()["id"]
            job.status = "running"

    def _refresh_ltx25(self, job: UnifiedJob) -> None:
        if job.status in TERMINAL:
            return
        if manager.active_backend_name() != "ltx25":
            job.status = "failed"
            job.error = "バックエンド ltx25 がジョブ完了前に停止しました(切替/unload)"
            job.finished_at = time.time()
            return
        base = BACKENDS["ltx25"].base_url()
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(base + f"/api/jobs/{job.backend_job_id}")
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("ltx25 ジョブ照会に失敗(%s): %s", job.backend_job_id, exc)
            return  # 一時的な失敗は前回状態のまま
        data = resp.json()
        status = data.get("status")
        job.progress = float(data.get("progress") or 0.0)
        if status in ("queued", "running"):
            job.status = "running"
        elif status == "completed":
            job.status = "completed"
            job.progress = 1.0
            job.finished_at = time.time()
            job.result = {
                "video_url": _rewrite_url("ltx25", data.get("video_url")),
                "image_url": _rewrite_url("ltx25", data.get("image_url")),
                "generation_seconds": data.get("generation_seconds"),
                "peak_vram_gb": data.get("peak_vram_gb"),
            }
        else:  # failed 系
            job.status = "failed"
            job.error = data.get("error") or f"ltx25 ジョブが失敗しました(status={status})"
            job.finished_at = time.time()

    # -- h3 ----------------------------------------------------------------

    def _submit_h3(self, job: UnifiedJob, path: str, data: dict, files: list) -> None:
        base = BACKENDS["h3"].base_url()

        def worker():
            job.status = "running"
            try:
                with httpx.Client(timeout=httpx.Timeout(
                        connect=10.0, read=None, write=None, pool=None)) as client:
                    resp = client.post(base + path, data=data,
                                       files=files if files else None)
                if resp.status_code == 200:
                    result = resp.json()
                    result["video_url"] = _rewrite_url("h3", result.get("video_url"))
                    if result.get("image_url"):
                        result["image_url"] = _rewrite_url("h3", result.get("image_url"))
                    job.result = result
                    job.status = "completed"
                    job.progress = 1.0
                else:
                    job.status = "failed"
                    try:
                        detail = resp.json().get("detail")
                    except Exception:
                        detail = resp.text[:500]
                    job.error = f"h3 が {resp.status_code} を返しました: {detail}"
            except Exception as exc:  # ネットワーク断・バックエンド停止等
                job.status = "failed"
                job.error = f"h3 呼び出しに失敗しました: {exc}"
            finally:
                job.finished_at = time.time()

        threading.Thread(target=worker, name=f"h3-job-{job.id}", daemon=True).start()

    def _refresh_h3(self, job: UnifiedJob) -> None:
        """running 中のみ /api/progress を中継して progress を更新する。"""
        if job.status != "running":
            return
        if manager.active_backend_name() != "h3":
            return  # worker スレッドが接続断で failed にする
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(BACKENDS["h3"].base_url() + "/api/progress")
                resp.raise_for_status()
        except httpx.HTTPError:
            return
        snap = resp.json()
        total = snap.get("total_steps") or 0
        step = snap.get("step") or 0
        phase = snap.get("phase") or "idle"
        if phase == "denoising" and total > 0:
            # denoise を 5%〜95% に割り当てる(前後にロード/デコードがあるため)
            job.progress = min(0.95, 0.05 + 0.9 * (step / total))
        elif phase in ("loading_text_encoder", "encoding", "loading_transformer",
                       "starting", "loading"):
            job.progress = max(job.progress, 0.02)
        elif phase == "decoding":
            job.progress = max(job.progress, 0.95)

    # -- query -------------------------------------------------------------

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.backend == "ltx25":
            self._refresh_ltx25(job)
        else:
            self._refresh_h3(job)
        return job.public()

    def list(self, limit: int = 50) -> list[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at,
                          reverse=True)[:max(1, min(limit, 200))]
        out = []
        for job in jobs:
            if job.backend == "ltx25":
                self._refresh_ltx25(job)
            else:
                self._refresh_h3(job)
            out.append(job.public())
        return out


registry = JobRegistry()
