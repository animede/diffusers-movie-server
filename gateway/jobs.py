"""統一ジョブモデル(gateway 側ジョブレジストリ)。

- メモリ内 dict + threading.Lock を一次ソースとし、**状態遷移時に SQLite
  (gateway/data/jobs.sqlite3)へ永続化**する(Phase 5b)。gateway 再起動時は
  SQLite から履歴を復元する。復元時に running のまま残っていたジョブは、
  h3 系は「interrupted」(失敗扱いの終端状態)へ、ltx25 系は backend_job_id で
  バックエンドへ問い合わせて実状態に同期する(未起動なら interrupted)。
- 統一ステータス: queued | running | completed | failed | interrupted
- ltx25: バックエンドの非同期ジョブ API(POST /api/jobs)へ委譲し backend_job_id を
  保持。GET 時にバックエンドへ問い合わせて状態を都度変換する
  (queued/running → running、completed → completed、failed 系 → failed)。
- h3: 同期 API を gateway のバックグラウンドスレッドで呼び、実行中は /api/progress を
  中継して progress(0-1)を返す。同時1件(バックエンド自体が排他)。
- Phase 2 ではキューイングしない: バックエンド busy 中の generate は 409。
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

import assets as assets_mod
import modes
from backends import BACKENDS
from procman import BusyError, manager

logger = logging.getLogger("gateway.jobs")

TERMINAL = {"completed", "failed", "interrupted"}

DB_PATH = Path(__file__).resolve().parent / "data" / "jobs.sqlite3"
RESTORE_LIMIT = 1000  # 復元する履歴の上限(新しい順)


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
    status: str = "queued"          # queued | running | completed | failed | interrupted
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
        self._db_lock = threading.Lock()
        self._init_db()
        self._restore()

    # -- 永続化(SQLite)---------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._db_lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    result TEXT,
                    notes TEXT,
                    created_at REAL NOT NULL,
                    finished_at REAL,
                    backend_job_id TEXT
                )""")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")

    def _persist(self, job: UnifiedJob) -> None:
        """状態遷移時に呼ぶ(作成/running/終端)。失敗してもジョブ実行は止めない。"""
        try:
            with self._db_lock, self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO jobs "
                    "(id, backend, mode, status, progress, error, result, notes,"
                    " created_at, finished_at, backend_job_id)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (job.id, job.backend, job.mode, job.status,
                     round(job.progress, 4), job.error,
                     json.dumps(job.result, ensure_ascii=False)
                     if job.result is not None else None,
                     json.dumps(job.notes, ensure_ascii=False),
                     job.created_at, job.finished_at, job.backend_job_id))
        except sqlite3.Error as exc:
            logger.warning("ジョブ永続化に失敗しました(%s): %s", job.id, exc)

    def _restore(self) -> None:
        """起動時に SQLite から履歴を復元する。

        running/queued のまま残っていた h3 ジョブは worker スレッドが失われている
        ため即 interrupted へ。ltx25 ジョブはこの時点では running のまま残し、
        resync_restored()(app 起動イベント、adopt_orphans 後)でバックエンドへ
        問い合わせて実状態に同期する。
        """
        try:
            with self._db_lock, self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (RESTORE_LIMIT,)).fetchall()
        except sqlite3.Error as exc:
            logger.warning("ジョブ履歴の復元に失敗しました: %s", exc)
            return
        interrupted: list[UnifiedJob] = []
        self._pending_resync: list[str] = []
        for row in rows:
            job = UnifiedJob(
                id=row["id"], backend=row["backend"], mode=row["mode"],
                status=row["status"], progress=row["progress"] or 0.0,
                error=row["error"],
                result=json.loads(row["result"]) if row["result"] else None,
                notes=json.loads(row["notes"]) if row["notes"] else [],
                created_at=row["created_at"], finished_at=row["finished_at"],
                backend_job_id=row["backend_job_id"])
            if job.status not in TERMINAL:
                if job.backend == "ltx25" and job.backend_job_id:
                    self._pending_resync.append(job.id)
                else:
                    job.status = "interrupted"
                    job.error = ("gateway 再起動によりジョブ追跡が中断されました"
                                 "(h3 同期呼び出しの結果は失われました)")
                    job.finished_at = job.finished_at or time.time()
                    interrupted.append(job)
            self._jobs[job.id] = job
        for job in interrupted:
            self._persist(job)
        if rows:
            logger.info("ジョブ履歴を %d 件復元しました(interrupted=%d, "
                        "resync待ち=%d)", len(rows), len(interrupted),
                        len(self._pending_resync))

    def resync_restored(self) -> None:
        """復元した running ltx25 ジョブをバックエンドの実状態に同期する。

        app の startup イベント(manager.adopt_orphans() の後)から呼ぶこと。
        バックエンド未起動・照会失敗・ジョブ不明(404)は interrupted にする。
        """
        pending = getattr(self, "_pending_resync", [])
        self._pending_resync = []
        for job_id in pending:
            job = self._jobs.get(job_id)
            if job is None or job.status in TERMINAL:
                continue
            base = BACKENDS["ltx25"].base_url()
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(base + f"/api/jobs/{job.backend_job_id}")
                    resp.raise_for_status()
            except httpx.HTTPError as exc:
                job.status = "interrupted"
                job.error = ("gateway 再起動後、バックエンド ltx25 のジョブ状態を"
                             f"確認できませんでした: {exc}")
                job.finished_at = time.time()
                self._persist(job)
                logger.info("復元ジョブ %s を interrupted にしました(%s)",
                            job.id, exc)
                continue
            self._apply_ltx25_state(job, resp.json())
            self._persist(job)
            logger.info("復元ジョブ %s をバックエンド実状態に同期しました"
                        "(status=%s)", job.id, job.status)

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
        self._persist(job)  # 受理確定(running)時点で永続化
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

    def _apply_ltx25_state(self, job: UnifiedJob, data: dict) -> None:
        """ltx25 バックエンドのジョブ応答を統一ステータスへ変換して反映する。"""
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
        elif status == "interrupted":
            # ユーザーが `POST /ltx25/api/interrupt` で止めた場合。失敗ではないので
            # 「ジョブが失敗しました」と言わない(統一ステータスに interrupted がある)。
            job.status = "interrupted"
            job.error = data.get("error") or "生成が中断要求により停止しました"
            job.finished_at = time.time()
        else:  # failed 系
            job.status = "failed"
            job.error = data.get("error") or f"ltx25 ジョブが失敗しました(status={status})"
            job.finished_at = time.time()

    def _refresh_ltx25(self, job: UnifiedJob) -> None:
        if job.status in TERMINAL:
            return
        if manager.active_backend_name() != "ltx25":
            job.status = "failed"
            job.error = "バックエンド ltx25 がジョブ完了前に停止しました(切替/unload)"
            job.finished_at = time.time()
            self._persist(job)
            return
        base = BACKENDS["ltx25"].base_url()
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(base + f"/api/jobs/{job.backend_job_id}")
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("ltx25 ジョブ照会に失敗(%s): %s", job.backend_job_id, exc)
            return  # 一時的な失敗は前回状態のまま
        self._apply_ltx25_state(job, resp.json())
        if job.status in TERMINAL:
            self._persist(job)

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
                    try:
                        detail = resp.json().get("detail")
                    except Exception:
                        detail = resp.text[:500]
                    if resp.status_code == 499:
                        # ユーザーが `POST /h3/api/interrupt` で止めた場合。失敗では
                        # ないので「失敗しました」と言わない(統一ステータスに
                        # interrupted がある)。ltx25 側の `_apply_ltx25_state()` と
                        # 扱いを揃えること。
                        job.status = "interrupted"
                        job.error = detail or "生成が中断要求により停止しました"
                    else:
                        job.status = "failed"
                        job.error = f"h3 が {resp.status_code} を返しました: {detail}"
            except Exception as exc:  # ネットワーク断・バックエンド停止等
                job.status = "failed"
                job.error = f"h3 呼び出しに失敗しました: {exc}"
            finally:
                job.finished_at = time.time()
                self._persist(job)  # 終端状態(completed/failed)を永続化

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

    def delete(self, job_id: str) -> bool:
        """履歴レコードを削除する(成果物ファイルは消さない)。"""
        with self._lock:
            existed = self._jobs.pop(job_id, None) is not None
        try:
            with self._db_lock, self._connect() as conn:
                cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                existed = existed or cur.rowcount > 0
        except sqlite3.Error as exc:
            logger.warning("ジョブレコード削除に失敗しました(%s): %s", job_id, exc)
        return existed


registry = JobRegistry()
