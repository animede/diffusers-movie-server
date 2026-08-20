from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Settings
from .generator import LTXGenerator
from .schemas import GenerateRequest

# Output filename prefixes for still-image modes (PNG instead of MP4).
STILL_IMAGE_PREFIXES = {"t2i": "t2i", "refine_image": "refine", "ref2i": "ref2i"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    session_number: int
    request: GenerateRequest
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    video_url: str | None = None
    image_url: str | None = None
    created_at: str = ""
    updated_at: str = ""
    generation_seconds: float | None = None
    peak_vram_gb: float | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "session_number": self.session_number,
            "status": self.status,
            "progress": self.progress,
            "error": self.error,
            "video_url": self.video_url,
            "image_url": self.image_url,
            "request": self.request,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "generation_seconds": self.generation_seconds,
            "peak_vram_gb": self.peak_vram_gb,
        }


class JobManager:
    def __init__(self, config: Settings, generator: LTXGenerator | None = None):
        self.config = config
        self.generator = generator or LTXGenerator(config)
        self.jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str] = queue.Queue(maxsize=config.max_queue_size)
        self._lock = threading.Lock()
        self._init_db()
        self._worker = threading.Thread(target=self._run, daemon=True, name="ltx-worker")
        self._worker.start()

    def _connect(self):
        self.config.history_db.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.config.history_db, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self):
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS sessions (number INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL)")
            db.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, session_number INTEGER NOT NULL, request_json TEXT NOT NULL,
                status TEXT NOT NULL, progress REAL NOT NULL, error TEXT, video_url TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"""
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "generation_seconds" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN generation_seconds REAL")
            if "peak_vram_gb" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN peak_vram_gb REAL")
            if "image_url" not in columns:
                # Still-image modes (t2i / refine_image / ref2i) record a PNG here.
                db.execute("ALTER TABLE jobs ADD COLUMN image_url TEXT")
            now = utc_now()
            db.execute(
                "UPDATE jobs SET status='failed', error='Server restarted before completion', updated_at=? "
                "WHERE status IN ('queued','running')",
                (now,),
            )

    def create_session(self) -> int:
        with self._connect() as db:
            cursor = db.execute("INSERT INTO sessions(created_at) VALUES (?)", (utc_now(),))
            return int(cursor.lastrowid)

    def _save(self, job: Job):
        job.updated_at = utc_now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO jobs (
                id, session_number, request_json, status, progress, error, video_url, image_url,
                created_at, updated_at, generation_seconds, peak_vram_gb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET status=excluded.status, progress=excluded.progress,
                error=excluded.error, video_url=excluded.video_url, image_url=excluded.image_url,
                updated_at=excluded.updated_at,
                generation_seconds=excluded.generation_seconds, peak_vram_gb=excluded.peak_vram_gb""",
                (
                    job.id, job.session_number, job.request.model_dump_json(), job.status, job.progress,
                    job.error, job.video_url, job.image_url, job.created_at, job.updated_at,
                    job.generation_seconds, job.peak_vram_gb,
                ),
            )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"], session_number=row["session_number"],
            request=GenerateRequest.model_validate(json.loads(row["request_json"])),
            status=row["status"], progress=row["progress"], error=row["error"],
            video_url=row["video_url"],
            image_url=row["image_url"] if "image_url" in row.keys() else None,
            created_at=row["created_at"], updated_at=row["updated_at"],
            generation_seconds=row["generation_seconds"], peak_vram_gb=row["peak_vram_gb"],
        )

    def submit(self, request: GenerateRequest) -> Job:
        session_number = request.session_number or self.create_session()
        request.session_number = session_number
        now = utc_now()
        job = Job(id=uuid.uuid4().hex, session_number=session_number, request=request, created_at=now, updated_at=now)
        with self._lock:
            self.jobs[job.id] = job
        self._save(job)
        try:
            self._queue.put_nowait(job.id)
        except queue.Full:
            with self._lock:
                self.jobs.pop(job.id, None)
            job.status = "failed"
            job.error = "Generation queue is full"
            self._save(job)
            raise
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            active = self.jobs.get(job_id)
        if active:
            return active
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._from_row(row) if row else None

    def history(self, session_number: int | None = None, limit: int = 50) -> list[Job]:
        query = "SELECT * FROM jobs"
        params: list[object] = []
        if session_number is not None:
            query += " WHERE session_number=?"
            params.append(session_number)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def delete(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        if job.status in {"queued", "running"}:
            raise RuntimeError("A queued or running job cannot be deleted")
        with self._lock:
            self.jobs.pop(job_id, None)
        with self._connect() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        (self.config.output_dir / f"{job_id}.mp4").unlink(missing_ok=True)
        for prefix in STILL_IMAGE_PREFIXES.values():
            (self.config.output_dir / f"{prefix}_{job_id}.png").unlink(missing_ok=True)
        return True

    def _update_progress(self, job: Job, value: float):
        previous_bucket = int(job.progress * 20)
        job.progress = value
        # Persist coarse progress only, avoiding a DB write for every callback tick.
        if value >= 1 or int(value * 20) > previous_bucket:
            self._save(job)

    def _run(self):
        while True:
            job_id = self._queue.get()
            job = self.jobs[job_id]
            job.status = "running"
            self._save(job)
            try:
                still_prefix = STILL_IMAGE_PREFIXES.get(job.request.mode)
                target = self.config.output_dir / (
                    f"{still_prefix}_{job.id}.png" if still_prefix else f"{job.id}.mp4"
                )
                started = time.monotonic()
                metrics = self.generator.generate(job.request, target, lambda value: self._update_progress(job, value))
                job.generation_seconds = time.monotonic() - started
                if metrics:
                    job.peak_vram_gb = metrics.get("peak_vram_gb")
                if still_prefix:
                    job.image_url = f"/outputs/{target.name}"
                else:
                    job.video_url = f"/outputs/{job.id}.mp4"
                job.status = "completed"
            except Exception as exc:
                job.error = str(exc)
                job.status = "failed"
            finally:
                self._save(job)
                self._queue.task_done()
