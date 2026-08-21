"""バックエンドプロセスマネージャ。

- 同時アクティブは1バックエンドのみ。切替は「busy でないこと確認 → 旧 stop 完了 →
  新 start」の直列。全操作は threading.Lock 1本で排他する(FastAPI 側は def
  エンドポイント=スレッドプール実行なので同期ロックで良い)。
- 起動: subprocess で run.sh を env 付き起動。ログは gateway/logs/<backend>.log、
  PID は gateway/run/<backend>.pid。
- 孤児処理: gateway 起動時に PID ファイル + ポートの生存を確認し、一致すれば
  adopt(管理下へ戻す)。PID ファイルと不一致のリスナーはエラー報告のみ
  (勝手に kill しない)。
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import httpx

from backends import BACKENDS, BackendDef, ValidationError, resolve_env

logger = logging.getLogger("gateway.procman")

GATEWAY_DIR = Path(__file__).resolve().parent
LOG_DIR = GATEWAY_DIR / "logs"
RUN_DIR = GATEWAY_DIR / "run"
LOG_DIR.mkdir(exist_ok=True)
RUN_DIR.mkdir(exist_ok=True)

STOP_GRACE_S = 20.0        # SIGTERM → SIGKILL までの猶予
PORT_CLOSE_TIMEOUT_S = 15.0


class BusyError(RuntimeError):
    """busy 中の stop/switch 要求(409 に変換)。"""


class ForeignListenerError(RuntimeError):
    """PID ファイルと一致しないプロセスがポートを占有している(勝手に kill しない)。"""


def _port_open(port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        # PermissionError は「存在するが他ユーザー」— このプロジェクトでは同一ユーザー
        # 運用のため生存扱いにしない
        return False


@dataclass
class ManagedProcess:
    backend: BackendDef
    pid: int
    preset: str
    env_extra: dict[str, str]
    started_at: float
    adopted: bool = False
    popen: subprocess.Popen | None = field(default=None, repr=False)

    def alive(self) -> bool:
        if self.popen is not None:
            return self.popen.poll() is None
        return _pid_alive(self.pid)

    def uptime_s(self) -> float:
        return time.time() - self.started_at


class ProcessManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._active: ManagedProcess | None = None
        self._foreign_listeners: dict[str, int] = {}  # backend name -> port

    # -- 起動時の孤児処理 --------------------------------------------------

    def adopt_orphans(self) -> None:
        with self._lock:
            for backend in BACKENDS.values():
                pid_file = RUN_DIR / f"{backend.name}.pid"
                pid = None
                if pid_file.is_file():
                    try:
                        pid = int(pid_file.read_text().strip())
                    except ValueError:
                        pid = None
                listening = _port_open(backend.port)
                if pid is not None and _pid_alive(pid) and listening:
                    if self._active is not None:
                        logger.error(
                            "複数バックエンドが同時に生存しています: %s(adopt済み)と %s。"
                            "%s は管理外のまま残します(手動で停止してください)",
                            self._active.backend.name, backend.name, backend.name)
                        self._foreign_listeners[backend.name] = backend.port
                        continue
                    self._active = ManagedProcess(
                        backend=backend, pid=pid, preset="(adopted: 不明)",
                        env_extra={}, started_at=_proc_start_time(pid) or time.time(),
                        adopted=True)
                    logger.info("孤児プロセスを adopt しました: %s pid=%d port=%d",
                                backend.name, pid, backend.port)
                elif listening:
                    self._foreign_listeners[backend.name] = backend.port
                    logger.error(
                        "ポート %d に PID ファイルと一致しないリスナーがあります"
                        "(backend=%s)。勝手に kill しません。手動確認が必要です",
                        backend.port, backend.name)
                elif pid is not None and not _pid_alive(pid):
                    pid_file.unlink(missing_ok=True)
                    logger.info("陳腐化した PID ファイルを削除: %s(pid=%d は既に消滅)",
                                pid_file, pid)

    # -- busy 判定 ---------------------------------------------------------

    def backend_busy(self, proc: ManagedProcess) -> bool:
        base = proc.backend.base_url()
        try:
            with httpx.Client(timeout=5.0) as client:
                if proc.backend.name == "h3":
                    resp = client.get(base + "/api/status")
                    resp.raise_for_status()
                    return bool(resp.json().get("busy"))
                if proc.backend.name == "ltx25":
                    resp = client.get(base + "/api/jobs", params={"limit": 10})
                    resp.raise_for_status()
                    return any(job.get("status") in ("queued", "running")
                               for job in resp.json())
        except httpx.HTTPError as exc:
            logger.warning("busy 判定に失敗(%s): %s — busy=不明のため安全側(True)扱い",
                           proc.backend.name, exc)
            return True
        return False

    def backend_health(self, proc: ManagedProcess) -> dict | None:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(proc.backend.base_url() + proc.backend.health_path)
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError:
            return None

    # -- start / stop / switch --------------------------------------------

    def load(self, backend_name: str, preset: str | None,
             overrides: dict[str, str] | None,
             toggles: dict[str, bool] | None = None) -> dict:
        """排他切替。既に同一バックエンド・同一構成なら no-op。"""
        env_extra, preset_used = resolve_env(backend_name, preset, overrides, toggles)
        backend = BACKENDS[backend_name]
        with self._lock:
            if backend_name in self._foreign_listeners:
                raise ForeignListenerError(
                    f"ポート {backend.port} を管理外のプロセスが占有しています。"
                    "手動で停止してから再試行してください")
            active = self._active
            if active is not None and active.alive():
                if (active.backend.name == backend_name
                        and active.env_extra == env_extra and not active.adopted):
                    return {"result": "no-op",
                            "detail": "同一バックエンド・同一構成が既に起動中です",
                            **self._proc_info(active)}
                if self.backend_busy(active):
                    raise BusyError(
                        f"アクティブなバックエンド {active.backend.name} が生成中(busy)です。"
                        "完了を待ってから再試行してください")
                self._stop_locked(active)
            elif active is not None:
                # プロセスが勝手に死んでいた
                logger.warning("アクティブプロセス %s (pid=%d) は既に消滅していました",
                               active.backend.name, active.pid)
                self._cleanup_locked(active)

            proc = self._start_locked(backend, preset_used, env_extra)
            self._wait_health_locked(proc)
            return {"result": "started", **self._proc_info(proc)}

    def unload(self) -> dict:
        with self._lock:
            active = self._active
            if active is None or not active.alive():
                if active is not None:
                    self._cleanup_locked(active)
                return {"result": "no-op", "detail": "アクティブなバックエンドはありません"}
            if self.backend_busy(active):
                raise BusyError(
                    f"バックエンド {active.backend.name} が生成中(busy)のため停止できません")
            name = active.backend.name
            self._stop_locked(active)
            return {"result": "stopped", "backend": name}

    def status(self) -> dict:
        with self._lock:
            active = self._active
            info: dict = {"active_backend": None, "process": None,
                          "backend_health": None, "busy": None}
            if active is not None:
                if not active.alive():
                    self._cleanup_locked(active)
                else:
                    info["active_backend"] = active.backend.name
                    info["process"] = self._proc_info(active)
                    info["backend_health"] = self.backend_health(active)
                    info["busy"] = (self.backend_busy(active)
                                    if info["backend_health"] is not None else None)
            if self._foreign_listeners:
                info["foreign_listeners"] = dict(self._foreign_listeners)
            return info

    # -- 内部(_lock 保持前提)---------------------------------------------

    def _proc_info(self, proc: ManagedProcess) -> dict:
        return {
            "backend": proc.backend.name,
            "pid": proc.pid,
            "port": proc.backend.port,
            "preset": proc.preset,
            "env_extra": proc.env_extra,
            "uptime_s": round(proc.uptime_s(), 1),
            "adopted": proc.adopted,
        }

    def _start_locked(self, backend: BackendDef, preset: str,
                      env_extra: dict[str, str]) -> ManagedProcess:
        if _port_open(backend.port):
            raise ForeignListenerError(
                f"ポート {backend.port} が既に使用中です(管理外プロセスの可能性)。"
                "手動で確認してください")
        log_path = LOG_DIR / f"{backend.name}.log"
        log_file = open(log_path, "ab")
        env = dict(os.environ)
        env.update(env_extra)
        script = backend.dir / backend.run_script
        popen = subprocess.Popen(
            ["bash", str(script)],
            cwd=str(backend.dir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # gateway 停止の巻き添えでシグナルを受けない
        )
        log_file.close()  # 子プロセス側に fd は複製済み
        proc = ManagedProcess(backend=backend, pid=popen.pid, preset=preset,
                              env_extra=env_extra, started_at=time.time(), popen=popen)
        (RUN_DIR / f"{backend.name}.pid").write_text(str(popen.pid))
        self._active = proc
        logger.info("起動: %s pid=%d preset=%s env=%s(ログ: %s)",
                    backend.name, popen.pid, preset, env_extra, log_path)
        return proc

    def _wait_health_locked(self, proc: ManagedProcess) -> None:
        deadline = time.time() + proc.backend.health_timeout_s
        url = proc.backend.base_url() + proc.backend.health_path
        while time.time() < deadline:
            if not proc.alive():
                self._cleanup_locked(proc)
                raise RuntimeError(
                    f"バックエンド {proc.backend.name} が起動中に終了しました。"
                    f"ログを確認してください: gateway/logs/{proc.backend.name}.log")
            try:
                with httpx.Client(timeout=5.0) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        logger.info("ヘルスOK: %s(%.1fs)", proc.backend.name,
                                    proc.uptime_s())
                        return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
        # タイムアウト: プロセスは残したままエラーにする(ロード継続中の可能性)
        raise RuntimeError(
            f"バックエンド {proc.backend.name} のヘルス待ちがタイムアウトしました"
            f"({proc.backend.health_timeout_s:.0f}s)。プロセスは起動したまま残しています"
            f"(pid={proc.pid})。ログ: gateway/logs/{proc.backend.name}.log")

    def _stop_locked(self, proc: ManagedProcess) -> None:
        name = proc.backend.name
        logger.info("停止開始: %s pid=%d", name, proc.pid)
        # run.sh は exec で uvicorn に置き換わるため pid 直接 SIGTERM でよい。
        # adopt したプロセスはセッションが不明なので pid 単位、自前起動は
        # start_new_session=True なのでプロセスグループへ送る。
        try:
            if proc.popen is not None:
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                os.kill(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.time() + STOP_GRACE_S
        while time.time() < deadline and proc.alive():
            time.sleep(0.5)
        if proc.alive():
            logger.warning("SIGTERM で停止しないため SIGKILL: %s pid=%d", name, proc.pid)
            try:
                if proc.popen is not None:
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc.popen is not None:
            try:
                proc.popen.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        # ポート閉鎖確認
        deadline = time.time() + PORT_CLOSE_TIMEOUT_S
        while time.time() < deadline and _port_open(proc.backend.port):
            time.sleep(0.5)
        if _port_open(proc.backend.port):
            logger.error("停止後もポート %d が開いたままです(%s)",
                         proc.backend.port, name)
        self._cleanup_locked(proc)
        logger.info("停止完了: %s", name)

    def _cleanup_locked(self, proc: ManagedProcess) -> None:
        (RUN_DIR / f"{proc.backend.name}.pid").unlink(missing_ok=True)
        if self._active is proc:
            self._active = None

    # パススルー用
    def active_backend_name(self) -> str | None:
        active = self._active
        if active is not None and active.alive():
            return active.backend.name
        return None


def _proc_start_time(pid: int) -> float | None:
    """/proc/<pid> の作成時刻から起動時刻を推定(adopt 用、精度は秒で十分)。"""
    try:
        return os.stat(f"/proc/{pid}").st_mtime
    except OSError:
        return None


manager = ProcessManager()
