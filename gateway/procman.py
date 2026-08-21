"""バックエンドプロセスマネージャ。

- 「VRAM を持てる(アクティブ)のは同時1バックエンドのみ」の排他原則。切替は
  「busy でないこと確認 → 旧の解放完了 → 新の有効化」の直列。全操作は
  threading.Lock 1本で排他する(FastAPI 側は def エンドポイント=スレッドプール
  実行なので同期ロックで良い)。
- Phase 5a: 切替戦略を2種類サポートする。
  - "process"(既定・従来どおり): 旧プロセスを停止(kill)してから新プロセスを起動。
  - "resident": 旧プロセスは生かしたまま `/api/admin/unload` で VRAM だけ解放し
    (nvidia-smi の per-process 実測で解放を確認)、新バックエンドのプロセスが
    既に生きていればプロセス起動をスキップして再有効化する(h3 は
    `/api/admin/reload` で preload、ltx25 は遅延ロードに任せる)。
    env(プリセット/overrides)が前回起動時と異なる場合は resident 不可のため
    自動でプロセス再起動へフォールバックする(env はプロセス起動時に固定のため)。
- 起動: subprocess で run.sh を env 付き起動。ログは gateway/logs/<backend>.log、
  PID は gateway/run/<backend>.pid。
- 孤児処理: gateway 起動時に PID ファイル + ポートの生存を確認し、一致すれば
  adopt(管理下へ戻す)。resident 運用では複数プロセスが同時に生きているのが
  正常なので、生存プロセスは全て adopt し、「アクティブ(重みロード済み)」は
  各バックエンドの自己申告(h3 runner status / ltx25 health.loaded)から推定する。
  PID ファイルと不一致のリスナーはエラー報告のみ(勝手に kill しない)。
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

# resident 解放後に per-process VRAM がこの値以下になれば「解放済み」とみなす。
# in-process unload 後も CUDA コンテキスト等が数百MB〜1GB台残るのが正常
# (実測: h3 96gb unload 後 ~1.2GB、ltx25 nf4 unload 後 ~0.7GB)。
UNLOAD_VRAM_THRESHOLD_MB = 2048
UNLOAD_VRAM_TIMEOUT_S = 90.0
ADMIN_TIMEOUT_S = 120.0        # /api/admin/unload の HTTP タイムアウト
RELOAD_TIMEOUT_S = 900.0       # /api/admin/reload(h3 96gb preload は数十秒〜)

VALID_STRATEGIES = ("process", "resident")


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


def _pid_vram_mb(pid: int) -> int | None:
    """nvidia-smi の per-process 実測(全GPU合算)。取得不可なら None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        total = 0
        for line in out.stdout.strip().splitlines():
            if not line.strip():
                continue
            p, used = [part.strip() for part in line.split(",")]
            if int(p) == pid:
                total += int(used)
        return total
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None


@dataclass
class ManagedProcess:
    backend: BackendDef
    pid: int
    preset: str
    env_extra: dict[str, str]
    started_at: float
    adopted: bool = False
    weights_loaded: bool = True   # gateway 視点の「VRAM 保持を許可された状態」
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
        self._procs: dict[str, ManagedProcess] = {}   # backend name -> live process
        self._active: str | None = None               # VRAM を持てる唯一のバックエンド
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
                    proc = ManagedProcess(
                        backend=backend, pid=pid, preset="(adopted: 不明)",
                        env_extra={}, started_at=_proc_start_time(pid) or time.time(),
                        adopted=True, weights_loaded=False)
                    self._procs[backend.name] = proc
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
            # アクティブ推定: 重みロード済みと自己申告するバックエンドを探す
            loaded = [name for name, proc in self._procs.items()
                      if self._backend_weights_loaded(proc)]
            if len(loaded) > 1:
                logger.error(
                    "複数バックエンドが同時に重みロード済みです: %s。排他原則違反の"
                    "状態のため先頭 %s のみアクティブ扱いにします(手動確認を推奨)",
                    loaded, loaded[0])
            if loaded:
                self._active = loaded[0]
                self._procs[loaded[0]].weights_loaded = True
                logger.info("adopt: アクティブバックエンド = %s", loaded[0])
            elif len(self._procs) == 1:
                # 自己申告なし(ltx25 の遅延ロード前など)でも、生存プロセスが1つ
                # だけなら従来どおりそれをアクティブ扱いにする(gateway 再起動で
                # パススルーが 502 になる退行を避ける)。2つ生きていて判別不能な
                # 場合のみ両方 parked のままにする(排他原則を優先)。
                only = next(iter(self._procs))
                self._active = only
                self._procs[only].weights_loaded = True
                logger.info("adopt: 生存プロセスが1つのため %s をアクティブ扱いにします", only)

    def _backend_weights_loaded(self, proc: ManagedProcess) -> bool:
        """バックエンド自己申告の「重みロード済み」判定(adopt 時のアクティブ推定用)。"""
        base = proc.backend.base_url()
        try:
            with httpx.Client(timeout=5.0) as client:
                if proc.backend.name == "h3":
                    resp = client.get(base + "/api/status")
                    resp.raise_for_status()
                    runner = resp.json().get("runner") or {}
                    return bool(runner.get("transformer_loaded")
                                or runner.get("transformer_ref_loaded")
                                or runner.get("text_encoder_loaded")
                                or runner.get("vae_loaded"))
                resp = client.get(base + "/api/health")
                resp.raise_for_status()
                return bool(resp.json().get("loaded"))
        except httpx.HTTPError:
            return False

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
             toggles: dict[str, bool] | None = None,
             strategy: str = "process",
             gpus: str | None = None) -> dict:
        """排他切替。既に同一バックエンド・同一構成でアクティブなら no-op。

        strategy="process": 旧プロセス停止 → 新プロセス起動(従来どおり)。
        strategy="resident": 旧プロセスは温存して VRAM のみ解放 → 新バックエンドの
        既存プロセスを再有効化(env 一致時)。env 不一致・プロセス無しは自動で
        プロセス起動へフォールバックする。
        """
        if strategy not in VALID_STRATEGIES:
            raise ValidationError(
                f"未知の strategy です: {strategy!r}(有効: {list(VALID_STRATEGIES)})")
        env_extra, preset_used = resolve_env(backend_name, preset, overrides, toggles,
                                             gpus=gpus)
        backend = BACKENDS[backend_name]
        with self._lock:
            if backend_name in self._foreign_listeners:
                raise ForeignListenerError(
                    f"ポート {backend.port} を管理外のプロセスが占有しています。"
                    "手動で停止してから再試行してください")
            self._reap_dead_locked()

            active = self._procs.get(self._active) if self._active else None
            if (active is not None and active.backend.name == backend_name
                    and active.env_extra == env_extra and active.weights_loaded
                    and not active.adopted):
                return {"result": "no-op",
                        "detail": "同一バックエンド・同一構成が既にアクティブです",
                        **self._proc_info(active)}

            # -- 旧アクティブの解放 ----------------------------------------
            note = None
            if active is not None:
                if self.backend_busy(active):
                    raise BusyError(
                        f"アクティブなバックエンド {active.backend.name} が生成中(busy)です。"
                        "完了を待ってから再試行してください")
                if strategy == "resident" and active.backend.name != backend_name:
                    self._deactivate_locked(active)
                else:
                    # process 戦略、または同一バックエンドの構成変更(env 固定のため
                    # resident でも再起動が必須)
                    if strategy == "resident" and active.backend.name == backend_name:
                        note = ("resident 要求ですが同一バックエンドの構成変更"
                                f"(旧={active.env_extra} 新={env_extra})のため"
                                "プロセス再起動へフォールバックしました")
                        logger.info("%s: %s", backend_name, note)
                    self._stop_locked(active)
            self._active = None

            # -- 新バックエンドの有効化 ------------------------------------
            existing = self._procs.get(backend_name)
            if existing is not None and existing.alive():
                if (strategy == "resident" and existing.env_extra == env_extra
                        and not existing.adopted):
                    t0 = time.time()
                    self._reactivate_locked(existing)
                    self._active = backend_name
                    return {"result": "reactivated",
                            "reactivate_s": round(time.time() - t0, 1),
                            **self._proc_info(existing)}
                if strategy == "resident":
                    note = ("resident 要求ですが既存プロセスの env が異なる"
                            f"(旧={existing.env_extra} 新={env_extra})ため"
                            "プロセス再起動へフォールバックしました")
                    logger.info("%s: %s", backend_name, note)
                self._stop_locked(existing)
            elif existing is not None:
                self._cleanup_locked(existing)

            proc = self._start_locked(backend, preset_used, env_extra)
            self._active = backend_name
            # 起動中に死んだ場合は _wait_health_locked → _cleanup_locked が
            # _procs/_active を掃除した上で例外を上げる。ヘルスタイムアウトは
            # プロセス温存のままエラー(従来どおり、ロード継続中の可能性)。
            self._wait_health_locked(proc)
            result = {"result": "started", **self._proc_info(proc)}
            if note:
                result["note"] = note
            return result

    def unload(self, strategy: str = "process") -> dict:
        """strategy="process": 管理下の全プロセスを停止(完全クリーン)。
        strategy="resident": アクティブの VRAM のみ解放してプロセスは温存。
        """
        if strategy not in VALID_STRATEGIES:
            raise ValidationError(
                f"未知の strategy です: {strategy!r}(有効: {list(VALID_STRATEGIES)})")
        with self._lock:
            self._reap_dead_locked()
            active = self._procs.get(self._active) if self._active else None
            if strategy == "resident":
                if active is None:
                    return {"result": "no-op",
                            "detail": "アクティブなバックエンドはありません"}
                if self.backend_busy(active):
                    raise BusyError(
                        f"バックエンド {active.backend.name} が生成中(busy)のため解放できません")
                self._deactivate_locked(active)
                self._active = None
                return {"result": "unloaded-resident",
                        "backend": active.backend.name,
                        "detail": "VRAM を解放しました(プロセスは温存)"}
            # process: アクティブが busy なら拒否、その後全プロセス停止
            if active is not None and self.backend_busy(active):
                raise BusyError(
                    f"バックエンド {active.backend.name} が生成中(busy)のため停止できません")
            stopped = []
            for proc in list(self._procs.values()):
                if proc.alive():
                    self._stop_locked(proc)
                else:
                    self._cleanup_locked(proc)
                stopped.append(proc.backend.name)
            self._active = None
            if not stopped:
                return {"result": "no-op", "detail": "アクティブなバックエンドはありません"}
            return {"result": "stopped", "backends": stopped,
                    "backend": stopped[0] if len(stopped) == 1 else None}

    def status(self) -> dict:
        with self._lock:
            self._reap_dead_locked()
            active = self._procs.get(self._active) if self._active else None
            info: dict = {"active_backend": None, "process": None,
                          "backend_health": None, "busy": None}
            if active is not None:
                info["active_backend"] = active.backend.name
                info["process"] = self._proc_info(active)
                info["backend_health"] = self.backend_health(active)
                info["busy"] = (self.backend_busy(active)
                                if info["backend_health"] is not None else None)
            # Phase 5a: process alive / weights loaded の2軸(全バックエンド)
            backends_info = {}
            for name in BACKENDS:
                proc = self._procs.get(name)
                if proc is None:
                    backends_info[name] = {"process_alive": False,
                                           "weights_loaded": False}
                else:
                    backends_info[name] = {
                        "process_alive": True,
                        "weights_loaded": proc.weights_loaded,
                        "pid": proc.pid,
                        "preset": proc.preset,
                        "uptime_s": round(proc.uptime_s(), 1),
                        "vram_mb": _pid_vram_mb(proc.pid),
                        "gpus": proc.env_extra.get("CUDA_VISIBLE_DEVICES"),
                    }
            info["backends"] = backends_info
            if self._foreign_listeners:
                info["foreign_listeners"] = dict(self._foreign_listeners)
            return info

    # -- 内部(_lock 保持前提)---------------------------------------------

    def _reap_dead_locked(self) -> None:
        """勝手に死んだプロセスを管理簿から掃除する。"""
        for name, proc in list(self._procs.items()):
            if not proc.alive():
                logger.warning("プロセス %s (pid=%d) は既に消滅していました",
                               name, proc.pid)
                self._cleanup_locked(proc)

    def _proc_info(self, proc: ManagedProcess) -> dict:
        return {
            "backend": proc.backend.name,
            "pid": proc.pid,
            "port": proc.backend.port,
            "preset": proc.preset,
            "env_extra": proc.env_extra,
            "uptime_s": round(proc.uptime_s(), 1),
            "adopted": proc.adopted,
            "weights_loaded": proc.weights_loaded,
            "gpus": proc.env_extra.get("CUDA_VISIBLE_DEVICES"),  # None = 全GPU可視
        }

    def _deactivate_locked(self, proc: ManagedProcess) -> None:
        """resident 解放: プロセスを残したまま /api/admin/unload → VRAM 実解放を確認。

        unload エンドポイントが無い(古いプロセス)・失敗した・VRAM が解放されない
        場合は、排他原則を守るためプロセス停止へフォールバックする。
        """
        name = proc.backend.name
        url = proc.backend.base_url() + "/api/admin/unload"
        t0 = time.time()
        try:
            with httpx.Client(timeout=ADMIN_TIMEOUT_S) as client:
                resp = client.post(url)
            if resp.status_code == 409:
                raise BusyError(
                    f"バックエンド {name} が生成中(busy)のため解放できません")
            resp.raise_for_status()
        except BusyError:
            raise
        except httpx.HTTPError as exc:
            logger.warning("resident 解放に失敗(%s): %s — プロセス停止へフォールバック",
                           name, exc)
            self._stop_locked(proc)
            return
        # VRAM 実解放の確認(nvidia-smi per-process。取得不可ならバックエンド報告に依存)
        deadline = time.time() + UNLOAD_VRAM_TIMEOUT_S
        vram = _pid_vram_mb(proc.pid)
        while vram is not None and vram > UNLOAD_VRAM_THRESHOLD_MB and time.time() < deadline:
            time.sleep(1.0)
            vram = _pid_vram_mb(proc.pid)
        if vram is not None and vram > UNLOAD_VRAM_THRESHOLD_MB:
            logger.warning(
                "resident 解放後も VRAM が %dMB 残っています(%s pid=%d、閾値 %dMB)。"
                "排他原則を守るためプロセス停止へフォールバックします",
                vram, name, proc.pid, UNLOAD_VRAM_THRESHOLD_MB)
            self._stop_locked(proc)
            return
        proc.weights_loaded = False
        logger.info("resident 解放完了: %s pid=%d(%.1fs、残VRAM=%sMB)",
                    name, proc.pid, time.time() - t0, vram)

    def _reactivate_locked(self, proc: ManagedProcess) -> None:
        """resident 再有効化: h3 は /api/admin/reload(preload_all)、ltx25 は
        遅延ロードに任せる(何もしない)。失敗時は例外(呼び出し元が 500 化)。"""
        name = proc.backend.name
        if name == "h3":
            url = proc.backend.base_url() + "/api/admin/reload"
            try:
                with httpx.Client(timeout=RELOAD_TIMEOUT_S) as client:
                    resp = client.post(url)
                if resp.status_code == 409:
                    raise BusyError(f"バックエンド {name} が生成中(busy)のため再有効化できません")
                resp.raise_for_status()
            except BusyError:
                raise
            except httpx.HTTPError as exc:
                raise RuntimeError(
                    f"バックエンド {name} の再有効化(admin/reload)に失敗しました: {exc}")
        proc.weights_loaded = True
        logger.info("resident 再有効化: %s pid=%d", name, proc.pid)

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
        self._procs[backend.name] = proc
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
        if self._procs.get(proc.backend.name) is proc:
            del self._procs[proc.backend.name]
        if self._active == proc.backend.name:
            self._active = None

    # パススルー用
    def active_backend_name(self) -> str | None:
        if self._active is None:
            return None
        proc = self._procs.get(self._active)
        if proc is not None and proc.alive():
            return self._active
        return None


def _proc_start_time(pid: int) -> float | None:
    """/proc/<pid> の作成時刻から起動時刻を推定(adopt 用、精度は秒で十分)。"""
    try:
        return os.stat(f"/proc/{pid}").st_mtime
    except OSError:
        return None


manager = ProcessManager()
