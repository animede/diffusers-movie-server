"""バックエンドプロセスマネージャ。

- 既定は「VRAM を持てるのは同時1バックエンドのみ」の排他原則。切替は
  「busy でないこと確認 → 旧の解放完了 → 新の有効化」の直列。全操作は
  threading.Lock 1本で排他する(FastAPI 側は def エンドポイント=スレッドプール
  実行なので同期ロックで良い)。
- Phase 7 (coresident): **VRAM は両方保持し、実行(生成)だけを排他する**モード。
  「重みの再ロードを一切起こさずに LTX⇔H3 を切り替えたい」用途のために足した。
  排他の軸が2つに分かれるので、状態も2軸で持つ:
    - `_loaded`: 重みを VRAM に持つことを許可されたバックエンドの**集合**
      (process/resident では常に 0〜1 個、coresident では複数になりうる)
    - 実行ゲート: 状態は持たず、投入時に「自分以外が busy でないか」を
      バックエンドへ問い合わせて判定する(`other_busy()`)。生成の直列化は
      これ1点で担保する。
  2026-09-15 の実機実測では GPU0(96GB)で LTX nvfp4-fast 常駐 + H3 int8 単騎が
  同居でき、ピーク 83.2GiB・余裕 12.4GiB・切替コスト 0・速度劣化なしだった。
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
  adopt(管理下へ戻す)。resident/coresident 運用では複数プロセスが同時に生きて
  いるのが正常なので、生存プロセスは全て adopt し、「重みロード済み」は
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
import uuid
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

VALID_STRATEGIES = ("process", "resident", "coresident")


# リアルタイム優先リース(会話セッション宣言)の TTL。
# 会話アプリはチャンクごとに renew するので短めで良い。切れれば自動で解放されるため、
# クライアントが落ちてもゲートが開きっぱなしにならない。
LEASE_TTL_DEFAULT_S = 60.0
LEASE_TTL_MIN_S = 5.0
LEASE_TTL_MAX_S = 600.0


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
class RealtimeLease:
    """「いま会話セッション中」の宣言。保持者以外の生成を入口で締め出す。

    なぜ busy 判定ではなくリースなのか:
    - LTX は会話していないときも待機プール補充で頻繁に busy になる。「LTX が busy なら
      H3 は待つ」にすると H3 がほとんど回らない。
    - 逆に H3 の生成(t2va 5秒で約25秒)が走り出してから会話が来ると、LTX の
      リアルタイム予算 4.8秒に対して25秒待たされる。**H3 は途中で止められない**ので、
      始まってから優先度を付けても間に合わない。
    → 会話の**開始を宣言**してもらい、その間は相手の投入を入口で断るしかない。
    """
    lease_id: str
    backend: str
    created_at: float
    expires_at: float
    renewals: int = 0

    def alive(self) -> bool:
        return time.time() < self.expires_at

    def to_dict(self) -> dict:
        return {"lease_id": self.lease_id, "backend": self.backend,
                "created_at": self.created_at, "expires_at": self.expires_at,
                "remaining_s": round(max(0.0, self.expires_at - time.time()), 1),
                "renewals": self.renewals}


def _gpu_vram_mb() -> list[dict] | None:
    """GPU ごとの使用量/総量。coresident の load 応答に載せて余裕を可視化する。"""
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
            gpus.append({"index": int(idx), "used_mb": int(used),
                         "total_mb": int(total), "free_mb": int(total) - int(used)})
        return gpus
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
        # 重みを VRAM に持つことを許可されたバックエンド。process/resident 戦略では
        # 常に 0〜1 個(従来の排他原則)、coresident 戦略でのみ複数になる。
        self._loaded: set[str] = set()
        self._foreign_listeners: dict[str, int] = {}  # backend name -> port
        self._lease: RealtimeLease | None = None      # リアルタイム優先リース

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
            # ロード状態の推定: 重みロード済みと自己申告するバックエンドを全て拾う。
            # coresident 戦略では複数が同時にロード済みなのが**正常**なので、
            # 以前あった「先頭のみ採用 + 排他原則違反エラー」は廃止した。
            loaded = [name for name, proc in self._procs.items()
                      if self._backend_weights_loaded(proc)]
            if loaded:
                for name in loaded:
                    self._loaded.add(name)
                    self._procs[name].weights_loaded = True
                logger.info("adopt: 重みロード済みバックエンド = %s", loaded)
            elif len(self._procs) == 1:
                # 自己申告なし(ltx25 の遅延ロード前など)でも、生存プロセスが1つ
                # だけならそれをロード済み扱いにする(gateway 再起動でパススルーが
                # 502 になる退行を避ける)。2つ生きていて判別不能な場合のみ
                # 両方 parked のままにする(誤ってVRAMを二重に見積もらないため)。
                only = next(iter(self._procs))
                self._loaded.add(only)
                self._procs[only].weights_loaded = True
                logger.info("adopt: 生存プロセスが1つのため %s をロード済み扱いにします", only)

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
        """バックエンドを有効化する。既に同一構成でロード済みなら no-op。

        strategy="process": 他バックエンドのプロセスを停止 → 新プロセス起動(従来どおり)。
        strategy="resident": 他バックエンドはプロセス温存で VRAM のみ解放 → 対象の
        既存プロセスを再有効化(env 一致時)。
        strategy="coresident": **他バックエンドに一切触らない**(重みは VRAM に
        載せたまま)。対象だけを起動/再有効化する。生成の排他は VRAM ではなく
        実行ゲート(`other_busy()`)側で担保する。

        resident/coresident とも、env(プリセット/overrides)が既存プロセスと
        異なる場合は env がプロセス起動時に固定されるためプロセス再起動へ
        自動フォールバックする。
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

            # 会話セッション中の相手から VRAM を奪う load は断る。coresident は
            # 相手に触らないので許可する(生成側は実行ゲートで直列化される)。
            blocker = self._lease_blocker_locked(backend_name)
            if blocker is not None and strategy != "coresident":
                raise BusyError(
                    f"バックエンド {blocker} が会話セッション中です。strategy={strategy!r} は "
                    f"{blocker} の VRAM を解放してしまうため実行できません"
                    '(strategy="coresident" なら同居で起動できます)')

            existing = self._procs.get(backend_name)
            if existing is not None and not existing.alive():
                self._cleanup_locked(existing)
                existing = None

            if (existing is not None and existing.env_extra == env_extra
                    and existing.weights_loaded and not existing.adopted):
                return {"result": "no-op",
                        "detail": "同一バックエンド・同一構成が既にロード済みです",
                        **self._proc_info(existing)}

            note = None

            # -- 他バックエンドの扱い --------------------------------------
            # coresident は「触らない」が唯一の正しい振る舞い。他は従来どおり退去。
            if strategy == "coresident":
                busy_other = self._other_busy_locked(backend_name)
                if busy_other is not None:
                    raise BusyError(
                        f"他バックエンド {busy_other} が生成中(busy)です。重みの確保と"
                        "競合するため、完了を待ってから再試行してください")
            else:
                for other in [p for name, p in self._procs.items()
                              if name != backend_name and p.weights_loaded]:
                    if self.backend_busy(other):
                        raise BusyError(
                            f"バックエンド {other.backend.name} が生成中(busy)です。"
                            "完了を待ってから再試行してください")
                    if strategy == "resident":
                        self._deactivate_locked(other)
                    else:
                        self._stop_locked(other)

            # -- 対象バックエンドの有効化 ----------------------------------
            if existing is not None:
                reusable = existing.env_extra == env_extra and not existing.adopted
                if reusable and strategy in ("resident", "coresident"):
                    # 重みだけ落ちている(parked)状態からの復帰。プロセスは再利用する。
                    t0 = time.time()
                    self._reactivate_locked(existing)
                    self._loaded.add(backend_name)
                    return {"result": "reactivated",
                            "reactivate_s": round(time.time() - t0, 1),
                            **self._proc_info(existing)}
                if not reusable and strategy in ("resident", "coresident"):
                    note = (f"{strategy} 要求ですが既存プロセスの env が異なる"
                            f"(旧={existing.env_extra} 新={env_extra})ため"
                            "プロセス再起動へフォールバックしました")
                    logger.info("%s: %s", backend_name, note)
                if self.backend_busy(existing):
                    raise BusyError(
                        f"バックエンド {backend_name} が生成中(busy)です。"
                        "完了を待ってから再試行してください")
                self._stop_locked(existing)

            proc = self._start_locked(backend, preset_used, env_extra)
            self._loaded.add(backend_name)
            # 起動中に死んだ場合は _wait_health_locked → _cleanup_locked が
            # _procs/_loaded を掃除した上で例外を上げる。ヘルスタイムアウトは
            # プロセス温存のままエラー(従来どおり、ロード継続中の可能性)。
            self._wait_health_locked(proc)
            result = {"result": "started", **self._proc_info(proc)}
            if strategy == "coresident":
                result["coresident_with"] = sorted(self._loaded - {backend_name})
                result["vram"] = _gpu_vram_mb()
            if note:
                result["note"] = note
            return result

    def unload(self, strategy: str = "process",
               backend_name: str | None = None) -> dict:
        """strategy="process": プロセスを停止(完全クリーン)。
        strategy="resident"/"coresident": VRAM のみ解放してプロセスは温存。

        backend_name を指定すると**そのバックエンドだけ**を対象にする。省略時は
        従来どおり全部が対象(coresident で片方だけ降ろしたい場合に指定する)。
        """
        if strategy not in VALID_STRATEGIES:
            raise ValidationError(
                f"未知の strategy です: {strategy!r}(有効: {list(VALID_STRATEGIES)})")
        if backend_name is not None and backend_name not in BACKENDS:
            raise ValidationError(
                f"未知のバックエンドです: {backend_name!r}(有効: {sorted(BACKENDS)})")
        with self._lock:
            self._reap_dead_locked()
            if backend_name is None:
                targets = list(self._procs.values())
            else:
                proc = self._procs.get(backend_name)
                targets = [proc] if proc is not None else []

            if not targets:
                return {"result": "no-op",
                        "detail": "対象のバックエンドは起動していません"}

            # 会話セッション中のバックエンドは降ろさせない
            lease = self._lease_info_locked()
            if lease is not None and any(p.backend.name == lease["backend"] for p in targets):
                raise BusyError(
                    f"バックエンド {lease['backend']} が会話セッション中のため解放できません"
                    f"(残り {lease['remaining_s']:.0f}秒)")

            # busy なものが1つでもあれば、何も壊さずに拒否する(全か無か)
            for proc in targets:
                if proc.weights_loaded and self.backend_busy(proc):
                    raise BusyError(
                        f"バックエンド {proc.backend.name} が生成中(busy)のため解放できません")

            if strategy in ("resident", "coresident"):
                freed = []
                for proc in targets:
                    if not proc.weights_loaded:
                        continue
                    self._deactivate_locked(proc)
                    freed.append(proc.backend.name)
                if not freed:
                    return {"result": "no-op",
                            "detail": "重みをロード済みのバックエンドはありません"}
                return {"result": "unloaded-resident", "backends": freed,
                        "backend": freed[0] if len(freed) == 1 else None,
                        "detail": "VRAM を解放しました(プロセスは温存)"}

            stopped = []
            for proc in targets:
                if proc.alive():
                    self._stop_locked(proc)
                else:
                    self._cleanup_locked(proc)
                stopped.append(proc.backend.name)
            return {"result": "stopped", "backends": stopped,
                    "backend": stopped[0] if len(stopped) == 1 else None}

    def status(self) -> dict:
        with self._lock:
            self._reap_dead_locked()
            loaded = sorted(n for n in self._loaded if n in self._procs)
            # active_backend / process は後方互換のため残す(単一バックエンド運用では
            # 従来と同じ値になる)。coresident で複数ロード中は loaded_backends を見ること。
            primary = self._procs.get(loaded[0]) if loaded else None
            info: dict = {"active_backend": None, "process": None,
                          "backend_health": None, "busy": None,
                          "loaded_backends": loaded}
            if primary is not None:
                info["active_backend"] = primary.backend.name
                info["process"] = self._proc_info(primary)
                info["backend_health"] = self.backend_health(primary)
                info["busy"] = (self.backend_busy(primary)
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
            info["realtime_lease"] = self._lease_info_locked()
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
        self._loaded.discard(name)
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
        self._loaded.discard(proc.backend.name)

    def _other_busy_locked(self, backend_name: str) -> str | None:
        """自分以外で生成中のバックエンド名。無ければ None(_lock 保持前提)。"""
        for name, proc in self._procs.items():
            if name == backend_name or not proc.weights_loaded:
                continue
            if self.backend_busy(proc):
                return name
        return None

    # -- 公開ヘルパ(パススルー・ジョブ投入から使う)-------------------------

    def is_loaded(self, backend_name: str) -> bool:
        """このバックエンドが「重みを持って生きている」か。

        coresident では複数が同時に True になりうる。従来の
        `active_backend_name() == name` の置き換え。
        """
        if backend_name not in self._loaded:
            return False
        proc = self._procs.get(backend_name)
        return proc is not None and proc.alive()

    def other_busy(self, backend_name: str) -> str | None:
        """**実行ゲート**: 自分以外のバックエンドが生成中ならその名前を返す。

        coresident では VRAM は同時に持てるが、生成を重ねると計算資源を食い合う
        (2026-09-15 実測で LTX は 1.89倍・H3 は 1.49倍に劣化し、LTX の
        リアルタイム予算 4.8秒を超える)。投入前にこれで直列化する。
        """
        with self._lock:
            return self._other_busy_locked(backend_name)

    def loaded_backends(self) -> list[str]:
        """重みを持って生きているバックエンド名(ソート済み)。"""
        return sorted(n for n in self._loaded
                      if (p := self._procs.get(n)) is not None and p.alive())

    # -- リアルタイム優先リース(会話セッション)------------------------------

    def acquire_lease(self, backend_name: str, ttl_s: float | None = None,
                      lease_id: str | None = None) -> dict:
        """会話セッションを宣言する。lease_id を渡すと延長(renew)になる。

        既に**別バックエンド**が保持している場合は BusyError(409)。同一バックエンドの
        取得は lease_id 無しでも新しいリースとして受け付ける(会話アプリが再起動した
        ケースを救うため — 同じ持ち主なら奪い合いにならない)。
        """
        if backend_name not in BACKENDS:
            raise ValidationError(
                f"未知のバックエンドです: {backend_name!r}(有効: {sorted(BACKENDS)})")
        ttl = LEASE_TTL_DEFAULT_S if ttl_s is None else float(ttl_s)
        if not LEASE_TTL_MIN_S <= ttl <= LEASE_TTL_MAX_S:
            raise ValidationError(
                f"ttl_s は {LEASE_TTL_MIN_S}〜{LEASE_TTL_MAX_S} 秒の範囲で指定してください: {ttl_s!r}")
        now = time.time()
        with self._lock:
            current = self._lease if (self._lease and self._lease.alive()) else None
            if current is not None and current.backend != backend_name:
                raise BusyError(
                    f"バックエンド {current.backend} が会話セッション中です"
                    f"(残り {current.expires_at - now:.0f}秒)。終了を待ってください")
            if current is not None and (lease_id is None or lease_id == current.lease_id):
                current.expires_at = now + ttl
                current.renewals += 1
                return {"result": "renewed", **current.to_dict()}
            lease = RealtimeLease(
                lease_id=uuid.uuid4().hex[:16], backend=backend_name,
                created_at=now, expires_at=now + ttl)
            self._lease = lease
            logger.info("リアルタイムリース取得: %s (ttl=%.0fs, id=%s)",
                        backend_name, ttl, lease.lease_id)
            return {"result": "acquired", **lease.to_dict()}

    def release_lease(self, lease_id: str | None = None) -> dict:
        """会話セッション終了。lease_id 指定時は一致するときだけ解放する
        (古いターンの release が新しいターンのリースを消さないようにするため)。"""
        with self._lock:
            lease = self._lease
            if lease is None or not lease.alive():
                self._lease = None
                return {"result": "no-op", "detail": "有効なリースはありません"}
            if lease_id is not None and lease_id != lease.lease_id:
                return {"result": "no-op",
                        "detail": f"lease_id が一致しないため解放しません"
                                  f"(現在の保持者: {lease.backend})"}
            self._lease = None
            logger.info("リアルタイムリース解放: %s (id=%s)", lease.backend, lease.lease_id)
            return {"result": "released", "backend": lease.backend,
                    "lease_id": lease.lease_id}

    def lease_info(self) -> dict | None:
        with self._lock:
            return self._lease_info_locked()

    def _lease_info_locked(self) -> dict | None:
        if self._lease is None:
            return None
        if not self._lease.alive():
            logger.info("リアルタイムリース期限切れ: %s (id=%s)",
                        self._lease.backend, self._lease.lease_id)
            self._lease = None
            return None
        return self._lease.to_dict()

    def _lease_blocker_locked(self, backend_name: str) -> str | None:
        """このバックエンドの投入を止めるリース保持者。無ければ None。"""
        info = self._lease_info_locked()
        if info is None or info["backend"] == backend_name:
            return None
        return info["backend"]

    def realtime_blocker(self, backend_name: str) -> str | None:
        """**優先ゲート**: 他バックエンドが会話セッション中ならその名前を返す。

        `other_busy()` が「いま生成中か」の事後判定なのに対し、こちらは
        「これから会話が続く」という事前宣言。H3 のような長尺ジョブは一度始まると
        止められないので、入口で断るにはこちらが必要になる。
        """
        with self._lock:
            return self._lease_blocker_locked(backend_name)


def _proc_start_time(pid: int) -> float | None:
    """/proc/<pid> の作成時刻から起動時刻を推定(adopt 用、精度は秒で十分)。"""
    try:
        return os.stat(f"/proc/{pid}").st_mtime
    except OSError:
        return None


manager = ProcessManager()
