"""バックエンド定義(宣言的カタログ)。

各バックエンドの起動方法・ポート・ヘルス/busy 判定・env プリセットを1か所に集約する。
プリセットの env セットは backends/minimax-h3/README.md「VRAM級別クイックスタート」
「VRAM対応表と主な環境変数(早見表)」、および backends/ltx2_5/app/config.py の
実測構成(docs/phase0-*.md)に基づく。

注意:
- H3_TE_DEVICE=cuda:1 を含む README の起動例は GPU:1 を使うため、プリセットからは
  既定で外してある(必要なら overrides={"H3_TE_DEVICE": "cuda:1"} で明示指定する)。
- overrides はホワイトリスト方式(下記 override_allowed())。任意 env 注入は拒否する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Preset:
    name: str
    env: dict[str, str]
    description: str
    vram_hint: str  # 想定VRAM(実測ベースの目安)


@dataclass(frozen=True)
class BackendDef:
    name: str
    dir: Path                 # run.sh のあるディレクトリ
    run_script: str           # 起動コマンド(dir 相対)
    port: int
    health_path: str          # ヘルスチェック用 GET パス
    health_timeout_s: float   # 起動→ヘルスOKまでの待ちタイムアウト
    presets: dict[str, Preset]
    default_preset: str
    description: str = ""
    toggles: dict[str, dict[str, str]] = field(default_factory=dict)  # 例: turbo

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# ---------------------------------------------------------------------------
# minimax-h3 (port 8631)
# ---------------------------------------------------------------------------

H3_PRESETS = {
    "96gb": Preset(
        name="96gb",
        env={},
        description="既定(env なし)。全モデル常駐、起動時プリロードで常駐 87.3GB。96GB 専有前提。",
        vram_hint="常駐 87.3GB / t2va peak ~92GB",
    ),
    "48gb-lowvram": Preset(
        name="48gb-lowvram",
        # README「48GB級(推奨: 高速化フル)」から H3_TE_DEVICE=cuda:1 を除いたもの
        # (GPU:1 は使わない方針。必要なら overrides で明示)。
        # 注意: H3_KEEP_TRANSFORMER=1 は runner.py のガードにより
        # (H3_TE_DEVICE または H3_TE_PROJ)+ H3_VIDEO_VAE_FP16=1 が必須のため、
        # TE_DEVICE を外したこのプリセットには含められない(Phase 1 実機で確認)。
        # 高速化フルにしたい場合は overrides で H3_TE_DEVICE=cuda:1(または
        # H3_TE_PROJ)+ H3_KEEP_TRANSFORMER=1 を明示すること。
        env={
            "H3_LOWVRAM": "1",
            "H3_TE_PRUNE": "1",
            "H3_VIDEO_VAE_FP16": "1",
        },
        description="48GB級のフェーズ循環 + TE prune + fp16 デコード。README 推奨構成から "
                    "H3_TE_DEVICE=cuda:1 と(それが前提条件の)H3_KEEP_TRANSFORMER=1 を除外"
                    "(overrides で追加可)。",
        vram_hint="t2va peak ~38.9GB",
    ),
    "32gb-group": Preset(
        name="32gb-group",
        env={"H3_LOWVRAM": "group", "H3_TE_PRUNE": "1"},
        description="32GB級以下の block offload(H3_LOWVRAM=group)+ TE prune。",
        vram_hint="t2va peak ~28.7GB(prune 併用で ~17.7GB)",
    ),
    "16gb-proj": Preset(
        name="16gb-proj",
        env={
            "H3_LOWVRAM": "group",
            "H3_TE_PROJ": "NicoLab28/ClipProj-MiniMax-H3",
            "H3_VIDEO_VAE_FP16": "1",
            "H3_ATTN_BACKEND": "default",
        },
        description="16GB 単体向け(投影TE + fp16 デコード + SDPA)。README の 16GB 起動例どおり "
                    "H3_ATTN_BACKEND=default を含む(sm_120 で sage を使うなら overrides で上書き)。",
        vram_hint="peak ~11.4GB",
    ),
}

H3 = BackendDef(
    name="h3",
    dir=REPO_ROOT / "backends" / "minimax-h3",
    run_script="run.sh",
    port=8631,
    health_path="/api/status",
    health_timeout_s=600.0,  # 96gb 既定は起動時プリロードで数分かかる
    presets=H3_PRESETS,
    default_preset="96gb",
    description="MiniMax-H3 動画+音声生成(同期API、/api/status ポーリング)",
    toggles={"turbo": {"H3_TURBO_LORA": "1"}},
)

# ---------------------------------------------------------------------------
# ltx2_5 (port 8632)
# ---------------------------------------------------------------------------

LTX25_PRESETS = {
    "nf4": Preset(
        name="nf4",
        env={},
        description="既定(bnb 4bit transformer、遅延ロード)。",
        vram_hint="peak ~17.2GB",
    ),
    "fp8": Preset(
        name="fp8",
        env={"LTX25_TRANSFORMER_PRECISION": "fp8"},
        description="fp8 layerwise casting(bf16 相当品質)。48GB級向け。",
        vram_hint="常駐 ~18GB / peak ~29GB",
    ),
    "bf16": Preset(
        name="bf16",
        env={"LTX25_TRANSFORMER_PRECISION": "bf16", "OFFLOAD_MODE": "none"},
        description="bf16 リリース重み全常駐(オフロードなし)。96GB級向け。",
        vram_hint="transformer ~38GB + TE/VAE",
    ),
}

# app/config.py の Settings フィールド由来のキーのみ許可(pydantic-settings は
# フィールド名の大文字化を env 名として解決する)。
LTX25_ALLOWED_KEYS = {
    "MODEL_ID", "MODEL_REVISION", "QUANTIZED_MODEL_DIR", "HF_TOKEN",
    "OFFLOAD_MODE", "OUTPUT_DIR", "INPUT_DIR", "LORA_DIR",
    "MAX_UPLOAD_SIZE_MB", "MAX_QUEUE_SIZE", "HISTORY_DB",
    "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "LLM_TIMEOUT_SECONDS",
    "LTX25_PORT",
}

LTX25 = BackendDef(
    name="ltx25",
    dir=REPO_ROOT / "backends" / "ltx2_5",
    run_script="run.sh",
    port=8632,
    health_path="/api/health",
    health_timeout_s=60.0,  # 遅延ロードのため起動は軽い
    presets=LTX25_PRESETS,
    default_preset="nf4",
    description="LTX-2.5 動画+音声生成(非同期ジョブ API、POST /api/jobs)",
)

BACKENDS: dict[str, BackendDef] = {"h3": H3, "ltx25": LTX25}


# ---------------------------------------------------------------------------
# overrides / 組み合わせバリデーション
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """400 に変換されるバリデーションエラー。"""


def override_allowed(backend: str, key: str) -> bool:
    if backend == "h3":
        return key.startswith("H3_")
    if backend == "ltx25":
        return key.startswith("LTX25_") or key in LTX25_ALLOWED_KEYS
    return False


def resolve_env(backend_name: str, preset_name: str | None,
                overrides: dict[str, str] | None,
                toggles: dict[str, bool] | None = None) -> tuple[dict[str, str], str]:
    """プリセット + トグル + overrides を合成した env dict を返す。

    戻り値: (env, 実際に使ったプリセット名)。不正入力は ValidationError。
    """
    backend = BACKENDS.get(backend_name)
    if backend is None:
        raise ValidationError(
            f"未知のバックエンドです: {backend_name!r}(有効: {sorted(BACKENDS)})")

    preset_name = preset_name or backend.default_preset
    preset = backend.presets.get(preset_name)
    if preset is None:
        raise ValidationError(
            f"未知のプリセットです: {preset_name!r}(backend={backend_name}, "
            f"有効: {sorted(backend.presets)})")

    env = dict(preset.env)

    for toggle_name, enabled in (toggles or {}).items():
        if toggle_name not in backend.toggles:
            raise ValidationError(
                f"未知のトグルです: {toggle_name!r}(backend={backend_name}, "
                f"有効: {sorted(backend.toggles)})")
        if enabled:
            env.update(backend.toggles[toggle_name])

    for key, value in (overrides or {}).items():
        if not isinstance(value, str):
            value = str(value)
        if not override_allowed(backend_name, key):
            raise ValidationError(
                f"overrides で許可されていないキーです: {key!r}(backend={backend_name})")
        env[key] = value

    _validate_combination(backend_name, env)
    return env, preset_name


def _validate_combination(backend_name: str, env: dict[str, str]) -> None:
    if backend_name == "h3":
        turbo = env.get("H3_TURBO_LORA") == "1"
        if turbo and env.get("H3_TRANSFORMER_QUANT") == "int8":
            raise ValidationError(
                "turbo(H3_TURBO_LORA=1)と H3_TRANSFORMER_QUANT=int8 は併用できません"
                "(torchao Int8Tensor に aten.cat が無い既知の禁止事項)")
        if turbo and env.get("H3_LOWVRAM") == "group":
            raise ValidationError(
                "turbo(H3_TURBO_LORA=1)と H3_LOWVRAM=group は併用できません"
                "(README 記載の既知の非互換)")
        if env.get("H3_KEEP_TRANSFORMER") == "1":
            # core/runner.py 側の起動時ガードを先取りして 400 にする(実機確認済み):
            # H3_KEEP_TRANSFORMER=1 は LOWVRAM != group かつ(TE_DEVICE または
            # TE_PROJ)かつ VIDEO_VAE_FP16=1 が必須。
            problems = []
            if env.get("H3_LOWVRAM") == "group":
                problems.append("H3_LOWVRAM=group と併用不可")
            if not env.get("H3_TE_DEVICE") and not env.get("H3_TE_PROJ"):
                problems.append("H3_TE_DEVICE か H3_TE_PROJ の指定が必要")
            if env.get("H3_VIDEO_VAE_FP16") != "1":
                problems.append("H3_VIDEO_VAE_FP16=1 が必要")
            if problems:
                raise ValidationError(
                    "H3_KEEP_TRANSFORMER=1 の前提条件を満たしていません: "
                    + " / ".join(problems))
    if backend_name == "ltx25":
        precision = env.get("LTX25_TRANSFORMER_PRECISION")
        if precision is not None and precision not in ("nf4", "fp8", "bf16"):
            raise ValidationError(
                f"LTX25_TRANSFORMER_PRECISION の値が不正です: {precision!r}"
                "(有効: nf4 / fp8 / bf16)")


def catalog() -> list[dict]:
    """GET /api/v1/backends 用のカタログ。"""
    items = []
    for backend in BACKENDS.values():
        items.append({
            "name": backend.name,
            "description": backend.description,
            "port": backend.port,
            "health_path": backend.health_path,
            "default_preset": backend.default_preset,
            "presets": [
                {
                    "name": p.name,
                    "description": p.description,
                    "vram_hint": p.vram_hint,
                    "env": p.env,
                }
                for p in backend.presets.values()
            ],
            "toggles": {k: v for k, v in backend.toggles.items()},
            "overrides_policy": (
                "H3_ プレフィックスのみ" if backend.name == "h3"
                else "LTX25_* / OFFLOAD_MODE / OUTPUT_DIR 等 config.py 由来キーのみ"
            ),
        })
    return items
