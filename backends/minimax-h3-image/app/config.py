"""MiniMax-H3-Image バックエンドの設定(環境変数 H3IMG_*)。

パスは全てこのバックエンドのディレクトリ(backends/minimax-h3-image)基準で解決する。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent          # backends/minimax-h3-image
REPO_ROOT = BASE_DIR.parent.parent                          # diffusers-movie-server

# H3 動画バックエンドの量子化済み text_encoder(nf4・51層)。H3-Image の TE は H3 と
# ビット一致で、どちらも hidden_states[50] を読むためそのまま流用する(probe で実証済み)。
DEFAULT_TE_DIR = REPO_ROOT / "backends" / "minimax-h3" / "models" / "prequant" / "te_bnb-4bit_prune1"

STEP_PRESETS = {
    # steps → (sample_sigmas の上書き(None=チェックポイント既定の 6 step)、キャッシュ名接尾辞、LoRA 要否)
    6: (None, "", False),
    # pDMD 2-step LoRA(akhaliq/minimax-h3-image-pdmd-2step)。shift 12、σ = 12b/(1+11b)。
    3: ([1.0, 0.96, 0.8571428571428571, 0.0], "_pdmd3", True),
    2: ([1.0, 0.9230769230769231, 0.0], "_pdmd2", True),
}


def _load_dotenv(path: Path) -> None:
    """`.env`(KEY=VALUE、# コメント)を環境変数へ(既存の環境変数が優先)。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


@dataclass
class Settings:
    port: int = int(_env("H3IMG_PORT", "8633"))
    model_id: str = _env("H3IMG_MODEL_ID", "MiniMaxAI/MiniMax-H3-Image")
    # ローカルのスナップショットを直接指す場合(未指定なら HF キャッシュの最新スナップショット)
    model_dir: str = _env("H3IMG_MODEL_DIR", "")
    te_dir: Path = Path(_env("H3IMG_TE_DIR", str(DEFAULT_TE_DIR)))
    # 使う GPU: H3IMG_GPU=1 のように番号で指定(H3IMG_DEVICE=cuda:1 と同じ)。TE は H3IMG_TE_GPU / H3IMG_TE_DEVICE。
    # 既定: transformer は cuda:0、TE は GPU が 2 枚以上あれば cuda:1(engine 側で解決)
    device: str = (f"cuda:{_env('H3IMG_GPU', '')}" if _env("H3IMG_GPU", "") else _env("H3IMG_DEVICE", "cuda:0"))
    te_device: str = (f"cuda:{_env('H3IMG_TE_GPU', '')}" if _env("H3IMG_TE_GPU", "") else _env("H3IMG_TE_DEVICE", ""))
    # VRAM 予算(GB、transformer 側 GPU)。auto(既定)= その GPU の総容量。起動時/ロード時に実際の空きと比べ
    # 小さい方を予算にして、48gb / 32gb / 16gb / 8gb の構成(engine.PROFILES)を自動選択する。
    # 生成中に CUDA OOM が起きたら一段下の構成へ自動で下げて 1 回だけやり直す(H3IMG_VRAM_FALLBACK=0 で無効)。
    vram_gb: str = _env("H3IMG_VRAM_GB", "auto")
    vram_fallback: bool = _env("H3IMG_VRAM_FALLBACK", "1") == "1"
    # 以下 3 つは auto(既定)なら VRAM 予算から決まる。明示すると予算より優先する。
    transformer_quant: str = _env("H3IMG_TRANSFORMER_QUANT", "auto")
    te_stream: str = _env("H3IMG_TE_STREAM", "auto")
    transformer_stream: str = _env("H3IMG_TRANSFORMER_STREAM", "auto")
    # 既定のステップ数(2 / 3 / 6)。リクエストごとに上書き可(transformer の差し替えが走る)
    default_steps: int = int(_env("H3IMG_STEPS", "3"))
    # 起動時にロードする(1)か、初回リクエストまで遅延する(0、既定)
    preload: bool = _env("H3IMG_PRELOAD", "0") == "1"
    lora_path: Path = Path(_env("H3IMG_LORA_PATH",
                                str(BASE_DIR / "models" / "lora" / "minimax_h3_image_pdmd_2step_lora.safetensors")))
    cache_dir: Path = Path(_env("H3IMG_CACHE_DIR", str(BASE_DIR / "cache")))
    output_dir: Path = Path(_env("H3IMG_OUTPUT_DIR", str(BASE_DIR / "outputs")))
    input_dir: Path = Path(_env("H3IMG_INPUT_DIR", str(BASE_DIR / "inputs")))
    max_upload_mb: int = int(_env("H3IMG_MAX_UPLOAD_MB", "200"))
    max_queue: int = int(_env("H3IMG_MAX_QUEUE", "4"))
    # 既定の生成キャンバス: "native" = 最低 4MP で生成して縮小(モデル推奨、1024² でも 2048² 相当の時間)、
    # "requested" = 要求サイズで生成(1024² が 2〜3 倍速い。品質は probe で良好)
    default_generation_area: str = _env("H3IMG_GENERATION_AREA", "requested")
    max_pixels: int = int(_env("H3IMG_MAX_PIXELS", str(4096 * 4096)))
    models_dir: Path = Path(_env("H3IMG_MODELS_DIR", str(BASE_DIR / "models")))  # prompts/ の置き場
    # プロンプト拡張に使う OpenAI 互換 LLM(未設定なら拡張は 501)。gateway 経由なら H3IMG_LLM_* を overrides で渡せる
    llm_base_url: str = _env("H3IMG_LLM_BASE_URL", "")
    llm_model: str = _env("H3IMG_LLM_MODEL", "")
    llm_api_key: str = _env("H3IMG_LLM_API_KEY", "")
    llm_timeout_s: float = float(_env("H3IMG_LLM_TIMEOUT_S", "180"))
    cache_tags: dict = field(default_factory=lambda: {k: v[1] for k, v in STEP_PRESETS.items()})

    def __post_init__(self) -> None:
        if self.default_steps not in STEP_PRESETS:
            raise ValueError(f"H3IMG_STEPS は {sorted(STEP_PRESETS)} のいずれかです: {self.default_steps}")
        if self.te_stream not in ("auto", "0", "1"):
            raise ValueError("H3IMG_TE_STREAM は auto / 0 / 1 のいずれかです")
        if self.transformer_stream not in ("auto", "0", "1"):
            raise ValueError("H3IMG_TRANSFORMER_STREAM は auto / 0 / 1 のいずれかです")
        if self.vram_gb != "auto":
            try:
                float(self.vram_gb)
            except ValueError:
                raise ValueError("H3IMG_VRAM_GB は auto か数値(GB)です") from None
        if self.transformer_quant not in ("auto", "int8wo", "ck-w4a8"):
            raise ValueError("H3IMG_TRANSFORMER_QUANT は auto / int8wo / ck-w4a8 のいずれかです")
        if self.default_generation_area not in ("native", "requested"):
            raise ValueError("H3IMG_GENERATION_AREA は native / requested のいずれかです")
        for d in (self.cache_dir, self.output_dir, self.input_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
