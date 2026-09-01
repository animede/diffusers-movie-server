from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_id: str = "Lightricks/LTX-2.5-Diffusers"
    # Pinned default: opt in to newer Hub weights by changing MODEL_REVISION.
    model_revision: str | None = "69009ff070135c693ad1ad1ef2cc149c227963da"
    quantized_model_dir: Path = Path("LTX-2.5-Diffusers-bnb-4bit")
    hf_token: str | None = None
    offload_mode: str = "model"
    output_dir: Path = Path("outputs")
    input_dir: Path = Path("inputs")
    lora_dir: Path = Path("loras")
    max_upload_size_mb: int = 500
    max_queue_size: int = 4
    history_db: Path = Path("outputs/history.sqlite3")
    # OpenAI-compatible chat-completions endpoint used only for prompt rewriting.
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 60.0
    # Decode path after the 2x latent upscale/refine stage: "diffusion"
    # (better fine detail, +~18s with NATTEN) or "vae" (fastest).
    # Non-upscaled jobs always use VAE. Env: LTX25_DECODER
    ltx25_decoder: str = "diffusion"
    # libx264 CRF for all output videos (lower = higher quality). Env: LTX25_VIDEO_CRF
    ltx25_video_crf: int = 18
    # Transformer weights: "nf4" (bnb 4bit, default), "fp8" (bf16-equivalent
    # quality via layerwise casting storage=fp8_e4m3fn / compute=bf16, resident
    # ~18GB / peak ~29GB, for 48GB-class GPUs; requires the ~38GB bf16
    # transformer shards in the HF cache) or "bf16" (release weights, ~38GB,
    # for 96GB-class GPUs) or "nvfp4" (official Blackwell-native FP4 distilled
    # transformer, resident ~19GB, FP4 tensor-core matmul via torch._scaled_mm;
    # requires sm_120+. See app/nvfp4.py). Env: LTX25_TRANSFORMER_PRECISION
    ltx25_transformer_precision: str = "nf4"
    # Optional local path to the ComfyUI-format nvfp4 checkpoint. When unset,
    # hf_hub_download("Lightricks/LTX-2.5", "diffusion_models/ltx-2.5-22b-
    # distilled-transformer-nvfp4.safetensors") resolves it (18.7GB, cached).
    # Env: LTX25_NVFP4_CKPT
    ltx25_nvfp4_ckpt: str | None = None


settings = Settings()
