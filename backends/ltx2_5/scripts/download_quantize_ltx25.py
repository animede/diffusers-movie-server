#!/usr/bin/env python3
"""Download LTX-2.5 one component at a time and persist NF4 weights.

Each large component gets an isolated Hugging Face cache.  That cache is
removed only after the saved quantized component has successfully reloaded.
The script is safe to rerun: verified components are skipped.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path


REPO_ID = "Lightricks/LTX-2.5-Diffusers"
REVISION = "69009ff070135c693ad1ad1ef2cc149c227963da"
TEMPORAL_REVISION = "871165de037793df7d75c23a02dd7f45fd364c97"
PIXEL_UPSCALER_REPO_ID = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
PIXEL_UPSCALER_FILENAME = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
DEFAULT_OUTPUT = Path("LTX-2.5-Diffusers-bnb-4bit")
DEFAULT_MIN_FREE_GIB = 80
GIB = 1024**3

# The first-stage LTX2Pipeline only needs these files besides the two large models.
# The latent upsampler is fetched separately because it is the optional second
# stage of the high-quality generation path. transformer_full, LoRA and the
# diffusion decoder are intentionally not downloaded.
BASE_ALLOW_PATTERNS = [
    "model_index.json",
    "audio_vae/*",
    "connectors/config.json",
    "connectors/diffusion_pytorch_model.safetensors",
    "duration_head/*",
    "scheduler/*",
    "tokenizer/*",
    "vae/*",
    "vocoder/*",
]

QUALITY_ALLOW_PATTERNS = [
    "latent_upsampler/config.json",
    "latent_upsampler/diffusion_pytorch_model.safetensors",
]

TEMPORAL_ALLOW_PATTERNS = [
    "temporal_latent_upsampler/config.json",
    "temporal_latent_upsampler/diffusion_pytorch_model.safetensors",
]


def log(message: str) -> None:
    print(message, flush=True)


def free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / GIB


def require_free_space(path: Path, minimum_gib: int, stage: str) -> None:
    available = free_gib(path)
    log(f"[{stage}] free disk space: {available:.1f} GiB")
    if available < minimum_gib:
        raise RuntimeError(
            f"Refusing to start {stage}: {available:.1f} GiB free, "
            f"but at least {minimum_gib} GiB is required."
        )


def resolve_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. Set HF_TOKEN or log in on the host before running."
        )
    return token


def marker_path(component_dir: Path) -> Path:
    return component_dir / ".ltx25-nf4-verified.json"


def is_complete(component_dir: Path) -> bool:
    marker = marker_path(component_dir)
    if not marker.is_file():
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    has_weights = any(component_dir.glob("*.safetensors"))
    return (
        state.get("repo_id") == REPO_ID
        and state.get("revision") == REVISION
        and state.get("quantization") == "bitsandbytes-nf4"
        and has_weights
    )


def mark_complete(component_dir: Path, source_component: str) -> None:
    state = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "source_component": source_component,
        "quantization": "bitsandbytes-nf4",
        "compute_dtype": "bfloat16",
    }
    temporary = component_dir / ".ltx25-nf4-verified.json.tmp"
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(marker_path(component_dir))


def remove_dedicated_cache(cache_dir: Path, cache_root: Path) -> None:
    """Remove only a direct child of our private cache root."""
    resolved_root = cache_root.resolve()
    resolved_cache = cache_dir.resolve()
    if resolved_cache.parent != resolved_root or not resolved_cache.name.endswith("-hub-cache"):
        raise RuntimeError(f"Unsafe cache deletion target: {resolved_cache}")
    if resolved_cache.exists():
        log(f"Removing verified component cache: {resolved_cache}")
        shutil.rmtree(resolved_cache)


def release_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def verify_4bit_model(model, component: str) -> None:
    import bitsandbytes as bnb

    quantized_layers = sum(1 for module in model.modules() if isinstance(module, bnb.nn.Linear4bit))
    if quantized_layers == 0:
        raise RuntimeError(f"{component} reloaded, but contains no bitsandbytes Linear4bit layers")
    log(f"[{component}] reload verified ({quantized_layers} Linear4bit layers)")


def normalize_model_index(output_dir: Path) -> None:
    """Remove optional components intentionally omitted from this local snapshot."""
    index_path = output_dir / "model_index.json"
    if not index_path.is_file():
        return
    model_index = json.loads(index_path.read_text(encoding="utf-8"))
    if model_index.pop("diffusion_decoder", None) is not None:
        index_path.write_text(json.dumps(model_index, indent=2) + "\n", encoding="utf-8")


def download_base(output_dir: Path, token: str, minimum_gib: int) -> None:
    from huggingface_hub import snapshot_download

    marker = output_dir / ".ltx25-base-complete.json"
    if marker.is_file():
        normalize_model_index(output_dir)
        log("[base] already complete; skipping")
        return
    require_free_space(output_dir.parent, minimum_gib, "base download")
    output_dir.mkdir(parents=True, exist_ok=True)
    log("[base] downloading only required non-quantized pipeline components")
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        token=token,
        local_dir=output_dir,
        allow_patterns=BASE_ALLOW_PATTERNS,
    )
    required = ("model_index.json", "vae/config.json", "vocoder/config.json")
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Base download incomplete; missing: {', '.join(missing)}")
    normalize_model_index(output_dir)
    marker.write_text(json.dumps({"repo_id": REPO_ID, "revision": REVISION}, indent=2) + "\n")
    local_metadata = output_dir / ".cache" / "huggingface"
    if local_metadata.exists():
        shutil.rmtree(local_metadata)
    log("[base] complete")


def download_quality_components(output_dir: Path, token: str, minimum_gib: int) -> None:
    """Fetch the small second-stage upsampler without retaining a Hub cache."""
    from huggingface_hub import snapshot_download

    config = output_dir / "latent_upsampler" / "config.json"
    weights = output_dir / "latent_upsampler" / "diffusion_pytorch_model.safetensors"
    if config.is_file() and weights.is_file():
        log("[quality] latent upsampler already complete; skipping")
        return
    require_free_space(output_dir.parent, minimum_gib, "quality component download")
    output_dir.mkdir(parents=True, exist_ok=True)
    log("[quality] downloading the second-stage latent upsampler")
    snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        token=token,
        local_dir=output_dir,
        allow_patterns=QUALITY_ALLOW_PATTERNS,
    )
    if not config.is_file() or not weights.is_file():
        raise RuntimeError("Latent upsampler download is incomplete")
    local_metadata = output_dir / ".cache" / "huggingface"
    if local_metadata.exists():
        shutil.rmtree(local_metadata)
    log(f"[quality] complete ({weights.stat().st_size / GIB:.2f} GiB)")


def download_temporal_component(output_dir: Path, token: str, minimum_gib: int) -> None:
    """Fetch the optional x2 temporal latent upsampler."""
    from huggingface_hub import snapshot_download

    component = output_dir / "temporal_latent_upsampler"
    config = component / "config.json"
    weights = component / "diffusion_pytorch_model.safetensors"
    if config.is_file() and weights.is_file():
        log("[temporal] latent upsampler already complete; skipping")
        return
    require_free_space(output_dir.parent, minimum_gib, "temporal component download")
    output_dir.mkdir(parents=True, exist_ok=True)
    log("[temporal] downloading the x2 temporal latent upsampler")
    snapshot_download(
        repo_id=REPO_ID,
        revision=TEMPORAL_REVISION,
        token=token,
        local_dir=output_dir,
        allow_patterns=TEMPORAL_ALLOW_PATTERNS,
    )
    if not config.is_file() or not weights.is_file():
        raise RuntimeError("Temporal latent upsampler download is incomplete")
    local_metadata = output_dir / ".cache" / "huggingface"
    if local_metadata.exists():
        shutil.rmtree(local_metadata)
    log(f"[temporal] complete ({weights.stat().st_size / GIB:.2f} GiB)")


def download_pixel_upscaler(output_dir: Path, cache_root: Path, token: str, minimum_gib: int) -> None:
    """Fetch and validate the official LTX-2.5 x2 Pixel Spatial Upscaler IC-LoRA."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    destination_dir = output_dir / "pixel_spatial_upscaler"
    destination = destination_dir / PIXEL_UPSCALER_FILENAME
    cache_dir = cache_root / "pixel-upscaler-hub-cache"

    def verify(path: Path) -> None:
        with safe_open(path, framework="pt", device="cpu") as weights:
            metadata = weights.metadata() or {}
            keys = list(weights.keys())
        factor = int(metadata.get("reference_downscale_factor", 2))
        if factor != 2 or not keys:
            raise RuntimeError(
                f"Unexpected Pixel Spatial Upscaler checkpoint (factor={factor}, tensors={len(keys)})"
            )

    if destination.is_file():
        verify(destination)
        log("[pixel_upscaler] already verified; skipping")
        remove_dedicated_cache(cache_dir, cache_root)
        return

    require_free_space(output_dir.parent, minimum_gib, "Pixel Spatial Upscaler download")
    cache_dir.mkdir(parents=True, exist_ok=True)
    log("[pixel_upscaler] downloading the official x2 IC-LoRA into its private cache")
    downloaded = Path(hf_hub_download(
        repo_id=PIXEL_UPSCALER_REPO_ID,
        filename=PIXEL_UPSCALER_FILENAME,
        token=token,
        cache_dir=cache_dir,
    ))
    verify(downloaded)
    destination_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    shutil.copy2(downloaded, temporary)
    verify(temporary)
    temporary.replace(destination)
    remove_dedicated_cache(cache_dir, cache_root)
    log(f"[pixel_upscaler] complete ({destination.stat().st_size / GIB:.2f} GiB)")


def quantize_text_encoder(output_dir: Path, cache_root: Path, token: str, minimum_gib: int) -> None:
    import torch
    from transformers import BitsAndBytesConfig, Gemma4UnifiedForConditionalGeneration

    destination = output_dir / "text_encoder_bnb_4bit"
    cache_dir = cache_root / "text-encoder-hub-cache"
    if is_complete(destination):
        log("[text_encoder] already verified; skipping")
        remove_dedicated_cache(cache_dir, cache_root)
        return
    require_free_space(output_dir.parent, minimum_gib, "text encoder download/quantization")
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    log("[text_encoder] downloading into its private cache and quantizing to NF4")
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        revision=REVISION,
        token=token,
        cache_dir=cache_dir,
        quantization_config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    del model
    release_cuda_memory()

    log("[text_encoder] verifying reload from the saved local directory")
    verified = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        destination, dtype=torch.bfloat16, device_map={"": 0}
    )
    verify_4bit_model(verified, "text_encoder")
    del verified
    release_cuda_memory()
    mark_complete(destination, "text_encoder")
    remove_dedicated_cache(cache_dir, cache_root)


def quantize_transformer(output_dir: Path, cache_root: Path, token: str, minimum_gib: int) -> None:
    import torch
    from diffusers import BitsAndBytesConfig, LTX2VideoTransformer3DModel

    destination = output_dir / "transformer_bnb_4bit"
    cache_dir = cache_root / "transformer-hub-cache"
    if is_complete(destination):
        log("[transformer] already verified; skipping")
        remove_dedicated_cache(cache_dir, cache_root)
        return
    require_free_space(output_dir.parent, minimum_gib, "transformer download/quantization")
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    log("[transformer] downloading referenced 4-shard checkpoint and quantizing to NF4")
    model = LTX2VideoTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        revision=REVISION,
        token=token,
        cache_dir=cache_dir,
        quantization_config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination, safe_serialization=True)
    del model
    release_cuda_memory()

    log("[transformer] verifying reload from the saved local directory")
    verified = LTX2VideoTransformer3DModel.from_pretrained(
        destination, dtype=torch.bfloat16, device_map={"": 0}
    )
    verify_4bit_model(verified, "transformer")
    del verified
    release_cuda_memory()
    mark_complete(destination, "transformer")
    remove_dedicated_cache(cache_dir, cache_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-free-gib", type=int, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument(
        "--component",
        choices=("all", "base", "quality", "temporal", "pixel_upscaler", "text_encoder", "transformer"),
        default="all",
        help="Run one stage or all stages in order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    cache_root = output_dir.parent / ".ltx25-component-caches"
    cache_root.mkdir(parents=True, exist_ok=True)
    token = resolve_token()

    log(f"Output: {output_dir}")
    log(f"Pinned source: {REPO_ID}@{REVISION}")
    stages = (
        "base", "quality", "temporal", "pixel_upscaler", "text_encoder", "transformer"
    ) if args.component == "all" else (args.component,)
    try:
        if "base" in stages:
            download_base(output_dir, token, args.min_free_gib)
        if "quality" in stages:
            download_quality_components(output_dir, token, args.min_free_gib)
        if "temporal" in stages:
            download_temporal_component(output_dir, token, args.min_free_gib)
        if "pixel_upscaler" in stages:
            download_pixel_upscaler(output_dir, cache_root, token, args.min_free_gib)
        if "text_encoder" in stages:
            quantize_text_encoder(output_dir, cache_root, token, args.min_free_gib)
        if "transformer" in stages:
            quantize_transformer(output_dir, cache_root, token, args.min_free_gib)
    except Exception as exc:
        log(f"ERROR: {exc}")
        log("The current stage cache was retained so the download can resume.")
        return 1

    log(f"All requested stages completed. Free space: {free_gib(output_dir.parent):.1f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
