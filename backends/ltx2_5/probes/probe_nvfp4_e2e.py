"""NVFP4 E2E スモーク+速度計測(稼働サーバ非依存のスタンドアロン)。

generator.py の nvfp4 分岐と同一手順でパイプラインを組み、テスト解像度
(512×288)の短尺 t2v を生成する。--precision nf4 で同一seedの比較対象を生成。
"""
import argparse, sys, time, json
sys.path.insert(0, "/home/animede/diffusers-movie-server/backends/ltx2_5")

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--precision", default="nvfp4", choices=["nvfp4", "nf4", "nvfp4-dequant"])
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--frames", type=int, default=121)  # 5s @ 24fps
ap.add_argument("--out", default="")
args = ap.parse_args()

MODEL_DIR = "/home/animede/diffusers-movie-server/backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit"
DISTILLED_SIGMA_VALUES = [1.0, 0.99609375, 0.9765625, 0.9375, 0.8515625, 0.578125, 0.28125, 0.109375]

from diffusers import LTX2ConditionPipeline, LTX2VideoTransformer3DModel
from transformers import Gemma4UnifiedForConditionalGeneration

t0 = time.time()
text_encoder = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    f"{MODEL_DIR}/text_encoder_bnb_4bit", dtype=torch.bfloat16
)
if args.precision.startswith("nvfp4"):
    from app.nvfp4 import load_nvfp4_transformer
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download("Lightricks/LTX-2.5",
                           "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors")
    cfg = json.load(open(f"{MODEL_DIR}/transformer_bnb_4bit/config.json"))
    transformer = load_nvfp4_transformer(ckpt, cfg, torch.device("cuda"), dequant_bf16=args.precision.endswith("dequant"))
else:
    transformer = LTX2VideoTransformer3DModel.from_pretrained(
        f"{MODEL_DIR}/transformer_bnb_4bit", dtype=torch.bfloat16
    )
pipe = LTX2ConditionPipeline.from_pretrained(
    MODEL_DIR, text_encoder=text_encoder, transformer=transformer,
    dtype=torch.bfloat16, local_files_only=True,
)
pipe.vae.enable_tiling()
pipe.to("cuda")
print(f"[{args.precision}] load: {time.time()-t0:.1f}s")

prompt = ("A young woman with shoulder-length dark hair sings into a vintage microphone "
          "on a small stage, warm spotlights, shallow depth of field, photorealistic")
gen = torch.Generator("cuda").manual_seed(args.seed)
torch.cuda.reset_peak_memory_stats()

# ウォームアップ(triton JIT / cuBLAS ヒューリスティクスの初回コストを分離)
t1 = time.time()
with torch.no_grad():
    video, audio = pipe(
        prompt=prompt, negative_prompt="",
        width=512, height=288, num_frames=args.frames, frame_rate=24,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0, audio_guidance_scale=1.0,
        stg_scale=0.0, audio_stg_scale=0.0,
        modality_scale=1.0, audio_modality_scale=1.0,
        generator=gen, output_type="np", return_dict=False,
    )
gen_s = time.time() - t1
peak = torch.cuda.max_memory_allocated() / 1024**3
dur = args.frames / 24
print(f"[{args.precision}] gen1: {gen_s:.1f}s ({gen_s/dur:.2f}x realtime, {dur:.1f}s clip) peak {peak:.1f}GB")

# 2回目(定常速度)
gen2 = torch.Generator("cuda").manual_seed(args.seed)
torch.cuda.reset_peak_memory_stats()
t2 = time.time()
with torch.no_grad():
    video, audio = pipe(
        prompt=prompt, negative_prompt="",
        width=512, height=288, num_frames=args.frames, frame_rate=24,
        sigmas=DISTILLED_SIGMA_VALUES,
        guidance_scale=1.0, audio_guidance_scale=1.0,
        stg_scale=0.0, audio_stg_scale=0.0,
        modality_scale=1.0, audio_modality_scale=1.0,
        generator=gen2, output_type="np", return_dict=False,
    )
gen_s2 = time.time() - t2
peak2 = torch.cuda.max_memory_allocated() / 1024**3
print(f"[{args.precision}] gen2: {gen_s2:.1f}s ({gen_s2/dur:.2f}x realtime) peak {peak2:.1f}GB")

if args.out:
    import numpy as np
    from diffusers.utils import export_to_video
    frames = [np.asarray(f) for f in video[0]]
    export_to_video(frames, args.out, fps=24)
    print("saved:", args.out)
