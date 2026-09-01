"""diffusion decoder のタイル構成別速度/品質プローブ(1024×576×121f 相当)。

A: 既定タイル(768×768×80f / stride 704×704×56f)= 現行(4タイル)
B: 1タイル化(タイルサイズ >= 出力サイズ)
決定論の decode(同一latent・同一seed)なので出力差はタイル境界ブレンドの有無のみ。
"""
import sys, time
sys.path.insert(0, "/home/animede/diffusers-movie-server/backends/ltx2_5")
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders import LTX2VideoDiffusionDecoderModel
from diffusers.pipelines.ltx2 import LTX2VideoDiffusionDecodePipeline
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoVaeNeighborhoodNattenProcessor,
)

MODEL_DIR = "/home/animede/diffusers-movie-server/backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit"
MODEL_ID = "Lightricks/LTX-2.5-Diffusers"
REV = "69009ff070135c693ad1ad1ef2cc149c227963da"

decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
    MODEL_ID, subfolder="diffusion_decoder", revision=REV, torch_dtype=torch.bfloat16
)
decoder.to("cuda")
decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_DIR, subfolder="scheduler")
pipe = LTX2VideoDiffusionDecodePipeline(diffusion_decoder=decoder, scheduler=scheduler, vae=None)

# 1024×576×121f 相当の latent(spatial 1/32, temporal 1/8, C=128)
lat = torch.randn(1, 128, 16, 18, 32, device="cuda", dtype=torch.bfloat16) * 0.5

def run(tag, **tiling):
    decoder.enable_tiling(**tiling)
    g = torch.Generator("cpu").manual_seed(1)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():  # warmup
        out = pipe(latents=lat, generator=g, output_type="np", return_dict=False, denormalize=False)[0]
    torch.cuda.synchronize(); t0 = time.time()
    g = torch.Generator("cpu").manual_seed(1)
    with torch.no_grad():
        out = pipe(latents=lat, generator=g, output_type="np", return_dict=False, denormalize=False)[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"{tag}: {dt:.2f}s peak {peak:.1f}GB out {out.shape}")
    return out, dt

out_a, ta = run("A 既定タイル(4分割)")
out_b, tb = run("B 1タイル", tile_sample_min_height=1088, tile_sample_min_width=1088,
                tile_sample_min_num_frames=128, tile_sample_stride_height=1024,
                tile_sample_stride_width=1024, tile_sample_stride_num_frames=120)
import numpy as np
a = np.asarray(out_a, dtype=np.float32); b = np.asarray(out_b, dtype=np.float32)
mse = ((a - b) ** 2).mean()
psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float("inf")
print(f"A vs B: PSNR {psnr:.1f}dB / speedup x{ta/tb:.2f}")
