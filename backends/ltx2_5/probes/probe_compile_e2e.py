"""compile(+CUDA Graph)の E2E 品質 A/B 用の動画生成プローブ。

probe_compile_customop.py / probe_compile_bisect.py の結論:
  - custom_op 化で fullgraph compile は通る(fp4 隠蔽・graph break 解消)
  - しかし Inductor カーネルの微小な系統誤差(ブロックあたり cosine ~0.9999、
    emulate_precision_casts でも消えず、attention を eager 島化しても残る)が
    48 ブロックで累積し、1 forward で cosine ~0.965-0.969 の軌道差になる
  - aot_eager は bit 一致 = 意味論は正しい。よって採否は実映像の品質判定で決める
本プローブは同一 seed の t2av を生成し mp4 を保存する。
  PROBE_MODE=eager   : 基準
  PROBE_MODE=fusion  : custom_op + per-block fullgraph compile + CUDA Graph
  PROBE_MODE=islands : 同上だが attention は eager 島(compile 18s、速度僅差)
"""
import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, "/home/animede/diffusers-movie-server/backends/ltx2_5")

import torch
from torch.library import custom_op, register_fake

MODE = os.environ.get("PROBE_MODE", "eager")
OUT = os.environ.get("PROBE_OUT", f"/tmp/claude-1000/ltx_compile_ab_{MODE}.mp4")
MODEL_DIR = "/home/animede/diffusers-movie-server/backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit"
SIGMAS = [1.0, 0.99609375, 0.9765625, 0.9375, 0.8515625, 0.578125, 0.28125, 0.109375]

from diffusers import LTX2ConditionPipeline
from diffusers.utils import export_to_video
from transformers import Gemma4UnifiedForConditionalGeneration

from app.nvfp4 import NVFP4Linear, load_nvfp4_transformer, nvfp4_quantize, to_blocked
from huggingface_hub import hf_hub_download


@custom_op("ltx25::nvfp4_linear", mutates_args=())
def nvfp4_linear_op(
    x: torch.Tensor, weight_u8: torch.Tensor, weight_scale_blocked: torch.Tensor,
    bias: Optional[torch.Tensor], input_scale: float, out_scale: float,
    out_features: int, in_features: int,
) -> torch.Tensor:
    orig_shape = x.shape
    x2 = x.reshape(-1, in_features)
    if x2.dtype != torch.bfloat16:
        x2 = x2.to(torch.bfloat16)
    M = x2.shape[0]
    pad = (-M) % 16
    if pad:
        x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
    xq, xs = nvfp4_quantize(x2, input_scale)
    out = torch._scaled_mm(
        xq, weight_u8.view(torch.float4_e2m1fn_x2).t(),
        scale_a=to_blocked(xs), scale_b=weight_scale_blocked, out_dtype=torch.bfloat16,
    )
    if pad:
        out = out[:M]
    out = out * out_scale
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], out_features)


@register_fake("ltx25::nvfp4_linear")
def _(x, weight_u8, weight_scale_blocked, bias, input_scale, out_scale, out_features, in_features):
    return x.new_empty(*x.shape[:-1], out_features, dtype=torch.bfloat16)


def forward_via_op(self, x):
    return torch.ops.ltx25.nvfp4_linear(
        x, self.weight.view(torch.uint8), self.weight_scale_blocked, self.bias,
        self.input_scale, self.out_scale, self.out_features, self.in_features)


t0 = time.time()
text_encoder = Gemma4UnifiedForConditionalGeneration.from_pretrained(
    f"{MODEL_DIR}/text_encoder_bnb_4bit", dtype=torch.bfloat16
)
ckpt = hf_hub_download(
    "Lightricks/LTX-2.5",
    "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
)
cfg = json.load(open(f"{MODEL_DIR}/transformer_bnb_4bit/config.json"))
transformer = load_nvfp4_transformer(ckpt, cfg, torch.device("cuda"))
pipe = LTX2ConditionPipeline.from_pretrained(
    MODEL_DIR, text_encoder=text_encoder, transformer=transformer,
    dtype=torch.bfloat16, local_files_only=True,
)
pipe.vae.enable_tiling()
pipe.to("cuda")
print(f"load: {time.time() - t0:.1f}s", flush=True)

if MODE != "eager":
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 256)
    NVFP4Linear.forward = forward_via_op
    fullgraph = True
    if MODE == "islands":
        cls = type(pipe.transformer.transformer_blocks[0].attn1)
        cls.forward = dynamo.disable(cls.forward)
        fullgraph = False
    for i, blk in enumerate(pipe.transformer.transformer_blocks):
        pipe.transformer.transformer_blocks[i] = torch.compile(blk, fullgraph=fullgraph)

    # CUDA Graph(app/cudagraph.py と同じ方式の簡易版)
    sys.path.insert(0, "/home/animede/diffusers-movie-server/backends/ltx2_5")
    from app.cudagraph import ForwardGraphRunner
    runner = ForwardGraphRunner(pipe.transformer, max_captures=4)
    runner.install()

prompt = ("A young woman with shoulder-length dark hair sings into a vintage microphone "
          "on a small stage, warm spotlights, shallow depth of field, photorealistic")


def gen(seed):
    g = torch.Generator("cuda").manual_seed(seed)
    t1 = time.time()
    with torch.no_grad():
        video, audio = pipe(
            prompt=prompt, negative_prompt="",
            width=512, height=288, num_frames=121, frame_rate=24,
            sigmas=SIGMAS, guidance_scale=1.0, audio_guidance_scale=1.0,
            stg_scale=0.0, audio_stg_scale=0.0,
            modality_scale=1.0, audio_modality_scale=1.0,
            generator=g, output_type="np", return_dict=False,
        )
    return video, time.time() - t1


import numpy as np

_, dt1 = gen(12345)  # warmup(compile/capture 込み)
video, dt2 = gen(12345)
print(f"[{MODE}] first gen: {dt1:.2f}s / steady gen: {dt2:.2f}s", flush=True)
arr = np.asarray(video)
if arr.ndim == 5:
    arr = arr[0]
frames = [arr[i] for i in range(arr.shape[0])]
export_to_video(frames, OUT, fps=24)
print(f"[{MODE}] saved {OUT}", flush=True)
