"""Inductor 誤コンパイルの局所化プローブ。

probe_compile_customop.py の結論: aot_eager は bit 一致 / Inductor は
emulate_precision_casts でも cosine 0.97 → codegen 段の実バグ。
ここでは block 0 に実入力(フル forward から hook で捕獲)を流し、
  1. block 全体 compile での出力偏差
  2. サブモジュール(attn1/attn2/ff 等)単位で compile した場合の偏差
を測って犯人を特定する。
"""
import json
import sys
import time
from typing import Optional

sys.path.insert(0, "/home/animede/diffusers-movie-server/backends/ltx2_5")

import torch
from torch.library import custom_op, register_fake

MODEL_DIR = "/home/animede/diffusers-movie-server/backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit"

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


NVFP4Linear.forward = forward_via_op

ckpt = hf_hub_download(
    "Lightricks/LTX-2.5",
    "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
)
cfg = json.load(open(f"{MODEL_DIR}/transformer_bnb_4bit/config.json"))
tr = load_nvfp4_transformer(ckpt, cfg, torch.device("cuda"))
print("loaded", flush=True)

B, VID_TOKENS, AUD_TOKENS, TXT = 1, 2304, 126, 128
dev = torch.device("cuda")
g = torch.Generator(device="cuda").manual_seed(0)

def mk(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, generator=g, device=dev, dtype=torch.float32).to(dtype)

kwargs = dict(
    hidden_states=mk(B, VID_TOKENS, cfg["in_channels"]),
    audio_hidden_states=mk(B, AUD_TOKENS, cfg["audio_in_channels"]),
    encoder_hidden_states=mk(B, TXT, cfg["cross_attention_dim"]),
    audio_encoder_hidden_states=mk(B, TXT, cfg["audio_cross_attention_dim"]),
    timestep=torch.full((B, VID_TOKENS), 500.0, device=dev),
    audio_timestep=torch.full((B, AUD_TOKENS), 500.0, device=dev),
    sigma=torch.full((B, 1), 0.5, device=dev),
    audio_sigma=torch.full((B, 1), 0.5, device=dev),
    encoder_attention_mask=torch.ones(B, TXT, device=dev),
    audio_encoder_attention_mask=torch.ones(B, TXT, device=dev),
    num_frames=16, height=9, width=16, fps=24.0,
    audio_num_frames=AUD_TOKENS,
    use_cross_timestep=True,
    return_dict=False,
)
kwargs["video_coords"] = tr.rope.prepare_video_coords(B, 16, 9, 16, dev, fps=24.0)
kwargs["audio_coords"] = tr.audio_rope.prepare_audio_coords(B, AUD_TOKENS, dev)

blk = tr.transformer_blocks[0]
captured = {}

def pre_hook(module, args, kw):
    captured["args"] = args
    captured["kwargs"] = kw

h = blk.register_forward_pre_hook(pre_hook, with_kwargs=True)
with torch.no_grad():
    tr(**kwargs)
h.remove()
cargs, ckw = captured["args"], captured["kwargs"]
print("captured block inputs:", len(cargs), "args,", sorted(ckw.keys()), flush=True)


def cmp(label, ref, out):
    for i, (a, b) in enumerate(zip(ref, out)):
        a, b = a.float(), b.float()
        d = (a - b).abs()
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
        tag = ["video", "audio"][i] if i < 2 else str(i)
        print(f"  {label}[{tag}]: max|d| {d.max().item():.5f} mean|d| {d.mean().item():.7f} "
              f"cosine {cos.item():.6f}", flush=True)


with torch.no_grad():
    ref = blk(*cargs, **ckw)

    # 1) ブロック全体 compile
    torch._dynamo.reset()
    cblk = torch.compile(blk, fullgraph=True)
    out = cblk(*cargs, **ckw)
    cmp("whole-block", ref, out)

    # 2) サブモジュール単位
    names = [n for n, _ in blk.named_children()]
    print("children:", names, flush=True)
    for name in names:
        torch._dynamo.reset()
        orig = getattr(blk, name)
        setattr(blk, name, torch.compile(orig, fullgraph=True))
        try:
            out = blk(*cargs, **ckw)
            cmp(f"compile:{name}", ref, out)
        except Exception as exc:
            print(f"  compile:{name} FAILED: {exc!r}"[:300], flush=True)
        finally:
            setattr(blk, name, orig)

# ---------------------------------------------------------------------------
# SDPA バックエンド固定での切り分け: compile の偏差が「バックエンド選択差」なのか
# 「本物の誤コンパイル」なのか
# ---------------------------------------------------------------------------
from torch.nn.attention import sdpa_kernel, SDPBackend

with torch.no_grad():
    with sdpa_kernel([SDPBackend.MATH]):
        ref_math = blk(*cargs, **ckw)
    cmp("eager-MATH vs eager(auto)", ref, ref_math)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        ref_eff = blk(*cargs, **ckw)
    cmp("eager-EFF  vs eager(auto)", ref, ref_eff)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        try:
            ref_fl = blk(*cargs, **ckw)
            cmp("eager-FLASH vs eager(auto)", ref, ref_fl)
        except Exception as exc:
            print("  eager-FLASH failed:", repr(exc)[:120], flush=True)

    torch._dynamo.reset()
    cblk2 = torch.compile(blk, fullgraph=True)
    with sdpa_kernel([SDPBackend.MATH]):
        out_math = cblk2(*cargs, **ckw)
    cmp("compiled-MATH vs eager-MATH", ref_math, out_math)
    torch._dynamo.reset()
    cblk3 = torch.compile(blk, fullgraph=True)
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
        out_eff = cblk3(*cargs, **ckw)
    cmp("compiled-EFF vs eager-EFF", ref_eff, out_eff)
