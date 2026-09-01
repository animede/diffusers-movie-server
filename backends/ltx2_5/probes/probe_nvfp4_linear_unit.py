"""NVFP4Linear の単体検証: 実checkpoint層 vs デクオンタイズ参照。

1. triton 量子化カーネル vs naive torch 実装の一致(コード列・スケール)
2. NVFP4Linear.forward vs bf16 デクオンタイズ参照(cosine ~0.995 期待)
3. 任意 M(非16倍数)のパディング動作
4. forward マイクロベンチ(bf16 Linear 比)
"""
import sys, time
sys.path.insert(0, "/home/animede/diffusers-movie-server/backends/ltx2_5")

import torch
from safetensors import safe_open
from app.nvfp4 import NVFP4Linear, nvfp4_quantize, to_blocked, dequantize_nvfp4_weight, FP4_MAX_F as FP4_MAX

P = "/home/animede/.cache/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/e8dc69fd26150afbfa20351f6bc9ac384257f9fd/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
pfx = "model.diffusion_model.transformer_blocks.0.attn1.to_q"
dev = torch.device("cuda")

with safe_open(P, framework="pt") as f:
    w_packed = f.get_tensor(pfx + ".weight")
    w_scale = f.get_tensor(pfx + ".weight_scale")
    w_scale2 = float(f.get_tensor(pfx + ".weight_scale_2"))
    in_scale = float(f.get_tensor(pfx + ".input_scale"))
    bias = f.get_tensor(pfx + ".bias")

# --- 参照: 重みデクオンタイズ ---
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
# checkpoint規約対応済みの正規デクオンタイズ(nibble順+blockedスケール逆写像)
w_deq = dequantize_nvfp4_weight(w_packed, w_scale, w_scale2).to(dev, torch.bfloat16)
N, K = w_deq.shape

# --- 1. triton カーネル vs naive ---
M = 333  # 敢えて非16倍数… ただし quantize 自体は 2D [M, K] ならOK
x = (torch.randn(M, K, device=dev) * 0.4).to(torch.bfloat16)
xq, xs = nvfp4_quantize(x, in_scale)

def naive_quant(t, gscale):
    tb = t.float().reshape(t.shape[0], -1, 16)
    bs = (tb.abs().amax(dim=-1) / (FP4_MAX * gscale)).clamp(min=2**-6)
    bs8 = bs.to(torch.float8_e4m3fn)
    q = (tb / (bs8.float().unsqueeze(-1) * gscale)).clamp(-FP4_MAX, FP4_MAX)
    lut = E2M1.to(t.device)
    idx = (q.unsqueeze(-1) - lut).abs().argmin(dim=-1)
    lo = idx[..., 0::2]; hi = idx[..., 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8).reshape(t.shape[0], -1)
    return packed, bs8

np_packed, np_s8 = naive_quant(x, in_scale)
t_packed = xq.view(torch.uint8)
code_match = (t_packed == np_packed).float().mean().item()
scale_match = (xs.view(torch.uint8) == np_s8.view(torch.uint8)).float().mean().item()
print(f"1) triton vs naive: code一致 {code_match*100:.3f}% / scale一致 {scale_match*100:.3f}%")

# --- 2. NVFP4Linear.forward vs 参照 ---
lin = NVFP4Linear(K, N)
lin.load_quantized(w_packed, w_scale, w_scale2, in_scale, bias, dev)
out = lin(x)
ref = x @ w_deq.t() + bias.to(dev, torch.bfloat16)
cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
print(f"2) forward vs dequant-ref (M={M}): cosine {cos.item():.6f} / rel {rel.item():.4f}")

# 3D入力(transformerの実形状 [B, T, C])
x3 = x.reshape(1, M, K)
out3 = lin(x3)
print(f"3) 3D入力: shape {tuple(out3.shape)} / 2D出力と一致 {torch.equal(out3.reshape(M, N), out)}")

# --- 4. マイクロベンチ ---
M2 = 8192
xb = (torch.randn(M2, K, device=dev) * 0.4).to(torch.bfloat16)
ref_lin = torch.nn.Linear(K, N, bias=True, dtype=torch.bfloat16, device=dev)
def bench(fn, iters=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000
ms_bf16 = bench(lambda: ref_lin(xb))
ms_nvfp4 = bench(lambda: lin(xb))
print(f"4) [{M2}x{K}x{N}] bf16 Linear {ms_bf16:.3f} ms / NVFP4Linear {ms_nvfp4:.3f} ms (x{ms_bf16/ms_nvfp4:.2f})")
