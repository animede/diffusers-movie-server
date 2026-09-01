"""Phase 2b: scale swizzle(cuBLAS blocked layout)を適用して正当性を再検証。"""
import torch
from safetensors import safe_open

P = "/home/animede/.cache/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/e8dc69fd26150afbfa20351f6bc9ac384257f9fd/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
p = "model.diffusion_model.transformer_blocks.0.attn1.to_q"
dev = "cuda"

def ceil_div(a, b): return -(-a // b)

def to_blocked(m):
    """torchao の to_blocked 相当: [rows, cols] scale 行列 → cuBLAS swizzled flat."""
    rows, cols = m.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded = torch.zeros(n_row_blocks * 128, n_col_blocks * 4, dtype=m.dtype, device=m.device)
    padded[:rows, :cols] = m
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()

with safe_open(P, framework="pt") as f:
    w_packed = f.get_tensor(p + ".weight").to(dev)
    w_scale = f.get_tensor(p + ".weight_scale").to(dev)
    w_scale2 = float(f.get_tensor(p + ".weight_scale_2"))
    in_scale = float(f.get_tensor(p + ".input_scale"))

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=dev)
lo = (w_packed & 0x0F).long(); hi = (w_packed >> 4).long()
vals = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(w_packed.shape[0], -1)
w_deq = vals * w_scale.float().repeat_interleave(16, dim=1) * w_scale2

N, K = w_deq.shape; M = 4096
x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
ref = x @ w_deq.to(torch.bfloat16).t()

def quant_nvfp4(t, gscale):
    tb = t.float().reshape(t.shape[0], -1, 16)
    bs = tb.abs().amax(dim=-1) / (6.0 * gscale)
    bs8 = bs.clamp(min=2**-6).to(torch.float8_e4m3fn)
    bsf = bs8.float().unsqueeze(-1) * gscale
    q = (tb / bsf).clamp(-6, 6)
    idx = (q.unsqueeze(-1) - E2M1).abs().argmin(dim=-1)
    lo = idx[..., 0::2]; hi = idx[..., 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8).reshape(t.shape[0], -1)
    return packed.view(torch.float4_e2m1fn_x2), bs8

xq, xs = quant_nvfp4(x, in_scale)
wq = w_packed.view(torch.float4_e2m1fn_x2)

out = torch._scaled_mm(xq, wq.t(),
                       scale_a=to_blocked(xs), scale_b=to_blocked(w_scale),
                       out_dtype=torch.bfloat16)
out = out * (in_scale * w_scale2)
rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
cos = torch.nn.functional.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
print(f"swizzled scaled_mm: relative error {rel.item():.4f}, cosine {cos.item():.6f}")
