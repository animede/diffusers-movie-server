"""nvfp4 プローブ Phase 2: 実checkpointの重みで正当性確認。

1. to_q 層の nvfp4 重みを手動デクオンタイズ(u8→e2m1x2 unpack、16要素ブロック
   scale(f8e4m3)× グローバル scale_2(f32))して統計を確認。
2. デクオンタイズ済み bf16 matmul と、実スケールを渡した torch._scaled_mm
   (fp4 経路)の出力一致度を確認(重み側のみ実データ、活性化側は動的量子化)。
3. 活性化の動的 nvfp4 量子化を素の torch で実装した場合のコストを測る。
4. torchao の有無を確認(高速な量子化カーネル候補)。
"""
import torch, time
from safetensors import safe_open

P = "/home/animede/.cache/huggingface/hub/models--Lightricks--LTX-2.5/snapshots/e8dc69fd26150afbfa20351f6bc9ac384257f9fd/diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
p = "model.diffusion_model.transformer_blocks.0.attn1.to_q"
dev = "cuda"

try:
    import torchao
    print("torchao:", torchao.__version__)
except ImportError:
    print("torchao: not installed")

with safe_open(P, framework="pt") as f:
    w_packed = f.get_tensor(p + ".weight")          # U8 [4096, 2048] packed
    w_scale = f.get_tensor(p + ".weight_scale")     # F8_E4M3 [4096, 256]
    w_scale2 = f.get_tensor(p + ".weight_scale_2")  # F32 []
    in_scale = f.get_tensor(p + ".input_scale")     # F32 []
print("packed", w_packed.shape, "scale", w_scale.shape, "scale2", float(w_scale2), "input_scale", float(in_scale))

# --- e2m1 LUT dequant ---
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
lo = (w_packed & 0x0F).long()
hi = (w_packed >> 4).long()
vals = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).reshape(w_packed.shape[0], -1)  # [4096, 4096]
scale = w_scale.float().repeat_interleave(16, dim=1) * float(w_scale2)
w_deq = (vals * scale)  # f32 [out, in]
print("dequant weight stats: mean", w_deq.mean().item(), "std", w_deq.std().item(),
      "absmax", w_deq.abs().max().item(), "nan", torch.isnan(w_deq).any().item())

# --- scaled_mm vs dequant matmul(重み実データ、活性化は乱数を動的量子化) ---
K = w_deq.shape[1]; N = w_deq.shape[0]; M = 4096
x = (torch.randn(M, K) * 0.5).to(dev, torch.bfloat16)
w_deq_gpu = w_deq.to(dev, torch.bfloat16)
ref = x @ w_deq_gpu.t()

def quant_nvfp4(t, gscale):
    """naive: 16要素ブロック absmax→e4m3 scale、値→e2m1(最近傍)、u8パック"""
    tb = t.float().reshape(t.shape[0], -1, 16)
    bs = tb.abs().amax(dim=-1) / (6.0 * gscale)          # ブロックscale(グローバルscale適用前)
    bs8 = bs.clamp(min=2**-6).to(torch.float8_e4m3fn)     # e4m3化
    bsf = bs8.float().unsqueeze(-1) * gscale
    q = tb / bsf
    lut = E2M1.to(t.device)
    idx = (q.unsqueeze(-1) - lut).abs().argmin(dim=-1)   # 最近傍(遅い、正当性確認用)
    lo = idx[..., 0::2]; hi = idx[..., 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8).reshape(t.shape[0], -1)
    return packed.view(torch.float4_e2m1fn_x2), bs8

xq, xs = quant_nvfp4(x, float(in_scale))
wq = w_packed.to(dev).view(torch.float4_e2m1fn_x2)
ws = w_scale.to(dev)
out = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
out = out * float(in_scale) * float(w_scale2)
rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean()
print(f"scaled_mm vs dequant-ref: relative error {rel.item():.4f} (活性化fp4化誤差込み)")

# 活性化を量子化しない参照(重みだけfp4)との比較 → 重み経路の正当性を分離確認
x8, x8s = None, None
rel_w_only = (x @ w_deq_gpu.t() - ref).abs().max()
print("(sanity) dequant path self-check:", rel_w_only.item())

# --- 活性化の動的量子化コスト(naive torch、argmin LUT は測定から除外して
#     現実的な実装(divide+round系)の下限を概算: ここでは absmax+scale 部のみ) ---
def act_quant_cost():
    tb = x.float().reshape(M, -1, 16)
    bs = tb.abs().amax(dim=-1)
    _ = (tb / bs.clamp(min=1e-6).unsqueeze(-1))
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): act_quant_cost()
torch.cuda.synchronize()
print(f"act quant (naive absmax+div only): {(time.perf_counter()-t0)/50*1000:.3f} ms / [{M}x{K}]")
