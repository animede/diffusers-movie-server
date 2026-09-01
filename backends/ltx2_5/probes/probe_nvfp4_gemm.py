"""nvfp4 速度プローブ Phase 1: sm_120 での FP4 GEMM カーネル可用性 + 速度上限測定。

LTX-2.5 transformer の代表的な Linear 形状で
  bf16 matmul vs fp8 scaled_mm vs nvfp4 scaled_mm
を比較する。VRAM は数百MBしか使わない(常駐サーバと共存可)。
"""
import torch
import time

dev = "cuda"
print("torch", torch.__version__, "| device:", torch.cuda.get_device_name(0),
      "| cc:", torch.cuda.get_device_capability(0))

# dtype 可用性
has_f4 = hasattr(torch, "float4_e2m1fn_x2")
print("float4_e2m1fn_x2:", has_f4)

# 代表形状: LTX-2.5 22B video block: hidden 4096, ff 16384? attn to_q [4096,2048]?
# checkpoint 実測: to_q weight U8 [4096, 2048] → 実 in=4096 (packed /2)。
# つまり Linear(in=4096, out=4096)。ff は net.0.proj でもっと大きい。
# トークン数: 512x288x121f ≈ latent seq ~ (お手軽に M=8192 と 32768 の2点)
shapes = [
    (8192, 4096, 4096),   # attn qkv/out 相当
    (8192, 4096, 16384),  # ff up 相当(概算)
    (32768, 4096, 4096),  # hires 相当の長い seq
]

def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def to_fp8_rowwise(t):
    # per-row absmax scale で fp8_e4m3 化
    scale = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    q = (t / scale).to(torch.float8_e4m3fn)
    return q, scale.float()


for M, K, N in shapes:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b = torch.randn(N, K, device=dev, dtype=torch.bfloat16)

    ms_bf16 = bench(lambda: a @ b.t())
    print(f"[{M}x{K}x{N}] bf16 matmul: {ms_bf16:.3f} ms")

    # fp8 scaled_mm(rowwise)
    try:
        aq, asc = to_fp8_rowwise(a)
        bq, bsc = to_fp8_rowwise(b)
        fn8 = lambda: torch._scaled_mm(aq, bq.t(), scale_a=asc, scale_b=bsc.t(),
                                       out_dtype=torch.bfloat16)
        _ = fn8()
        ms_fp8 = bench(fn8)
        print(f"[{M}x{K}x{N}] fp8 scaled_mm: {ms_fp8:.3f} ms (x{ms_bf16/ms_fp8:.2f})")
    except Exception as e:
        print(f"[{M}x{K}x{N}] fp8 scaled_mm: FAILED: {type(e).__name__}: {e}")

    # nvfp4 scaled_mm: packed u8 (e2m1x2) + blockwise fp8 scale (block=16)
    if has_f4:
        try:
            # ランダムなパック済み fp4 データと e4m3 ブロックスケールを直接生成
            a4 = torch.randint(0, 255, (M, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            b4 = torch.randint(0, 255, (N, K // 2), device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
            asc4 = torch.ones(M, K // 16, device=dev, dtype=torch.float8_e4m3fn)
            bsc4 = torch.ones(N, K // 16, device=dev, dtype=torch.float8_e4m3fn)
            fn4 = lambda: torch._scaled_mm(a4, b4.t(), scale_a=asc4, scale_b=bsc4,
                                           out_dtype=torch.bfloat16)
            _ = fn4()
            ms_fp4 = bench(fn4)
            print(f"[{M}x{K}x{N}] nvfp4 scaled_mm: {ms_fp4:.3f} ms (x{ms_bf16/ms_fp4:.2f})")
        except Exception as e:
            print(f"[{M}x{K}x{N}] nvfp4 scaled_mm: FAILED: {type(e).__name__}: {str(e)[:300]}")
    print()
