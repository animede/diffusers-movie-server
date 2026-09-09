"""torch.compile(regional=transformer_blocks)× CUDA Graph の GPU 時間プローブ。

probe_cudagraph_gputime.py の結論(graph 単体は GPU 側 1.10x のみ、律速は
小カーネルの GPU 実行オーバーヘッド)を受け、カーネル融合(Inductor)で
per-forward GPU 時間がどこまで縮むかを測る。手順:
  1. eager ベースライン(再掲)
  2. transformer_blocks を個別 torch.compile(diffusers の compile_repeated_blocks
     相当。fullgraph=True で nvfp4 の Triton カーネル・_scaled_mm が trace できるか
     の検証を兼ねる)して GPU 時間
  3. さらに全 forward を CUDA Graph capture して replay の GPU 時間
"""
import json
import sys
import time

sys.path.insert(0, "/home/animede/diffusers-movie-server/backends/ltx2_5")

import torch

MODEL_DIR = "/home/animede/diffusers-movie-server/backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit"

from app.nvfp4 import load_nvfp4_transformer
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download(
    "Lightricks/LTX-2.5",
    "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
)
cfg = json.load(open(f"{MODEL_DIR}/transformer_bnb_4bit/config.json"))
t0 = time.time()
tr = load_nvfp4_transformer(ckpt, cfg, torch.device("cuda"))
print(f"load: {time.time() - t0:.1f}s", flush=True)

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


def gpu_time(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    h0 = time.time()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    hl = time.time() - h0
    torch.cuda.synchronize()
    return s.elapsed_time(e) / 1000.0, hl


with torch.no_grad():
    out_ref = tr(**kwargs)
    eg, el = gpu_time(lambda: tr(**kwargs))
    print(f"eager          : GPU {eg:.3f}s / launch {el:.3f}s", flush=True)

    t1 = time.time()
    # NVFP4Linear(自前 Triton カーネル + float4_e2m1fn_x2 view)を Inductor に
    # 通すと Triton の PassManager::run failed で codegen が落ちる(実機確認)。
    # dynamo.disable で NVFP4Linear.forward を eager 島にし、その周囲の
    # norm/modulation/elementwise 連鎖だけを融合させる。
    import torch._dynamo as dynamo
    from app.nvfp4 import NVFP4Linear

    NVFP4Linear.forward = dynamo.disable(NVFP4Linear.forward)
    dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 256)
    for i, blk in enumerate(tr.transformer_blocks):
        tr.transformer_blocks[i] = torch.compile(blk)
    label = "per-block torch.compile (nvfp4 eager-island)"
    # 初回コンパイル(warmup 1回で発火)
    tr(**kwargs)
    torch.cuda.synchronize()
    print(f"{label}: first-call compile {time.time() - t1:.1f}s", flush=True)

    cg, cl = gpu_time(lambda: tr(**kwargs))
    print(f"compiled eager : GPU {cg:.3f}s / launch {cl:.3f}s", flush=True)

    out_c = tr(**kwargs)
    a, b = out_ref[0].float(), out_c[0].float()
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    print(f"compiled vs eager: max|diff| {d.max().item():.6f} mean {d.mean().item():.8f} "
          f"mean|ref| {a.abs().mean().item():.4f} cosine {cos.item():.6f}", flush=True)

    # compiled + CUDA graph
    static = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            tr(**static)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    torch._C._cuda_clearCublasWorkspaces()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_g = tr(**static)
    gg, gl = gpu_time(graph.replay)
    print(f"compiled+graph : GPU {gg:.3f}s / launch {gl:.3f}s", flush=True)
    print(f"per-fwd: eager {eg/8*1000:.1f}ms -> compiled {cg/8*1000:.1f}ms -> "
          f"compiled+graph {gg/8*1000:.1f}ms ({eg/gg:.2f}x total)", flush=True)
