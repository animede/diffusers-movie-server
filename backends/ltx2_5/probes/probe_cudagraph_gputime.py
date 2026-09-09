"""CUDA イベントで transformer forward の GPU 実行時間を測る(eager vs graph replay)。

probe_cudagraph.py の callback 計測はホスト側ループ時間であり、非同期実行下では
GPU 時間を反映しない(launch-bound なら過大、GPU-bound なら無意味)。ここでは
transformer 単体 + 実物 shape の合成入力で、cuda Event 差分により
  - eager 8 連続 forward の GPU 時間
  - graph replay 8 連続の GPU 時間
を直接比較する。差が小さければ「denoise は GPU 実行そのものが律速で、
CUDA Graph の利得は CPU 起動スラック分のみ」という結論になる。
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

# 512x288x121f 相当の実 shape(パイプライン実測値):
# latent 16x9 spatial(/32)、latent frames 16(121f/8+1)、video tokens=16*9*16=2304
# audio: ~5s -> audio tokens 126(実測近似)。text 128 tokens。
B = 1
VID_TOKENS = 2304
AUD_TOKENS = 126
TXT = 128
D_IN = cfg["in_channels"]
D_AUD = cfg.get("audio_in_channels", 128)
# パイプラインは connector 済みプロンプト埋め込みを渡す(video 4096 / audio 2048)
CAP = cfg["cross_attention_dim"]
CAP_AUD = cfg["audio_cross_attention_dim"]
dev = torch.device("cuda")
g = torch.Generator(device="cuda").manual_seed(0)

def mk(*shape, dtype=torch.bfloat16):
    return torch.randn(*shape, generator=g, device=dev, dtype=torch.float32).to(dtype)

kwargs = dict(
    hidden_states=mk(B, VID_TOKENS, D_IN),
    audio_hidden_states=mk(B, AUD_TOKENS, D_AUD),
    encoder_hidden_states=mk(B, TXT, CAP),
    audio_encoder_hidden_states=mk(B, TXT, CAP_AUD),
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
# 実パイプライン同様、RoPE 座標は forward の外で事前計算して渡す(渡さないと
# forward 内の prepare_*_coords が CPU テンソルを作り、capture 中の H2D copy で
# 落ちる -- 実機で確認。パイプライン経由では常に渡されるため本番は非該当)。
kwargs["video_coords"] = tr.rope.prepare_video_coords(B, 16, 9, 16, dev, fps=24.0)
kwargs["audio_coords"] = tr.audio_rope.prepare_audio_coords(B, AUD_TOKENS, dev)

def gpu_time(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    host0 = time.time()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    host_launch = time.time() - host0
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0, host_launch

with torch.no_grad():
    eager_gpu, eager_launch = gpu_time(lambda: tr(**kwargs))
    print(f"eager : GPU {eager_gpu:.3f}s / host-launch {eager_launch:.3f}s (8 fwd)", flush=True)

    # capture
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
        out = tr(**static)
    graph_gpu, graph_launch = gpu_time(graph.replay)
    print(f"graph : GPU {graph_gpu:.3f}s / host-launch {graph_launch:.3f}s (8 replay)", flush=True)
    print(f"GPU speedup: {eager_gpu / graph_gpu:.2f}x  per-fwd: "
          f"{eager_gpu / 8 * 1000:.1f}ms -> {graph_gpu / 8 * 1000:.1f}ms", flush=True)
