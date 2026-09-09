"""nvfp4 linear を torch custom_op 化して torch.compile(fullgraph)+ CUDA Graph を試す。

probe_compile_gputime.py の続き。前回の結論:
  - fullgraph=True: Inductor が fp4 dtype 絡みのカーネルを codegen して Triton
    PassManager::run failed でクラッシュ
  - dynamo.disable の eager 島方式: 動くが graph break コストで compiled eager は
    むしろ遅く、かつ出力が cosine 0.968(相対誤差 ~23%)と壊れる
本プローブでは `torch.library.custom_op`("ltx25::nvfp4_linear")で量子化+GEMM を
opaque 化する。Dynamo はカスタム op として1ノードで trace し(graph break なし)、
Inductor は extern call として扱う(fp4 dtype を見ない)ため、両問題を同時に回避
できるはず。数値が eager と整合するか(前回の 23% が graph break 由来か Inductor
の融合バグ由来かの切り分けにもなる)を必ず確認する。
"""
import json
import sys
import time
from typing import Optional

sys.path.insert(0, "/home/animede/diffusers-movie-server/backends/ltx2_5")

import torch
from torch.library import custom_op, register_fake

MODEL_DIR = "/home/animede/diffusers-movie-server/backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit"

from app import nvfp4 as nvfp4_mod
from app.nvfp4 import NVFP4Linear, load_nvfp4_transformer, nvfp4_quantize, to_blocked
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# custom op: NVFP4Linear.forward の本体をそのまま opaque 化
# ---------------------------------------------------------------------------
@custom_op("ltx25::nvfp4_linear", mutates_args=())
def nvfp4_linear_op(
    x: torch.Tensor,
    weight_u8: torch.Tensor,  # float4_e2m1fn_x2 は schema で扱えないため u8 view で渡す
    weight_scale_blocked: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: float,
    out_scale: float,
    out_features: int,
    in_features: int,
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
        xq,
        weight_u8.view(torch.float4_e2m1fn_x2).t(),
        scale_a=to_blocked(xs),
        scale_b=weight_scale_blocked,
        out_dtype=torch.bfloat16,
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


def forward_via_op(self, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.ltx25.nvfp4_linear(
        x,
        self.weight.view(torch.uint8),
        self.weight_scale_blocked,
        self.bias,
        self.input_scale,
        self.out_scale,
        self.out_features,
        self.in_features,
    )


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


def cmp(label, a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0)
    print(f"{label}: max|d| {d.max().item():.6f} mean|d| {d.mean().item():.8f} "
          f"mean|ref| {a.abs().mean().item():.4f} cosine {cos.item():.6f}", flush=True)


with torch.no_grad():
    out_ref = tr(**kwargs)
    eg, el = gpu_time(lambda: tr(**kwargs))
    print(f"eager               : GPU {eg:.3f}s / launch {el:.3f}s", flush=True)

    # 1) custom_op 差し替えのみ(compile なし)-- 等価性の確認
    NVFP4Linear.forward = forward_via_op
    out_op = tr(**kwargs)
    cmp("custom_op vs eager  ", out_ref[0], out_op[0])
    og, ol = gpu_time(lambda: tr(**kwargs))
    print(f"custom_op eager     : GPU {og:.3f}s / launch {ol:.3f}s", flush=True)

    # 2) per-block torch.compile(fullgraph=True)
    # PROBE_BACKEND=aot_eager: Inductor codegen を外して trace/decomposition だけ検証
    # PROBE_EMULATE=1: Inductor に eager 同等の bf16 丸めを強制(fp32 保持による
    #   「高精度側への乖離」なのか codegen バグなのかの切り分け)
    import os
    import torch._dynamo as dynamo
    dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 256)
    backend = os.environ.get("PROBE_BACKEND", "inductor")
    fullgraph = True
    if os.environ.get("PROBE_SKIP_ATTN") == "1":
        # attention モジュールを eager 島化(bisect の結論: 偏差の発生源は attention
        # 内部の融合のみ。norm/FF/変調は bit 一致)。graph break が入るため
        # fullgraph=False。break の CPU コストは CUDA Graph replay で消える。
        attn_classes = {type(tr.transformer_blocks[0].attn1),
                        type(tr.transformer_blocks[0].audio_attn1),
                        type(tr.transformer_blocks[0].audio_to_video_attn)}
        for cls in attn_classes:
            cls.forward = dynamo.disable(cls.forward)
        fullgraph = False
        print(f"attention eager-island: {[c.__name__ for c in attn_classes]}", flush=True)
    if os.environ.get("PROBE_EMULATE") == "1":
        import torch._inductor.config as ind_cfg
        ind_cfg.emulate_precision_casts = True
        print("inductor.emulate_precision_casts = True", flush=True)
    t1 = time.time()
    for i, blk in enumerate(tr.transformer_blocks):
        tr.transformer_blocks[i] = torch.compile(blk, fullgraph=fullgraph, backend=backend)
    tr(**kwargs)
    torch.cuda.synchronize()
    print(f"per-block compile(fullgraph=True): first-call {time.time() - t1:.1f}s", flush=True)

    out_c = tr(**kwargs)
    cmp("compiled vs eager   ", out_ref[0], out_c[0])
    cg, cl = gpu_time(lambda: tr(**kwargs))
    print(f"compiled eager      : GPU {cg:.3f}s / launch {cl:.3f}s", flush=True)

    # 3) compiled + CUDA graph
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
    print(f"compiled+graph      : GPU {gg:.3f}s / launch {gl:.3f}s", flush=True)
    graph.replay(); torch.cuda.synchronize()
    cmp("compiled+graph vs eager", out_ref[0], out_g[0])
    print(f"per-fwd: eager {eg/8*1000:.1f}ms -> custom_op {og/8*1000:.1f}ms -> "
          f"compiled {cg/8*1000:.1f}ms -> compiled+graph {gg/8*1000:.1f}ms "
          f"({eg/gg:.2f}x)", flush=True)
