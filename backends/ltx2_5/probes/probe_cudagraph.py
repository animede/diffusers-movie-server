"""CUDA Graph capture 可否・数値一致・速度のプローブ(スタンドアロン)。

Lightricks 公式 LTX-2 パッケージの cudagraph_capture.py(ブロックループのみ capture)を
参考にしつつ、こちらは transformer.forward 全体を capture する(パイプラインが
video_coords/audio_coords を事前計算して渡し、STG/CFG 無効の蒸留経路では forward 内に
CPU 依存の分岐が残らないため、より広い境界が狙える)。

手順: 同一 seed で (1) eager 生成 → (2) graph 生成 を1プロセスで実行し、
出力 np 配列の一致(bit)と、denoise 区間・全体の時間を比較する。
"""
import argparse
import json
import sys
import time

sys.path.insert(0, "/home/animede/diffusers-movie-server/backends/ltx2_5")

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--frames", type=int, default=121)
ap.add_argument("--width", type=int, default=512)
ap.add_argument("--height", type=int, default=288)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--warmup-iters", type=int, default=3)
args = ap.parse_args()

MODEL_DIR = "/home/animede/diffusers-movie-server/backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit"
SIGMAS = [1.0, 0.99609375, 0.9765625, 0.9375, 0.8515625, 0.578125, 0.28125, 0.109375]

from diffusers import LTX2ConditionPipeline
from transformers import Gemma4UnifiedForConditionalGeneration

from app.nvfp4 import load_nvfp4_transformer
from huggingface_hub import hf_hub_download

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


# ---------------------------------------------------------------------------
# Graph runner: transformer.forward 全体を (shape/dtype/非テンソル引数) キーで capture
# ---------------------------------------------------------------------------
class ForwardGraphRunner:
    def __init__(self, module: torch.nn.Module, warmup_iters: int = 3):
        self._module = module
        self._orig_forward = module.forward
        self._warmup_iters = warmup_iters
        self._pool = None
        self._captures = {}
        self.replays = 0
        self.captures_done = 0

    def _key(self, kwargs):
        parts = []
        for k in sorted(kwargs):
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                parts.append((k, "T", tuple(v.shape), str(v.dtype)))
            elif isinstance(v, (int, float, str, bool, type(None))):
                parts.append((k, "S", repr(v)))
            else:
                parts.append((k, "O", repr(v)))
        return tuple(parts)

    def __call__(self, *fargs, **kwargs):
        assert not fargs, "positional args not supported (pipeline passes kwargs only)"
        key = self._key(kwargs)
        cap = self._captures.get(key)
        if cap is None:
            cap = self._capture(kwargs)
            self._captures[key] = cap
            self.captures_done += 1
        else:
            for k, static in cap["inputs"].items():
                static.copy_(kwargs[k])
        cap["graph"].replay()
        self.replays += 1
        return cap["outputs"]

    def _capture(self, kwargs):
        statics = {}
        static_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                s = v.clone()
                statics[k] = s
                static_kwargs[k] = s
            else:
                static_kwargs[k] = v
        # warmup on a side stream (JIT/autotune/cuBLAS workspace を capture 外へ)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self._warmup_iters):
                self._orig_forward(**static_kwargs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()
        # warmup で書き換わっていないはずだが公式に倣い入力を再コピー
        for k, s in statics.items():
            s.copy_(kwargs[k])
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        t = time.time()
        with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="global"):
            outputs = self._orig_forward(**static_kwargs)
        print(f"[graph] captured in {time.time() - t:.2f}s "
              f"(key tensors: {len(statics)})", flush=True)
        return {"graph": graph, "inputs": statics, "outputs": outputs}


RUNNER = None  # プロセスで1つ(常駐サーバと同じ定常状態: capture は初回のみ)


def run_once(label, use_graph):
    global RUNNER
    gen = torch.Generator("cuda").manual_seed(args.seed)
    step_times = []

    def on_step(_pipe, step, _ts, cb_kwargs):
        step_times.append(time.time())
        return cb_kwargs

    runner = None
    if use_graph:
        if RUNNER is None:
            RUNNER = ForwardGraphRunner(pipe.transformer, warmup_iters=args.warmup_iters)
        runner = RUNNER
        runner.replays = runner.captures_done = 0
        pipe.transformer.forward = runner
    try:
        torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        with torch.no_grad():
            video, audio = pipe(
                prompt=("A young woman with shoulder-length dark hair sings into a "
                        "vintage microphone on a small stage, warm spotlights, "
                        "shallow depth of field, photorealistic"),
                negative_prompt="",
                width=args.width, height=args.height, num_frames=args.frames,
                frame_rate=24, sigmas=SIGMAS[: args.steps],
                guidance_scale=1.0, audio_guidance_scale=1.0,
                stg_scale=0.0, audio_stg_scale=0.0,
                modality_scale=1.0, audio_modality_scale=1.0,
                generator=gen, output_type="np", return_dict=False,
                callback_on_step_end=on_step,
            )
        total = time.time() - t1
    finally:
        if use_graph:
            pipe.transformer.forward = runner._orig_forward
    denoise = (step_times[-1] - step_times[0]) if len(step_times) > 1 else float("nan")
    peak = torch.cuda.max_memory_allocated() / 1024**3
    extra = ""
    if runner is not None:
        extra = f" captures={runner.captures_done} replays={runner.replays}"
    print(f"[{label}] total={total:.2f}s denoise(step1..N)={denoise:.2f}s "
          f"peak={peak:.1f}GB{extra}", flush=True)
    return video, audio, total, denoise


# 1回目 eager はウォームアップ兼ベースライン(2回目 eager を正式ベースラインに)
run_once("eager-warm", False)
v_e, a_e, te, de = run_once("eager", False)
v_g, a_g, tg, dg = run_once("graph", True)
v_g2, a_g2, tg2, dg2 = run_once("graph-2nd", True)

import numpy as np
v_e = np.asarray(v_e); v_g = np.asarray(v_g)
print(f"video equal(bit): {np.array_equal(v_e, v_g)}  "
      f"max|diff|: {np.max(np.abs(v_e.astype(np.float32) - v_g.astype(np.float32))):.6f}")
if a_e is not None and a_g is not None:
    if isinstance(a_e, torch.Tensor):
        print(f"audio equal(bit): {torch.equal(a_e, a_g)}  "
              f"max|diff|: {(a_e.float() - a_g.float()).abs().max().item():.6f}")
    else:
        a_e = np.asarray(a_e); a_g = np.asarray(a_g)
        print(f"audio equal(bit): {np.array_equal(a_e, a_g)}  "
              f"max|diff|: {np.max(np.abs(a_e.astype(np.float32) - a_g.astype(np.float32))):.6f}")
print(f"speedup denoise: {de:.2f}s -> {dg2:.2f}s ({de / dg2:.2f}x)  "
      f"total: {te:.2f}s -> {tg2:.2f}s", flush=True)
