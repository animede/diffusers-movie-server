"""AdaLN modulation precompute for the MiniMax-H3 transformer (bf16 path only).

Ported from NVIDIA Sol-Engine's `sana-sol-engine` repo
(`models/minimax_h3/GB200/adaln.py`, Apache-2.0) with the mechanism kept verbatim --
only the call sites (this project's `MiniMaxH3Runner._ensure_transformer(_ref)`
instead of Sol-Engine's own GB200 loader script) and the module-level env-var/guard
wiring are new.

Why this works (Sol-Engine's own reasoning, cross-checked against this project's
pinned diffusers commit -- see `enable_adaln_precompute()`'s docstring below for the
exact places this was verified against `transformer_minimax_h3.py` /
`modular_pipelines/minimax_h3/denoise.py` in THIS venv, not assumed from upstream):

Roughly 13B of MiniMax-H3's 33B parameters sit in the per-block `adaln_proj`, a
`Linear(2688 -> 6 * 5376 * 3)` that every one of the 50 blocks evaluates on every
denoising step. Its input is the timestep embedding alone: a `(num_timesteps, 2688)`
tensor with a handful of rows that depends on nothing but the sampling schedule, which
this project's own `MiniMaxH3SetTimestepsStep` call fixes before the denoise loop
starts (`row_timestep_plan` is fully built by then -- see `core/runner.py`'s
`generate()`, right before `MiniMaxH3DenoiseStep`/`MiniMaxH3LoopDenoiser` run). So the
whole modulation table for the whole trajectory is knowable up front, and diffusers
recomputes it once per (block, step) anyway -- this just does that computation once,
up front, into a small table, and drops the weights that would otherwise sit
GPU-resident doing nothing else for the rest of the request.

What this buys (Sol-Engine's own measurement, in order of size):

* ~24 GB of GPU memory. ~26 GB of `adaln_proj` weights (50 blocks) are replaced by a
  ~1.5 GB table (50 steps x 50 blocks x up to 9 rows x 32256 values, bfloat16 -- the
  row count in THIS project's case is capped by `MINIMAX_H3_MODALITY_NUM * distinct
  timesteps per step`, see `precompute()`'s own docstring). The bf16 transformer's
  resident footprint drops from ~66.3GB to roughly ~42GB.
* ~26 GB of HBM reads per step eliminated. Each block used to stream ~520MB of weights
  to produce nine rows of output -- a pure bandwidth tax on an operation with
  essentially no arithmetic intensity relative to attention/FFN.

The precompute runs one GEMM per (block, step) at exactly the shapes the reference
`MiniMaxH3AdaLayerNormModulation.forward()` uses, rather than one batched GEMM per
block over all steps -- slower by a few tens of milliseconds, once, in exchange for
values that are bitwise identical to what the unmodified model would have computed.
This technique therefore needs no quality gate: the verification this task ran
(`docs/h3-adaln-precompute-20260826.md`) is an `ffmpeg framemd5` exact-match check,
not an A/B similarity threshold.

v1 scope (see `core/runner.py`'s `H3_ADALN_PRECOMP` block for the enforced guards):
bf16 transformer only (`H3_TRANSFORMER_QUANT=none`), no `H3_LOWVRAM`/`H3_LOWVRAM_GROUP`,
and rejected outright against turbo (`H3_TURBO_LORA` env default, and any per-request
`turbo=True` override -- see `core/settings.py`'s `validate_instant_settings()`). The
turbo-incompatibility is a real structural conflict, not just an unverified
combination: `core/runner.py`'s `apply_turbo_lora()`/`apply_diffusers_turbo_lora()`
wrap every block's `adaln_proj.linear` in a `_TurboLoRALinear` (see
`_turbo_lora_key_map()`'s docstring, `blocks.N.adaln_proj.linear` key), and that
wrapping is toggled per-REQUEST via `_TurboLoRALinear.enabled` on an already-resident
transformer instance (`set_turbo_lora_enabled()`), not fixed at load time. A precompute
table is baked once, from whatever `adaln_proj.linear` behaviour existed when
`precompute()` ran, for one fixed schedule -- it cannot simultaneously serve a
turbo=True request (different LoRA-adjusted modulation, and typically a different
video-schedule shift, see `_apply_turbo_video_shift()`) and a turbo=False request
(plain modulation) off the same table, and `precompute()` deletes `block.adaln_proj`
outright (the LoRA-wrapped `.linear` submodule goes with it), so there is nothing left
to toggle after precompute runs. Serving both would need a per-turbo-state table pair
rebuilt on every toggle, which defeats the "compute once" design this technique exists
for -- out of scope here, guarded instead of silently mishandled.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger("minimax_h3.adaln_precompute")


class _StepCursor:
    """Which denoising step the block stack is currently evaluating.

    The index lives in a device tensor rather than a Python int so that a future
    `torch.compile` pass over the block stack would see one graph for the whole
    trajectory instead of specializing (and recompiling) on every step's Python int.
    This project does not currently `torch.compile` the H3 transformer, but keeping
    this device-tensor shape costs nothing and keeps the port faithful to Sol-Engine's
    own reasoning (see its `_StepCursor` docstring) in case that changes later.
    """

    __slots__ = ("step",)

    def __init__(self, device: torch.device) -> None:
        self.step = torch.zeros((), dtype=torch.long, device=device)

    def set(self, index: int) -> None:
        self.step.fill_(index)


class PrecomputedModulation(nn.Module):
    """Drop-in replacement for `MiniMaxH3AdaLayerNormModulation` that indexes a table.

    Holds `(num_steps, num_rows, 6 * hidden_size)` and returns the six chunks of the
    row block belonging to the current step, matching
    `MiniMaxH3AdaLayerNormModulation.forward()`'s own return shape exactly (see
    `transformer_minimax_h3.py`: `temb.chunk(6, dim=-1)`).
    """

    def __init__(self, table: torch.Tensor, cursor: _StepCursor) -> None:
        super().__init__()
        self.register_buffer("table", table, persistent=False)
        self.cursor = cursor

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # `table[step_tensor]` would be advanced indexing on a data-dependent value,
        # which `torch.compile(fullgraph=True)` rejects (cannot prove the result's
        # static shape). `index_select` with a length-1 index tensor has a statically
        # known output shape, so the step index can stay a runtime value without
        # breaking a future compiled graph. Not exercised today (this project does not
        # compile the H3 transformer), kept for parity with the reference and because
        # it costs nothing over the advanced-indexing form.
        rows = self.table.index_select(0, self.cursor.step.reshape(1))[0]
        return rows.chunk(6, dim=-1)


def _timestep_embedding(transformer, timestep: torch.Tensor) -> torch.Tensor:
    """The `temb` the block stack would have been handed for this step.

    Mirrors `MiniMaxH3Transformer3DModel.forward()`'s own two lines exactly (see
    `transformer_minimax_h3.py`, around `temb = self.time_proj(timestep); temb =
    self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))`) -- the cast
    to `time_embedder`'s own weight dtype matters because this checkpoint's
    `time_embedder` is a float32 module inside an otherwise-bf16 transformer (see that
    method's own comment), and reproducing that cast exactly is part of what makes the
    precomputed table bitwise identical to the uncached path.
    """
    temb = transformer.time_proj(timestep)
    return transformer.time_embedder(temb.to(transformer.time_embedder.linear_1.weight.dtype))


@torch.no_grad()
def precompute(transformer, row_timestep_plan: list) -> dict:
    """Build every block's modulation table for the whole trajectory and free the
    `adaln_proj` weights that produced it.

    `row_timestep_plan` is the pipeline's own `[(timestep, timestep_indices), ...]`
    (`block_state.row_timestep_plan`, built by `MiniMaxH3SetTimestepsStep` before the
    denoise loop starts -- see this module's own docstring). Only the timestep *values*
    are read here; the per-row indices stay with the caller/transformer forward
    unchanged (`adaln_indices` in `MiniMaxH3TransformerBlock.forward()` still indexes
    into this table's rows exactly as it always indexed into the uncached projection's
    output rows -- the row layout is unchanged, only how those rows are produced is).

    Raises if this transformer already has a precompute table installed (call
    `is_precomputed()` first, or just check `H3_ADALN_PRECOMP`'s own `_wanted` flag --
    see `enable_adaln_precompute()`) -- a repeat call would try to read
    `block.adaln_proj.linear` off a block that no longer has a `.linear` submodule
    (`PrecomputedModulation` does not expose one), so failing loudly here beats a
    confusing `AttributeError` deep inside the loop.
    """
    if is_precomputed(transformer):
        raise RuntimeError("AdaLN precompute is already installed on this transformer.")

    device = next(transformer.parameters()).device
    embeddings = [_timestep_embedding(transformer, ts.to(device)) for ts, _ in row_timestep_plan]

    cursor = _StepCursor(device)
    freed_bytes = 0
    table_bytes = 0

    # A step's table has one row per (timestep, modality) pair, and the number of
    # *distinct* timesteps a step carries is not constant: the video and audio
    # schedules run at different shifts (see `core/runner.py`'s `H3_VIDEO_SHIFT`/
    # `H3_AUDIO_SHIFT` module comment) and coincide on some steps, so a plan mixes
    # 1-row and 2-row steps (x `MINIMAX_H3_MODALITY_NUM` = 3 modalities each). Pad every
    # step out to the widest one. A block reads row `timestep_indices *
    # MINIMAX_H3_MODALITY_NUM + token_tags`, which never reaches past that step's own
    # rows (`timestep_indices` only ever indexes into that step's *own* distinct-
    # timestep count), so the padding is written and never read.
    from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

    max_rows = max(int(timestep.numel()) for timestep, _ in row_timestep_plan) * MINIMAX_H3_MODALITY_NUM

    def padded(rows: torch.Tensor) -> torch.Tensor:
        if rows.shape[0] == max_rows:
            return rows
        return torch.cat([rows, rows.new_zeros(max_rows - rows.shape[0], rows.shape[1])])

    for block in transformer.transformer_blocks:
        projection = block.adaln_proj
        # One GEMM per step at the reference's own shape keeps the result bitwise
        # identical to the uncached path (a single batched GEMM over all steps would
        # differ in floating-point summation order/kernel selection for a
        # differently-shaped input -- not assumed safe, not used).
        table = torch.stack([padded(torch.cat(projection(temb), dim=-1)) for temb in embeddings])
        table_bytes += table.numel() * table.element_size()

        for parameter in projection.linear.parameters():
            freed_bytes += parameter.numel() * parameter.element_size()

        block.adaln_proj = PrecomputedModulation(table, cursor)
        del projection

    transformer._h3opt_adaln_cursor = cursor
    torch.cuda.empty_cache()

    stats = {
        "steps": len(row_timestep_plan),
        "blocks": len(transformer.transformer_blocks),
        "table_gb": table_bytes / 1024**3,
        "freed_gb": freed_bytes / 1024**3,
    }
    logger.info(
        "[h3opt.adaln] cached %d blocks x %d steps: table %.2f GB, freed %.2f GB of weights",
        stats["blocks"], stats["steps"], stats["table_gb"], stats["freed_gb"],
    )
    return stats


def is_precomputed(transformer) -> bool:
    """Whether `precompute()` has already replaced this transformer's `adaln_proj`
    modules with `PrecomputedModulation`. Used both by `precompute()`'s own guard and by
    `core/runner.py` to decide whether a fresh load needs (re-)arming (see
    `enable_adaln_precompute()`'s docstring: every full free+reload of the transformer,
    which this project's default `H3_TE_QUANT=bnb-4bit` steady state does around every
    request's decode window, drops this attribute along with the rest of the module and
    needs to re-arm from scratch on the next load).
    """
    return getattr(transformer, "_h3opt_adaln_cursor", None) is not None


def enable_adaln_precompute(transformer) -> None:
    """Arm the precompute on `transformer`: it fires on the first denoise step of the
    next request that drives this transformer instance through
    `MiniMaxH3LoopDenoiser`/`MiniMaxH3Ref2VALoopDenoiser`.

    The schedule is not known until the pipeline has built `row_timestep_plan` (inside
    the denoise loop's own step 0, not at load time -- see this module's own docstring),
    so the actual precompute work is hung off the first iteration of the loop denoiser
    rather than done here. The cursor is advanced from that same patched call, since it
    is the only place that knows the current step index `i`.

    Idempotent per-process for the *class*-level monkeypatch (`MiniMaxH3LoopDenoiser
    .__call__` is only ever wrapped once, tracked via `_h3opt_patched` on the class
    itself -- verified against this project's pinned diffusers commit,
    `diffusers.modular_pipelines.minimax_h3.denoise.MiniMaxH3LoopDenoiser.__call__` and
    `MiniMaxH3Ref2VALoopDenoiser` share the exact same bound method by inheritance,
    neither subclass overrides `__call__` -- so wrapping the base class's `__call__`
    once covers both `transformer` (t2va/fl2va) and `transformer_ref` (ref2va) call
    sites without a second patch). NOT idempotent per-transformer-instance: call this
    again on the same (already-armed-or-precomputed) `transformer` object only after a
    fresh load has replaced it with a brand new instance (`core/runner.py`'s
    `_ensure_transformer()`/`_ensure_transformer_ref()` call this once per fresh load,
    right after the load succeeds and before the request reaches the denoise loop --
    every full free+reload gets a fresh instance with no `_h3opt_adaln_wanted`/
    `_h3opt_adaln_cursor` attributes yet, so re-arming here is always correct, never a
    double-arm of the same instance).
    """
    from diffusers.modular_pipelines.minimax_h3 import denoise as h3_denoise

    if not getattr(h3_denoise.MiniMaxH3LoopDenoiser, "_h3opt_patched", False):
        original_call = h3_denoise.MiniMaxH3LoopDenoiser.__call__

        @torch.no_grad()
        def call_with_precomputed_adaln(self, components, block_state, i: int, t):
            transformer_component = getattr(components, self.transformer_name)
            if getattr(transformer_component, "_h3opt_adaln_wanted", False) and i == 0:
                precompute(transformer_component, block_state.row_timestep_plan)
                transformer_component._h3opt_adaln_wanted = False
            cursor = getattr(transformer_component, "_h3opt_adaln_cursor", None)
            if cursor is not None:
                cursor.set(i)
            return original_call(self, components, block_state, i, t)

        h3_denoise.MiniMaxH3LoopDenoiser.__call__ = call_with_precomputed_adaln
        h3_denoise.MiniMaxH3LoopDenoiser._h3opt_patched = True
        logger.info("MiniMaxH3LoopDenoiser.__call__ patched for AdaLN precompute (covers t2va/fl2va and ref2va)")

    transformer._h3opt_adaln_wanted = True
