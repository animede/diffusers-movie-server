"""
Instant-apply per-request settings resolution + reload-group settings application.

Split out of core/runner.py to keep the two concerns (huge, already-dense VRAM
choreography file vs. small, mostly-validation glue) apart. Imports runner module
globals lazily (inside functions) rather than at module load time, since several of
them (H3_CACHE, H3_ATTN_BACKEND, H3_TURBO_LORA, and the whole reload-group set) are
mutated in place by `apply_reload_settings()` below -- a `from core.runner import
H3_CACHE` at this module's own import time would bind a stale local name that never
sees later reassignments (Python name binding, not a reference into the other
module's namespace). Every read in this file goes through `core.runner.<NAME>` (module
attribute access) instead, exactly like `runner.py`'s own request-time code already
does implicitly for its top-level globals -- see `apply_reload_settings()`'s docstring
for why this matters for the reload group specifically.

Two independent groups (see the task brief this file implements):

- Instant (per-request, no reload): FirstBlockCache on/off + threshold, attention
  backend, turbo LoRA on/off. Resolved per-request by `resolve_instant_settings()`
  (falls back to whatever the process-wide env-var defaults are when a request field is
  left unset) and applied by `MiniMaxH3Runner.apply_instant_settings()`.
- Reload (process-wide, needs every big model dropped and reloaded under the new
  config): transformer int8, TE quant, TE layer prune, lowvram mode, video VAE fp16,
  projected TE (te_proj, 4B+learned linear map replacing the 32B TE) on/off and its own
  quantization (te_proj_quant). Changed via `apply_reload_settings()`, which validates
  the new combination using the exact same rules `core/runner.py`'s own import-time
  validation uses (kept in sync by hand -- see that function's docstring), then unloads
  everything and calls `runner.preload_all()` under the new config.
"""
from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("minimax_h3.settings")

# Guards apply_reload_settings() so two concurrent /api/settings/apply calls cannot
# interleave their unload/reload sequences (the app-level generation lock already
# keeps this from overlapping with an actual generate() call -- see app.py -- but two
# settings-apply calls racing each other is a separate, narrower race this lock alone
# closes cheaply without complicating app.py's own lock semantics).
_reload_lock = threading.Lock()

INSTANT_CACHE_CHOICES = ("fbc", "none")
# ck-w4a8 (comfy-kitchen W4A8) は 2026-10-07 の A/B を経て本採用になった方式
# (core/runner.py:883 のランタイム受理値と一致させる)。runner と同じ制約として
# H3_PRUNED=1 の起動構成が前提 (apply_reload_settings() で検証する)。
RELOAD_TRANSFORMER_QUANT_CHOICES = ("none", "int8", "ck-w4a8")
RELOAD_TE_QUANT_CHOICES = ("none", "bnb-4bit")
RELOAD_LOWVRAM_CHOICES = ("0", "1", "group")
# 投影TE (H3_TE_PROJ, Qwen3-VL-4B + 学習済み線形写像) 自身の量子化。既定は
# core/runner.py の H3_TE_PROJ_QUANT と同じ "bnb-4bit" (2026-08-10 実測: 相対RMS
# 0.61〜0.96%、cosine 1.0000、動画劣化なし)。
RELOAD_TE_PROJ_QUANT_CHOICES = ("none", "bnb-4bit", "bnb-8bit")

# ---- turbo LoRA 構成プリセット (再ロードグループ) -------------------------------
# turbo の on/off 自体は instant (リクエスト単位) だが、「どの LoRA を使うか」
# (repo/file/scale/renoise/steps 既定) は従来 env 起動時固定だった。検証済みの
# 組み合わせだけをプリセットとして選択可能にする (生の repo/file を UI から
# 自由入力させると scale/shift/renoise の整合が壊れるため、curated 方式)。
#   - lightx2v: ref2v 4step v1.0 専用 LoRA (2026-10-22 A/B で最良) + base(fl2va 待機)は
#     fl2v 4step 768p (video shift 6 自動)。scale はチェックポイント metadata から自動解決。
#   - pika-flex: Pika H3 Flex (ref2va/fl2va アダプタ + renoise サンプラー、7step 既定、
#     scale 1.0)。2026-10-30 に「バランス」として採用された構成。
# HyperFlow (H3_HYPERFLOW) は H3_PRUNED と排他のためここには含めない (プリセット/
# 再起動でのみ切替可能)。
RELOAD_TURBO_PRESET_CHOICES = ("lightx2v", "pika-flex")
TURBO_LORA_PRESETS = {
    "lightx2v": {
        "repo": "lightx2v/Minimax-h3-Turbo",
        "file": "minimax_h3_ref2v_turbo_4step_v1.0_768p_bf16.safetensors",
        "repo_base": "lightx2v/Minimax-h3-Turbo",
        "file_base": "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
        "scale_raw": "",
        "pika_renoise": False,
        "steps_default": 4,
    },
    "pika-flex": {
        "repo": "Pika-Labs/Pika-H3-Flex",
        "file": "ref2va/adapter.safetensors",
        "repo_base": "Pika-Labs/Pika-H3-Flex",
        "file_base": "fl2va/adapter.safetensors",
        "scale_raw": "1.0",
        "pika_renoise": True,
        "steps_default": 7,
    },
}


def _turbo_state_tuple(runner) -> tuple:
    return (
        runner.H3_TURBO_LORA_REPO,
        runner.H3_TURBO_LORA_FILE,
        runner.H3_TURBO_LORA_REPO_BASE,
        runner.H3_TURBO_LORA_FILE_BASE,
        runner.H3_TURBO_LORA_SCALE_RAW,
        runner.H3_PIKA_RENOISE,
        runner.H3_TURBO_STEPS_DEFAULT,
    )


def _preset_state_tuple(preset: dict) -> tuple:
    return (
        preset["repo"],
        preset["file"],
        preset["repo_base"],
        preset["file_base"],
        preset["scale_raw"],
        preset["pika_renoise"],
        preset["steps_default"],
    )


def current_turbo_preset() -> str:
    """現在の runner グローバル群がどのプリセットと一致するか。env で独自構成が
    組まれている場合は "custom" (UI は変更しない限りそのまま維持する)。"""
    import core.runner as runner

    current = _turbo_state_tuple(runner)
    for name, preset in TURBO_LORA_PRESETS.items():
        if current == _preset_state_tuple(preset):
            return name
    return "custom"


def resolve_instant_settings(
    cache: str | None,
    cache_threshold: float | None,
    attn: str | None,
    turbo: bool | None,
) -> dict:
    """Fill in any unset instant-apply field from the current process-wide default
    (whatever core.runner's own H3_CACHE/H3_CACHE_THRESHOLD/H3_ATTN_BACKEND/
    H3_TURBO_LORA env vars resolved to, possibly since changed by a prior
    apply_reload_settings() call for cache_threshold's H3_CACHE_THRESHOLD -- that one is
    actually instant-group by nature but is read from the same module attribute either
    way). A request that omits every one of these fields gets back exactly today's
    server-default behaviour, unchanged.

    Validates `cache`/`attn` against known choices and raises `ValueError` (mapped to
    400 by the caller) rather than silently passing through an unrecognised string to
    `enable_cache`/`set_attention_backend`, where the failure would be much harder to
    attribute to the request.
    """
    import core.runner as runner

    if cache is None:
        cache = runner.H3_CACHE if not runner.H3_TURBO_LORA else "none"
    cache = cache.strip().lower()
    if cache not in INSTANT_CACHE_CHOICES:
        raise ValueError(f"cache must be one of {INSTANT_CACHE_CHOICES}, got {cache!r}")

    if cache_threshold is None:
        cache_threshold = runner.H3_CACHE_THRESHOLD
    cache_threshold = float(cache_threshold)
    if not (0.0 <= cache_threshold <= 1.0):
        raise ValueError(f"cache_threshold must be between 0.0 and 1.0, got {cache_threshold}")

    if attn is None:
        attn = runner.H3_ATTN_BACKEND or "default"
    attn = attn.strip().lower()
    if attn in ("", "none"):
        attn = "default"

    if turbo is None:
        turbo = runner.H3_TURBO_LORA
    turbo = bool(turbo)

    if getattr(runner, "H3_HYPERFLOW", False):
        # HyperFlow(8step flow-map 蒸留)構成では turbo と FBC を強制無効にする:
        # turbo の _TurboLoRALinear は HyperFlow の PEFT アダプタと同じ Linear 群へ
        # 二重適用になり、FBC は蒸留短工程に安全な窓が無い(turbo と同じ理屈)。
        turbo = False
        cache = "none"

    # A handful of turbo steps (4-8) leaves no redundant-computation window for FBC's
    # residual-similarity skip to safely exploit, and caching on top of an
    # already-short trajectory risks compounding drift for no measured benefit -- same
    # reasoning `H3_TURBO_LORA`'s own module comment in core/runner.py gives for why
    # the old load-time-only wiring force-disabled FBC whenever turbo was on. Preserved
    # here for the instant-apply path: `cache` is force-downgraded to "none" whenever
    # `turbo` resolves True, regardless of what the request asked for, and the
    # `effective_cache`/`cache_forced_off_by_turbo` fields let callers report this
    # back to the client instead of silently ignoring their `cache` request.
    cache_forced_off_by_turbo = turbo and cache == "fbc"
    effective_cache = "none" if turbo else cache

    resolved = {
        "cache": cache,
        "cache_threshold": cache_threshold,
        "attn": attn,
        "turbo": turbo,
        "effective_cache": effective_cache,
        "cache_forced_off_by_turbo": cache_forced_off_by_turbo,
    }
    validate_instant_settings(resolved)
    return resolved


def validate_instant_settings(resolved: dict) -> None:
    """Reject combinations confirmed NOT to work together, mirroring the exact same
    rules `core/runner.py` enforces at import time for the equivalent env vars
    (H3_TURBO_LORA vs H3_LOWVRAM_ANY/H3_TRANSFORMER_BOTH_RESIDENT, see that module's own
    comment on the `H3_TURBO_LORA` block) -- except checked per-request here, since
    turbo is now an instant (not process-wide-fixed) setting and the *current* reload
    group config can be anything by the time a turbo=True request arrives.

    Actually verified (not just "unverified") by this task's own A/B run, 2026-08-06:
    turbo=1 + (lowvram=1 or lowvram=group or transformer_both_resident, i.e. any path
    where `H3_TRANSFORMER_QUANT=int8`) reproducibly fails with `NotImplementedError:
    Int8Tensor dispatch: attempting to run unimplemented operator/function:
    func=<OpOverload(op='aten.cat', overload='default')>` -- `apply_turbo_lora()`'s
    call to `attn.fuse_projections()` (`core/runner.py`) does `torch.cat([to_q.weight,
    to_k.weight, to_v.weight])` to build the fused `to_qkv` Linear the checkpoint's LoRA
    targets, and torchao's `Int8Tensor` (`H3_INT8_MODULES_TO_NOT_CONVERT` does not skip
    `to_q`/`to_k`/`to_v`, so they ARE quantized under `H3_TRANSFORMER_QUANT=int8`) has no
    registered `aten.cat` kernel (grepped `torchao/quantization/quantize_/workflows/
    int8/int8_tensor.py`'s `@implements(...)` list -- no `aten.cat` entry). This is not
    an offload-hook-ordering problem (group mode's "wrap LoRA before enable_group_
    offload()" fix that helped a sibling project, diffusers-server CLAUDE.md #44, does
    NOT apply here): the failure happens inside `fuse_projections()` itself, before any
    group offload hook is ever consulted, so reordering the two calls cannot help --
    the base weights themselves are the wrong tensor subclass for `torch.cat`. Confirmed
    identical for both `H3_LOWVRAM=1` and `H3_LOWVRAM=group` (both force
    `H3_TRANSFORMER_QUANT=int8`, same `modules_to_not_convert` recipe as plain
    `transformer_both_resident` int8 mode) -- one 500 response with the exact same
    `Int8Tensor`/`aten.cat` error text from each, request rejected cleanly with no VRAM
    leak or lasting corruption (confirmed by a follow-up plain, non-turbo generation
    succeeding right after on the same still-running server).
    """
    import core.runner as runner

    # 2026-08-08 更新: この拒否が必要なのは **comfy 形式 (融合QKV、Ostris 版) の LoRA の
    # ときだけ**になった。既定になった diffusers ネイティブ形式 (lightx2v 版) は
    # fuse_projections() を呼ばないため上記の torch.cat 問題を踏まず、int8/低VRAM でも
    # 適用できる (スパイク実測済み、README「lightx2v 版」節)。形式は
    # `runner.turbo_lora_expected_format()` (キャッシュ済みなら実ファイルのキー、
    # 未DLならリポジトリ名) で判定する。予備判定が外れた場合も
    # `_apply_turbo_lora_checkpoint()` の実ファイルチェックが 400 で拒否する。
    # group offload は LoRA の形式を問わず拒否: enable_group_offload() の
    # cpu_param_dict は有効化時点で固定され、後から追加される _TurboLoRALinear の
    # lora_a/lora_b バッファが offload サイクルから欠落するリスクがある (未実測の
    # まま解禁しない。runner.py の import 時ガードと同じ理由 -- レビュー指摘)。
    # 2026-11-17: 自作ストリーマ(H3_GROUP_STREAM=own)で、起動時(H3_TURBO_LORA=1)にストリーマ構築前へ
    # LoRA を巻いてある場合は、lora_a / lora_b もストリーマの転送対象に入っているので通す。
    if resolved["turbo"] and runner.H3_LOWVRAM_GROUP and not (runner.H3_GROUP_STREAM == "own" and runner.H3_TURBO_LORA):
        raise ValueError(
            "turbo=1 と lowvram=group は H3_GROUP_STREAM=own かつ起動時 H3_TURBO_LORA=1 のときだけ使えます "
            "(ブロックストリーマの転送対象は構築時に固定されるため、ロード後に LoRA を巻けない)。"
            "diffusers の group offload 経路は従来どおり拒否: "
            "turbo=1 と lowvram=group は併用できません (group offload の cpu_param_dict "
            "は有効化時点で固定されるため、後から追加される LoRA バッファが offload "
            "サイクルから欠落するリスクがある -- 未検証)。lowvram=1 なら turbo を使えます。"
        )
    if (
        resolved["turbo"]
        and (runner.H3_LOWVRAM_ANY or runner.H3_TRANSFORMER_BOTH_RESIDENT)
        and runner.turbo_lora_expected_format() == "comfy"
    ):
        raise ValueError(
            "turbo=1 with the comfy-format (fused-QKV) LoRA is only supported against "
            "the default transformer path (transformer_quant=none, lowvram=0): "
            "apply_turbo_lora()'s fuse_projections() call does torch.cat() on the "
            "transformer's to_q/to_k/to_v weights, and those are torchao Int8Tensor "
            "under transformer_quant=int8 -- Int8Tensor has no aten.cat kernel, so this "
            "reproducibly raises NotImplementedError. Use the default diffusers-native "
            "LoRA (lightx2v/Minimax-h3-Turbo) or change the reload-group settings."
        )

    # H3_ADALN_PRECOMP + turbo=True (v2 coexistence, core/adaln_precompute.py's module
    # docstring has the full derivation/verification): NOT a blanket conflict any more.
    # The default diffusers-native turbo LoRA (H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-
    # Turbo) never wraps `adaln_proj` at all (verified against every non-comfyui-mirror
    # checkpoint file it ships), so a precomputed table serves turbo=True and
    # turbo=False requests identically off the same table -- no rebuild, no rejection
    # needed. Only the comfy-format LoRA (`_TURBO_COMFY_REPOS`) genuinely conflicts (its
    # checkpoint DOES carry an adaln_proj/norm_out delta): that combination is still
    # rejected, both here (fast, no download needed -- same repo-name heuristic
    # `runner.turbo_lora_expected_format()` uses) and, as the actual enforcement point,
    # by `core/adaln_precompute.py`'s `_reject_turbo_wrapped_adaln()` once the real
    # checkpoint keys are known (catches an operator pointing H3_TURBO_LORA_REPO/
    # H3_TURBO_LORA_FILE at some other, not-yet-known comfy-format checkpoint under a
    # different repo name, which this heuristic alone would miss).
    if resolved["turbo"] and runner.H3_ADALN_PRECOMP and runner.turbo_lora_expected_format() == "comfy":
        raise ValueError(
            "turbo=1 with the comfy-format turbo LoRA is not supported while "
            "H3_ADALN_PRECOMP=1 (this process's env config): that checkpoint format "
            "wraps adaln_proj.linear/norm_out.linear with a real LoRA delta, which a "
            "precomputed table (baked once from a fixed schedule, adaln_proj deleted "
            "afterwards) cannot represent. See core/adaln_precompute.py's module "
            "docstring for the full derivation. Use the default diffusers-native turbo "
            "LoRA (H3_TURBO_LORA_REPO=lightx2v/Minimax-h3-Turbo, or leave "
            "H3_TURBO_LORA_REPO/H3_TURBO_LORA_FILE unset) to use turbo with "
            "H3_ADALN_PRECOMP=1, or restart with H3_ADALN_PRECOMP=0."
        )


def validate_instant_settings_for_upscale(resolved: dict, do_upscale: bool) -> None:
    """`upscale=1` (hires-fix) + turbo=1 works and is the recommended way to run
    hires-fix: verified by this task's own A/B run, 2026-08-06 (RTX PRO 6000 96GB,
    768->1536 hires-fix, seed=12345). 8-step turbo: 210.3s total (82.6s pass1+pass2
    denoise + 24.4s decode + ~103s model load/preload overhead already counted in
    total_elapsed_s), pass1=5/pass2=2 steps (`H3_HIRES_DENOISE=0.35` default split of
    the 7 scheduler timesteps an 8-step turbo run produces), peak VRAM 88.09GB -- vs
    the non-turbo 30-step baseline's 645s (README's own upscale=1 row). 16-step turbo:
    331.8s, pass1=10/pass2=5, peak VRAM 88.35GB, visibly sharper than 8-step. Both
    frame-sampled through the full clip (start/mid/end) with no checkerboard corruption
    or color-space breakage (the exact failure mode an earlier hires-fix iteration hit
    per `_upscale_block_state_2x`'s own docstring) and non-silent, non-clipped audio
    (RMS/peak matched the API's own reported values via an independent wav decode).
    `settings.py`'s no-FBC-with-turbo rule (see `resolve_instant_settings()`) already
    covers the "FBC bookkeeping under hires-fix is turbo-unaware" concern the previous
    version of this function's docstring raised: `effective_cache` is force-downgraded
    to "none" before this is ever reached, so `_fbc_last_step_was_skip()`'s calls in the
    hires-fix branch are harmless no-ops (try/except-wrapped, return 0) whenever turbo
    is on, exactly like the non-hires-fix path.
    """


def current_settings_snapshot() -> dict:
    """Everything `GET /api/settings` reports: current values for both groups, plus the
    choice lists the UI needs to render selects/checkboxes without hardcoding them
    twice (once server-side, once in static/index.html).
    """
    import core.runner as runner

    return {
        "instant": {
            "cache": runner.H3_CACHE if not runner.H3_TURBO_LORA else "none",
            "cache_threshold": runner.H3_CACHE_THRESHOLD,
            "attn": runner.H3_ATTN_BACKEND or "default",
            "turbo": runner.H3_TURBO_LORA,
        },
        "reload": {
            "transformer_quant": runner.H3_TRANSFORMER_QUANT,
            "te_quant": runner.TE_QUANT,
            "te_prune": runner.H3_TE_PRUNE,
            "lowvram": runner.H3_LOWVRAM_RAW,
            "video_vae_fp16": runner.H3_VIDEO_VAE_FP16,
            # 投影TE (H3_TE_PROJ) はこの API から変更可能 (te_proj bool)。ON にすると
            # `runner.H3_TE_PROJ` が既定リポジトリID (apply_reload_settings 参照) へ
            # 書き換えられ、32B TE の代わりに 4B+投影を使う。ON のとき te_quant/
            # te_prune の変更は apply 側で 400 拒否される (32B 自体をロードしないため
            # 意味を持たない) -- UI はその理由を表示できるよう、状態自体は常に見える。
            "te_proj": bool(runner.H3_TE_PROJ),
            "te_proj_quant": runner.H3_TE_PROJ_QUANT,
            # turbo LoRA 構成 (プリセット名、独自 env 構成なら "custom")。参考情報として
            # 実際の repo/file/steps 既定も返す (UI のツールチップ・デバッグ用、read-only)。
            "turbo_preset": current_turbo_preset(),
            "turbo_lora_repo": runner.H3_TURBO_LORA_REPO,
            "turbo_lora_file": runner.H3_TURBO_LORA_FILE,
            "turbo_lora_file_base": runner.H3_TURBO_LORA_FILE_BASE,
            "turbo_steps_default": runner.H3_TURBO_STEPS_DEFAULT,
        },
        "choices": {
            "cache": list(INSTANT_CACHE_CHOICES),
            # ck-w4a8 は起動時専用 (apply_reload_settings 参照) だが、現在値として表示できるように含める
            "transformer_quant": list(RELOAD_TRANSFORMER_QUANT_CHOICES)
            + (["ck-w4a8"] if runner.H3_TRANSFORMER_QUANT == "ck-w4a8" else []),
            "te_quant": list(RELOAD_TE_QUANT_CHOICES),
            "te_proj_quant": list(RELOAD_TE_PROJ_QUANT_CHOICES),
            "lowvram": list(RELOAD_LOWVRAM_CHOICES),
            "turbo_preset": list(RELOAD_TURBO_PRESET_CHOICES),
        },
        "constraints": {
            # Mirrors the exact incompatibilities core/runner.py enforces at import
            # time / generate()-time, exposed so the UI can grey out turbo instead of
            # only finding out via a 400 after clicking generate. turbo+upscale is
            # verified to work together (see validate_instant_settings_for_upscale's
            # own docstring, 2026-08-06 A/B) and is no longer greyed out.
            # int8/lowvram との非互換は **comfy 形式 (融合QKV) の LoRA のときだけ**
            # (validate_instant_settings のコメント参照) なので動的に判定する --
            # 既定の diffusers ネイティブ LoRA (lightx2v) では False になり、UI は
            # 低VRAMモードでも turbo を有効にできる。
            "turbo_incompatible_with_lowvram": runner.turbo_lora_expected_format() == "comfy",
            # 2026-11-17: 自作ストリーマ (H3_GROUP_STREAM=own) で起動時 (H3_TURBO_LORA=1) に巻いてあれば
            # group でも turbo を使える。それ以外 (diffusers の group offload、または起動時に巻いて
            # いない) はロード後に巻けないので不可 (validate_instant_settings 参照)。
            "turbo_incompatible_with_lowvram_group": not (runner.H3_GROUP_STREAM == "own" and runner.H3_TURBO_LORA),
            "turbo_incompatible_with_transformer_both_resident": runner.turbo_lora_expected_format() == "comfy",
            "turbo_incompatible_with_upscale": False,
            # H3_ADALN_PRECOMP (process-wide env flag, not a reload-group field -- see
            # runner.py's own module comment) only rejects turbo=True for the
            # comfy-format LoRA (v2 coexistence -- see validate_instant_settings()'s
            # matching guard above and core/adaln_precompute.py's module docstring for
            # the full derivation). Format-conditional like the three turbo_incompatible_
            # with_* fields above it, not the blanket True v1 had.
            "turbo_incompatible_with_adaln_precomp": bool(
                runner.H3_ADALN_PRECOMP and runner.turbo_lora_expected_format() == "comfy"
            ),
            "transformer_both_resident": runner.H3_TRANSFORMER_BOTH_RESIDENT,
            # te_proj (H3_TE_PROJ, 4B+投影) が ON のとき te_quant/te_prune は無効:
            # 32B TE 自体をロードしないため、この2つを変えても何も起きない (適用しても
            # 静かに無視されるだけになる、という事故を防ぐため apply 側は 400 で拒否する
            # -- apply_reload_settings() 参照)。UI はこのフラグを見て te_quant/te_prune
            # のコントロールを disable する。
            "te_quant_te_prune_incompatible_with_te_proj": bool(runner.H3_TE_PROJ),
        },
        "reload_eta_s": RELOAD_ETA_S,
    }


# Rough reload time estimates (see the task brief's own numbers / this task's own
# measurements, logged in README once verified) -- purely informational, shown in the
# UI's "apply" button before the user commits to a ~30-100s wait. Keyed by what is
# actually changing costs the most: TE swap and int8 transformer quantize are the two
# slow steps; lowvram=group additionally pays a one-time CPU-resident pin cost.
RELOAD_ETA_S = {
    "transformer_quant": 36,
    "te_quant": 40,
    "te_prune": 5,
    "lowvram": 90,
    "video_vae_fp16": 5,
    # te_proj は 32B (66GB級) の代わりに 4B (~5-9GB) をロードするだけなので、
    # te_quant (32B NF4 量子化、40s) より軽い見込み -- 4Bロード + 量子化 + 投影行列
    # ロードの合計として te_prune 相当に寄せる (未実測、実機で要調整)。
    # 実測 (2026-08-10, 96GB機): OFF→ON 47.9s (4Bロード+投影行列)、ON→OFF 29.4s
    # (32B TE nf4 再ロード)、ON→ON量子化変更相当 27.3s。控えめに大きい方へ丸める。
    "te_proj": 50,
    "te_proj_quant": 50,
    # turbo LoRA 構成の切替は unload_all + preload_all のみ (LoRA 本体のダウンロード・
    # 適用は次回 turbo=1 リクエスト時に遅延実行されるため reload 自体には乗らない)。
    "turbo_preset": 40,
}


def estimate_reload_seconds(changed_fields: list[str]) -> int:
    if not changed_fields:
        return 0
    return max(RELOAD_ETA_S.get(f, 30) for f in changed_fields)


def apply_reload_settings(runner_instance, **fields) -> dict:
    """Validate + apply a new reload-group configuration, unloading every big model and
    reloading under the new settings. Never restarts the process (CLAUDE.md-style rule
    for this project, per the task brief: only `core.runner`'s existing unload/preload
    machinery is used -- no os.execv, no self-kill).

    `fields` may contain any of: transformer_quant, te_quant, te_prune, lowvram,
    video_vae_fp16, te_proj, te_proj_quant, turbo_preset. Fields not present keep their current value
    (partial updates are fine -- the caller, `POST /api/settings/apply`, only sends what
    the user actually changed via its diff-against-current-snapshot logic, but this
    function itself does not require that; passing every field with its unchanged value
    is equally valid).

    Validation mirrors `core/runner.py`'s own import-time rules for the equivalent env
    vars (H3_LOWVRAM/H3_TRANSFORMER_QUANT/H3_TE_QUANT interactions -- see that module's
    big comment block above H3_LOWVRAM_RAW) as closely as possible; kept as a hand
    port rather than a shared function because the source rules run at *import* time
    (raise-and-crash-the-process semantics: wrong env vars should never start the
    server) while this runs at *request* time (raise-and-report-400 semantics: a bad
    apply must never leave the process in a half-reconfigured state). The two therefore
    cannot trivially share one function body without a mode flag threaded through every
    branch -- duplicated instead, with this docstring as the pointer to keep both in
    sync if the source rules ever change.

    Returns the new settings snapshot (see `current_settings_snapshot()`) plus timing.
    """
    import core.runner as runner

    with _reload_lock:
        t0 = time.time()

        # ---- resolve new te_proj / te_proj_quant first: the te_quant/te_prune conflict
        # guard further down needs to know the *post-request* te_proj state (new_te_proj),
        # not the current one (runner.H3_TE_PROJ) -- see that guard's own comment for why.
        new_te_proj = bool(fields.get("te_proj", runner.H3_TE_PROJ))
        new_te_proj_quant = str(fields.get("te_proj_quant", runner.H3_TE_PROJ_QUANT)).strip().lower()
        if new_te_proj_quant not in RELOAD_TE_PROJ_QUANT_CHOICES:
            raise ValueError(
                f"te_proj_quant must be one of {RELOAD_TE_PROJ_QUANT_CHOICES}, got {new_te_proj_quant!r}"
            )

        # ---- resolve new values (unset fields keep the current one) ----
        new_transformer_quant = fields.get("transformer_quant", runner.H3_TRANSFORMER_QUANT)
        new_te_quant = fields.get("te_quant", runner.TE_QUANT)
        new_te_prune = fields.get("te_prune", runner.H3_TE_PRUNE)
        new_lowvram_raw = fields.get("lowvram", runner.H3_LOWVRAM_RAW)
        new_video_vae_fp16 = fields.get("video_vae_fp16", runner.H3_VIDEO_VAE_FP16)

        new_transformer_quant = str(new_transformer_quant).strip().lower()
        new_te_quant = str(new_te_quant).strip().lower()
        new_lowvram_raw = str(new_lowvram_raw).strip().lower()
        new_te_prune = bool(new_te_prune)
        new_video_vae_fp16 = bool(new_video_vae_fp16)

        if new_transformer_quant == "ck-w4a8" or runner.H3_TRANSFORMER_QUANT == "ck-w4a8":
            # ck-w4a8 は H3_PRUNED=1 / comfy-kitchen / pinned 退避と結び付いた起動時設定 (runner.py の
            # import 時ガード参照) で、再ロード API からの切替は未対応。現在値のままなら通す (他の項目だけ
            # 変えられるように)、ck-w4a8 への / からの変更は拒否する。
            if new_transformer_quant != runner.H3_TRANSFORMER_QUANT:
                raise ValueError(
                    "transformer_quant の ck-w4a8 への/からの切替は再ロード設定からはできません "
                    "(H3_PRUNED=1 と comfy-kitchen を前提にした起動時の環境変数で指定してください)。"
                )
        elif new_transformer_quant not in RELOAD_TRANSFORMER_QUANT_CHOICES:
            raise ValueError(
                f"transformer_quant must be one of {RELOAD_TRANSFORMER_QUANT_CHOICES}, "
                f"got {new_transformer_quant!r}"
            )
        if new_transformer_quant == "ck-w4a8" and not runner.H3_PRUNED:
            # core/runner.py の import 時ガードと同じ制約 (ck-w4a8 は pruned transformer_ref
            # 側のみ実装)。H3_PRUNED は env 起動時固定で reload では切替できないため、
            # 非 pruned プロセスでは選択自体を拒否する。
            raise ValueError(
                "transformer_quant=ck-w4a8 は H3_PRUNED=1 の起動構成が前提です"
                "(非 pruned への適用は未実装)。gateway のプリセット"
                "(48gb-pruned-lowvram / dual-realtime 系)で起動し直してください。"
            )
        if new_te_quant not in RELOAD_TE_QUANT_CHOICES:
            raise ValueError(f"te_quant must be one of {RELOAD_TE_QUANT_CHOICES}, got {new_te_quant!r}")
        if new_lowvram_raw not in RELOAD_LOWVRAM_CHOICES:
            raise ValueError(f"lowvram must be one of {RELOAD_LOWVRAM_CHOICES}, got {new_lowvram_raw!r}")

        # ---- turbo LoRA 構成プリセット ----
        new_turbo_preset = fields.get("turbo_preset")
        new_turbo: dict | None = None
        if new_turbo_preset is not None:
            new_turbo_preset = str(new_turbo_preset).strip().lower()
            if new_turbo_preset not in RELOAD_TURBO_PRESET_CHOICES:
                raise ValueError(
                    f"turbo_preset must be one of {RELOAD_TURBO_PRESET_CHOICES}, "
                    f"got {new_turbo_preset!r}"
                )
            new_turbo = TURBO_LORA_PRESETS[new_turbo_preset]
        turbo_changed = new_turbo is not None and _turbo_state_tuple(runner) != _preset_state_tuple(new_turbo)

        # ---- 投影TE (te_proj) 有効化後は TE 系 (32B 用) の変更を拒否する ----
        # 判定は「このリクエスト適用後の値」(new_te_proj) で行う -- 「現在の値」
        # (runner.H3_TE_PROJ) で判定すると、te_proj を OFF にしながら同時に
        # te_quant/te_prune も変える (32B に戻すのだから意味がある) 正当なリクエストが
        # 誤って拒否されてしまう。逆に、te_proj を ON にする (または ON のまま) リクエスト
        # で te_quant/te_prune が実際に**現在値と違う値へ変わる**場合は、実際のロードが
        # `_load_text_encoder` の投影分岐で 32B 側をそもそも見ないため、変更しても
        # 静かに無視される -- 適用されない変更を受け取った時点で 400 にする。
        #
        # 判定基準は `k in fields` (フィールドが送られてきたか) ではなく「新しい値が
        # 現在値と違うか」にすること: app.py/static側は既存フィールドと同じパターンで
        # te_quant/te_prune を**毎回送る** (変更の有無に関わらずフォーム全体を POST
        # する) ため、`k in fields` で判定すると te_proj を ON にする度に
        # (te_quant/te_prune の値を一切変えていなくても) 誤って 400 になってしまう。
        if new_te_proj:
            _te_fields = [
                name
                for name, old, new in (
                    ("te_quant", runner.TE_QUANT, new_te_quant),
                    ("te_prune", runner.H3_TE_PRUNE, new_te_prune),
                )
                if old != new
            ]
            if _te_fields:
                raise ValueError(
                    "投影TE (te_proj) を有効にする(または有効のままにする)リクエストで "
                    f"te_quant / te_prune の値も変更されています ({', '.join(_te_fields)})。"
                    "投影TE 有効時は 32B TE 自体をロードしないため、これらの変更は適用されず"
                    "無視されるだけになります。先に te_proj=false で適用してから "
                    "te_quant/te_prune を変更してください。"
                )

        new_lowvram = new_lowvram_raw == "1"
        new_lowvram_group = new_lowvram_raw == "group"
        new_lowvram_any = new_lowvram or new_lowvram_group

        # ---- validation, mirrors core/runner.py's import-time rules ----
        if new_lowvram_any:
            if new_transformer_quant == "none" and "transformer_quant" in fields:
                # Only reject an *explicit* none -- if the caller did not touch
                # transformer_quant at all, silently auto-upgrade it to int8 below,
                # exactly like core/runner.py's own H3_LOWVRAM_ANY block does for the
                # env-var case (distinguishing "left alone" from "explicitly chosen"
                # the same way, via `fields.get(...)` presence rather than a value
                # comparison against the default).
                raise ValueError(
                    f"lowvram={new_lowvram_raw!r} requires an int8 transformer (bf16's "
                    "66.3GB does not fit a 48GB-class card even alone) but "
                    "transformer_quant=none was explicitly requested. Drop "
                    "transformer_quant from the request (it will default to int8 under "
                    "lowvram) or set transformer_quant=int8 explicitly."
                )
            # 旧実装は無条件で int8 へ強制していたが、ck-w4a8 + lowvram=1 は本番採用済み
            # の実在構成 (48gb-pruned-lowvram プリセット) のため、補正は none のときに限る。
            if new_transformer_quant == "none":
                new_transformer_quant = "int8"
            if new_te_quant != "bnb-4bit":
                raise ValueError(
                    f"lowvram={new_lowvram_raw!r} requires te_quant=bnb-4bit, got "
                    f"te_quant={new_te_quant!r}. bf16 TE (~66.3GB) cannot coexist with "
                    "anything else on a 24-48GB-class card."
                )

        # core/runner.py:919 と同じ判定 (int8 / ck-w4a8 の両方が両常駐対象)。旧実装は
        # int8 のみを見ており ck-w4a8 で不整合だった。
        new_transformer_both_resident = (
            new_transformer_quant in ("int8", "ck-w4a8") and not new_lowvram_any
        )

        # turbo (instant-group) compatibility with the *new* reload config: reject the
        # apply outright if turbo is currently on and would become a confirmed-broken
        # combo under the new settings (see validate_instant_settings()'s own docstring
        # for the Int8Tensor/aten.cat root cause), rather than silently leaving a
        # now-invalid turbo=on state for the next request to trip over. Mirrors
        # validate_instant_settings()'s own rule, evaluated against the *proposed* new
        # state instead of the current one.
        # comfy 形式(融合QKV)の LoRA だけが int8/lowvram と非互換 (Int8Tensor に
        # aten.cat カーネルが無い)。diffusers ネイティブ形式 (lightx2v、2026-08-08 の
        # c49ce1e で解禁・現在の既定) は int8/both_resident/lowvram=1 と併用可能で、
        # validate_instant_settings() (上) と constraints (下) は当時この条件で緩和済み
        # だったが、**この apply 側ガードだけ条件が付け忘れられていた** (2026-08-13 の
        # 回帰監査で発覚)。その結果、H3_TURBO_LORA=1 のプロセスでは new_transformer_
        # both_resident が int8 で常に真になり、te_proj 切替を含む**あらゆる**
        # /api/settings/apply が一律 400 になっていた (リロード設定 UI が実質使用不能)。
        # turbo プリセットも同時に切り替わる場合は「新しい repo」で comfy 形式かどうかを
        # 判定する (両プリセットとも diffusers ネイティブ形式なので、comfy env 構成からの
        # 脱出リクエストが旧 repo 判定で誤って 400 にならないようにする)。
        if new_turbo is not None:
            _new_turbo_is_comfy = new_turbo["repo"] in runner._TURBO_COMFY_REPOS
        else:
            _new_turbo_is_comfy = runner.turbo_lora_expected_format() == "comfy"
        if (
            runner.H3_TURBO_LORA
            and _new_turbo_is_comfy
            and (new_lowvram_any or new_transformer_both_resident)
        ):
            raise ValueError(
                "Cannot apply this reload configuration while the turbo LoRA default "
                "(H3_TURBO_LORA env var) is on: lowvram/transformer_both_resident are "
                "confirmed incompatible with the comfy-format turbo LoRA (Int8Tensor "
                "has no aten.cat kernel, see validate_instant_settings()'s docstring). "
                "Note this only refers to the startup env var default, not any single "
                "request's instant turbo=1 override (those are validated independently, "
                "per request, by validate_instant_settings())."
            )

        # ---- resolve the actual H3_TE_PROJ string to commit ----
        # ON (new_te_proj=True): if `runner.H3_TE_PROJ` is already non-empty (operator
        # set a local path or a different repo via env), keep it as-is -- only fill in
        # the default repo ID when it was empty (i.e. env-unconfigured), so an env
        # override is never silently clobbered by the UI's own default.
        # OFF (new_te_proj=False): clear to "" so `_load_text_encoder`'s `if H3_TE_PROJ:`
        # branch is skipped and the 32B path (TE_QUANT-driven) runs instead.
        if new_te_proj:
            new_te_proj_repo = runner.H3_TE_PROJ or runner.H3_TE_PROJ_DEFAULT_REPO
        else:
            new_te_proj_repo = ""

        changed_fields = [
            name
            for name, old, new in (
                ("transformer_quant", runner.H3_TRANSFORMER_QUANT, new_transformer_quant),
                ("te_quant", runner.TE_QUANT, new_te_quant),
                ("te_prune", runner.H3_TE_PRUNE, new_te_prune),
                ("lowvram", runner.H3_LOWVRAM_RAW, new_lowvram_raw),
                ("video_vae_fp16", runner.H3_VIDEO_VAE_FP16, new_video_vae_fp16),
                ("te_proj", runner.H3_TE_PROJ, new_te_proj_repo),
                ("te_proj_quant", runner.H3_TE_PROJ_QUANT, new_te_proj_quant),
            )
            if old != new
        ]
        if turbo_changed:
            changed_fields.append("turbo_preset")
        if not changed_fields:
            logger.info("apply_reload_settings: no changes, skipping unload/reload")
            return {**current_settings_snapshot(), "changed_fields": [], "reload_time_s": 0.0}

        logger.info(
            "apply_reload_settings: changed=%s -> transformer_quant=%s te_quant=%s "
            "te_prune=%s lowvram=%s video_vae_fp16=%s te_proj=%s te_proj_quant=%s",
            changed_fields, new_transformer_quant, new_te_quant, new_te_prune,
            new_lowvram_raw, new_video_vae_fp16, bool(new_te_proj_repo), new_te_proj_quant,
        )

        # ---- unload everything (models only -- pipe shells are cheap/idempotent to
        # rebuild and are left alone) ----
        # `unload_all()` -> `_free_text_encoder(force=True)` also clears any stale
        # `self._pipe._te_projection` cache (see that method's comment) -- important for
        # both directions of the te_proj toggle: OFF must not leave a stale projection
        # matrix that a later `_te_projection_for()` call could accidentally pick back
        # up, and ON must not reuse a projection loaded under a *different* repo/tap
        # from a previous ON period.
        runner_instance.unload_all()

        # ---- commit new globals (module attribute assignment -- every other function
        # in core/runner.py reads these as plain module-level names, which in Python
        # resolve through the module's __dict__ at call time, so this is visible to all
        # of them immediately, no re-import needed) ----
        runner.H3_TRANSFORMER_QUANT = new_transformer_quant
        runner.TE_QUANT = new_te_quant
        runner.H3_TE_PRUNE = new_te_prune
        runner.H3_LOWVRAM_RAW = new_lowvram_raw
        runner.H3_LOWVRAM = new_lowvram
        runner.H3_LOWVRAM_GROUP = new_lowvram_group
        runner.H3_LOWVRAM_ANY = new_lowvram_any
        runner.H3_TRANSFORMER_BOTH_RESIDENT = new_transformer_both_resident
        runner.H3_VIDEO_VAE_FP16 = new_video_vae_fp16
        runner.H3_TE_PROJ = new_te_proj_repo
        runner.H3_TE_PROJ_QUANT = new_te_proj_quant
        if turbo_changed:
            runner.H3_TURBO_LORA_REPO = new_turbo["repo"]
            runner.H3_TURBO_LORA_FILE = new_turbo["file"]
            runner.H3_TURBO_LORA_REPO_BASE = new_turbo["repo_base"]
            runner.H3_TURBO_LORA_FILE_BASE = new_turbo["file_base"]
            runner.H3_TURBO_LORA_SCALE_RAW = new_turbo["scale_raw"]
            runner.H3_PIKA_RENOISE = new_turbo["pika_renoise"]
            runner.H3_TURBO_STEPS_DEFAULT = new_turbo["steps_default"]
            # 派生 video shift を import 時と同じファイル名規則で再計算する
            # (`_fl2v_` かつ `_768p` のときだけ 6.0 -- core/runner.py:1755 のコメント参照)。
            # H3_TURBO_VIDEO_SHIFT の明示 env 指定があるときは import 時同様それが最優先
            # (再計算しない)。
            if not runner.H3_TURBO_VIDEO_SHIFT_RAW:
                runner.H3_TURBO_VIDEO_SHIFT = (
                    6.0 if ("_fl2v_" in new_turbo["file"] and "_768p" in new_turbo["file"]) else None
                )
                runner.H3_TURBO_VIDEO_SHIFT_BASE = (
                    6.0
                    if ("_fl2v_" in new_turbo["file_base"] and "_768p" in new_turbo["file_base"])
                    else None
                )
            # 解決済み LoRA パスは「プロセスに1回」前提のインスタンスキャッシュ
            # (_download_turbo_lora_if_needed) に乗っているため、repo/file を変えたら
            # 必ず無効化する (しないと旧 LoRA が適用され続ける)。
            runner_instance._turbo_lora_path = None
            runner_instance._turbo_lora_path_base = None
            logger.info(
                "apply_reload_settings: turbo preset -> %s (repo=%s file=%s file_base=%s "
                "scale_raw=%r renoise=%s steps_default=%d)",
                new_turbo_preset, new_turbo["repo"], new_turbo["file"],
                new_turbo["file_base"], new_turbo["scale_raw"],
                new_turbo["pika_renoise"], new_turbo["steps_default"],
            )

        # ---- reload steady-state residents under the new config ----
        runner_instance.preload_all()

        elapsed = time.time() - t0
        logger.info("apply_reload_settings: done in %.1fs. status=%s", elapsed, runner_instance.status())
        return {**current_settings_snapshot(), "changed_fields": changed_fields, "reload_time_s": round(elapsed, 1)}
