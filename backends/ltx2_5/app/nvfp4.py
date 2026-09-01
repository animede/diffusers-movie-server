"""NVFP4(Blackwell ネイティブ FP4)量子化 transformer のローダと Linear 実装。

対象: Lightricks/LTX-2.5 公式配布の
`diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors`
(ComfyUI 命名の単一ファイル、18.7GB)。

チェックポイント形式(実機検分済み):
  - 量子化層: `<module>.weight` U8 [out, in/2](e2m1 を2値/byteパック)
    + `<module>.weight_scale` F8_E4M3 [out, in/16](16要素ブロックスケール)
    + `<module>.weight_scale_2` F32 スカラー(グローバル)
    + `<module>.input_scale` F32 スカラー(活性化の静的グローバルスケール)
  - 非量子化(norm / bias / connectors / adaln 等)は bf16。
  - ヘッダ `_quantization_metadata` に量子化層一覧(format: "nvfp4")。

行列積は torch 2.11 の `torch._scaled_mm`(cuBLAS block-scaled FP4、sm_120
ネイティブ)を使う。実測(probes/probe_nvfp4_gemm.py): bf16 比 3.2〜3.8 倍。

【最重要の罠】`torch._scaled_mm` に渡すブロックスケールは素の行順ではなく
cuBLAS の swizzled layout(128行×4列タイル、`to_blocked()`)でなければならない。
素のまま渡すと**エラーにならず黙って壊れた値**が出る(probe で相対誤差 0.89 →
swizzle 適用で cosine 0.9955 を実機確認)。重み側はロード時に一度だけ、
活性化側は forward ごとに適用する。

活性化の動的 nvfp4 量子化は triton カーネル(`_nvfp4_quant_kernel`)。
torch には bf16→float4_e2m1fn_x2 の cast が存在しない(実機確認)ため自前実装。
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl

logger = logging.getLogger("ltx25.nvfp4")

import triton.language as _tl_const
FP4_MAX = _tl_const.constexpr(6.0)  # e2m1 の最大絶対値(triton kernel から参照するため constexpr)
FP4_MAX_F = 6.0  # python 側計算用
BLOCK = 16     # nvfp4 のスケールブロック長


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def from_blocked(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """`to_blocked` の逆写像: cuBLAS blocked flat → [rows, cols] 行順。

    checkpoint の weight_scale は**最初から blocked layout で格納されている**
    (実機で確定、下記 load_quantized の注記参照)ため、デクオンタイズ時のみ
    この逆写像で行順に戻す。
    """
    nrb, ncb = ceil_div(rows, 128), ceil_div(cols, 4)
    t = flat.reshape(-1, 32, 16).reshape(-1, 32, 4, 4).transpose(1, 2)
    t = t.reshape(nrb, ncb, 128, 4).permute(0, 2, 1, 3).reshape(nrb * 128, ncb * 4)
    return t[:rows, :cols]


def swap_nibbles(u8: torch.Tensor) -> torch.Tensor:
    """byte 内の上下 nibble を入れ替える(checkpoint → cuBLAS パック順の変換)。"""
    return (u8 << 4) | (u8 >> 4)


def to_blocked(m: torch.Tensor) -> torch.Tensor:
    """[rows, cols] のスケール行列を cuBLAS blocked layout へ並べ替える。

    torchao の `to_blocked` と同一(128行×4列タイル、32×16 の内部再配置)。
    出力は flat 1D。rows/cols がタイル境界に満たない分はゼロパディングされる
    (パディング部は行列積で使われないので値は不問)。
    """
    rows, cols = m.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded = m
    if rows != n_row_blocks * 128 or cols != n_col_blocks * 4:
        padded = torch.zeros(
            n_row_blocks * 128, n_col_blocks * 4, dtype=m.dtype, device=m.device
        )
        padded[:rows, :cols] = m
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()


@triton.jit
def _nvfp4_quant_kernel(
    x_ptr,          # 入力 bf16 [M, K](行連続)
    out_ptr,        # 出力 u8 [M, K//2]
    scale_ptr,      # 出力 e4m3 [M, K//16]
    K,              # 列数(16の倍数)
    inv_gscale,     # 1.0 / input_scale(グローバルスケールの逆数)
    BLOCK_K: tl.constexpr,  # 1プログラムが処理する列数(16の倍数)
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    k0 = pid_k * BLOCK_K
    n_blocks: tl.constexpr = BLOCK_K // 16

    # 偶数/奇数インデックスを別グリッドでロードする(triton はテンソルの
    # スライスを持たないため、パック対象のペアを最初から分けて扱う)。
    offs_b = tl.arange(0, n_blocks)                        # ブロック番号
    offs_p8 = tl.arange(0, 8)                              # ブロック内ペア番号
    base = k0 + offs_b[:, None] * 16 + offs_p8[None, :] * 2
    mask = base < K
    x_lo = tl.load(x_ptr + pid_m * K + base, mask=mask, other=0.0).to(tl.float32)
    x_hi = tl.load(x_ptr + pid_m * K + base + 1, mask=mask, other=0.0).to(tl.float32)

    # 16要素ブロック absmax → e4m3 スケール(グローバルスケールで割ってから格納)
    absmax = tl.maximum(tl.max(tl.abs(x_lo), axis=1), tl.max(tl.abs(x_hi), axis=1))
    s = absmax * (inv_gscale / FP4_MAX)
    s = tl.maximum(s, 0.015625)                            # 2^-6 下限(ゼロ割回避)
    # e4m3 の上限 448 でクランプする。実活性化の absmax が checkpoint の
    # 校正値(input_scale)を超えるブロックでは s が 448 を超え、triton の
    # float8e4nv 変換は飽和保証が無い(NaN 化しうる)。クランプ後は当該
    # ブロックが FP4_MAX 超の値を持つが、続く q の ±6 クランプで飽和する
    # (=通常の量子化サチュレーションに落ちる)。
    s = tl.minimum(s, 448.0)
    s8 = s.to(tl.float8e4nv)                               # e4m3 へ丸め
    s_eff = s8.to(tl.float32) * (1.0 / inv_gscale)         # 実効スケール(丸め後)

    q_lo = tl.minimum(tl.maximum(x_lo / s_eff[:, None], -FP4_MAX), FP4_MAX)
    q_hi = tl.minimum(tl.maximum(x_hi / s_eff[:, None], -FP4_MAX), FP4_MAX)

    # e2m1 への最近傍丸め: 表現値 {0, .5, 1, 1.5, 2, 3, 4, 6} の中点閾値で符号化
    a_lo = tl.abs(q_lo)
    c_lo = (
        (a_lo > 0.25).to(tl.int32) + (a_lo > 0.75).to(tl.int32)
        + (a_lo > 1.25).to(tl.int32) + (a_lo > 1.75).to(tl.int32)
        + (a_lo > 2.5).to(tl.int32) + (a_lo > 3.5).to(tl.int32)
        + (a_lo > 5.0).to(tl.int32)
    )
    c_lo = tl.where(q_lo < 0, c_lo + 8, c_lo)
    a_hi = tl.abs(q_hi)
    c_hi = (
        (a_hi > 0.25).to(tl.int32) + (a_hi > 0.75).to(tl.int32)
        + (a_hi > 1.25).to(tl.int32) + (a_hi > 1.75).to(tl.int32)
        + (a_hi > 2.5).to(tl.int32) + (a_hi > 3.5).to(tl.int32)
        + (a_hi > 5.0).to(tl.int32)
    )
    c_hi = tl.where(q_hi < 0, c_hi + 8, c_hi)

    packed = (c_lo | (c_hi << 4)).to(tl.uint8)             # [n_blocks, 8]

    offs_pk = base // 2                                    # パック後の byte 位置
    tl.store(out_ptr + pid_m * (K // 2) + offs_pk, packed, mask=offs_pk < K // 2)
    offs_s = k0 // 16 + offs_b
    tl.store(scale_ptr + pid_m * (K // 16) + offs_s, s8, mask=offs_s < K // 16)


def nvfp4_quantize(x: torch.Tensor, input_scale: float):
    """bf16 [M, K] → (packed fp4x2 [M, K/2], e4m3 scales [M, K/16])。K は16の倍数。"""
    M, K = x.shape
    out = torch.empty(M, K // 2, dtype=torch.uint8, device=x.device)
    scales = torch.empty(M, K // 16, dtype=torch.float8_e4m3fn, device=x.device)
    BLOCK_K = 256 if K >= 256 else K
    grid = (M, ceil_div(K, BLOCK_K))
    _nvfp4_quant_kernel[grid](
        x.contiguous(), out, scales, K, 1.0 / input_scale, BLOCK_K=BLOCK_K
    )
    return out.view(torch.float4_e2m1fn_x2), scales


class NVFP4Linear(nn.Module):
    """nvfp4 量子化済み Linear。forward で活性化を動的量子化し FP4 GEMM を叩く。

    - `weight`(u8 パック済み)・`weight_scale_blocked`(swizzle 済み)は
      チェックポイント名と形が異なるため通常の load_state_dict では読まない。
      ローダ(`load_nvfp4_transformer`)が `load_quantized()` で直接注入する。
    - M(トークン数)は任意長に対応: `_scaled_mm` 前に 16 の倍数へパディングする
      (スケール swizzle 自体が 128 行へパディングするため、行データ側だけ揃える)。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", None, persistent=False)
        self.register_buffer("weight_scale_blocked", None, persistent=False)
        self.register_buffer("bias", None, persistent=False)
        self.input_scale = 1.0
        self.out_scale = 1.0  # input_scale * weight_scale_2(forward の最後に掛ける)

    def load_quantized(
        self,
        weight_packed_u8: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_scale_2: float,
        input_scale: float,
        bias: Optional[torch.Tensor],
        device: torch.device,
    ) -> None:
        # 【checkpoint フォーマットの罠(2026-09-01、bf16 リファレンスとの
        # 全数値照合で確定)】ComfyUI 配布の nvfp4 checkpoint は:
        #   (1) nibble 順が cuBLAS パック規約と逆(high nibble = 先頭要素)。
        #       ロード時に byte 内スワップして cuBLAS 順へ直す。
        #   (2) weight_scale [out, in/16] は**既に cuBLAS blocked layout
        #       (to_blocked 済み)の並びで格納されている**。ここで to_blocked を
        #       重ねると二重 swizzle になり、モデル出力が決定論的なノイズになる
        #       (実機で再現。ブロック単位のスケール置換なので分布・中央値は
        #       正常に見え、層単体の cosine 検査でも自作 dequant と自作 GEMM が
        #       「同じ誤解釈」で一致してしまい検出できなかった。検出には
        #       公式 bf16 重みとの直接照合が必要だった)。
        self.weight = (
            swap_nibbles(weight_packed_u8.to(device)).view(torch.float4_e2m1fn_x2)
        )
        rows, cols = weight_scale.shape
        if rows % 128 == 0 and cols % 4 == 0:
            # blocked layout のパディングが不要な形状(実測: 全1176層が該当)は
            # 格納バイト列がそのまま cuBLAS の期待する並び。
            self.weight_scale_blocked = weight_scale.to(device).flatten()
        else:
            # 将来の非整列形状への保険: 一度行順へ戻してから正規の swizzle。
            self.weight_scale_blocked = to_blocked(
                from_blocked(weight_scale.flatten(), rows, cols).to(device)
            )
        self.input_scale = float(input_scale)
        self.out_scale = float(input_scale) * float(weight_scale_2)
        self.bias = None if bias is None else bias.to(device, torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        M = x2.shape[0]
        pad = (-M) % 16
        if pad:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
        xq, xs = nvfp4_quantize(x2, self.input_scale)
        out = torch._scaled_mm(
            xq,
            self.weight.t(),
            scale_a=to_blocked(xs),
            scale_b=self.weight_scale_blocked,
            out_dtype=torch.bfloat16,
        )
        if pad:
            out = out[:M]
        out = out * self.out_scale
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)


def _fix_prompt_adaln_keys(sd: Dict[str, torch.Tensor]) -> None:
    """diffusers の LTX2 変換関数が取りこぼす 12 キーを追加リネームする。

    `convert_ltx2_transformer_to_diffusers()` の adaln ハンドラは
    `adaln_single.` / `audio_adaln_single.` の完全一致プレフィックスしか
    処理しないため、`prompt_adaln_single.` / `audio_prompt_adaln_single.` が
    素通りする(LTX-2.3 統合時と同一の既知ギャップ。実測 12 キー)。
    """
    for key in list(sd.keys()):
        if key.startswith("prompt_adaln_single."):
            sd["prompt_adaln." + key[len("prompt_adaln_single."):]] = sd.pop(key)
        elif key.startswith("audio_prompt_adaln_single."):
            sd["audio_prompt_adaln." + key[len("audio_prompt_adaln_single."):]] = sd.pop(key)


E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def dequantize_nvfp4_weight(
    w_packed: torch.Tensor, w_scale: torch.Tensor, w_scale_2: float
) -> torch.Tensor:
    """パック済み nvfp4 重みを f32 [out, in] へ展開する(デバッグ/検証用)。

    checkpoint 規約(load_quantized の注記参照): high nibble = 先頭要素、
    weight_scale は blocked layout 格納 → 行順へ逆写像してから適用する。
    公式 bf16 重みとの照合で cosine ≈ 1.0 を確認済み。
    """
    lo = (w_packed & 0x0F).long()
    hi = (w_packed >> 4).long()
    vals = torch.stack([E2M1_LUT[hi], E2M1_LUT[lo]], dim=-1).reshape(
        w_packed.shape[0], -1
    )
    rows, cols = w_scale.shape
    s_rm = from_blocked(w_scale.flatten(), rows, cols).float()
    return vals * s_rm.repeat_interleave(16, dim=1) * w_scale_2


def load_nvfp4_transformer(
    ckpt_path: str,
    config: dict,
    device: torch.device,
    dequant_bf16: bool = False,
):
    """ComfyUI 形式の nvfp4 単一ファイルから LTX2VideoTransformer3DModel を組み立てる。

    手順:
      1. 全テンソルを CPU へロード(18.7GB、RAM 内で完結)
      2. diffusers の `convert_ltx2_transformer_to_diffusers()` でリネーム
         (+ prompt_adaln 12 キーの追加リネーム)
      3. meta skeleton を構築し、`weight_scale` を持つモジュールを NVFP4Linear へ置換
      4. 量子化層は `load_quantized()` で GPU へ直接注入、残り(bf16)は
         `load_state_dict(assign=True)` で materialize してから GPU へ
    """
    from safetensors import safe_open
    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.loaders.single_file_utils import (
        convert_ltx2_transformer_to_diffusers,
    )

    logger.info("[nvfp4] loading checkpoint tensors from %s", ckpt_path)
    sd: Dict[str, torch.Tensor] = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
            sd[k] = f.get_tensor(k)

    sd = convert_ltx2_transformer_to_diffusers(sd)
    _fix_prompt_adaln_keys(sd)

    cfg = {k: v for k, v in config.items() if not k.startswith("_")}
    cfg.pop("quantization_config", None)
    with init_empty_weights():
        model = LTX2VideoTransformer3DModel.from_config(cfg)

    # weight_scale の存在 = nvfp4 量子化層(ヘッダのメタデータではなく実キーで判定)
    quant_modules = sorted(
        {k[: -len(".weight_scale")] for k in sd if k.endswith(".weight_scale")}
    )
    logger.info("[nvfp4] quantized linear modules: %d", len(quant_modules))

    modules = dict(model.named_modules())
    for name in quant_modules:
        parent_name, _, child_name = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        old = modules[name]
        if not isinstance(old, nn.Linear):
            raise RuntimeError(f"[nvfp4] {name} is {type(old).__name__}, not Linear")
        w_packed = sd.pop(name + ".weight")
        w_scale = sd.pop(name + ".weight_scale")
        w_scale_2 = float(sd.pop(name + ".weight_scale_2"))
        input_scale = float(sd.pop(name + ".input_scale"))
        bias = sd.pop(name + ".bias", None)
        if (w_packed.shape[0] != old.out_features
                or w_packed.shape[1] * 2 != old.in_features):
            raise RuntimeError(
                f"[nvfp4] {name}: packed weight {tuple(w_packed.shape)} does not "
                f"match Linear({old.in_features}, {old.out_features})"
            )
        if dequant_bf16:
            # デバッグ用: 量子化GEMMを使わず、デクオンタイズ済み bf16 の
            # 素の Linear として組む(重み値は nvfp4 と同一、計算だけ bf16)。
            lin = nn.Linear(old.in_features, old.out_features,
                            bias=bias is not None, dtype=torch.bfloat16, device=device)
            with torch.no_grad():
                lin.weight.copy_(
                    dequantize_nvfp4_weight(w_packed, w_scale, w_scale_2)
                    .to(torch.bfloat16)
                )
                if bias is not None:
                    lin.bias.copy_(bias.to(torch.bfloat16))
            setattr(parent, child_name, lin)
            continue
        qlin = NVFP4Linear(old.in_features, old.out_features, bias=old.bias is not None)
        qlin.load_quantized(
            weight_packed_u8=w_packed,
            weight_scale=w_scale,
            weight_scale_2=w_scale_2,
            input_scale=input_scale,
            bias=bias,
            device=device,
        )
        setattr(parent, child_name, qlin)

    # 残りの F32 テンソル(scale_shift_table 系 290個、実機検分)を bf16 へ
    # 揃える。assign=True はチェックポイントの dtype をそのまま採用するため、
    # ここで揃えないと変調演算 `(1+scale)*x+shift` が活性化を fp32 へ昇格させ、
    # 後段の bf16 Linear で dtype 不一致になる(実機で再現)。bf16 経路の
    # `from_pretrained(torch_dtype=bf16)` と同じ扱いに合わせる。量子化層の
    # weight_scale_2 / input_scale(F32 スカラー)は上で pop 済みのため無傷。
    for key, val in sd.items():
        if val.dtype == torch.float32:
            sd[key] = val.to(torch.bfloat16)

    # 残り(非量子化 bf16)を materialize。NVFP4Linear のバッファは
    # persistent=False のため state_dict に現れず、strict 検査とは干渉しない。
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    problem_missing = [
        k for k in missing if not any(k.startswith(q + ".") for q in quant_modules)
    ]
    if problem_missing or unexpected:
        raise RuntimeError(
            f"[nvfp4] state_dict mismatch: missing={problem_missing[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    model.to(device)
    return model
