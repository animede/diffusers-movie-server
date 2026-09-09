"""transformer_blocks の per-block torch.compile(LTX25_COMPILE_BLOCKS)。

【結論(2026-09-09): 本番では利得なし → 既定 off の実験的フラグとして残置。
 nvfp4-fast プリセットには焼き込まない。graph 単体(app/cudagraph.py)が本番構成】
  - probe(transformer 単体・512x288x121f 相当 2304 tokens)では compiled+graph が
    graph 単体比 225→185-190ms/fwd と勝つが、**サーバ E2E では同 shape で誤差範囲**
    (islands 3.34-3.36s vs graph単体 3.38-3.40s)。
  - **リアルタイム shape(384x288x81f・1188 tokens)では明確に退行**
    (islands 2.39s vs graph単体 1.83s、同時刻ベースラインで確認)。極小テンソル
    では Inductor 生成 Triton カーネルの固定オーバーヘッドが eager の ATen
    カーネルに負ける。dynamic=False(shape ごと静的特殊化)適用後もこの結果
    (適用前は dynamic-shape 汎用カーネルが capture され 2.77s とさらに悪かった)。
  - 再挑戦するなら: mode="max-autotune-no-cudagraphs"(未検証)、torch 更新後の
    再計測、quality tier の大 shape(2段目 refine)限定適用、あたりが残り筋。

機構: nvfp4 の量子化+FP4 GEMM を torch custom_op("ltx25::nvfp4_linear")で opaque 化
することで fullgraph compile を解禁し(fp4 dtype を Inductor から隠蔽 + graph break
解消)、ブロック内の norm/変調/FF の小カーネル群を Inductor に融合させる。**必ず
CUDA Graph(app/cudagraph.py)と併用すること** -- compiled eager 単体は graph break /
dispatch コストで素の eager より遅い(実測 250→338-360ms/fwd)。probe レベルの
steady E2E(512x288x121f t2av 8step、probe_compile_e2e.py):
  eager 2.76s / graph のみ 2.60s / islands 2.36s / fusion 2.18s

モード:
  - "islands": attention モジュール(LTX2Attention)を dynamo.disable で eager 島化し、
    fullgraph=False で compile。初回 compile ~20s。リアルタイム向け既定。
  - "fusion" : 全融合(fullgraph=True)。最速だが初回 compile 167s
    (Inductor キャッシュ warm でも ~55s)。長時間セッションの手動選択向け。

数値特性(重要、probe_compile_customop.py / probe_compile_bisect.py で確定):
  - custom_op 差し替え自体は eager と bit 一致。
  - compile 後は eager と bit 一致しない: Inductor 融合カーネルの微小な系統誤差
    (ブロックあたり cosine ~0.9999、emulate_precision_casts でも不変)が 48 ブロック
    で累積し 1 forward あたり cosine ~0.965-0.969 の軌道差になる。aot_eager は
    bit 一致のため意味論は正しい。実映像では構図の微ドリフト(PSNR 18-25dB、
    int8 prequant ON/OFF と同クラス)として現れ、品質はユーザー判定で
    「問題なし」(2026-09-09)。**bit 再現が要る対照実験では必ず off にすること。**

前提: nvfp4 精度(NVFP4Linear)のみ対応。LoRA ジョブとの併用は未検証
(nvfp4 では LoRA 注入自体が元々非対応の構成)。
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.library import custom_op, register_fake

from .nvfp4 import NVFP4Linear, nvfp4_quantize, to_blocked


@custom_op("ltx25::nvfp4_linear", mutates_args=())
def nvfp4_linear_op(
    x: torch.Tensor,
    weight_u8: torch.Tensor,  # float4_e2m1fn_x2 は op schema で扱えないため u8 view
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


def _forward_via_op(self, x: torch.Tensor) -> torch.Tensor:
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


def apply_block_compile(transformer: torch.nn.Module, mode: str) -> None:
    """transformer_blocks を per-block compile する(mode: "islands" | "fusion")。

    呼び出し順の制約: CUDA Graph runner の install() より**前**に呼ぶこと
    (graph は compile 済みブロックの呼び出しを capture する必要がある)。
    NVFP4Linear.forward をクラスレベルで custom_op 経由に差し替える
    (compile 無しの経路と bit 一致、probe で確認済み)。
    """
    import torch._dynamo as dynamo

    if mode not in ("islands", "fusion"):
        raise ValueError(f"unknown compile mode: {mode!r} (islands|fusion)")
    NVFP4Linear.forward = _forward_via_op
    dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, 256)
    fullgraph = mode == "fusion"
    if mode == "islands":
        attn_cls = type(transformer.transformer_blocks[0].attn1)
        if not getattr(attn_cls, "_ltx25_dynamo_disabled", False):
            attn_cls.forward = dynamo.disable(attn_cls.forward)
            attn_cls._ltx25_dynamo_disabled = True
    for i, blk in enumerate(transformer.transformer_blocks):
        # dynamic=False: 2つ目の shape で dynamic-shape 再コンパイル(汎用カーネル化)
        # に切り替わるのを禁止し、shape ごとに静的特殊化させる。dynamic カーネルを
        # CUDA Graph に capture すると graph 単体より遅くなることを実機で確認
        # (rt shape 1.83s -> 2.77s の退行)。静的なら shape ごとに compile コスト
        # (~17s)を払う代わりに最適カーネルが capture される。
        transformer.transformer_blocks[i] = torch.compile(blk, fullgraph=fullgraph, dynamic=False)
    print(
        f"[ltx25] compile_blocks={mode}: {len(transformer.transformer_blocks)} blocks "
        f"wrapped (fullgraph={fullgraph}; compile fires on first denoise, "
        f"~{'20s' if mode == 'islands' else '55-170s'})",
        flush=True,
    )
