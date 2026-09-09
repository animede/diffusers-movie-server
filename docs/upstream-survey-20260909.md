# 上流サーベイ 2026-09-09(MiniMax-H3 / LTX-2.5 高速化)

前回: `upstream-survey-20260826.md` + メモリ `h3-upstream-speedups-20260902`(〜09-03)。
本書はそこからの**差分のみ**。高速化最優先の観点で調査(Web調査エージェント2本並列、一次ソース確認済み)。

---

## MiniMax-H3

### ★最大の新規物件: Sol-H3 スタンドアロンランタイム(NVlabs/Sana PR #484、09-07 マージ)
- https://github.com/NVlabs/Sana/pull/484
- H3 専用ランタイム。**T2V / I2V / Ref2VA 対応、4-step + SOL/BSA sparse attention 既定、
  「1/2/4/8-GPU execution」と単GPU実行を明記**、"Diffusers-compatible preprocessing mode" あり。
- 09-05 時点のコミュニティ記録(matsuo-koya)では「Sol の sparse カーネルは単GPU拒否」
  だったのが、この PR で変わった。sol-attn は既知の RTX 5090(sm_120)プロファイルを
  持つため、**sm_120 単GPUで ref2va を sparse+4step で回せる可能性がある唯一の上流物**。
- 検証実績は 8×B300 のみ(T2V 124f 1.669s 等)。**sm_120 動作は未確認 → 精査が次アクション最有力**。
  選択肢は「ランタイムごと使う」か「BSA/sol-attn 部分だけ diffusers サーバへ移植」の二択。

### すぐ試せる: lightx2v ref2v turbo 8-step v1.0 768p(09-03 公開)
- https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/51
- `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`。**非-ComfyUI 標準形式も同時配布**
  (下記の重み順序の罠を回避可)。推奨: 8 steps / video shift 12 / audio shift 3 / euler / 〜768p。
- 参照保持・音声品質の向上を主張(コミュニティ評は賛否混在)。8step なので速度は現行
  4step v0.1(53s)のほぼ倍 → **品質優先モードとして併存**が現実的。`H3_TURBO_LORA_FILE`
  差し替えだけで試せる。**注意: ファイル名 `_768p` → shift 自動判定の挙動を要確認**
  (このLoRAの推奨 video shift は 12 で、8step_v1.0_768p の 6 とは異なる)。
- 併せて FL2V turbo 4-step v1.2 768p(09-04)も出た(fl2v 利用時のみ関係)。

### ref2va への sparse 化の設計図: VSA ゲート転植(Kablex/ComfyUI-Ref2VA-VSA)
- https://github.com/Kablex/ComfyUI-Ref2VA-VSA
- FastVideo VSA の学習済みゲート行列50個を **ref2va ベースモデルへ無学習トランスプラント**。
  参照トークンは dense のまま生成側だけ疎化(`Ref2VAVSAGatePatch`)。RTX 4090 で 5秒動画
  ~72s(ネイティブ比 9x)、ピーク 13.5GB。カーネルは sol_attn(CUDA)。
- ref2va 学生モデル不在の間、こちらの ref2v 経路に最も直接効きうる新アイデア。
  turbo 4step と併用できれば 53s からの上積み候補。ComfyUI 実装なので移植工数あり。
- 関連: barelymining/ComfyUI-MiniMax-H3-FastVideo が **Triton 版 VSA を RTX 3090Ti/5090 で
  実動**(sm_90a 必須の CUDA カーネルを回避、topk_ratio=0.10 で ~90% スキップ、
  RTX 3090Ti 0.8MP 5s で 20-step 比 ~10x。Triton は CUDA 比 ~2.5x 遅いが全GPUで動く)。
  → **VSA が sm_120 で原理的に動くことのコミュニティ実証**。

### FastH3 監視条件の結論(前回の2条件)
- **ref2va 学生モデル: 依然未公開**。HF `FastVideo/FastVideo-Minimax-FastH3-Preview-v0.2` は
  08-23頃から更新なし、`transformer_ref` 無し・T2VA のみ。ロードマップ文言も変化なし。
- **VSA 公式 CUDA カーネル: 依然 sm100a のみ**(ただし上記 Triton 経路が抜け道)。
- 新: LoRA 抽出版 `drozbay/MiniMax-H3-FastH3-Preview-LoRA`(rank128・1.33GB、full attention で
  動作)。t2va/fl2va 系のみで現状は動機薄。**FastH3 の蒸留差分は adaln 重みに集中**という
  分析(matsuo-koya)があり、**適用時は AdaLN precompute テーブルの再構築が必須**。

### コミュニティ実測ノート(有用な副産物): matsuo-koya/minimax-h3-notes
- https://github.com/matsuo-koya/minimax-h3-notes(RTX 5090 / ComfyUI、08-14〜09-05)
- **【罠】diffusers と ComfyUI で fused FC の重み順序が異なる(`[value;gate]` vs `[gate;value]`)。
  shape が一致したまま LoRA 変換が壊れる**。変換ツール `h3_lora_convert.py` 公開。
  → ComfyUI 形式 LoRA を扱うときの必読事項(lightx2v は標準形式配布があるので回避可)。
- INT8 attention(comfy-kitchen)でサンプラー 2x の実測。
- 「効かなかったもの」記録: Sage attention / FastVideo VSA(CUDA版)/ Sol-H3 sparse
  (当時=09-05、単GPU拒否 → 09-07 の PR #484 で状況変化)/ Zironic H3MemoryOptimization。

### 変化なし・不可
- cache-dit: H3 上流対応なし(supported matrix に依然無し)。
- MiniMax 公式: H3.5 / 公式蒸留のアナウンスなし。
- diffusers 本体: 9月の H3 関連マージなし(既知: 08-14 #14408 で LoRA 公式サポート、
  08-05 #14371 に breaking change あり — ピン更新時の注意点)。
- **Sol-Engine の fused MXFP8(PR #491、09-09)は SM100 専用で sm_120 明示 reject → 使えない**。
- 参考(多GPU・技法カタログ): vLLM-Omni ブログ 09-01(QKNorm+RoPE融合・online FP8・
  SVDQuant W4A4 等を列挙、RTX 5090 / RTX PRO 5000 向けレシピの存在に言及)、
  LMSYS 08-27(8×H200、lossless 1.95x)。

---

## LTX-2.5

### ★本命課題への直接回答: 公式 CUDA Graph capture 参照実装が存在する
- https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/model/transformer/cudagraph_capture.py
  (v1.2.0=08-11 導入、v1.3.0=08-26 改良)
- **transformer ブロックループ全体を `torch.cuda.CUDAGraph` で手動 capture** し static buffer
  から replay。**RoPE prepare と出力 projection は eager のまま**(capture 範囲をブロック
  ループに限定 → torch.compile の graph break 問題を丸ごと回避する設計)。
  graph は (input shape, perturbation signature) ごとに1本、`_InputBufferPool` で複数 capture
  が最大トークン数サイズの入力バッファを共有(v1.3.0 の `--compile max_video_tokens=N`)。
- **純 PyTorch なので sm_120 / torch 2.11 / diffusers 自前サーバへ移植障害なし**。
  realtime 用途(288×384 固定 shape 連投)は shape 数が少なく理想的なユースケース。
  固定費 ~2.0s(カーネル起動律速)の大半を replay に畳める見込み。
- 代替案: PyTorch 公式ブログ(04-08)の diffusers レシピ
  `compile_repeated_blocks(fullgraph=True)` + `reduce-overhead`(CUDA Graphs)。
  両立には「各ブロックを入力 clone するラッパで包む」トリックが要る
  (https://github.com/sayakpaul/diffusers-blackwell-quants/pull/1)。固定 shape 連投なら
  手動 capture の方が制御しやすい。

### 公式 prequant NVFP4 distilled チェックポイントの更新に注意
- `Lightricks/LTX-2.5` の `ltx-2.5-22b-distilled-transformer-nvfp4.safetensors` が
  **08-16 に「Restore keyframes support」で更新**されている。古い snapshot を
  キャッシュしている場合は要確認・要比較(自前 nvfp4 キャストとの品質比較材料にも)。

### 次の伸びしろ候補: TurboT2VA(thu-ml、08-25 論文 / 08-27 実装 / 09-07 v2)
- https://arxiv.org/abs/2608.24674 / https://github.com/thu-ml/TurboDiffusion/tree/main/turbot2va
- LTX-2 **19B**(2.5 の 22B ではない)を rCM で 4-step 蒸留 + SageSLA(topk=0.3)+
  text-context trimming で累積 54.67x(H20)。蒸留重み公開済みだが **2.5 には直接使えない**。
- 価値は「**SageSLA topk=0.3 でも品質維持・text-context trimming が LTX 系 DiT で成立**」の
  実証。CUDA Graphs 完了後の attention 側の 2〜2.8x 余地の示唆。ただし Sage 系の
  sm_120 動作検証が前提(CLAUDE.md 21番の系譜)。W8A8+FastNorm(TileLang)単体 1.37x も実測あり。

### LTX-2 パッケージ v1.3.0(08-26)のその他
- DFR 4K を tiled spatial epilogue 化 / keyframe-aware diffusion-VAE decode(要対応VAE ckpt)/
  `--diffvae-optimization combined_compile` が NATTEN 必須でなくなった(Triton→PyTorch
  フォールバック)/ dfr_mgpu(多GPU、非該当)。

### 変化なし・その他
- **LTX-2.6 / 新モデル: なし**(HF org・GitHub org の日付を直接確認)。
- diffusers 本体: #14567(08-31)で **LTX-2.5 DFR pipeline が本体入り**(品質系)。
  #14694(open)は diffusion decoder forward のリファクタ予告(bit-exact、性能影響なし)。
  高速化目的のマージなし。
- taehv が LTX-2.5 対応を正式明記(08-24 README、2.3 用重みの互換確認。新規学習ではない)。
- TeaCache/FBC の LTX-2.5 新事例なし。Nunchaku/SVDQuant の LTX 対応の証拠なし。
- コミュニティのローカル real-time 達成事例は未発見 → **こちらの 0.78x(nvfp4-fast+4step)は
  公開事例より進んでいる**。

---

## 推奨アクション(優先度順)

1. **[LTX] 公式 cudagraph_capture.py を参照に denoise ブロックループの手動 CUDA Graph
   capture を移植** — 残っていた本命課題への完成した参照実装。realtime の固定費 ~2.0s 圧縮に直結。
2. **[H3] Sol-H3 ランタイム(Sana PR #484)の sm_120 精査** — 単GPU+Ref2VA+4step+sparse を
   謳う唯一の上流物。動作可否と「移植 vs ランタイム利用」の判断。
3. **[H3] lightx2v ref2v 8-step v1.0 の A/B** — env 差し替えだけ。品質優先モードとして併存評価。
4. **[LTX] NVFP4 prequant ckpt の 08-16 版確認**(キャッシュ整合+品質比較)。
5. **[H3] VSA ゲート転植(Kablex 方式)** — 2 の結果次第で。sol_attn sm_120 と組み合わせ。
6. **[LTX] SageSLA/context trimming(TurboT2VA)** — 1 の後の attention 側伸びしろ。sm_120 検証が前提。
7. ウォッチ継続: FastH3 ref2va 学生(公開時は AdaLN precompute 再構築が必要な点をセットで)。
