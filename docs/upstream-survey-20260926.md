# 上流サーベイ 2026-09-26(MiniMax-H3 / LTX-2.5)

前回: `upstream-survey-20260909.md`(+9/10 の検証・確定事項)。本書はそこからの**差分のみ**。
Web調査エージェント2本並列、一次ソース確認済み。

---

## MiniMax-H3

### ★最重要: HyperFlow 8-step LoRA(Video Rebirth、9/19)— ref2va を含む初の few-step LoRA
- https://github.com/Video-Rebirth/hyperflow / https://huggingface.co/videorebirth/hyperflow
  (Sana sol-engine への統合 PR #507 も 9/19 マージ)
- 49 NFE→8step の PEFT LoRA(rank 256 / alpha 256、316 モジュール、2.8GB)。
  data-free flow self-distillation。**t2va / fl2va / ref2va の3タスク全対応**
  (1ファイルで transformer / transformer_ref 両方に適用)。
- **公式実装が diffusers Modular Pipeline 前提**(`pip install hyperflow-h3`)。
  ただし独自機構 **TwoTimeEmbedder**(現時刻 t + step 終端 r の2時刻条件付け、
  base time_embedder をラップ)があり **標準 `load_lora_weights()` では読めない**
  (専用ローダ必須)。sigma グリッドは焼き込み、shift 12/3 = うちの現行と同一。
- 早期評: 「他の turbo LoRA よりカメラ制御・一貫性・ディテールが base に近い」。
  H200 単GPUで ~130s vs ~395s(offload込み、ピーク~80GB)。
- **適用見立て**: 8step 固定なので速度は現行 lightx2v 4step の約2倍 — 価値は
  **ref2va の品質**(FastH3 転用 A/B の品質不採用の穴を埋める候補、
  V3「バランス」tier の代替候補)。移植の肝は TwoTimeEmbedder パッチと
  int8 prequant / AdaLN precompute / Sage の共存確認。
  ライセンス: MiniMax H3 Community License(EU/UK/韓国/米国は別途許諾。日本は制限外)。

### FastH3 V2(9/15)— ref2va は依然未蒸留
- https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2
- LoRA ではなくフル蒸留 ckpt(35B bf16 / GGUF Q4_K_M ~19.8GB)。DMD2 + VSA 80%
  sparsity、8 forward、**shift 10**。**「FL2VA と Ref2VA は蒸留していない」と明記**
  → 最重要ウォッチ(ref2va 学生)は変化なし。VSA sm_120 ネイティブも未着地
  (Triton フォールバックのみ)。うち用途では引き続き見送り。

### Sol-H3 / Sol-Engine — 単GPU方面で前進(ただし優先度は低いまま)
- **Sol-H3 Spark**(9/11): 単一 DGX Spark(GB10, sm_121)公式サポート
  (VAE-free latent transfer + refiner prompt cache、3.92x)。SGLang runtime のまま。
- **Sol-Attn ドキュメントが単GPU(kv_splits 1/2/4)と SM120(RTX 5090)対応を明記**
  — 9/10 の「単GPU拒否」から状況変化。ただしコミュニティの SM120 Triton 実測
  (sumeetprashant/ComfyUI-SolAttn)は **tau 1.4 で +6%**(tau 1.8 は +19% だが破綻)
  と小さく、Sage 導入済みのうちでは上乗せ限定的 → **見送り推奨のまま**。

### その他
- lightx2v: 9/10 以降の新 LoRA なし(fl2v v1.1 fp8 版 ~9/10 が最後)。
- MiniMax 公式: H3.5 等なし。
- **diffusers PR #14839(H3 の single-file/GGUF ロード対応)が 9/23 承認済み・
  0.41.0 マイルストーン** — ピン更新時の選択肢としてウォッチ。
- PDD: ComfyUI ノードの採用は広がるが Experimental のまま。head bank 構造で
  diffusers 移植コスト高 → HyperFlow の後。
- cache-dit: H3 対応依然なし。
- 名前だけ記録: rzgar/minimax-h3_ref2va_8Step_motion_enhancer(ref2va 8step 用
  モーション補強 LoRA、未検証)。

---

## LTX-2.5

### ★fps 分布外仮説の公式裏付け: Slow-Motion-Control LoRA(~9/15)
- https://huggingface.co/Lightricks/LTX-2.5-22b-LoRA-Slow-Motion-Control
- `motion_fps = frame_rate / speed` の定式で **fps 条件値を「モーション速度ノブ」
  として明示的に使う** LoRA(出力 24fps 固定、speed=0.2 で motion 120fps)。
  学習分布を高 motion-fps 側に張り直している。
  = **「fps 条件はモーション時間スケール分布に直結、分布外はダメ」という
  うちの fps=16 揺らぎ発見(9/10)の公式追認**。コミュニティにも「30fps 以上で
  アーティファクト軽減」報告(Phr00t discussions)。20fps 運用の裏付け確定。
  speed 値のパイプライン配線を読む価値あり(可変 fps 対応の一次資料)。

### 検討候補: Pixel-Spatial-Upscaler IC-LoRA(~9/14)
- https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler
- 低解像度動画を参照に 2x 解像度で生成的に再レンダリングする V2V IC-LoRA
  (bf16・rank32)。**公式推奨は「~280p 下書き → 本 LoRA で最終出力」の2段**。
  リアルタイム経路(小解像度 denoise の速さ)を保ったまま最終画質を上げる
  構成として親和性あり。未知数: ComfyUI ワークフローのみ(diffusers 手動ロード
  は可能なはず)/ 蒸留 4step + NVFP4 への LoRA 適用品質 / 第2段の時間予算 /
  CUDA Graph との併用(LoRA ジョブは現状自動 eager)。
- 同時期の Creative Lab IC-LoRA 群(Colorization/Deblur 等)は高速化と無関係。
- InvokeAI が 9/22 に同発想の2段 upscale-and-refine を統合(PR #297)。

### 変化なし
- LTX-2.6 なし、パッケージ v1.3.0 のまま(main 8/26 から動きなし)。
- TurboT2VA の 2.5(22B)対応なし、新蒸留なし。
- sparse attention 新実装なし。SpargeAttn sm_120 の実測所見「小ワークロードでは
  dense Sage が勝つ」= うちの「小 shape では固定費が勝てない」結論の傍証。
- diffusers 本体: docs/minor のみ。
- ローカルリアルタイムの公開事例でうちの 0.37x を上回るものなし。
- 論文: Uncertainty DMD(9月中旬、AR蒸留の多様性回復)、SVEET(9/21、
  ストリーミング編集)— いずれも即戦力ではない。

---

## 推奨アクション(優先度順)

1. **[H3] HyperFlow の A/B**(品質狙い): ref2va を蒸留対象に含む初の few-step LoRA。
   現行 lightx2v ref2v(4step=fast / 8step v1.0=balance)との3つ巴比較。
   diffusers 公式実装ありで移植障壁は TwoTimeEmbedder のみ。8step なので
   「バランス tier の品質更新」が主目的。
2. **[LTX] Pixel-Spatial-Upscaler の2段構成 probe**: 「低解像度リアルタイム +
   生成的アップスケール」の成立性(品質・時間予算・NVFP4 との相性)。
3. **[LTX] Slow-Motion LoRA の speed 配線読解**(軽作業): fps 条件の内部実装の
   一次資料として。
4. ウォッチ: FastH3 の ref2va 蒸留(V2 でも未対応明記)/ diffusers #14839
   (H3 single-file、0.41.0)/ Sol-Attn sm_120 の成熟。
