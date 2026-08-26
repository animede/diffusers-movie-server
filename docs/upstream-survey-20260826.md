# 上流サーベイ(2026-08-26): diffusers / Turbo LoRA / エコシステム

バージョン固定中のバックエンドと上流の差分調査。結論: **固定継続で問題なし。
個別に拾う価値があるのは3点**(下記アクション)。

## diffusers(固定コミット vs 上流)

- **0.40.0 正式リリース済み(08-20)**。MiniMax-H3(Modular のみ)・LTX-2.5 を含む
- **H3 側(固定 f37ab93e)**: 固定以降の実質変更は3件のみ。
  結合点(before_denoise.py / denoise.py / scheduler、= Audio Drive・中断・参照短辺が
  依存する内部)は**無傷**
  - #14407 context-parallel 対応(opt-in、無関係)
  - #14408 `MiniMaxH3LoraLoaderMixin`(将来 LoRA を公式機構で載せるなら)
  - **#14464 `d5baa4fb`: VAE の低精度 dtype バグ修正 + audio VAE への
    `_supports_group_offloading=False` ガード → 個別に取り込む価値あり**
- **LTX-2.5 側(固定 11a82a15)**: 実質差分なし(typo 2件 + LTX2Guidance の移動、
  非結合を grep 確認済み)。**実質最新**
- 既知の未解決 upstream issue: #14379(マルチGPU device mismatch — こちらの
  probe_te_on_second_gpu.py が扱う問題そのもの。公式未修正)

## Turbo LoRA(lightx2v/Minimax-h3-Turbo)

- **`minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` が存在**(ファイル一覧を API で
  実確認済み)。**現在は fl2v 用 v1.0 768p を ref2va に転用しており、専用版を未使用**
- fl2v は v1.1(08-20)まで進行。コミュニティ推奨強度 1.2〜1.3(discussions/42)。
  ref2v 版は v0.1 のまま。**ref2va 専用の本命 Turbo は「訓練中・時期未定」**(discussions/14)
- turbo 時の ghosting はコミュニティでも既知(discussions/33, /44)— 当方の実測
  (seed 9999 二重露光、30steps ではクリーン)と整合
- 切替は `H3_TURBO_LORA_FILE` env で可能(runner.py:933)。**注意: shift の自動判定は
  ファイル名の `_768p` 有無で行われる**(runner.py:991-997)ため、ref2v v0.1(768p 表記
  なし)は非 768p 側の shift になる。A/B 時にこの妥当性を確認すること

## LTX-2.5 エコシステム

- 公式の目玉は diffusion video decoder(既存メモの品質診断と一致)
- **NATTEN 無しのフォールバックは約5倍遅い**(SGLang doc)→ 既知の残課題
  「torch2.11+NATTEN」の優先度上げ

## アクション(優先順)

1. **ref2v turbo v0.1 の A/B**(即可能・10分): ghost seed 9999 を再現条件に、
   `H3_TURBO_LORA_FILE` 切替で現行転用と比較。高速モードの破綻率が下がるか
2. **H3 の diffusers 固定を `d5baa4fb` 以降へ更新**(低リスク・実利あり): dtype バグ+
   group offload ガード。venv 更新+回帰確認をセットで
3. **LTX-2.5 の NATTEN 導入**(別タスク): decode 5倍差
4. fl2v v1.1 は「キャラクタイメージMV」で FLF を使う段に採用検討(強度 1.2〜1.3 から)
