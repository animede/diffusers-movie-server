# H3 transformer int8 事前量子化キャッシュ (H3_TRANSFORMER_PREQUANT)— 2026-08-27

33B transformer(`transformer` / `transformer_ref` の2インスタンス)の torchao int8
量子化済み重みをディスクへ保存し、次回以降のロードで「bf16 66GB 読み込み + その場
量子化」を丸ごと省略する。`H3_TE_PREQUANT`(bnb-4bit TE の事前量子化キャッシュ)と
同じ設計思想の transformer 版。

## Phase 1: 直列化の実現可能性(probe、実装前に確認)

- **結論: diffusers の `save_pretrained()`/`from_pretrained()` の既定
  (`safe_serialization=True`)がそのまま使える。**
- torchao 0.17.0 は `torchao.prototype.safetensors.safetensors_support` の
  `flatten_tensor_state_dict` / `unflatten_tensor_state_dict` を提供し、pinned
  diffusers(f37ab93)の `TorchAoHfQuantizer.get_state_dict_and_metadata()` /
  `maybe_update_state_dict()` がこれを正しく呼ぶ(torchao>=0.16.0 ゲート、
  ソース確認済み)。`Int8WeightOnlyConfig(version=2)` が生む `Int8Tensor` は
  `ALLOWED_TENSORS_SUBCLASSES` に含まれる。**safetensors ネイティブ対応であり、
  `torch.load(weights_only=False)` のような pickle 経路は不要。**
- 小型 ModelMixin(Linear 数層 + modules_to_not_convert 相当の除外層)での
  round-trip probe(scratchpad `h3_prequant_tf/probe_tf_prequant.py`):
  量子化 → save → 別インスタンスへ load → 固定入力 forward が
  **`torch.equal` で完全一致(max_abs_diff 0.0)**。

## 実装(core/runner.py)

- 新規 env:
  - `H3_TRANSFORMER_PREQUANT`(既定 `"1"`、TE と同じ既定 ON)
  - `H3_TRANSFORMER_PREQUANT_DIR`(既定 = `H3_TE_PREQUANT_DIR` = `models/prequant/`)
  - `H3_TRANSFORMER_PREQUANT_MIN_FREE_GB`(既定 40。TE の 25 より大きい —
    保存物が ~34GB/インスタンスのため)
  - `H3_TRANSFORMER_PREQUANT_MIN_RAM_GB`(既定 15。保存前のホスト RAM ガード)
- キャッシュ先: `models/prequant/transformer_int8/` と
  `models/prequant/transformer_ref_int8/`(インスタンスごとに別ディレクトリ)。
  中身は `config.json` + sharded safetensors(10GB シャード×4)+ `meta.json`。
- **メタデータ無効化キー**(`meta.json`、ロード時に完全一致を要求。不一致 =
  キャッシュ無効として通常経路へフォールバック + 再保存):
  `model_id` / `source_snapshot`(HF キャッシュのスナップショットパス =
  コミットハッシュ入り、`try_to_load_from_cache` で安価に取得)/
  `torchao_version` / `torch_version` / `quant_config` / `modules_to_not_convert`。
- 保存点: `_ensure_transformer` / `_ensure_transformer_ref` の量子化直後、
  **turbo LoRA 構造 wrap・attention backend・FBC・AdaLN precompute より前**。
  turbo LoRA は Linear モジュール自体を差し替えるため、キャッシュに焼き込むと
  turbo 無効リクエストが壊れる。attn/FBC/AdaLN は `state_dict()` に影響しない
  (ソース確認済み)。
- fail-open: 保存失敗(ディスク不足・RAM 不足・例外)は警告ログのみで生成続行
  (`_save_te_prequant` と同じ方針)。tmp ディレクトリ + rename のアトミック保存。
- `H3_LOWVRAM_GROUP`(group offload、CPU ロード)経路は対象外(無変更)。

### 実装中に発見した罠: `modules_to_not_convert` の in-place 汚染

diffusers の `TorchAoHfQuantizer._process_model_before_weight_loading()` は
`quantization_config.modules_to_not_convert`(= runner.py がそのまま渡した
モジュールレベルのリスト `H3_INT8_MODULES_TO_NOT_CONVERT`)を **in-place で
extend する**(keep_in_fp32_modules の "rope" 追加 + ロードのたびに重複追加)。
初回実装のメタデータはこの汚染されたリストを記録してしまい、次回起動時の
pristine なリストと不一致 → キャッシュが永遠にヒットしないバグになった。
対策: 定義直後に `_H3_INT8_MODULES_TO_NOT_CONVERT_PRISTINE = tuple(...)` の
スナップショットを取り、メタデータはそれを使う(量子化の実効レシピは重複や
"rope" の有無で変わらないため比較キーとしては pristine 版が正しい)。

## Phase 3 実測(2026-08-27、RTX PRO 6000 Blackwell 96GB、gateway 経由)

### ロード時間(h3.log の実測値)

| 対象 | OFF(quantize-at-load) | ON 初回(quantize+save) | ON キャッシュヒット |
|---|---|---|---|
| transformer(起動時 preload) | 40.7s | 39.3s(うち保存 26.2s、bf16 シャードがページキャッシュ温存で量子化自体は短縮) | **14.3s / 16.2s** |
| transformer(毎リクエスト後の restore) | 40.1s | - | **6.9s / 6.8s** |
| transformer_ref(初回 ref2va 時) | 38.3s | 63.5s(うち保存 26.2s) | **17.6s** |

- キャッシュヒットは 2.4〜5.8 倍高速。**bnb-4bit TE モードでは毎 t2va リクエストの
  decode 後に transformer を再ロードするため、per-request 固定費も 40s → 7s へ短縮
  される**(起動時だけの恩恵ではない)。
- 保存時のピーク VRAM に注意: `flatten_tensor_state_dict` が state_dict 全体を
  GPU 上で `.detach().clone()` するため、**モデルサイズ分(~34GB)の一時 VRAM が
  追加で要る**。実測: 起動時の transformer 保存 66.33GB、TE 常駐下の
  transformer_ref 保存 **89.64GB**(96GB カードでぎりぎり成立。両 transformer +
  TE 常駐状態で保存が走る構成では OOM しうる — その場合も fail-open で生成は
  続行し、キャッシュが書かれないだけ)。

### ディスク・RAM

- キャッシュサイズ: **34.03GB × 2 = 68.06GB**(`du -sh` で各 32GiB)。
  検証後の `df -h /`: 66GB free(開始時 130GB)。
- 保存中のホスト RAM: available 70GB 以上を維持(10GB シャード単位の直列化のため
  モデル全体の CPU 複製は発生しない)。スワップ増加なし。

### 重み等価性(完全検証、bit-exact)

`scratchpad/h3_prequant_tf/probe_layer_equiv_full.py` / `probe_plain_equiv.py`
(GPU:1 で実行、キャッシュ safetensors vs 元 bf16 シャードからの独立再量子化):

- `transformer_int8`: 量子化 350 層すべて qdata+scale が `torch.equal` で一致、
  非量子化 288 テンソルすべて元シャードと bit 一致、zero_point 350 個すべてゼロ。
- `transformer_ref_int8`: 同上(350 + 288 すべて一致)。
- **結論: キャッシュは quantize-at-load が生成する重みと完全に bit 一致する。**

### 出力比較(framemd5)と、そこで判明した重要知見

| 比較 | 結果 |
|---|---|
| t2va ON run1 vs ON run2(**同一プロセス内**、seed=12345) | 映像・音声とも **framemd5 完全一致** |
| ref2va OFF(boot E1)vs OFF(boot E2)(**別プロセス・同一構成**、対照実験) | **framemd5 完全一致**(PSNR ∞) |
| ref2va ON(boot D)vs ON(boot F)(**別プロセス・同一構成・両方キャッシュヒット**、対照実験) | **framemd5 完全一致** |
| t2va OFF(boot A)vs ON(boot C) | 全 124 フレーム相違、PSNR avg 25.7dB(視覚的にはほぼ同一の構図・被写体) |
| ref2va OFF(boot E1)vs ON(boot D) | 全 192 フレーム相違、PSNR avg 20.0dB(同上) |

- **各ロード経路は完全に決定的で自己一貫**: 同一構成なら別プロセスでも bit 一致
  (OFF↔OFF、ON↔ON とも)。生成のプロセス内再実行も bit 一致。
- **OFF↔ON(quantize-at-load ↔ キャッシュロード)だけが恒常的に食い違う。**
  重みは完全 bit 一致・単一層の forward も bit 一致(実7168×5376層で
  qdata/scale/zero_point/block_size/属性/forward すべて一致を確認済み)なので、
  原因はモデル側ではなく**ロード経路ごとに異なる GPU メモリ割り当てレイアウト
  (bf16 66GB を段階確保→量子化 vs int8 を直接確保)に依存する
  cuBLAS/attention カーネル選択の違い**(縮約順序の違い → bf16 丸めの違い →
  denoise 30/4 steps で増幅)と推定される。sage attention・cuBLASLt はどちらも
  アドレスアライメント依存のヒューリスティクスを持ち、割り当て履歴が
  構成ごとに決定的なため「同一構成間は一致・構成間は恒常的相違」という
  観測と完全に整合する。
- 当初の検証基準(「int8 量子化は決定的だから出力 mp4 が bit 一致するはず」)は
  **「同一重み → 同一出力」がロード経路の違いを跨ぐと成立しない**という点で
  前提が崩れていた。キャッシュの正しさは (1) 重み全テンソルの bit 一致、
  (2) 単一層 forward の bit 一致、(3) キャッシュヒット同士の出力 bit 一致、の
  3点で立証した。**キャッシュ有効化は「seed 固定の過去出力とはわずかに違う
  (品質的には等価な)出力になる」一度きりのシフトを伴う**(ライブラリ更新と
  同種の影響)。以後はキャッシュヒット同士で完全に再現的。

## 運用上の注意

1. **既定 ON のため、次回プロセス再起動からキャッシュが使われる**(初回起動は
   保存 26s×2 が乗り、以後は起動 40.7s→14s・per-request restore 40s→7s・
   初回 ref2va 38s→17s)。
2. 保存時のみ一時 VRAM +34GB(state_dict の GPU 上クローン)。TE 常駐下の
   transformer_ref 保存は実測ピーク 89.64GB(96GB 専有でのみ安全)。失敗時は
   fail-open(生成続行・キャッシュなし)。
3. seed 固定の再現性が過去出力と厳密に一致する必要がある場合は
   `H3_TRANSFORMER_PREQUANT=0` で旧経路に戻せる(戻すと OFF 系の出力に完全一致
   する — 対照実験で確認済み)。
4. HF snapshot 更新・torchao/torch 更新・量子化レシピ変更は meta.json 不一致 →
   自動再quantize+再保存(古い重みを黙って読む事故は構造的に防止)。
