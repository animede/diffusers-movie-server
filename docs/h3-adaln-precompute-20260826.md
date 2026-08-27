# H3 AdaLN precompute の移植と検証(2026-08-26)

`docs/h3-baseline-comparison-20260826.md` の「関連」節で予告されていた、
NVIDIA Sol-Engine(`sana-sol-engine` ブランチ、Apache-2.0)の AdaLN precompute を
`backends/minimax-h3` へ移植した。目標(bit-exact・bf16 transformer ~24GB削減)は
達成、ただし移植先の常駐パターン(`H3_TE_QUANT=bnb-4bit` の decode窓ごと
transformer 解放+再ロード)により「削減がどこで観測できるか」が Sol-Engine の
想定と異なる点を確認したので、その内訳も記録する。

## 移植元の機構(要約)

`models/minimax_h3/GB200/adaln.py`。MiniMax-H3 の 33B パラメータのうち約13B が
各ブロックの `adaln_proj`(`Linear(2688 -> 6*5376*3)`、50ブロック)に集中しており、
入力はタイムステップ埋め込み(サンプリングスケジュールのみに依存)のみ。
このスケジュールはデノイズループに入る前に確定する(`row_timestep_plan`)ため、
軌道全体の変調テーブルを一度だけ計算してキャッシュし、`adaln_proj` の重みを
解放できる。1ステップ1GEMMを参照実装と全く同じ形状で回すため、キャッシュ後の
値は無キャッシュ経路とビット単位で同一(品質ゲート不要、`ffmpeg framemd5` の
完全一致で検証する設計)。

キーとなる仕組み:
- `PrecomputedModulation`(`nn.Module`)が `(num_steps, num_rows, 6*hidden)` の
  テーブルを保持し、`_StepCursor`(デバイス上のテンソル、`torch.compile` 互換の
  ための設計。本プロジェクトでは compile 未使用だが移植元の設計をそのまま踏襲)で
  現在のステップを指す。
- `MiniMaxH3LoopDenoiser.__call__` をクラスレベルで monkeypatch し、リクエストの
  最初のデノイズステップ(`i==0`)で `precompute()` を実行、`adaln_proj` を丸ごと
  `PrecomputedModulation` に差し替えて元の重みを解放する。
- `MiniMaxH3Ref2VALoopDenoiser` は `MiniMaxH3LoopDenoiser` のサブクラスで
  `__call__` を override していないため、ベースクラスへの1回のパッチで
  t2va/fl2va(`transformer`)・ref2va(`transformer_ref`)の両方をカバーできる
  (本プロジェクトのpinned diffusersコミットで実機確認済み)。

## 移植内容

- 新規ファイル `backends/minimax-h3/core/adaln_precompute.py`: 上記機構を
  ほぼそのまま移植(`precompute()`/`PrecomputedModulation`/`_StepCursor`/
  `enable_adaln_precompute()`)。呼び出し口だけこのプロジェクト向けに書き換えた
  (`self.transformer_name` を使い `transformer`/`transformer_ref` の両方を
  汎用的にカバーする形にした -- 移植元の GB200 スクリプトは `components.transformer`
  固定で ref2va を扱っていなかったため、ここは移植元を単純コピーではなく
  一般化した拡張)。
- `core/runner.py`: 新しい環境変数 `H3_ADALN_PRECOMP`(既定 `"0"` = 完全に無変更)。
  - `_ensure_transformer()` / `_ensure_transformer_ref()` の、ロード成功直後
    (FBC 有効化・turbo LoRA 適用と同じ地点)で `enable_adaln_precompute()` を呼び、
    新しくロードされたインスタンスを「武装」する。この呼び出しは「ロードのたびに」
    必要:本プロジェクトの既定 `H3_TE_QUANT=bnb-4bit` 常駐パターンは、
    **リクエストごとの decode 窓の前後で `transformer` を丸ごと解放・再ロード**
    するため(`_free_transformer()` → decode → `_ensure_transformer()`)、
    precompute で構築したテーブルもその解放でリセットされる。再ロード時に
    再武装しないと、2件目以降のリクエストで永久に無キャッシュ経路に戻ってしまう。
  - `status()` に `adaln_precomp`(env フラグ)と `adaln_precomp_built`
    (現在常駐している `transformer`/`transformer_ref` インスタンスに実際に
    テーブルが構築済みかどうか -- 武装はロード時点、構築はそのインスタンスの
    最初のデノイズステップ時点なので両者はロード直後にはズレる)を追加。
- `core/settings.py`: `validate_instant_settings()` に
  `turbo=True かつ H3_ADALN_PRECOMP=True` を 400 で拒否するガードを追加。
  `current_settings_snapshot()` の `constraints` にも
  `turbo_incompatible_with_adaln_precomp` を追加(UI 用)。

## ガード行列(v1スコープ: bf16 transformer のみ)

| 組み合わせ | 結果 | 検証箇所 |
|---|---|---|
| `H3_ADALN_PRECOMP=1` + `H3_TRANSFORMER_QUANT=int8` | import時 `RuntimeError` | 単体テスト、実機確認済み |
| `H3_ADALN_PRECOMP=1` + `H3_LOWVRAM=1` または `group` | import時 `RuntimeError`(int8自動昇格経由で同一メッセージ) | 単体テスト、実機確認済み |
| `H3_ADALN_PRECOMP=1` + `H3_TURBO_LORA=1`(env既定) | import時 `RuntimeError` | 単体テスト、実機確認済み |
| `H3_ADALN_PRECOMP=1`(env)+ リクエスト単位 `turbo=true` | `POST /api/t2va` が 400(サーバは健全なまま) | 実機確認済み(下記) |
| `H3_ADALN_PRECOMP=1` + `cache=fbc` | 正常動作、キャッシュも有効 | 実機確認済み(下記) |
| `H3_ADALN_PRECOMP=0`(既定) | 完全に無変更(diffのみ、既存ロジック削除なし) | コード差分確認 |

### turbo との組み合わせが構造的に不可能な理由

turbo LoRA チェックポイント(`larryvrh/MiniMax-H3-Turbo-Lora` /
`lightx2v/Minimax-h3-Turbo` のどちらも)は各ブロックの `adaln_proj.linear` を
`_TurboLoRALinear` でラップする対象キー(`blocks.N.adaln_proj.linear`)を持つ。
このラップは **リクエスト単位で on/off がトグルされる**
(`_TurboLoRALinear.enabled`、`set_turbo_lora_enabled()`)、常駐 transformer
インスタンスに対する仕組みである一方、AdaLN precompute は**固定スケジュールから
一度だけテーブルを焼き `adaln_proj` を削除**する。同じ常駐インスタンスで
turbo=True と turbo=False の両方に応えることはできず(削除された
`adaln_proj.linear` に後から LoRA を再ラップする経路も存在しない)、
出力を静かに間違えるくらいなら明確な 400 で拒否する方針とした
(タスク指示どおり)。

## 検証手順と結果

環境: gateway(`http://127.0.0.1:8630`)経由で `backends/minimax-h3`
(port 8631)をロード。GPU は RTX PRO 6000(96GB、共有)、`CUDA_VISIBLE_DEVICES=0`。

### 1. 復元用ベースライン確認

検証開始時点で稼働していた本番設定を記録:
`preset=96gb-int8`、`env_extra={H3_TRANSFORMER_QUANT: int8,
H3_REF_PREFIX_CACHE_SINGLE: 1, H3_VOCAL_LOCK: 1,
H3_TURBO_LORA_FILE: minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors}`、
`gpus=0`。常駐VRAM 55.04GB。

### 2. `H3_ADALN_PRECOMP=1` でロード

`POST /api/v1/backend/load {"backend":"h3","preset":"96gb","gpus":"0",
"overrides":{"H3_ADALN_PRECOMP":"1"}}`。プリロード完了後、
`/api/status` で `adaln_precomp: true`、`transformer_quant: none`(bf16)、
常駐 `gpu.allocated_gb: 87.29`(= bf16 transformer 66.28GB + TE-nf4 21GB、
precompute 未発火の初期状態)を確認。

### 3. baseline replica(`turbo=false`, `cache=none`)

`t2va`、768x768、5.0s、seed=12345、同一プロンプト
(「ジャズクラブで歌う女性」)を2回連続で実行。

**ログで precompute の発火を確認**(1回目):

```
[h3opt.adaln] cached 50 blocks x 29 steps: table 0.52 GB, freed 24.23 GB of weights
```

**framemd5 完全一致検証**(`ffmpeg -map 0:v -f framemd5` / `-map 0:a -f framemd5`、
比較対象は同日生成済みの基準 `t2va_1787735332.mp4` = r1、
`docs/h3-baseline-comparison-20260826.md` 参照):

| 実行 | 出力 | video framemd5 | audio framemd5 |
|---|---|---|---|
| replica 1 | `t2va_1787740500.mp4` | **IDENTICAL** | **IDENTICAL** |
| replica 2 | `t2va_1787740775.mp4` | **IDENTICAL** | **IDENTICAL** |

`diff` の出力が空(完全一致)であることを2回とも確認した -- bit-exact の主張は
検証済み。2回目の実行でも `[h3opt.adaln] cached ...` のログが再度出ており
(1回目の decode 窓での transformer 解放後、再ロード時に再武装されたことの
裏付け)、2回目の出力も同一seedで完全一致するため、**「解放→再ロード→
再武装→再構築」のサイクルが決定論的であること**も合わせて確認できた。

### 4. VRAM(削減が実際にどこで観測できるか)

**重要な発見**: レスポンスの `peak_vram_gb`(request開始時に
`torch.cuda.reset_peak_memory_stats()` する設計)は、**このプロジェクトの
`H3_TE_QUANT=bnb-4bit` 常駐パターンでは削減を直接反映しない**。
理由: reset は「transformer(未precompute、bf16、66.28GB)+ TE-nf4」が
**既にロード済みの状態**で呼ばれるため、その後 precompute が重みを解放しても、
reset 時点のフロア(≈87.29GB)より下がることはなく、`peak_vram_gb` は
むしろこのフロア+活性化メモリの合算になる(実測 87.59GB / 87.85GB、
`docs/h3-baseline-comparison-20260826.md` の r1 実測91.24GBとほぼ同水準)。

削減は **request の中で `/api/status` を直接ポーリングする**ことで確認できた:

| 時点 | `gpu.allocated_gb` | `adaln_precomp_built.transformer` |
|---|---|---|
| プリロード直後(precompute未発火) | 87.29 | `false` |
| デノイズ step 1(precompute発火直後) | 64.80 | `true` |
| デノイズ step 3〜28(発火後、安定) | 63.07〜64.80 | `true` |
| decode窓(transformer解放) | 37.66 | `false`(モジュールごと消える) |
| 次リクエストへ向け再ロード完了 | 87.56(bf16、precompute前) | `false` |

87.29GB(precompute前、TE-nf4常駐込み)→ 63〜65GB(precompute後、
デノイズ全域で安定)で、**transformer 単体では 66.28GB → 約42GB相当の削減**
(87.29 - 66.28 = TE-nf4分21.01GBを差し引くと transformer 単独の常駐は
66.28GB、precompute後の63〜65GBからTE-nf4 21GBを引くと transformer側は
約42〜44GB -- ログの `freed 24.23 GB` とほぼ整合)が確認でき、
**目標の「66.3GB → ~42GB」を実測でも裏付けた**。

**教訓(このプロジェクト固有の事情)**: Sol-Engine の GB200/H100 環境は
transformer が常駐したままの定常状態を前提にしているため、`peak_vram_gb`
のようなプロセス全体のピーク値がそのまま「削減量」を表す。本プロジェクトの
既定運用(`bnb-4bit`、decode窓ごとの解放+再ロード)ではリクエストの
レスポンス側指標だけでは削減が見えず、**request 中の `/api/status` ポーリング
でしか削減を直接観測できない**。将来この技術を他の常駐パターン
(例: `H3_KEEP_TRANSFORMER=1`)と組み合わせれば、`peak_vram_gb` 自体にも
削減が反映されるはずだが、本タスクでは v1 スコープ(bf16、既定の
`bnb-4bit` decode窓解放パターン)のみ検証した。

### 5. turbo replica

`turbo=true` でのリクエストは、タスクブリーフが想定していた
「turbo baseline との比較」ではなく、**設計どおり 400 で拒否**された
(上記ガード行列参照)。実際のレスポンス:

```
{"detail":"turbo=1 is not supported while H3_ADALN_PRECOMP=1 ..."}
```
`HTTP_STATUS:400`

拒否後もサーバは健全(`busy: false`、`transformer_loaded: true`、
VRAM 変化なし)であることを確認した。これは「turbo=true replica を
baseline と比較する」というタスクの検証手順そのものが、v1スコープの
設計(turbo と adaln precompute は併用不可)の下では実行不可能であることを
意味する -- 検証の失敗ではなく、ガードが意図どおり機能した結果。

### 6. FBC(`cache=fbc`)との併用確認

`H3_ADALN_PRECOMP=1` + `turbo=false` + `cache=fbc` で1回実行、完走を確認:

- `cache_mode: "fbc"`, `cache_skipped_steps: 7`(29ステップ中7ステップをスキップ)
- `denoise_time_s: 101.84`(同条件・cache無しは 132.15s / 134.39s、
  約22-24%短縮)
- 出力 `t2va_1787740984.mp4`(バイト単位の一致検証は未実施 -- FBCの
  residual-similarity skip はしきい値ベースで厳密な bit-exact 保証を
  意図した機構ではないため、baseline との framemd5 比較は行っていない。
  FBCが「完走し、想定通りステップをスキップして速くなる」ことの確認に
  スコープを絞った)。

### 7. 復元確認

`POST /api/v1/backend/load` で `preset=96gb-int8` + 元の overrides
(`H3_REF_PREFIX_CACHE_SINGLE=1, H3_VOCAL_LOCK=1,
H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`)を
再ロード。`/api/status` で `preset: "96gb-int8"`、
`transformer_quant: "int8"`、`text_encoder_loaded: true`、
`transformer_loaded: true`、`adaln_precomp: false`、
常駐VRAM 55.04GB(検証開始前の初期値と完全一致)を確認。本番設定への
復元は完了している。

## 未解決事項・今後の検討

1. **v1スコープの制約は上記のとおり構造的なもの**(int8/lowvram/turbo は
   いずれも根拠のある拒否)であり、今回のタスクでは解消していない。
   turbo と併用したい場合は、turbo LoRA が有効なリクエストと無効なリクエストで
   別々の precompute テーブルを持つ(turbo on/off それぞれで
   `adaln_proj.linear` の実効的な forward が異なるため、2セットの
   `PrecomputedModulation` テーブルを保持し `_TurboLoRALinear.enabled` の
   状態に応じて切り替える)設計が必要になるが、これは「一度だけ計算する」
   という本来の設計思想を大きく複雑化するため、今回は着手しなかった。
2. **`H3_TE_QUANT=bnb-4bit` 以外の常駐パターン**(例: `H3_KEEP_TRANSFORMER=1`、
   TE 側を `H3_TE_DEVICE` で別GPUに置く構成)では、transformer が
   decode窓で解放されずリクエスト間で常駐したままになる可能性があり、
   その場合は `peak_vram_gb` 自体に削減が反映されるはずだが未検証。
3. `H3_LOWVRAM_ANY`/`int8` との組み合わせ(削減対象の重みが重複するため
   v1では拒否)を将来的に解禁するかどうかは、torchao の `Int8Tensor`
   バックエンドで `precompute()` のGEMM-per-stepパターンが安全に動くかの
   個別検証が必要(本タスクでは未検証、対象外として明示的にガード)。
4. FBC併用時の出力は「完走・想定通り高速化」のみ確認し、baseline との
   framemd5完全一致検証は行っていない(FBCの性質上、無キャッシュ経路との
   厳密なbit-exact一致はそもそも期待される設計ではないため)。

## 変更ファイル

- 新規: `backends/minimax-h3/core/adaln_precompute.py`
- 変更: `backends/minimax-h3/core/runner.py`
  (`H3_ADALN_PRECOMP` 環境変数ブロック、`_ensure_transformer`/
  `_ensure_transformer_ref` への武装呼び出し、`status()` フィールド追加、
  `_adaln_precompute_status()` ヘルパー追加)
- 変更: `backends/minimax-h3/core/settings.py`
  (`validate_instant_settings()` に turbo+adaln_precomp ガード追加、
  `current_settings_snapshot()` の `constraints` に
  `turbo_incompatible_with_adaln_precomp` 追加)

git commit は行っていない(タスク指示どおり)。
