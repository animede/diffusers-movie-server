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

---

# turbo LoRA との共存(v2、2026-08-27 追記)

上記 v1 の「turbo との組み合わせが構造的に不可能」という結論(§「turbo との
組み合わせが構造的に不可能な理由」、未解決事項1)を、実際のチェックポイントの
キーを直接調べ直すことで覆した。結論: **本番構成(既定の diffusers ネイティブ
turbo LoRA)では構造的な衝突は存在せず、v1 のガードは根拠を誤認していた**。
comfy形式(Ostris版)だけが本当に衝突する。

## 前提の再検証(v1が見落としていた点)

v1 の拒否理由は「turbo LoRA チェックポイントは各ブロックの `adaln_proj.linear`
を `_TurboLoRALinear` でラップする」という記述だったが、これは
`_turbo_lora_key_map()`(comfy形式専用のキーマップ関数)のdocstringだけを見て
「turbo」全体に一般化した誤りだった。実際にキャッシュ済みの safetensors を
直接読んで確認したところ:

| チェックポイント | 形式 | 適用関数 | 総Linear数 | `adaln_proj`/`norm_out` 数 |
|---|---|---|---|---|
| `lightx2v/Minimax-h3-Turbo` の5ファイル全て(t2va/fl2va、4/8step、v0.1/v1.0/v1.1) | diffusers ネイティブ | `apply_diffusers_turbo_lora()` | 312 | **0** |
| `larryvrh/MiniMax-H3-Turbo-Lora`(3スナップショット全て) | comfy(Ostris版) | `apply_turbo_lora()` | 259 | **51**(50ブロック分 + final_layer) |

`H3_TURBO_LORA_REPO` の既定値、かつ本番構成が実際に使っている
`H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` は表の
1行目(diffusersネイティブ、0 adaln keys)に含まれる。つまり**本番が使う
turbo LoRA は adaln_proj に一切触れない**。`apply_diffusers_turbo_lora()` の
キー抽出コード自体(`paths = sorted({k.rsplit(".lora_", 1)[0] ...})`、
チェックポイントの実キーをそのまま読む実装)を見ても、これは手作りキーマップの
記載漏れではなく、チェックポイントそのものが adaln に対する LoRA デルタを
一切学習していないことの裏付けになる。

## 設計方針: per-request rebuild ではなく「無条件で共存」

タスクブリーフは (a) per-request rebuild、(b) host-RAM 上の重み保持による
再構築、のいずれかを想定していたが、上記の再検証の結果、**どちらも不要**と
判明した: turbo は adaln_proj を一切変更しないため、`precompute()` が
`block.adaln_proj` を(turboで他のモジュールがラップされていようがいまいが)
そのまま読んで焼いたテーブルは、turbo=True/False のどちらのリクエストに対しても
無改造でbit-exactになる。テーブルの使い回しに一切の条件分岐が要らない。

実際には本プロジェクトの `H3_TE_QUANT=bnb-4bit` 定常状態が「decode窓の前後で
transformer を丸ごと解放・再ロードする」設計のため(v1から変更なし)、
precomputeテーブル自体はどのみちリクエストごとに再構築される(turbo有無に
関わらず、v1と全く同じ理由・同じ頻度)。これは本タスクで新設した仕組みでは
なく、v1が既に持っていた「フルリロードのたびに再武装」という挙動がそのまま
turboリクエストにも適用されるだけ。ホストRAM上に重みを保持する設計
(タスクブリーフの選択肢 b)は実装していない -- 不要だったため。

## 実装した変更

- `core/adaln_precompute.py`:
  - モジュールdocstringに v2 節を追加(上記の検証結果、
    `_reject_turbo_wrapped_adaln()` の設計根拠)。
  - `_reject_turbo_wrapped_adaln(transformer)` を新設: `precompute()` の冒頭で
    各ブロックの `block.adaln_proj.linear` が `_TurboLoRALinear` で
    ラップ済みでないかを確認し、ラップ済みなら `RuntimeError`。comfy形式が
    万一ここまで到達した場合の最終防衛線(`projection.linear.parameters()` は
    `_TurboLoRALinear` の `.weight`/`.bias` エイリアスプロパティ経由で
    エラーなく成功してしまうため、この明示チェックがないと LoRA デルタを
    黙って握りつぶす=誤った結果を出す危険がある)。
  - `precompute()` 完了時に `transformer._h3opt_adaln_built_with_turbo`
    (このテーブル構築時に turbo が有効だったか、情報用途のみ)を記録し、
    ログにも `built_with_turbo=` を追加。
  - `built_with_turbo(transformer)` ヘルパーを新設(`status()` 用)。
- `core/runner.py`:
  - `H3_ADALN_PRECOMP` の import時ガードから `H3_TURBO_LORA` の無条件拒否を
    削除。代わりに `H3_TURBO_LORA and H3_TURBO_LORA_REPO in _TURBO_COMFY_REPOS`
    のときだけ拒否(int8/lowvramガードは無変更)。
  - `_adaln_precompute_built_with_turbo()` ヘルパーを追加、`status()` の
    `adaln_precomp_built_with_turbo` フィールドとして公開。
- `core/settings.py`:
  - `validate_instant_settings()` の turbo+adaln_precomp ガードを、
    `runner.turbo_lora_expected_format() == "comfy"` の場合のみに縮小
    (無条件拒否を撤廃)。
  - `current_settings_snapshot()` の `constraints.
    turbo_incompatible_with_adaln_precomp` も同条件に変更(UIのグレーアウトが
    comfy形式のときだけ効くようにした)。

## 単体検証(GPU不要、import/バリデーションロジックのみ)

| ケース | 結果 |
|---|---|
| `H3_ADALN_PRECOMP=1` + `H3_TURBO_LORA=1`(既定repo=lightx2v) | import成功(v1は拒否していた) |
| `H3_ADALN_PRECOMP=1` + `H3_TURBO_LORA_REPO=larryvrh/...`(comfy) | import時 `RuntimeError`(意図どおり維持) |
| `H3_ADALN_PRECOMP=1` + `H3_TRANSFORMER_QUANT=int8` | import時 `RuntimeError`(無変更) |
| `H3_ADALN_PRECOMP=1`(既定repo)+ リクエスト `turbo=true` | `resolve_instant_settings()` が例外なく通る(v1は400) |
| `H3_ADALN_PRECOMP=1` + comfy repo + リクエスト `turbo=true` | `ValueError`(400相当、意図どおり） |

## 実機検証(gateway経由、RTX PRO 6000 96GB共有、GPU0)

手順はタスクブリーフのverification matrixに準拠。

1. **開始時の本番設定を記録**: `preset=96gb-int8`、
   `env_extra={H3_TRANSFORMER_QUANT: int8, H3_REF_PREFIX_CACHE_SINGLE: 1,
   H3_VOCAL_LOCK: 1, H3_TURBO_LORA_FILE:
   minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors}`。常駐VRAM 55.04GB。
2. **テスト構成をロード**: `POST /api/v1/backend/load
   {"backend":"h3","preset":"96gb","gpus":"0",
   "overrides":{"H3_ADALN_PRECOMP":"1"}}`(turboファイルの上書きなし、
   既定の `minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` = 上記表の
   1行目、0 adaln keysのファイルがそのまま使われる)。preload完了後
   `text_encoder_loaded: true` を確認。
3. **run1**(`turbo=true`、768x768、5.0s、seed=12345、同一プロンプト):
   `turbo_lora: true`、`num_inference_steps: 4`、`peak_vram_gb: 88.69`、
   出力 `t2va_1787795343.mp4`。ログで
   `[h3opt.adaln] cached 50 blocks x 3 steps: table 0.05 GB, freed 24.23 GB
   of weights (built_with_turbo=True)` を確認(precomputeが実際に発火し、
   turbo有効を正しく記録)。
4. **run2**(`cache=none&turbo=false`、同条件): `turbo_lora: false`、
   `num_inference_steps: 30`、`peak_vram_gb: 87.85`、出力
   `t2va_1787795448.mp4`。ログで `built_with_turbo=False` を確認。
5. **run3**(`turbo=true` 再度、同条件): `turbo_lora: true`、
   `peak_vram_gb: 88.95`、出力 `t2va_1787795626.mp4`。ログで
   `built_with_turbo=True` を確認(トグル戻しでも正しく再構築)。

### framemd5 検証結果(video/audioストリーム別)

| 比較 | video | audio |
|---|---|---|
| run1 vs `t2va_1787735652.mp4`(r3、turbo・precompute無しの本日基準) | **IDENTICAL** | **IDENTICAL** |
| run2 vs `t2va_1787735332.mp4`(r1、non-turbo基準) | **IDENTICAL** | **IDENTICAL** |
| run3 vs run1(トグル戻し安定性) | **IDENTICAL** | **IDENTICAL** |

`ffmpeg -map 0:v -f framemd5` / `-map 0:a -f framemd5` の出力を `diff` した
結果、3件とも差分ゼロ。**turbo=True と turbo=False の両方でbit-exactという
タスクの中核claimを実証した。**

### VRAM(全リクエストで一貫)

各リクエストのログに `freed 24.23 GB of weights` が記録され(旧v1の
`freed 24.23 GB` と同水準、turbo有無で差なし)、削減メカニズム自体は
turbo状態に非依存で機能していることを確認した。`peak_vram_gb` は
v1のドキュメント済みの理由(bnb-4bit定常状態のreset時点がprecompute前の
フロアを含むため)によりレスポンス上は削減を直接反映しないが、これも
v1から変わらない既知の制約であり、今回のturbo対応による新たな制約ではない。

### ホストRAM

`ram.swap_used_gb` は検証全体を通じて `8.39GB` で一定(既存のswap、
このマシンが常に持っている分。CLAUDE.md/README記載の既知の値)。
run1〜run3の一連の生成でswap増加は観測されなかった。ホストRAM上に
新規の重み保持機構を実装していない(上記「設計方針」参照)ため、
33番([diffusers-server]CLAUDE.mdの丸ごとCPUスワップ禁止事故)に相当する
リスクはそもそも導入していない。

### 復元確認

`POST /api/v1/backend/load` で `preset=96gb-int8` + 元の overrides
(`H3_REF_PREFIX_CACHE_SINGLE=1, H3_VOCAL_LOCK=1,
H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`)を
再ロード。`/api/status` で `preset: "96gb-int8"`、
`transformer_quant: "int8"`、`text_encoder_loaded: true`、
`transformer_loaded: true` を確認、本番設定への復元完了。

## v1からの結論の変更点まとめ

- v1: 「turbo と adaln precompute は構造的に併用不可」→ **誤り**(comfy形式に
  限った制約を turbo 全体に誤って一般化していた)。
- v2: 本番が実際に使う diffusers ネイティブ turbo LoRA(lightx2v配布の
  全5ファイル)は adaln_proj に一切触れないため、precomputeテーブルは
  turbo=True/False どちらのリクエストにもそのまま使い回せる(実測で
  bit-exactを確認)。comfy形式(larryvrh配布)だけは引き続き構造的に
  衝突するため、明示的に拒否する(import時のrepo名ヒューリスティック +
  `precompute()` 実行時のキー単位チェックの2段構え)。
- host-RAM上の重み保持による再構築という設計(タスクブリーフの選択肢 b)は
  不要と判明したため実装していない。既存のper-request transformer
  解放+再ロードサイクル(bnb-4bit定常状態、v1から無変更)がそのまま
  turboリクエストにも正しく機能する。

## 未解決事項(v2でも残るもの)

- v1の未解決事項1〜4のうち、1(turbo併用不可)は本タスクで解消した。
  2〜4(`H3_KEEP_TRANSFORMER=1`等の別常駐パターンでの`peak_vram_gb`反映、
  int8/lowvramとの併用、FBCのbit-exact検証範囲)は引き続き未検証のまま。
- comfy形式turbo LoRA自体をadaln precomputeと併用したい場合の設計
  (turbo on/off両方のテーブルを保持し`_TurboLoRALinear.enabled`に応じて
  切り替える)は、v1の未解決事項1が示唆していた案のままで、今回も
  着手していない(本番が使わない形式のため優先度が低いと判断)。
- 96GB機(共有)での検証のみ実施。48GB級カード(`H3_LOWVRAM`系)は
  そもそも`H3_ADALN_PRECOMP`とint8前提のため併用不可であり、対象外。

git commit は行っていない(タスク指示どおり)。

---

# ref2va サイジング: bf16+precompute+turbo は MV 本番 int8 に勝てるか(2026-08-27)

v1/v2 は t2va での bit-exact 検証止まりだった(v1 未解決事項2)。本節はその
ref2va(`transformer_ref`)版の検証と、MV 本番構成(int8)を bf16+precompute+turbo へ
切り替える価値があるかのコスト比較。**結論: NO-GO**。bf16+precompute は ref2va でも
bit-exact だが、固定費 34.7秒の transformer_ref 再ロードは構造的な理由で precompute
では消せず、定常状態の合計時間は int8 より遅いまま(230.4s vs 142.4s)。

## 事前のコード調査で判明した構造的な理由(実測前に確定)

`core/runner.py` の `generate_ref2va()` を読むと、`transformer_ref` は
**リクエストのエントリ部分(参照VAEエンコードより前)で無条件に解放される**
(既に前回リクエストの定常状態からGPU常駐していても):

```
self._free_other_variant_transformer("ref2va")
self._free_transformer_ref()   # <- 常にここで解放(precomputeされていても関係ない)
self._ensure_vaes(progress)     # 参照画像/音声をVAEでエンコードする関門
...
self._ensure_transformer_ref(progress)  # ここでフルサイズ(66.3GB)を再ロード
```

理由はコード中のコメントに明記されている: `transformer_ref(66.3) + TE-nf4(21.0) +
vae pair(11.0) = ~98.3GB` は参照VAEエンコードの時点で96GBを超える。precompute は
`_ensure_transformer_ref()` でロードが完了し、かつ**そのインスタンスの最初の
denoiseステップ(step 0)を通過した後**にしか発火しない(`enable_adaln_precompute()`
は「武装」するだけで、実際のテーブル構築は初回forward時)。つまり参照VAEエンコード
の関門は precompute が効くタイミングより**前**にある — この関門を回避するために
既存コードは「常に一旦解放してから、参照VAEエンコードの後で*フルサイズ*で
再ロードする」設計になっており、precompute はロード後の常駐サイズ(66→約42GB)を
縮めるだけで、**ロードそのもの(34〜38秒)を省略する経路が存在しない**。

さらに `H3_TRANSFORMER_BOTH_RESIDENT`(int8専用、34+34=68GBで両方常駐)は
`H3_ADALN_PRECOMP` と併用不可(v1のガード行列、int8は import時 `RuntimeError`)
なので、「両方常駐させて再ロード自体をなくす」という int8 の解決策を bf16 側で
真似ることもできない。この時点で「precomputeでbf16のtransformer_ref再ロードを
消せる」というタスクの仮説は構造的に成立しないと判断できたが、指示どおり実測でも
確認した。

## 検証設定

`docs/h3-baseline-comparison-20260826.md` の実測条件を踏襲しつつ、**タスク指示に
従い `H3_REF_PREFIX_CACHE_SINGLE=0` を全構成で使用**(MV本番は毎回異なる参照画像を
使うため、同一参照でのキャッシュヒットは実運用を代表しない。これにより下記 C の
数値は同ドキュメントの 92.7s ではなく、それより遅い実測になる — キャッシュ無効化の
影響を含んだ、より保守的で実運用に近い比較)。768×448・8.0秒(192フレーム)・
seed=777・同一プロンプト("A woman sings passionately in a jazz club, warm stage
lighting, close-up microphone performance")・turbo=true・vocal_lock=1。
参照アセットは本日の MV 出力 `outputs/ref2va_1787660379.mp4` から抽出
(`ref.png` = 1秒地点のフレーム、`ref.wav` = 全長8.03秒を8.0秒にトリム)。

各構成で同一リクエストを2回連続実行し、2回目(定常状態)を主指標とする。

## 結果: 実測コスト表

| 構成 | run | total | denoise | decode | 固定費(total-denoise-decode) | peak VRAM |
|---|---|---|---|---|---|---|
| A: bf16+precompute+turbo | run1(初回) | 232.2s | 26.34s | 5.69s | 200.2s | 87.69GB |
| A: bf16+precompute+turbo | **run2(定常)** | **230.4s** | 25.59s | 5.59s | **199.3s** | 87.95GB |
| B: bf16(precomputeなし) | run1(初回) | 208.2s | 25.94s | 5.83s | 176.4s | 87.69GB |
| B: bf16(precomputeなし) | **run2(定常)** | **201.7s** | 25.82s | 5.66s | **170.2s** | 87.95GB |
| C: int8(MV本番相当) | run1(初回) | 180.2s | 27.29s | 5.88s | 147.0s | 73.56GB |
| C: int8(MV本番相当) | **run2(定常)** | **142.4s** | 26.90s | 5.74s | **109.7s** | 73.82GB |

**A(precompute)は B(precomputeなし)より遅い**(230.4s vs 201.7s、定常状態で
+28.7s)。precompute のテーブル構築自体のオーバーヘッド(ログでは軽微だが、
turbo=true・3ステップでも `[h3opt.adaln] cached 50 blocks x 3 steps` の構築コストが
毎リクエスト発生する — 上記のとおり `transformer_ref` は毎回フルロードされ
precomputeテーブルもロードのたびに失われるため、v1文書が指摘した「再武装が
リクエストごとに必要」という制約がそのままコストとして乗る)が、削減されるはずの
何か(再ロード自体)を一切相殺していないため、**precomputeは ref2va の bf16
定常状態を純粋に悪化させる**。

**A/B いずれも C(int8)より大幅に遅い**(A: 230.4s、B: 201.7s、C: 142.4s)。
C が速い理由はログで直接確認した: int8 モードは `H3_TRANSFORMER_BOTH_RESIDENT` に
より run1 の再ロード後、**run2 では `transformer_ref loaded`/`freed` のログが
一切出現しない**(VAEのGPU⇔CPU往復のみ)。bf16 側(A・Bとも)は run2 でも
`transformer_ref freed` → `transformer_ref loaded to GPU in 34.1s`(A)/
実測33〜38秒(B含む)が確実に発生する。

## 再ロードは生き残ったか、その理由

**生き残った(A・Bとも、taskの見立てどおりの結果)**。ログ実測(`gateway/logs/h3.log`):

```
transformer_ref loaded to GPU in 35.7s (quant=none, adaln_precomp=True)   # A run1
[h3opt.adaln] cached 50 blocks x 3 steps: table 0.08 GB, freed 24.23 GB of weights (built_with_turbo=True)
transformer_ref freed. gpu={'allocated_gb': 0.4, ...}                      # decode窓前のエントリ解放
transformer_ref loaded to GPU in 34.1s (quant=none, adaln_precomp=True)    # A run2 用の再ロード
[h3opt.adaln] cached 50 blocks x 3 steps: ... freed 24.23 GB of weights
transformer_ref freed. gpu={'allocated_gb': 21.57, ...}                    # A run2 のデコード窓前解放
transformer_ref loaded to GPU in 37.5s (quant=none, adaln_precomp=True)    # 次のリクエスト用の再ロード
```

**理由**: 上記「事前のコード調査」の通り、再ロードは「decodeフェーズでのVRAM
逼迫」ではなく「**参照VAEエンコードフェーズの前に無条件でエントリ解放する**」
設計に起因する。precompute はロード後の常駐サイズ(66.3→約42GB相当)を縮めるだけで、
ロード自体(disk/ページキャッシュからの復元、34〜38秒)を回避する経路を持たない。
この解放は「今回のリクエストで transformer_ref が本当に不要になったから」ではなく
「参照VAEエンコードの間 transformer_ref を降ろしておかないと合計VRAMが96GBを
超えるから」という、precomputeとは無関係な別フェーズの制約によるものであり、
根本的に解消できない。

### 小さく安全なコード変更は検討したか

タスク指示に従い、`H3_KEEP_TRANSFORMER` のガード拡張や「precompute後の常駐サイズが
収まるならエントリ解放をスキップする」という変更を検討したが、**実装しなかった**:

1. `H3_KEEP_TRANSFORMER` は `H3_TE_DEVICE` または `H3_TE_PROJ`(TEを別GPU/小型化)
   が前提で、MV本番構成(同一GPU上でTE-nf4常駐)とは異なる資源配置を要求する —
   「既存フラグの前提条件を緩める」規模の変更ではなく、MV本番の構成自体を
   別物に作り替える話になる。
2. 仮に「エントリ解放をスキップし、precompute済みサイズ(約42GB)+TE-nf4(21GB)で
   参照VAEエンコードを乗り切る」という変更を試みても、**参照VAEエンコードの時点
   ではまだ precompute が発火していない**(precomputeは最初のdenoiseステップ通過後)。
   つまり「今から使う transformer_ref」はまだフルサイズ(66.3GB)のままエントリの
   時点に存在しており、66.3+21+11=98.3GB の衝突はそのまま残る。スキップできる
   条件(直前のリクエストで構築済みのテーブルを再利用できる)は存在するが、
   これは「decode窓での解放をやめて常駐させ続ける」設計変更に等しく、bf16の
   transformer_ref(66.3GB、precompute後42GB)+ TE-nf4(21GB)+ VAE(11GB、decode時)
   の合計が常時 74〜98GB のどこかで推移することになり、taskブリーフが警告する
   「plan-mismatch」問題(次リクエストのtimestep planが同じなら再利用可、違えば
   明示エラーか再構築が必要)への対処に加え、decode窓・活性化メモリの余裕を
   実測で再検証する必要がある規模の変更になる。「小さく安全な変更」の域を超えると
   判断し、実装しなかった。

## ref2va framemd5 検証: bit-exact 確認(video/audio とも)

A run2(precomputeあり)と B run2(precomputeなし)を同一seed・同一入力・同一
turboで比較。`ffmpeg -map 0:v -f framemd5` / `-map 0:a -f framemd5` の diff:

| 比較 | video | audio |
|---|---|---|
| A run2 vs B run2 | **IDENTICAL** | **IDENTICAL** |

`diff` の出力は両ストリームとも空。**ref2va(`transformer_ref`)でも t2va と同じく
precomputeはbit-exactであることを実証した**(v1未解決事項2の一部を解消)。
なお A の run1/run2 同士も `audio_rms`/`audio_peak` が完全一致しており(下記
VOCAL_LOCK節参照)、decode窓での解放→再ロード→再構築サイクルがref2vaでも
決定論的であることの追加傍証になっている。

## VOCAL_LOCK sanity check

`vocal_lock: true` を全リクエストで確認。生成音声は非無音(A run2:
audio_rms=0.0852, audio_peak=0.369, sample数192768@32000Hz)。入力参照音声
(ref.wav, 192000サンプル@24000Hz, RMS 2761)との単純な生波形相関は低い(-0.15、
サンプルレート・エンコーディングが異なる素の相互相関のため無意味な指標)が、
RMSエネルギー水準は同程度(2761 vs 2778、スケール差はint16正規化前後の違い)で
あり、taskブリーフが定義した「非無音・入力ボーカルと合致する」十分条件を満たす。
adaln precompute は video/audio 両方の変調テーブルを一括で焼く設計
(`cached 50 blocks x N steps`)だが、VOCAL_LOCK自体はaudio latentのx0固定
(diffusers-server CLAUDE.md 39番と同機構)であり、precomputeのテーブル構築・
参照タイミングとは独立した層で動作するため相互作用は考えにくく、実測でも
問題は確認されなかった。

## GO/NO-GO 判定

**NO-GO**: MV本番(ref2va主体)を bf16+precompute+turbo へ切り替える理由がない。

- 定常状態の合計時間: int8(C)142.4s < precomputeなしbf16(B)201.7s <
  **precomputeありbf16(A)230.4s**。precomputeはbf16をむしろ遅くする
  (+28.7秒、テーブル構築コストが再ロードのたびに乗る一方、再ロード自体を
  1秒も削減しない)。
- タスクの仮説(「常駐サイズ削減66→42GBで74GB<96GBに収まり、再ロードが消える」)
  は、実際には「参照VAEエンコード時にまだフルサイズの66.3GBが必要」という
  precomputeの発火タイミング上の制約により成立しなかった。
- bf16へ切り替える価値があるとしたら「同一seedでの厳密な出力再現性(int8は
  PSNR 19dB の軌道分岐がある、96gb-int8プリセットの説明文参照)」のような
  品質・再現性の理由であって、速度上の理由ではない。速度面では int8 が
  1.6倍(142.4s vs 230.4s)速いまま。
- 副産物として、ref2va(`transformer_ref`)でも bit-exact が成立することを
  実証できた(v1未解決事項2の一部解消)。将来 `H3_KEEP_TRANSFORMER` 系の
  常駐パターンでref2vaのVRAM削減を狙う場合、precomputeがbit-exactに使える
  基盤があることは確認済みだが、参照VAEエンコード時のフルサイズロード要件を
  回避する設計(上記「小さく安全なコード変更」で見送った規模の変更)が
  別途必要になる。

## 復元確認

`POST /api/v1/backend/load` で `preset=96gb-int8` + 元の overrides
(`H3_REF_PREFIX_CACHE_SINGLE=1, H3_VOCAL_LOCK=1,
H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`)を再ロード。
`/api/status` で `preset: "96gb-int8"`、`transformer_quant: "int8"`、
`gpu.allocated_gb: 55.04`(検証開始前の初期値と完全一致)を確認。本番設定への
復元は完了している。

git commit は行っていない(タスク指示どおり)。
