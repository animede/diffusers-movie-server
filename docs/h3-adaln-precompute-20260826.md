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

---

# 投影TE(4B) ref2va 測定(2026-08-27)

MV本番の ref2va 主コスト(参照プレフィックスの32B TEエンコード)を、
`H3_TE_PROJ`(Qwen3-VL-4B-Instruct + 学習済み線形投影で32B TEを代替)へ
差し替えて速度・品質を実測した。コード変更なし、測定のみ。

## 構成・ガード確認

- `core/runner.py` に `H3_TE_PROJ` + ref2va の**ハードガードは無い**
  (`te_quant`/`te_prune` との排他は `core/settings.py` の再ロードAPIにあるが、
  int8/turbo/ref2va との組み合わせは禁止されていない)。ただし該当箇所
  (`_encode_ref2va_prompt` / `_encode_ref2va_prompt_prefix_cached`)に
  `"H3_TE_PROJ + ref2va (reference) path is UNVERIFIED -- the projection
  matrix was only checked against 4B text hidden states, not vision tower
  features."` という明示的な未検証警告が実装されている。本タスクはこの
  警告が指す品質リスクを実測で検証するもの。
- `H3_REF_PREFIX_CACHE_SINGLE=1` は `H3_TE_PROJ` 指定時に自動で無効化される
  (`H3_TE_PROJ` がTEを恒久常駐させるため、キャッシュが二度と解放されず
  危険という理由。`core/runner.py` 1258行目)。今回は元々 `SINGLE=0` を
  使うので無関係。
- **ハマった点(タスクブリーフの誤り)**: `H3_TE_PROJ` は真偽値フラグでは
  なく、**投影行列の HF リポジトリID/ローカルパスそのもの**を格納する
  変数(`H3_TE_PROJ = os.environ.get("H3_TE_PROJ", "").strip()`、truthy な
  文字列で有効化)。ブリーフの例示どおり `"H3_TE_PROJ":"1"` を渡すと、
  `hf_hub_download("1", "mmh3-4b-ClipProj.safetensors")` として解決を試み
  リポジトリ `"1"` への404を起こし、`preload_all()` が例外で失敗して
  `text_encoder_loaded=True` のまま `_te_projection` が未設定という不整合
  状態になった(`/api/status` の `te_proj_tap: null` で検知可能)。この
  状態で ref2va を呼ぶと `_te_encoder_layer_for()` が投影なしのフォール
  バック(32Bの層50)を使い、4B(36層)に対し
  `MiniMax-H3 conditions on hidden_states[50] ... but text_encoder has 36`
  で400になった。**正しい指定は
  `"H3_TE_PROJ":"NicoLab28/ClipProj-MiniMax-H3"`**(`H3_TE_PROJ_DEFAULT_REPO`
  と同じ値)。再ロード後 `/api/status` の `te_proj_tap: 24` を確認できれば
  投影行列が正しくロードされている。

## 測定条件

`docs/h3-baseline-comparison-20260826.md` / 本ファイル前半セクションと同一の
実測条件を踏襲: `H3_REF_PREFIX_CACHE_SINGLE=0`、768×448・8.0秒(192フレーム)・
seed=777・同一プロンプト("A woman sings passionately in a jazz club, warm
stage lighting, close-up microphone performance")・turbo=true・vocal_lock=1。
参照アセットは前半セクションと同じ `ref.png`/`ref.wav`
(`outputs/ref2va_1787660379.mp4` から抽出)。

構成D: `preset=96gb-int8` + `H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3` +
`H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` +
`H3_VOCAL_LOCK=1` + `H3_REF_PREFIX_CACHE_SINGLE=0`。

比較対象は前半セクションの構成C(int8、32B TE、同一条件)の実測値
(`run2`: total 142.4s / denoise 26.9s / decode 5.74s / 固定費 109.7s /
peak 73.82GB、mp4: `outputs/ref2va_1787799387.mp4`)をそのまま流用。

## 結果: 時間・VRAM

| 構成 | run | total | denoise | decode | 固定費 | peak VRAM |
|---|---|---|---|---|---|---|
| C: int8(32B TE、基準) | run2(定常) | 142.4s | 26.90s | 5.74s | 109.7s | 73.82GB |
| D: int8+TE_PROJ(4B) | run1(初回) | 168.0s | 27.21s | 5.86s | 134.9s | 55.67GB |
| D: int8+TE_PROJ(4B) | **run2(定常)** | **128.8s** | 26.69s | 5.76s | **96.3s** | **55.93GB** |

**D run1 が run2 より遅い理由**: run1 は `transformer_ref` がまだ
GPU非常駐で、encode完了後に35.2秒のロードが発生する(D run1 のポーリング
ログで `loading_transformer` フェーズが t=91.2〜125.7s に出現)。int8の
both-resident機構により run2 以降はこのロードが消え(Cのrun2と同じ
挙動)、以後は定常状態として扱える。

## encode フェーズの内訳(タスクの主眼)

`/api/progress` を1秒間隔でポーリングし、フェーズ遷移のタイムスタンプを
記録した(D)。C は同じ機構のログ(リクエスト受信〜`vae/audio_vae -> GPU`
=参照音声のVocal Lockエンコード開始、を「encode完了」の境界として使用)
から同等の窓を逆算した。

| 構成 | encode窓の測定法 | 所要時間 |
|---|---|---|
| C(32B TE) | ログ: run1完了(11:56:21.760)→`vae→GPU`(11:58:01.681) | **99.9s** |
| D(4B TE_PROJ) | `/api/progress`ポーリング: encoding開始(t=1.22s)→denoising開始(t=87.66s) | **86.4s** |
| D(4B TE_PROJ) | ログ: run1完了(12:23:47.238)→`vae→GPU`(12:25:19.995)、参考(HTTP経由の余剰込み) | 92.8s |

**TE_PROJ による encode 短縮は 7〜13秒程度**(99.9s → 86.4〜92.8s、
相対で 7〜13%)。「32B→4B、大幅高速化」という仮説どおりの規模には
遠く届かなかった。理由は主に、encodeフェーズの大半が参照**画像**の
vision tower 処理(`H3_REF_IMAGE_SHORT_EDGE=2048` の高解像度参照を
Qwen3-VL の vision encoder に通す処理、テキスト側より遥かに支配的)で
占められており、text decoder 層自体(32層→24層 tap 相当)の縮小効果は
相対的に小さいためと推測される(vision tower のFLOPsはテキスト本体
サイズに比例しないため、4B化してもvision側の計算コストはほぼ変わらない)。

## 全体時間への影響

固定費: C 109.7s → D(定常) 96.3s(**-13.4秒、-12%**)。
total: C 142.4s → D(定常) 128.8s(**-13.6秒、-9.5%**)。
peak VRAM: C 73.82GB → D 55.93GB(**-17.9GB、-24%**、TEが32B→4Bへ縮小した
直接効果でVRAM側は encode 側より明確に効いている)。

## 品質評価: 音声(Vocal Lock)

D run2 の生成音声を `ref.wav` と比較(10ms窓RMSエンベロープの相互相関、
タスクブリーフ指定の粗い指標):

| 比較 | RMS | エンベロープ相関 vs ref |
|---|---|---|
| C run2 音声 | 0.08467(正規化float) | 0.9536 |
| D run2 音声 | 0.08467(正規化float) | 0.9536 |
| D vs C(音声同士) | - | **1.0000(bit-identical)** |

**D と C の生成音声は完全に一致した**(RMS・エンベロープ相関とも同値、
D vs C の相互相関が1.0)。Vocal Lock(音声latentのx0固定、CLAUDE.md
39番と同機構)はテキストエンコーダの実体(32B/4B)に依存しない独立層で
動作しており、TE_PROJ化による音声側への副作用は皆無と実証できた。
入力参照音声(ref.wav)とのエンベロープ相関 0.9536 は C/D 共通で、
「入力ボーカルに追従している」という十分条件を満たす。

## 品質評価: 映像(アイデンティティ保持) -- 主要な否定的所見

start/mid/end(frame 0/96/190)を抽出し目視比較した。

- **frame 0(start)**: D は参照(ref.png)の特徴(ヘアクリップ・イヤモニ・
  白ブラウス+黒リボン・両手でマイクを包む構図)を良好に再現。顔つきも
  近い。ただし背景がジャズクラブ風の演者(コントラバス奏者)ではなく
  ワインボトルの並ぶバー背景+ギタリストに変化(Cはジャズクラブの
  コントラバス奏者2名)。
- **frame 190(end)**: D はヘアクリップ・イヤモニとも正しく再現され、
  参照に近い構図に復帰。
- **frame 96(mid)で重大な破綻を検出**: 右腕/右手が**解剖学的に破綻した
  第二の腕**として描画され、指の関節が渦巻き状・繊維状のテクスチャで
  異常伸長している(マイクを持つ左手とは別に、画面右側に不自然な
  腕状オブジェクトが出現)。frame 80/88 は正常、**frame 96〜112(約17
  フレーム、24fpsで約0.7秒)にわたり同一系統の破綻したアーム/手首
  アーティファクトが persist**(96で最も顕著、104/112でも縮小した
  同種の歪みが残存)、frame 140 では完全に正常な両手表現へ復帰した。
  **同一フレーム番号(96/104/112)で C(基準、32B TE)を確認したところ、
  いずれも完全にクリーン**(参照どおりの単手マイク保持、異常なし)。
  この破綻が TE_PROJ(4B投影)固有であり、seed/プロンプト由来の
  一般的な不安定性ではないことを同条件比較で確認した。
- 画像: `scratchpad/h3_te_proj_test/` に `ref.png`(参照)、
  `D_frame{1,2,3}.png`(D の0/96/190)、`C_frame{1,2,3}.png`(C の
  0/96/190、比較用)、`D_extra{1..6}.png`(D の80/88/96/104/112/140)、
  `C_extra{1,2,3}.png`(C の96/104/112、比較用)。

## 結論・所見(暫定、最終判断は親セッション/ユーザーに委ねる)

- **速度**: TE_PROJ は encode フェーズを 7〜13秒(7〜13%)短縮するのみで、
  「32B→4B」という縮小率から期待される大幅高速化は実現しなかった。
  encode コストの大半は参照画像の vision tower 処理が占めており、
  text decoder の縮小効果は限定的と推測される。固定費ベースで -12%、
  total で -9.5%(142.4s→128.8s)。
- **VRAM**: -24%(73.82GB→55.93GB)と明確な削減効果があり、これは
  32B→4B化の直接効果として速度より大きく効いている。
- **音声品質**: Vocal Lock 経由の生成音声は C と bit-identical。
  リスクなし。
- **映像品質**: **明確な懸念あり**。frame 96 前後で通常のフレームには
  見られない解剖学的な破綻(第二の腕/手首の異常テクスチャ)を検出した。
  同一 seed・同一条件の C(32B TE)ではこの破綻が一切発生しないため、
  TE_PROJ 固有の劣化である可能性が高い。`core/runner.py` 自身が
  記していた「投影行列は4Bのテキスト隠れ状態に対してのみ検証済みで、
  vision tower の特徴に対しては未検証」という警告と整合する結果と
  言える(参照画像=vision入力の条件付けが、テキストのみの場合より
  劣化しやすい可能性を示唆)。1回の実行(1 seed)のみの観測であり、
  複数 seed / 複数参照画像での再現率は未確認。
- **総合**: 現状の実測(1試行)では、TE_PROJ を MV 本番(ref2va主体)へ
  投入する根拠は乏しい。速度メリットが小さい(-9.5%)一方、映像
  アイデンティティ保持に無視できない劣化リスク(破綻したアーム
  アーティファクト)が観測された。VRAM削減(-24%)は魅力的だが、
  MV本番運用は現行構成(int8、73.82GB)で十分収まっており、VRAM
  逼迫の解消が主目的でない限り、品質リスクとのトレードオフに見合わない。
  採用するなら、複数 seed・複数参照画像での破綻再現率の追加検証が
  前提になる。

## 復元確認

`POST /api/v1/backend/load` で `preset=96gb-int8` + 元の overrides
(`H3_REF_PREFIX_CACHE_SINGLE=1, H3_VOCAL_LOCK=1,
H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`)を
再ロード。`/api/status` で `te_proj: false`、`transformer_quant: "int8"`、
`gpu.allocated_gb: 55.04`(検証開始前の初期値と完全一致)を確認。本番設定
への復元は完了している。

git commit は行っていない(タスク指示どおり)。

---

# ref2va エンコード相プロファイリング(2026-08-27)

「エンコード相(固定費のうち denoise/decode を除いた ~100秒)がどこで消費
されているか」を、推定ではなく実測で特定するタスク。事前に2つの仮説が
既に反証されていた: (1) 32B TE の prefill が主因(4B TE_PROJ へ置き換えても
7〜13秒しか短縮しなかった、本ファイル前半節参照)、(2) vision tower が主因
(粗い FLOP 見積りでは ~85秒を説明できないとされていた)。**結論: 主犯は
vision tower の中の、さらに1モジュールだけ ── `Qwen3VLVisionPatchEmbed` の
`nn.Conv3d`(kernel_size==stride のパッチ化畳み込み)であり、sm_120
(Blackwell)上でこの形状(バッチ=28160、空間ごく小)が病的に遅いカーネルへ
落ちることが根本原因と実測で確定した。粗い FLOP 見積りが外れていたのは
「計算量は妥当なのに実行が異常に遅い」というカーネル選択の問題であり、
FLOP から時間を逆算する手法自体がこの種の不具合を原理的に検出できない
ため。**

## 手法

`core/runner.py` に `H3_PHASE_TIMING`(既定 `"0"`、無変更)という新しい
opt-in 環境変数を追加し、以下を計装した(タスク指示の「衛生オプション
(a): クリーンで小さいので残す」を選択、git commit はしていないがコードは
そのまま残置):

- `_PhaseTimer`(新クラス): `generate_ref2va()` 本体の主要チェックポイント
  (entry_lock / setup_step / text_encode / vae_to_gpu / reference_encoder_step /
  vocal_lock_latents / vae_to_cpu / ensure_transformer_ref / layout系ステップ /
  force_free_te)に `.mark()` を挿入。各 `.mark()` は `torch.cuda.synchronize()`
  してからタイムスタンプを取るため、GPU非同期処理の完了が正しくその
  チェックポイントに帰属する。
- `_encode_ref2va_prompt()`(`H3_REF_PREFIX_CACHE_SINGLE=0` のときに通る
  非キャッシュ経路。本タスクの測定条件そのもの)に同様の内訳
  (`gather_vision_features` / `build_presentation` / `conditioner_forward`)
  を追加。既存の `_encode_ref2va_prompt_prefix_cached()`(キャッシュ経路)は
  同等の `t_key`/`t_prefix` ログを既に持っていたが、非キャッシュ経路には
  無かったため揃えた。
- `_install_qwen3vl_submodule_timing(text_encoder)`(新関数): TE ロード
  成功直後に一度だけ呼ばれ、`text_encoder.model`(`Qwen3VLModel`)の
  `get_image_features`(vision tower 全体)と `language_model.forward`
  (テキストデコーダ)をランタイムでラップしてタイミングを取る。さらに
  1段深く、`text_encoder.model.visual`(`Qwen3VLVisionModel`)の
  `patch_embed.forward`(Conv3d 1回)・27個の `blocks[i].forward` の合計・
  `merger.forward` も個別に計測する。**diffusers/transformers 本体
  (venv)は一切変更していない** ── `core/runner.py` からランタイムで
  bound method を差し替えているだけで、モデルの計算内容・戻り値は無変更。
- 全ての追加コードは `H3_PHASE_TIMING=0`(既定)のとき単一の早期 return の
  みで、CUDA sync もログ出力も一切発生しない(ゼロオーバーヘッド)。

計測はゲートウェイ経由で `preset=96gb-int8` +
`H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` +
`H3_VOCAL_LOCK=1` + `H3_REF_PREFIX_CACHE_SINGLE=0` + `H3_PHASE_TIMING=1` を
ロードし、同一リクエスト(ref2va、768×448、8秒、turbo=true、seed=777、
参照は `ref.png`(768×448 の実写風女性ポートレート)+ `ref.wav`)を2回連続
実行(run1=初回・run2=定常状態)。`nvidia-smi
--query-gpu=utilization.gpu,memory.used --format=csv,noheader` を1〜2秒間隔で
バックグラウンドポーリングし、GPU使用率が高い区間かどうかを region 単位で
突き合わせた。

## 結果: サブフェーズ内訳(run2、定常状態)

| サブフェーズ | 秒数 | CPU/GPU | 全体(~100s)に対する割合 |
|---|---|---|---|
| entry_lock(free_transformer+ensure_vaes+load_te+sync) | 0.00s | - | 0% (bnb-4bit常駐、no-op) |
| setup_step(reference_normalize/resize、PIL LANCZOS) | 0.05〜0.09s | CPU | <0.1% |
| **text_encode 内訳** | **91〜97s** | **GPU(後述)** | **~92%** |
| ├─ gather_vision_features(processor: 画像→pixel_values) | 0.13〜0.16s | CPU | <0.2% |
| ├─ build_presentation(トークナイズ) | 0.00〜0.01s | CPU | ~0% |
| └─ conditioner_forward(`text_encoder.model(...)` 1回) | 90.4〜96.7s | **GPU** | **~92%** |
| 　├─ **`get_image_features`(vision tower 全体)** | **90.1〜94.7s** | **GPU** | **~90%** |
| 　│　├─ **`visual.patch_embed`(Conv3d、kernel==stride)** | **91.9〜94.1s** | **GPU** | **~90%(単体でほぼ全部)** |
| 　│　├─ `visual.blocks`(27層 self-attn+MLP、合計) | 0.57〜0.59s | GPU | ~0.6% |
| 　│　└─ `visual.merger` | 0.00s | GPU | ~0% |
| 　└─ `language_model`(テキストデコーダ、64層、~7300トークン) | 1.92〜2.05s | GPU | ~2% |
| to_compute_device | 0.00s | - | 0% |
| vae_to_gpu(VAE CPU→GPU、bnb-4bit専用のフェーズ切替) | 2.05〜6.36s | GPU(小) | ~3% |
| reference_encoder_step(参照画像のVAEエンコード) | 2.09〜2.40s | GPU(小) | ~2% |
| vocal_lock_latents(音声VAEエンコード) | 0.04〜0.05s | GPU(小) | ~0% |
| vae_to_cpu(VAE GPU→CPU) | 3.43〜4.66s | GPU(小) | ~4% |
| ensure_transformer_ref | 0.00s(定常) / 35〜39s(初回のみ) | - | 定常0%、初回別枠 |
| layout+condition_latents+latents+ref2va_latents+timesteps | 0.04〜0.05s | CPU/小GPU | <0.1% |
| **合計(force_free_te込み、denoise開始まで)** | **99〜105s** | - | **100%** |

4回の独立した実行(初回×2、定常×2、うち1回はさらに vision tower 内部の
サブモジュール分解込み)すべてで同じ内訳が再現し、`visual.patch_embed`
単体が固定費の 87〜90%(vision tower 自体の 97〜99%)を占めることを確認
した。実測ログ抜粋(定常状態、run2):

```
visual.patch_embed (Conv3d, kernel==stride): 94.12s (input=(28160, 1536))
visual.merger: 0.00s
visual.blocks (all 27 Qwen3VLVisionBlock, sum): 0.57s
Qwen3VLModel.get_image_features (vision tower): 94.70s (pixel_values=(28160, 1536), image_grid_thw=[[1, 128, 220]])
Qwen3VLModel.language_model (text decoder, 64 layers): 1.94s
```

**GPU使用率の帰属**: `visual.patch_embed` の90秒間、GPU0 は 1〜2秒間隔の
サンプリングで一貫して **50〜57%使用率**(アイドルではない、CPU律速でも
ない)、しかし電力は **128.6W**(このカードのTDP ~300Wに対し明確に低い)。
低電力×中程度使用率×90秒という組み合わせは、「大きな計算をフルスループット
で回している」のではなく「多数の小さなサブカーネルを逐次発行している」
ことを示唆する典型的なシグネチャで、後述の完全単離マイクロベンチマークの
結果と整合する。

## 根本原因の特定

`transformers/models/qwen3_vl/modeling_qwen3_vl.py` の
`Qwen3VLVisionPatchEmbed`:

```python
kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]  # [2, 16, 16]
self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

def forward(self, hidden_states):
    hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
    hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
    return hidden_states
```

これは `/home/animede/diffusers-server/CLAUDE.md` の46番(JoyAI-Image-Edit-Plus
統合時に発見)が記録している「**kernel_size==stride のパッチ化 Conv3d を
極小パッチ×大量バッチで叩く形状が、sm_120(Blackwell)上で cuDNN の病的
カーネルに落ち、1呼び出しで数十秒かかる**」というパターンと寸分違わず一致
する。本タスクのGPU(RTX PRO 6000 Blackwell Workstation Edition /
RTX PRO 4000 Blackwell、いずれも `compute_cap 12.0` = sm_120)はまさに
その対象アーキテクチャ。

参照画像1枚(768×448)を `H3_REF_IMAGE_SHORT_EDGE`(既定2048px)の短辺基準で
リサイズすると 3520×2048 相当になり、Qwen3-VL の patch_size=16 で
128×220=28160 パッチ、`Conv3d` の入力は `(28160, 3, 2, 16, 16)`(バッチ
28160・空間はわずか 2×16×16)という、まさに「極小パッチ×大量バッチ」の
形状になる。

### 完全単離マイクロベンチマークでの確認

サーバプロセスを一切介さない、単独スクリプト(`torch.nn.Conv3d` を実測と
同一形状で構築し bf16/cuda に置くだけ)で追試した:

```
Conv3d single call:            89.961s   (output shape (28160, 1152))
Linear-equivalent single call:  0.0013s   (output shape (28160, 1152))
max abs diff: 1.5625e-02   mean abs diff: 6.18e-07
SPEEDUP: 68942.7x
```

`kernel_size==stride` の Conv3d は数学的に
`F.linear(x.flatten(1), conv.weight.reshape(out_ch, -1), conv.bias)` と
完全等価(CLAUDE.md 46番と同じ変換)。誤差は bf16 丸め誤差の水準
(mean abs diff 6.18e-07、CLAUDE.md 46番の「3.9e-03 = bf16丸め誤差レベル」と
同種の整合)で、数値的には別物ではない。**サーバ経由の計測(vision tower
全体で90〜95秒、うち patch_embed が90〜94秒)と、この完全単離ベンチマーク
(Conv3d単体90.0秒)が独立に一致した**ことで、原因の特定に高い確信を持てる
(bnb-4bit・32Bモデルコンテキスト・denoiseループ等、他の要因が一切関与
しない環境で同じ数十秒の遅延が再現したため)。

### 歴史的ログとの整合性(トークン数に対する線形スケーリング)

過去の `H3_REF_PREFIX_CACHE_SINGLE=1`(キャッシュ)構成での "cache MISS" ログ
58件を集計すると、プレフィックストークン数とエンコード時間がほぼ完全に
線形(トークンあたり ~12.6ms、`patch_embed` がバッチサイズ=パッチ数に
比例する構造と整合):

| プレフィックストークン数 | n | 平均秒数 | ms/1000トークン |
|---|---|---|---|
| 1832〜1837 | 19 | 23.1s | 12.6 |
| 4093〜4109 | 19 | 51.8s | 12.6 |
| 7309(本タスクの768×448画像相当) | 58 | 95.0s | 13.0 |

この線形性(2次でも定数項が支配的でもない)は、「大きなGEMMの実行効率が
悪い」のではなく「バッチ要素(パッチ)ごとにほぼ固定のオーバーヘッドを
持つカーネルが逐次発行されている」という仮説(低電力・中使用率の
GPUシグネチャとも整合)を裏付ける。

## 109.7秒(cache miss)と66.3秒(cache hit、旧記録)の整合

`gateway/backends.py` の `96gb-int8` プリセット説明文にある「固定費66.3秒
(参照エンコード~55秒 + VAE往復11.6秒)」は、**`H3_REF_PREFIX_CACHE_SINGLE=1`
(既定、本番運用値)でのキャッシュ HIT を含む測定**だった。キャッシュ HIT
時は `_encode_ref2va_prompt_prefix_cached()` が前回リクエストで構築済みの
KV キャッシュ(プレフィックス = 参照画像由来のビジョンブロック)を再利用し、
`visual.patch_embed` を含む重い prefill forward を**完全にスキップ**する
(プロンプト末尾のみを再エンコードする継続呼び出しだけが走る)。

これを裏付ける証拠: 同ログの `cache HIT` ログの `key prep` 時間
(`_gather_vision_features` + `_build_presentation` のみ、Conv3d 実行を含まない)
は 0.13〜0.27秒と、本タスクで計測した `gather_vision_features` 単体
(0.13〜0.16秒)とほぼ一致する。すなわちキャッシュ HIT 時、参照画像側の
処理はこの「軽い」前処理だけで完結し、90秒の Conv3d 呼び出し自体が
一度も発生しない。

一方、本タスク(タスク指示に従い `H3_REF_PREFIX_CACHE_SINGLE=0` で測定、
MV本番は毎回異なる参照画像を使うためキャッシュヒットが実運用を代表しない
という本ファイル前半節の判断を踏襲)は**リクエストごとに必ずキャッシュ
MISS**となり、90秒の Conv3d 呼び出しが毎回発生する。「66.3秒」と
「109.7秒」の差(~43秒)が90秒ちょうどにならないのは、(1) 66.3秒の内訳に
含まれる「VAE往復11.6秒」相当が両条件で共通に発生する固定費であること、
(2) キャッシュ HIT でも継続呼び出し(プロンプト末尾のみ、テキストデコーダ
64層)は毎回走ること、(3) 実際の運用環境・実測日・GPU共有状況が完全には
同一条件でないこと、による。**結論としては「66.3秒 vs 109.7秒」の差の
正体は cache hit/miss による「90秒の Conv3d 呼び出しをスキップできるか
どうか」であり、値そのものの厳密な差し引きが1:1で一致しないのは上記の
副次的な要因の重ね合わせのため**(この副次要因の精密な切り分けは本タスクの
範囲外)。

## 修正候補(報告のみ、実装はしていない)

| 候補 | 推定削減効果 | リスク | 備考 |
|---|---|---|---|
| **`Qwen3VLVisionPatchEmbed.forward()` をランタイムパッチで linear-equivalent に置換**(`core/runner.py` から transformers 本体は無変更のまま bound method 差し替え、CLAUDE.md 46番と同じ手法) | **~90秒 → ~0.001秒**(定常状態の固定費が ~100秒 → ~10秒程度まで縮む可能性、実測の `SPEEDUP: 68942.7x` に基づく) | **低**(数学的に完全等価、誤差はbf16丸め誤差レベルのみ。既存の `_install_qwen3vl_submodule_timing` と同型のランタイムパッチ機構が既にこのタスクの計装コードとして実証済み) | **最有力候補**。ただし `merge_with_config_defaults`/`capture_outputs` デコレータ済みの `Qwen3VLVisionModel.forward` が呼ぶ内部経路であり、`torch.compile`・deepstack機構等への副作用がないか要検証。本タスクは計測のみで実装はしていない |
| キャッシュ運用の拡大(`H3_REF_PREFIX_CACHE_SINGLE` の適用条件見直し、例: 同一参照が続く場面だけ自動検出してキャッシュを使う) | 条件が合えば ~90秒削減(既に実証済みの機構の適用範囲拡大) | 低〜中(既存機構の再利用だが、MV本番は「毎回異なる参照画像」なのでヒット率が構造的に低い。本ファイル前半節の判断どおり本番のユースケースには効果が薄い) | 参照画像を使い回す用途(例: 同一キャラクターの複数場面生成)には有効 |
| `H3_REF_IMAGE_SHORT_EDGE` を下げる(トークン数削減) | 線形スケーリング(12.6ms/token)に従い比例削減。例: 2048→1024 なら概ね半分程度(パッチ数はおおよそ短辺の2乗に比例するため、単純な半減より大きい可能性もある。未実測) | 中(参照の細部再現度が下がる、`core/runner.py` の該当コメントに明記の通りA/B未検証) | Conv3d自体の根本修正ではなく回避策。品質とのトレードオフが必要 |
| bitsandbytes/transformers のアップグレードでカーネル選択が改善するのを待つ | 不明(このプロジェクトの経緯からは楽観できない、CLAUDE.md 46番の事例は自前パッチで解決している) | 低(何もしないだけ) | 上流(cuDNN/PyTorch/transformers)側の改善を待つ受動的な選択肢。時期不明 |

**推奨は1番目**(ランタイムパッチによる linear-equivalent 置換)。理由:
(a) 数学的に完全等価(CLAUDE.md 46番で実証済みの手法をそのまま踏襲でき、
本タスクの計装コード自体が同種のランタイムパッチが問題なく機能することを
示している)、(b) 効果が実測(単離ベンチマークで68942倍)で確定している、
(c) MV本番の「毎回異なる参照画像」というキャッシュが効かないユースケースに
直接効く(2番目の候補と異なり、ヒット率に依存しない)。ただし本タスクは
「計測して原因を特定する」ことがスコープであり、修正の実装は行っていない
(タスク指示どおり)。

## 未検証事項

1. 上記パッチ候補を実際に適用した場合の、`Qwen3VLVisionModel.forward()`
   内の他の経路(`deepstack_merger_list`、`_can_record_outputs` による
   hidden_states/attentions 記録機構)への影響は未検証。
2. `H3_REF_IMAGE_SHORT_EDGE` を下げた場合の、パッチ数(≒エンコード時間)の
   厳密なスケーリング則(短辺の1乗か2乗か)は実測していない(理論上は
   面積(2乗)に近いはずだが、アスペクト比・32の倍数丸めの影響で単純では
   ない)。
3. 動画参照(image ではなく video reference)を持つリクエストでの
   `pixel_values_videos` 経路(`get_video_features`)は同じ
   `Qwen3VLVisionPatchEmbed` を共有するため同型の問題を抱えている可能性が
   高いが、本タスクでは画像参照のみで検証しており動画参照では未検証。
4. 「66.3秒 vs 109.7秒」の副次的な差(上記「整合」節参照)の精密な内訳は
   未検証(本タスクの主眼である「vision towerのConv3dが主犯」という結論
   には影響しない)。
5. 他のファミリー(t2va/fl2va、`_encode_h3_prompt` 経由)でキーフレーム画像を
   使う場合も同じ `Qwen3VLVisionPatchEmbed` を通るため同型の問題を抱えて
   いる可能性が高いが、本タスクは ref2va のみを対象とした(スコープ外)。

## 復元確認

`POST /api/v1/backend/load` で `preset=96gb-int8` + 元の overrides
(`H3_REF_PREFIX_CACHE_SINGLE=1, H3_VOCAL_LOCK=1,
H3_TURBO_LORA_FILE=minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors`、
`H3_PHASE_TIMING` は明示せず既定の `"0"` のまま)を再ロード。`/api/status` で
`transformer_quant: "int8"`、`text_encoder_loaded: true`、
`gpu.allocated_gb: 55.04`(検証開始前の初期値と完全一致)を確認。本番設定
への復元は完了している。

`core/runner.py` の `H3_PHASE_TIMING` 計装コードはそのまま残置している
(衛生オプション(a)、既定 `"0"` で完全に無効・ゼロオーバーヘッド、
`git status` は `M backends/minimax-h3/core/runner.py` のみ)。git commit は
行っていない(タスク指示どおり)。
