# Phase 1: ゲートウェイ基盤(2026-08-21)

port 8630 の統一ゲートウェイ。プロセス管理(排他切替・孤児 adopt)・env プリセット・
パススルーを実装し、GPU:0(RTX PRO 6000 96GB)で実機検証済み。

## 実装ファイル

| ファイル | 内容 |
|---|---|
| `gateway/venv` | 専用 venv(fastapi 0.141.1 / uvicorn 0.52.4 / httpx 0.28.1 のみ。CUDA系なし) |
| `gateway/requirements.txt` | pip freeze(16行) |
| `gateway/backends.py` | バックエンド宣言(port/health/プリセット/overrides ホワイトリスト/組み合わせバリデーション) |
| `gateway/procman.py` | プロセスマネージャ(start/stop/switch/busy判定/孤児adopt、threading.Lock 1本で排他) |
| `gateway/app.py` | FastAPI 本体(管理API + パススルー) |
| `run.sh`(リポジトリ直下) | gateway 起動(`GW_PORT` で上書き可、既定 8630) |
| `gateway/logs/` | gateway.log / h3.log / ltx25.log(バックエンド stdout+stderr) |
| `gateway/run/` | `<backend>.pid` |

## API 仕様

### GET /api/v1/backends
プリセットカタログ。各バックエンドの presets(name / description / vram_hint / env)、
toggles、overrides ポリシーを返す。

### GET /api/v1/status
```json
{"active_backend": "h3",
 "process": {"backend": "h3", "pid": 123, "port": 8631, "preset": "48gb-lowvram",
             "env_extra": {...}, "uptime_s": 10.1, "adopted": false},
 "backend_health": { ...バックエンドのヘルス応答を中継... },
 "busy": false,
 "vram": [{"index": 0, "memory_used_mb": 3119, "memory_total_mb": 97887}, ...]}
```

### POST /api/v1/backend/load
```json
{"backend": "h3", "preset": "48gb-lowvram",
 "overrides": {"H3_TE_DEVICE": "cuda:1"}, "toggles": {"turbo": true}}
```
- 排他切替: busy でないこと確認 → 旧 stop 完了(SIGTERM→SIGKILL、ポート閉鎖確認)→
  新 start → ヘルスOK まで待って 200(h3 タイムアウト 600s / ltx25 60s)
- 同一バックエンド・同一構成が起動中なら no-op 200(`{"result": "no-op"}`)
- busy 中は 409、未知 backend/preset/トグル・不正 overrides・禁止組み合わせは 400
- 起動失敗(プロセス即死)は 500 + ログパス案内

### POST /api/v1/backend/unload
アクティブ停止。busy は 409。アクティブなしは no-op 200。

### パススルー
`/h3/{path}` → `127.0.0.1:8631/{path}`、`/ltx25/{path}` → `127.0.0.1:8632/{path}`。
httpx AsyncClient で method/ヘッダ/ボディ(multipart 含む)/クエリを素通し、
レスポンスはストリーミング転送(read タイムアウト無制限 — h3 同期APIは数分かかる)。
対象未起動時は 502 + 日本語の起動案内。

## プリセット

### h3(port 8631、health=/api/status、busy=`status.busy`)
| preset | env | 想定VRAM |
|---|---|---|
| `96gb`(既定) | なし | 常駐 87.3GB(起動時プリロード、96GB専有前提) |
| `48gb-lowvram` | `H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_VIDEO_VAE_FP16=1` | t2va peak ~38.9GB |
| `32gb-group` | `H3_LOWVRAM=group H3_TE_PRUNE=1` | peak ~17.7〜28.7GB |
| `16gb-proj` | `H3_LOWVRAM=group H3_TE_PROJ=NicoLab28/ClipProj-MiniMax-H3 H3_VIDEO_VAE_FP16=1 H3_ATTN_BACKEND=default` | peak ~11.4GB |

- トグル: `turbo`(`H3_TURBO_LORA=1`)
- overrides: `H3_` プレフィックスのみ許可
- バリデーション(400):
  - turbo + `H3_TRANSFORMER_QUANT=int8`(torchao Int8Tensor に aten.cat が無い既知の禁止事項)
  - turbo + `H3_LOWVRAM=group`(README 記載の非互換)
  - `H3_KEEP_TRANSFORMER=1` の前提条件(下記「発見した問題点」1)

### ltx25(port 8632、health=/api/health、busy=`GET /api/jobs?limit=10` に queued/running があるか)
| preset | env | 想定VRAM |
|---|---|---|
| `nf4`(既定) | なし | peak ~17.2GB(遅延ロード) |
| `fp8` | `LTX25_TRANSFORMER_PRECISION=fp8` | 常駐 ~18GB / peak ~29GB |
| `bf16` | `LTX25_TRANSFORMER_PRECISION=bf16 OFFLOAD_MODE=none` | transformer ~38GB |

- overrides: `LTX25_*` プレフィックス + config.py 由来キー
  (`OFFLOAD_MODE` / `OUTPUT_DIR` / `INPUT_DIR` / `LORA_DIR` / `MODEL_ID` /
  `MODEL_REVISION` / `QUANTIZED_MODEL_DIR` / `HF_TOKEN` / `MAX_*` / `HISTORY_DB` / `LLM_*`)

## 孤児処理

gateway 起動時に PID ファイル + ポート生存を確認:
- PID 生存 + ポート open + 一致 → adopt(`adopted: true`、preset は「(adopted: 不明)」)
- ポート open だが PID 不一致 → エラー報告のみ(勝手に kill しない。該当バックエンドの
  load 要求は 409)
- PID ファイルだけ残存 → 削除

adopt したプロセスは gateway の shutdown では停止しない(gateway 再起動で再 adopt 可能)。
adopt 中は起動時 env が不明のため、同一構成でも load 要求は no-op にせず再起動する。

## 実機検証結果(2026-08-21、GPU:0 = RTX PRO 6000 96GB、ベースライン 3.1GB)

| # | ステップ | 結果 |
|---|---|---|
| 1 | gateway 起動 → status(active なし)/ backends | 200、カタログ・nvidia-smi VRAM 正常 |
| 2 | load ltx25(nf4)→ パススルー `/ltx25/api/health` | started 2.1s → 200 |
| 3 | パススルーで t2i ジョブ(512²・seed=42)| queued→running→**completed**(約60s、image_url あり)。完了後 GPU 4.6GB |
| 4 | 2本目 running 中に load h3(48gb-lowvram) | **409**「生成中(busy)です」 |
| 5 | 完了後に切替 ltx25→h3 | 12.7s で started、8632 閉鎖・8631 開放。同一 load 再送は **no-op** |
| 6 | パススルーで h3 t2i(512²・seed=42) | 200・**63.7s**(初回ロード込み、denoise 8.66s)・**peak_vram_gb 34.96** |
| 7 | unload | stopped、8631/8632 閉鎖、GPU:0 **3.1GB(ベースライン)復帰** |
| 8 | h3 起動中に gateway だけ再起動 | **adopt 成功**(`adopted: true`、status が active/health を正しく中継)。adopt 後の unload も正常 |
| - | バリデーション | turbo+int8 → 400、KEEP_TRANSFORMER 前提不足 → 400、overrides に `PATH` → 400、未起動バックエンドへのパススルー → 502 |

GPU:1 は全工程で 19MiB(未使用)のまま。

## 発見した問題点

1. **h3 の `H3_KEEP_TRANSFORMER=1` は単独GPU構成では起動できない**(実機で発見)。
   README「48GB級(推奨: 高速化フル)」の env セットから `H3_TE_DEVICE=cuda:1` を
   外すと、`core/runner.py` の起動時ガード
   「KEEP_TRANSFORMER=1 requires (TE_DEVICE or TE_PROJ) AND VIDEO_VAE_FP16=1 AND
   LOWVRAM != group」で即死する。そのため `48gb-lowvram` プリセットからは
   `H3_KEEP_TRANSFORMER=1` も外した(高速化フルにしたい場合は overrides で
   `H3_TE_DEVICE` か `H3_TE_PROJ` と共に明示)。同ガードは gateway 側でも先取りして
   400 を返す。
2. **h3 `/api/t2i` の `resolution` に `512x512` プリセットは無い**
   (`unknown resolution preset` 400)。512² は `height=512&width=512` の明示で指定する
   (Phase 2 の統一APIでモード変換する際に注意)。
3. h3 のヘルスタイムアウトは 600s にしてあるが、`48gb-lowvram`(遅延ロード)は
   起動 ~10s で完了する。600s が実際に必要なのは `96gb` 既定(起動時プリロード ~55s+
   TE ダウンロード時間)のみ。
4. ltx25 の busy 判定はジョブ履歴 API(`/api/jobs?limit=10`)ベースのため、
   11件以上のジョブを一気に積むケースでは古い queued を見逃す理論上の穴がある
   (max_queue_size=4 のため実際には起きない)。

## 未実施(後続フェーズ)

- 統一ジョブ API・モード変換・アセット中継(Phase 2)
- GUI タブシェル(Phase 3)
- h3 `96gb` プリセットでの起動実測(Phase 0 で単独起動済みのため省略。gateway 経由でも
  run.sh 素通しで同一挙動の見込み)
