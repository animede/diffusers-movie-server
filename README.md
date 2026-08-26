# diffusers-movie-server

MiniMax-H3(旧 `/home/animede/minimax-h3`、port 8611)と LTX-2.5
(旧 `/home/animede/diffusers-ltx2_5`、port 8000)を1つのゲートウェイに統合した
動画+音声生成サーバ。torch のメジャーバージョンが衝突する2バックエンドを
「専用 venv の別プロセス + 統一ゲートウェイ」で共存させる。

- 計画・経緯: `docs/INTEGRATION_PLAN.md`
- API 完全仕様: `docs/API_SPEC.md`
- 旧サーバからの移行: `docs/MIGRATION.md`
- 各フェーズの作業記録: `docs/phase0-*.md` 〜 `docs/phase4-acceptance.md`

> ### `diffusers-server`(8601)との切り分け(2026-08-26 方針決定)
>
> 動画生成の**最新版はこのリポジトリに集約**します。
>
> | | `diffusers-server`(8601) | このサーバ(8630) |
> |---|---|---|
> | 画像系(T2I/I2I/Edit/ControlNet/Tポーズ 等) | **あり** | なし |
> | 動画 | LTX-**2.3** | LTX-**2.5** / MiniMax-**H3** |
> | リップシンク(歌唱の口パク) | なし | **あり**(vocal_lock) |
>
> `diffusers-server` にも LTX-2.3 の動画機能がありますが、**両方を追従して保守するのは
> 手間に見合わない**ため、LTX-2.5 / H3 への更新は行いません(現状維持)。
> 動画系の新規開発・上流追従はこのリポジトリだけで行います。
>
> アプリ側は「画像は 8601(または 8620)、動画は 8630」と**接続先を分けて併用**する構成を
> 想定しています(実例: `mv_studio_V3`)。

## アーキテクチャ

```
クライアント / ブラウザ
        │
        ▼
┌──────────────────────────── gateway (port 8630, 軽量venv) ────────────────────────────┐
│  GET  /                    タブ切替GUI(iframe で既存SPAを表示 + バックエンド管理)      │
│  /api/v1/*                 統一API(backends/status/load/unload/generate/jobs/assets…)│
│  /h3/*    → 127.0.0.1:8631 パススルー(未起動時 502)                                  │
│  /ltx25/* → 127.0.0.1:8632 パススルー(未起動時 502)                                  │
│  /h3/outputs /ltx25/outputs 静的配信(バックエンド停止後も成果物URLが生きる)           │
│  procman: 同時アクティブ1バックエンド。切替 = 旧stop → 新start(env プリセット付き)   │
└───────────────┬───────────────────────────────┬───────────────────────────────────────┘
                ▼                               ▼
   backends/minimax-h3 (port 8631)   backends/ltx2_5 (port 8632)
   torch 2.9.0+cu128 / 同期API        torch 2.11.0+cu130 / 非同期ジョブAPI
   venv → /home/animede/minimax-h3    .venv → /home/animede/diffusers-ltx2_5
```

1プロセス統合が不可能な理由(torch/diffusers/transformers の非互換)は
`docs/INTEGRATION_PLAN.md` 冒頭の表を参照。

## ポート表

| ポート | 用途 |
|---|---|
| **8630** | gateway(統一API・GUI・パススルー)。`GW_PORT` で上書き可 |
| **8631** | MiniMax-H3 バックエンド(内部)。`H3_PORT` で上書き可 |
| **8632** | LTX-2.5 バックエンド(内部)。`LTX25_PORT` で上書き可 |

LAN クライアントから GUI を使う場合、iframe がバックエンドポートへ直接接続するため
**8630 だけでなく 8631/8632 も開放**が必要(既知の制約参照)。

## セットアップ

### 前提(symlink 依存 — 旧ディレクトリを消さないこと)

venv・モデル・量子化済み重みは旧ディレクトリへの **symlink 共有**(Phase 0 方針)。
以下が存在しないと動かない:

| symlink | 実体 |
|---|---|
| `backends/minimax-h3/venv` | `/home/animede/minimax-h3/venv`(torch2.9+cu128、diffusers PR#14355 ピン留め) |
| `backends/minimax-h3/models` | `/home/animede/minimax-h3/models`(prequant TE キャッシュ 36GB) |
| `backends/ltx2_5/.venv` | `/home/animede/diffusers-ltx2_5/.venv`(torch2.11+cu130、NATTEN) |
| `backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit` | 旧ディレクトリの量子化済み 27GB |
| `backends/ltx2_5/loras` | 旧ディレクトリ(IC-LoRA 2本、1.6GB) |

**旧ディレクトリ(`/home/animede/minimax-h3`・`/home/animede/diffusers-ltx2_5`)は
削除禁止**。完全独立化(venv 再構築)は Phase 5 の任意タスク
(手順: `backends/*/VENV_REBUILD.md`)。

### gateway venv の構築

gateway だけは自前の軽量 venv(CUDA 系依存なし):

```bash
cd gateway
python3 -m venv venv
venv/bin/pip install -r requirements.txt   # fastapi / uvicorn / httpx / python-multipart
```

## 起動

```bash
./run.sh          # gateway を port 8630 で起動(バックエンドは起動しない)
```

バックエンドは API か GUI から起動する:

```bash
curl -X POST http://127.0.0.1:8630/api/v1/backend/load \
  -H 'Content-Type: application/json' -d '{"backend":"h3"}'          # 既定 96gb
curl -X POST http://127.0.0.1:8630/api/v1/backend/load \
  -H 'Content-Type: application/json' \
  -d '{"backend":"h3","preset":"48gb-lowvram"}'                       # プリセット指定
curl -X POST http://127.0.0.1:8630/api/v1/backend/unload             # 停止
```

同時にアクティブなのは1バックエンドのみ。別バックエンドの load は自動で
旧 stop → 新 start(排他切替)。生成中(busy)の load/unload/generate は 409。

## プリセット表(想定VRAM は実測ベース)

### h3(MiniMax-H3)

| preset | env | 想定/実測 VRAM |
|---|---|---|
| `96gb`(既定) | なし | 常駐 87.3GB、**t2v 768²×5s peak 91.93GB(実測)**。96GB 専有前提 |
| `48gb-lowvram` | `H3_LOWVRAM=1 H3_TE_PRUNE=1 H3_VIDEO_VAE_FP16=1` | t2v peak ~38.9GB |
| `32gb-group` | `H3_LOWVRAM=group H3_TE_PRUNE=1` | peak ~17.7〜28.7GB |
| `16gb-proj` | `H3_LOWVRAM=group H3_TE_PROJ=… H3_VIDEO_VAE_FP16=1 H3_ATTN_BACKEND=default` | peak ~11.4GB |

- トグル: `turbo`(`H3_TURBO_LORA=1`)。int8 量子化・`H3_LOWVRAM=group` とは併用不可(400)
- overrides は `H3_` プレフィックスのみ許可
- `H3_KEEP_TRANSFORMER=1` は `H3_TE_DEVICE`(または `H3_TE_PROJ`)+
  `H3_VIDEO_VAE_FP16=1` が前提(満たさないと 400。単独GPU構成の既定プリセットには含めていない)

### ltx25(LTX-2.5)

| preset | env | 想定/実測 VRAM |
|---|---|---|
| `nf4`(既定) | なし | t2i peak ~17.2GB。**t2v 121f(2段アップスケール込み)は nvidia-smi 観測 31.5GB(実測)** |
| `fp8` | `LTX25_TRANSFORMER_PRECISION=fp8` | 常駐 ~18GB / peak ~29GB |
| `bf16` | `LTX25_TRANSFORMER_PRECISION=bf16 OFFLOAD_MODE=none` | transformer ~38GB + TE/VAE。96GB級向け |

- overrides は `LTX25_*` + config.py 由来キー(`OFFLOAD_MODE` / `OUTPUT_DIR` 等)のみ許可

## 統一 API の使用例

完全仕様は `docs/API_SPEC.md`。

```bash
# 生成(未起動なら auto_load で自動起動。202 → ジョブID をポーリング)
curl -X POST http://127.0.0.1:8630/api/v1/generate \
  -H 'Content-Type: application/json' -d '{
    "backend": "ltx25", "mode": "t2v",
    "params": {"prompt": "a red fox in a snowy forest", "width": 512, "height": 512,
               "num_frames": 121, "seed": 42},
    "auto_load": true, "preset": "nf4"}'

# ジョブ照会 / 一覧
curl http://127.0.0.1:8630/api/v1/jobs/<job_id>
curl 'http://127.0.0.1:8630/api/v1/jobs?limit=20'

# 画像アセットをアップロードして i2v
curl -X POST http://127.0.0.1:8630/api/v1/assets -F file=@input.png     # → {"id": "..."}
curl -X POST http://127.0.0.1:8630/api/v1/generate \
  -H 'Content-Type: application/json' -d '{
    "backend": "ltx25", "mode": "i2v",
    "params": {"prompt": "the scene comes alive", "seconds": 2},
    "asset_ids": ["<asset_id>"]}'

# 成果物の統合一覧 / プロンプト強化
curl http://127.0.0.1:8630/api/v1/outputs
curl -X POST http://127.0.0.1:8630/api/v1/prompt/enhance \
  -H 'Content-Type: application/json' -d '{"backend":"h3","prompt":"a cat"}'
```

統一 mode: `t2v` / `i2v` / `flf2v` / `ref2v` / `t2i` / `ref2i`(両対応)、
`a2v` / `extend` / `retake` / `iclora` / `refine_image`(ltx25 のみ、h3 は 400)。

## 既存 API 対応表(旧サーバ → 新 URL)

旧クライアントはパススルー URL に置き換えるだけで移行できる(API 仕様は不変)。
詳細は `docs/MIGRATION.md`。

### 旧 minimax-h3(`http://<host>:8611`)

| 旧 | 新 |
|---|---|
| `POST :8611/api/t2va` | `POST :8630/h3/api/t2va` |
| `POST :8611/api/fl2va` | `POST :8630/h3/api/fl2va` |
| `POST :8611/api/ref2va`(`/ref2va_batch` `/ref2i_batch`) | `POST :8630/h3/api/ref2va`(同) |
| `POST :8611/api/t2i`(`/t2i_batch`) | `POST :8630/h3/api/t2i`(同) |
| `GET :8611/api/status` / `/api/progress` | `GET :8630/h3/api/status` / `/h3/api/progress` |
| `GET :8611/api/settings` / `POST /api/settings/apply` | `:8630/h3/api/settings`(同) |
| `GET :8611/api/outputs` / `POST /api/outputs/delete` / `/api/outputs/concat` | `:8630/h3/api/outputs`(同) |
| `POST :8611/api/prompt/enhance` | `POST :8630/h3/api/prompt/enhance` |
| `GET :8611/outputs/<file>` | `GET :8630/h3/outputs/<file>`(gateway 静的配信) |
| `GET :8611/`(SPA) | `GET :8630/`(GUI の H3 タブ)または直接 `:8631/` |

### 旧 diffusers-ltx2_5(`http://<host>:8000`)

| 旧 | 新 |
|---|---|
| `POST :8000/api/jobs` / `GET /api/jobs` / `GET・DELETE /api/jobs/{id}` / `POST /api/jobs/concat` | `:8630/ltx25/api/jobs…`(同) |
| `POST :8000/api/assets` | `POST :8630/ltx25/api/assets` |
| `GET :8000/api/health` / `/api/loras` | `:8630/ltx25/api/health` / `/ltx25/api/loras` |
| `POST :8000/api/sessions` | `POST :8630/ltx25/api/sessions` |
| `POST :8000/api/prompts/enhance` | `POST :8630/ltx25/api/prompts/enhance` |
| `GET :8000/outputs/<file>` | `GET :8630/ltx25/outputs/<file>`(gateway 静的配信) |
| `GET :8000/`(SPA) | `GET :8630/`(GUI の LTX タブ)または直接 `:8632/` |

## GUI

`GET http://127.0.0.1:8630/` — タブ3つ(MiniMax-H3 / LTX-2.5 / バックエンド管理)。

- バックエンドタブは既存 SPA を iframe 表示。未起動ならプリセット選択付きの
  起動オーバーレイ(別バックエンド稼働中は確認ダイアログ付き切替)
- 管理タブ: 状態カード(アクティブ/preset/PID/VRAM バー)・起動/切替/アンロード・
  統一ジョブ一覧(進捗バー・成果物リンク、DOM 差分更新でちらつきなし)
- 詳細・検証記録: `docs/phase3-gui.md`

## 実測ベンチ(GPU:0 = RTX PRO 6000 Blackwell 96GB、詳細: docs/phase4-acceptance.md)

| 操作 | 所要時間 | VRAM |
|---|---|---|
| h3 `96gb` 起動(クリーン → ヘルスOK) | 45s(切替込みの2回目 64.1s) | 常駐 86.6GB |
| h3 t2v 768²×5秒(124f・30steps) | 155.6s(denoise 103.7s) | **peak 91.93GB** |
| h3 t2i 512² | 54.1s | peak 87.7GB |
| h3 `48gb-lowvram` 起動 | ~10〜14s | t2v peak 38.94GB / t2i 34.96GB |
| h3 t2v 768²×5秒(48gb-lowvram) | 177.5s | peak 38.94GB |
| ltx25 `nf4` 起動 | 2.1s(遅延ロード) | — |
| ltx25 t2i 512²(初回ロード込み) | 61.2s | peak 17.2GB |
| ltx25 t2v 512²×121f | 97.5s | 20.9GB(nvidia-smi 観測 31.5GB) |
| ltx25 i2v 512²×2秒 | 16.3s | peak 19.0GB |
| バックエンド切替(h3⇔ltx25) | 9.1s(→ltx25)/ 64.1s(→h3 96gb) | — |
| unload → ベースライン復帰 | 4.1s | GPU:0 2.8GB |

## 既知の制約

1. **h3 `96gb` は 96GB 専有前提**。t2v 768²×5s の peak 91.93GB はベースライン
   ~3GB との合算で 94.7GB/97.9GB と際どい。他プロセスが VRAM を使う環境では
   `48gb-lowvram` を使うこと(Phase 4 発見事項)。
2. **`H3_KEEP_TRANSFORMER=1` は単独GPU構成の既定プリセットでは使えない**
   (TE_DEVICE/TE_PROJ + VIDEO_VAE_FP16=1 が前提。overrides で明示、Phase 1 発見事項)。
3. **h3 の progress は混線しうる**: `/api/progress` はグローバル状態のため、統一 API の
   ジョブ実行中にパススルーで別生成を直接叩くと progress 値が混ざる(統一 API のみの
   単一飛行なら問題なし)。また起動プリロード中のフェーズ表示が
   `loading_transformer` のまま TE ロードを表示する場面がある(表示上のみ)。
4. **LAN クライアントは 8631/8632 も開放が必要**(GUI の iframe がバックエンドポート
   へ直接接続するため)。
5. **ltx25 の `.env` に HF_TOKEN は入っていない**。HF キャッシュ共有 + pinned revision
   で動作しており、キャッシュ削除時や gated repo 利用時は
   `backends/ltx2_5/.env` への `HF_TOKEN=` 追記が必要(Phase 0 発見事項)。
6. **ltx25 nf4 の VRAM ヒント(~17.2GB)は t2i 実測**。長尺 t2v(121f、既定2段
   アップスケール)は nvidia-smi 観測 31.5GB に達する。24GB 級では要調整
   (Phase 4 発見事項)。
7. **統一ジョブは非永続**(gateway 再起動で消える。成果物・ltx25 履歴 DB は残る)。
   `GET /api/v1/outputs` は既定 outputs/ 固定(OUTPUT_DIR overrides に非追従)。
8. **h3 `/api/t2i` に `512x512` の resolution プリセットは無い**
   (`height=512&width=512` を明示。統一 API は常に width/height を明示送信する)。
9. h3 の `negative_prompt` / `guidance_scale` / `fps` は無視される(notes に記録)。
10. venv・モデルは旧ディレクトリへの symlink 共有 — **旧ディレクトリ削除禁止**
    (`docs/MIGRATION.md` 参照)。
