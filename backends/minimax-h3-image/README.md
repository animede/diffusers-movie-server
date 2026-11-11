# MiniMax-H3-Image バックエンド(diffusers-movie-server、内部 port 8633)

本家 [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)(静止画生成 +
最大 16 枚参照の編集、6 step 蒸留・CFG なし)を、動画 H3 と同じ流儀で軽量化して載せたもの。
調査・実測の経緯は `probes/h3-image/RESULTS.md`。

## 構成(実測に基づく)

| 部品 | 方式 | 常駐 |
|---|---|---|
| transformer(33B) | **AdaLN 射影を固定時刻の事前計算テーブルへ(bit 一致)** + 312 層 torchao int8wo。`app/pruned_image.py` | ~19GB(GPU0) |
| steps 2 / 3 | pDMD 2-step LoRA(akhaliq、Apache-2.0)を bf16 へマージしてから int8wo 化。別キャッシュ | 同上 |
| text_encoder | H3 動画の量子化済みキャッシュ(nf4・51層、`backends/minimax-h3/models/prequant/te_bnb-4bit_prune1`)を流用 | ~17GB(GPU が 2 枚なら cuda:1) |
| VAE | bf16 | 2.3GB |

量子化キャッシュ(各 19GiB)は `cache/int8wo_adaln_table{,_pdmd2,_pdmd3}` に置く
(この機では実体は `~/models/minimax-h3-image/` にあり、`cache/`・`models/lora/`・動画 H3 の
`models/prequant/te_bnb-4bit_prune1` は symlink。単独リポジトリ `~/minimax-h3-image-server` と共有、2026-11-11)。無ければ bf16 シャード
(HF キャッシュの `transformer_diffusers`、66GB)から約 40 秒で構築して保存する(bf16 を CPU/GPU に全展開しない)。

## 実測(RTX PRO 6000 96GB + RTX PRO 5000 48GB、seed 固定)

| ケース | 6 step | **3 step(既定)** | 2 step |
|---|---|---|---|
| t2i 1024² | 2.4s | **1.8s** | 1.5s |
| t2i 2048² | 7.0s | **4.1s** | 3.0s |
| 編集 2048²(参照1枚) | 11.3s | **6.7s** | 5.0s |
| peak VRAM(GPU0) | 21GB | 21GB | 21GB |

品質: 3 step は 6 step とほぼ同等(等倍目視)。2 step は織り目状の微細パターンが出るためプレビュー向け。
初回ロード: TE 3〜8s + transformer キャッシュ 12s + VAE 1s ≈ 20 秒(遅延ロード、`H3IMG_PRELOAD=1` で起動時)。

## API(ltx25 バックエンドと同じ形。gateway の「ジョブ型」経路で扱われる)

| メソッド | パス | 内容 |
|---|---|---|
| GET | `/api/health` | `{status, model, loaded, steps}` |
| GET | `/api/status` | health + engine 状態(device / te_device / vram_alloc_gb)+ busy |
| POST | `/api/assets` | 参照画像(PNG/JPG/WebP/BMP、multipart `file`)→ `{id, kind, filename, size}` |
| POST | `/api/jobs` | 生成ジョブ(202)。body は下記 |
| GET | `/api/jobs?limit=N` / `/api/jobs/{id}` | 履歴 / 状態(`queued/running/completed/failed/interrupted`、`image_url`) |
| DELETE | `/api/jobs/{id}` | 履歴削除(実行中なら中断) |
| POST | `/api/interrupt` | `{job_id?}` 中断 |
| POST | `/api/admin/unload` | VRAM 解放・プロセス温存(実行中は 409)。次の生成で遅延再ロード |
| POST | `/api/admin/load` | `{steps?}` 明示ロード |

```json
{"mode": "t2i" | "edit", "prompt": "...", "width": 2048, "height": 2048, "seed": 42,
 "steps": 2 | 3 | 6, "generation_area": "native" | "requested",
 "conditions": [{"asset_id": "..."}]}
```
- `edit` はプロンプト中で `<Picture 1>`, `<Picture 2>` … と参照する(配布元の流儀)。
- `generation_area`: `native` = モデル推奨の最低 4MP キャンバスで生成して縮小(1024² でも 2048² と同じ時間)、
  `requested` = 要求サイズで生成(既定、1024² が 2〜3 倍速い)。
- `steps` をリクエストで変えると transformer を差し替える(キャッシュ読込 ~12s)。

統一 API(gateway)からは `backend: "h3image"`、`mode: "t2i" | "ref2i"`、`params.steps`、`extra.generation_area`。

## 環境変数(H3IMG_*)

| 変数 | 既定 | 内容 |
|---|---|---|
| `H3IMG_PORT` | 8633 | |
| `H3IMG_STEPS` | 3 | 既定ステップ(2/3/6)。gateway プリセット `3step` / `6step` / `2step` |
| `H3IMG_DEVICE` / `H3IMG_TE_DEVICE` | cuda:0 / (2枚あれば cuda:1) | TE の配置先 |
| `H3IMG_GENERATION_AREA` | requested | 既定キャンバス |
| `H3IMG_PRELOAD` | 0 | 1 で起動時ロード |
| `H3IMG_TE_DIR` / `H3IMG_LORA_PATH` / `H3IMG_CACHE_DIR` / `H3IMG_MODEL_DIR` | 上記既定 | 所在の上書き |
| `H3IMG_MAX_PIXELS` | 4096² | 1 リクエストの上限画素 |

## venv

`.venv`(uv、python3.12): torch 2.11.0+cu130 / diffusers main `875373ac7f`(H3-Image 対応、0.42.1 未収録)/
transformers 5.22.2 / torchao 0.18.0 / bitsandbytes 0.49.0。`requirements.txt` と `venv-freeze.txt` 参照。
動画 H3 の venv(torch 2.9・diffusers 固定)とは共存できないため別 venv・別プロセス。

## 既知の注意
- このマシンは GPU 間 P2P コピーが黙って全ゼロを返すため、TE(cuda:1)→transformer(cuda:0)の受け渡しは
  ホスト経由にしている(`engine._route_text_encoder`)。
- 時刻集合(= AdaLN テーブルのキー)はキャッシュごとに固定。スケジュールを変える場合はキャッシュを作り直す。
- ライセンス: MiniMax-H3-Image Community License(商用は別途書面許諾、地域制限あり)。
