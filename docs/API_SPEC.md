# 統一 API 仕様(gateway port 8630)

Phase 1(管理・パススルー)+ Phase 2(統一生成 API)の全仕様。
バックエンド固有 API はパススルー(`/h3/*` `/ltx25/*`)でそのまま使える。

## 管理 API(Phase 1)

| メソッド/パス | 内容 |
|---|---|
| `GET /api/v1/backends` | プリセットカタログ(presets / toggles / overrides ポリシー) |
| `GET /api/v1/status` | active_backend / process / backend_health / busy / vram(nvidia-smi)+ `backends`(各バックエンドの `process_alive` / `weights_loaded` / `vram_mb` 2軸、Phase 5a) |
| `POST /api/v1/backend/load` | `{"backend","preset?","overrides?","toggles?","strategy?"}`。排他切替。busy 409、同一構成 no-op |
| `POST /api/v1/backend/unload` | `{"strategy?"}`(body 省略可)。busy 409 |
| `/h3/{path}` `/ltx25/{path}` | パススルー(未起動時 502) |

詳細は `docs/phase1-gateway.md`。

### strategy(Phase 5a、docs/phase5a-resident.md)

- `"process"`(既定・従来どおり): load = 旧プロセス停止 → 新プロセス起動。
  unload = **管理下の全プロセス停止**(resident で parked 中のプロセスも含む)。
- `"resident"`: load = 旧プロセスを温存して `/api/admin/unload` で VRAM のみ解放
  (nvidia-smi per-process 実測で解放確認)→ 新バックエンドの既存プロセスを再有効化
  (h3 は `/api/admin/reload`(preload_all)、ltx25 は遅延ロード)。既存プロセスの
  env(プリセット/overrides)が要求と異なる場合は自動でプロセス再起動へ
  フォールバックする(レスポンス `note` で通知)。unload = アクティブの VRAM 解放のみ
  (プロセス温存)。
- レスポンス `result`: `started`(プロセス起動)/ `reactivated`(resident 再有効化、
  `reactivate_s` 付き)/ `no-op` / `stopped` / `unloaded-resident`。
- バックエンド側の追加エンドポイント(直接利用も可):
  `POST {h3|ltx25}/api/admin/unload`(busy 409)、`POST /h3/api/admin/reload`。

## 統一生成 API(Phase 2)

### POST /api/v1/generate → 202

```json
{
  "backend": "ltx25",          // "h3" | "ltx25"
  "mode": "t2i",               // 下記モード対応表
  "params": {                   // 共通パラメータのみ(それ以外は 400)
    "prompt": "a red apple", "negative_prompt": "...",
    "width": 512, "height": 512,
    "seconds": 2, "num_frames": 49, "fps": 24,
    "steps": 30, "guidance_scale": 3.0, "seed": 42
  },
  "asset_ids": ["<asset_id>", "..."],  // POST /api/v1/assets の ID(順序が意味を持つ)
  "extra": {},                  // バックエンド固有パラメータ(ホワイトリスト)
  "auto_load": true,            // 未起動時に自動起動(false なら 409)
  "preset": "48gb-lowvram"     // auto_load 時のみ使用(省略で既定プリセット)
}
```

レスポンス(= 統一ジョブ):

```json
{"id": "04790e75774243aa", "backend": "ltx25", "mode": "t2i",
 "status": "running",            // queued | running | completed | failed
 "progress": 0.45,               // 0-1
 "error": null,
 "result": {"image_url": "/ltx25/outputs/t2i_....png", "video_url": null,
            "generation_seconds": 60.4, "peak_vram_gb": 17.2},
 "notes": ["seconds=2 を num_frames=49 に換算しました(fps=24、8n+1 丸め)"],
 "created_at": 1787277066.7, "elapsed_s": 61.2,
 "backend_job_id": "617f..."}    // ltx25 のみ(バックエンド側ジョブ ID)
```

- **ltx25**: バックエンドの非同期ジョブ API へ委譲。GET のたびにバックエンドへ問い合わせて
  変換(queued/running → running、completed → completed、failed 系 → failed)。
- **h3**: 同期 API を gateway のバックグラウンドスレッドで実行。running 中は
  `/api/progress` を中継して progress を返す(denoise を 5〜95% に割り当て)。
- **busy 中の generate は 409**(Phase 2 ではゲートウェイ側キューイングをしない。
  ltx25 自体はキューを持つが統一 API は使わない)。
- result の URL は常にパススルー形式(`/h3/outputs/...` `/ltx25/outputs/...`)。
  この2プレフィックスは gateway の静的配信が直接担うため、**バックエンド停止後も
  成果物 URL は生きる**。
- h3 の result にはバックエンドの生レスポンス(denoise_time_s / peak_vram_gb 等)が
  そのまま含まれる。

### GET /api/v1/jobs/{id} / GET /api/v1/jobs?limit=N

gateway 発行ジョブの照会/一覧(新しい順)。**メモリ内レジストリのみ
(永続化は Phase 5 予定)** — gateway 再起動でジョブ履歴は消える(成果物は残る)。

### POST /api/v1/assets → 201(multipart `file`)

`gateway/data/assets/` に保存して asset_id を発行。
`{"id","kind"("image"|"video"|"audio"),"filename","size"}`。上限 500MB、
未対応拡張子は 415。generate 時の変換:

- ltx25 向け: gateway がバックエンドの `/api/assets` へ転送してバックエンド側 ID に
  変換し、conditions / audio_asset_id を組み立てる
- h3 向け: gateway が multipart(image / last_image / references)を組み立てて添付

### GET /api/v1/outputs / POST /api/v1/outputs/delete

両バックエンドの `outputs/` を統合列挙(backend / filename / size / mtime / url、
新しい順、対象拡張子 .mp4 .png .jpg .jpeg .webp .wav)。
delete は `{"backend","filename"}`(`/` `..` を含む名前・outputs 外への解決は 400)。
制限: OUTPUT_DIR を overrides で変更した構成では既定ディレクトリのみを見る。

### POST /api/v1/prompt/enhance

`{"backend","prompt","mode?","seconds?","task?","lang?","shots?"}` を振り分け:

- h3 → `POST /api/prompt/enhance`(Form: text/mode/seconds/task/lang。
  mode 既定 "storyboard")
- ltx25 → `POST /api/prompts/enhance`(JSON: prompt/mode/shots。mode 既定 "t2av")

バックエンドの応答(ステータスコード含む)をそのまま中継する。対象バックエンド
未起動は 502、LLM サーバ未起動時はバックエンド由来の 502/503 が返る。

## モード対応表

| 統一 mode | h3 | ltx25 | 必要アセット(asset_ids の順序) |
|---|---|---|---|
| `t2v` | `/api/t2va` | mode `t2av` | なし |
| `i2v` | `/api/fl2va`(image のみ) | mode `i2v` | 画像1(先頭フレーム) |
| `flf2v` | `/api/fl2va`(image+last_image) | mode `flf2v` | 画像2(先頭, 末尾) |
| `ref2v` | `/api/ref2va` | mode `condition` | 参照1つ以上(h3: 画像/動画/音声、ltx25: 画像/動画) |
| `t2i` | `/api/t2i`(height/width 明示) | mode `t2i` | なし |
| `ref2i` | `/api/ref2va` + `still=1` | mode `ref2i` | 画像1つ以上 |
| `a2v` | **400** | mode `a2v` | 音声1(+任意で先頭フレーム画像1) |
| `extend` | **400** | mode `extend` | 動画1 |
| `retake` | **400** | mode `retake` | 動画1(extra.retake_start/end 必須) |
| `iclora` | **400** | mode `iclora` | 画像/動画1(extra.loras に IC-LoRA 1つ必須) |
| `refine_image` | **400** | mode `refine_image` | 画像1 |

## 共通パラメータ対応表

| params | h3 | ltx25 |
|---|---|---|
| `prompt` | `prompt`(必須) | `prompt`(必須) |
| `negative_prompt` | **無視**(notes に記録) | `negative_prompt` |
| `width`/`height` | `width`/`height`(両方指定必須。省略時 t2v/i2v/flf2v/t2i は 768² を明示送信、ref2v/ref2i は参照から自動) | `width`/`height`(32 の倍数、バックエンドが検証) |
| `seconds` | `seconds`(0.5〜15、丸めはバックエンド) | `num_frames = 8n+1` へ換算(fps 基準、notes に記録) |
| `num_frames` | `seconds = num_frames / 24` へ換算 | `num_frames`(8n+1、9〜481) |
| `fps` | **無視**(24 固定) | `fps` |
| `steps` | `num_inference_steps` | `steps` |
| `guidance_scale` | **無視** | `guidance_scale` |
| `seed` | `seed` | `seed` |

## extra ホワイトリスト

- h3: `cache` `cache_threshold` `attn` `turbo` `mute`(即反映)、`upscale`(t2v のみ)、
  `frames`(t2i/ref2i の still フレーム数 22|5)
- ltx25: `upscale` `upscale_method` `temporal_upscale` `decoder` `min_seconds`
  `max_seconds` `enhance_prompt` `retake_start` `retake_end` `regenerate_video`
  `regenerate_audio` `extend_direction` `extend_seconds` `extend_context_seconds`
  `audio_start` `audio_duration` `loras` `strength` `frame_position` `session_number`
  `conditions`(自動生成された conditions の index/strength を asset_ids と同順の
  dict 配列で上書き)

## エラーコード

| コード | 条件 |
|---|---|
| 400 | 未知 mode / backend 非対応 mode / params・extra の未知キー / アセット数・種別不一致 / 不明 asset_id / バックエンド側バリデーション不通過(422 等を変換) / 不正ファイル名 |
| 404 | 不明ジョブ ID / 削除対象ファイルなし |
| 409 | バックエンド busy / auto_load=false かつ未起動 / 管理外リスナー |
| 413/415 | アセットサイズ超過(400) / 未対応拡張子 |
| 502 | prompt/enhance 対象バックエンド未起動、LLM 未接続(バックエンド由来) |
| 500 | バックエンド起動失敗ほか内部エラー |
