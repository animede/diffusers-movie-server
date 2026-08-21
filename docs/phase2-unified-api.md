# Phase 2: 統一 API(2026-08-21)

`POST /api/v1/generate` と統一ジョブモデルを gateway(8630)に追加し、
GPU:0(RTX PRO 6000 96GB)で実機検証済み。Phase 1 のコードは拡張のみ
(procman.py は無変更、app.py はエンドポイント追加のみ)。
API の完全仕様は `docs/API_SPEC.md`。

## 追加ファイル

| ファイル | 内容 |
|---|---|
| `gateway/jobs.py` | 統一ジョブレジストリ(メモリ内 dict + Lock。**永続化は Phase 5 送り**)。ltx25 = バックエンドジョブへの委譲+都度変換、h3 = 同期 API をバックグラウンドスレッド化+ /api/progress 中継 |
| `gateway/modes.py` | 統一 mode → バックエンド別エンドポイント/パラメータの宣言的変換テーブル(extra はホワイトリスト素通し) |
| `gateway/assets.py` | アセット保存(gateway/data/assets/、サイドカー JSON メタ、上限 500MB) |
| `gateway/app.py`(拡張) | /api/v1/generate /jobs /assets /outputs /outputs/delete /prompt/enhance + `/h3/outputs` `/ltx25/outputs` の静的配信マウント |
| `gateway/requirements.txt` | python-multipart 追加(UploadFile 用。gateway/venv のみ) |

## 設計上の決定

- **busy 中の generate は 409**(Phase 2 ではゲートウェイ側キューイングをしない)。
  ltx25 自体のキュー(max_queue_size=4)も統一 API からは使わない
  (submit 前に busy 判定して弾く)。
- **h3 の i2v** は `/api/fl2va` の先頭フレームのみ指定で代替(実機仕様上
  image / last_image の片方だけで受理されることを確認済み)。
- **h3 t2i/t2v は width/height を常に明示送信**(resolution プリセット不使用。
  512x512 プリセットが無い問題 = Phase 1 発見事項 2 への対応)。
- **成果物 URL はパススルー形式**(`/h3/outputs/...` `/ltx25/outputs/...`)だが、
  この2プレフィックスは StaticFiles マウント(パススルー catch-all より先に登録)が
  gateway 自身で配信する。バックエンド切替・停止後も URL が生きる。
- 単位換算は notes フィールドに記録(seconds→num_frames 8n+1、num_frames→seconds、
  h3 非対応パラメータの無視)。
- ltx25 へのアセットは generate のたびに転送(ローカル転送で軽量なためキャッシュ
  しない。バックエンド再起動で ID が無効になる問題も回避)。

## 実機検証結果(2026-08-21、GPU:0、ベースライン 3.1GB)

| # | ステップ | 結果 |
|---|---|---|
| 1 | generate ltx25 t2i(512²・seed42、auto_load で nf4 起動)| 202 → polling → **completed**(61.2s、peak 17.2GB)。image_url GET 200(PNG 770KB、1024²=2段仕様どおり) |
| 2 | ltx25 t2v(512²・2秒→num_frames49 換算)| completed 21.6s、video_url GET 200(MP4 440KB) |
| 3 | 手順1の PNG を /api/v1/assets → ltx25 i2v | asset 201 → completed 16.3s(peak 19.0GB) |
| 4 | generate h3 t2i(auto_load preset=48gb-lowvram)| **ltx25 停止→h3 起動 13.7s** → completed 62.3s(peak 34.96GB)。progress が 0.02→0.11→0.77→0.95 と中間値を返すことを確認 |
| 5 | h3 t2v(768²・5秒)| completed 177.5s(peak 38.94GB)。video_url GET 200(MP4 1.4MB) |
| 6 | GET /api/v1/outputs | h3 7件 + ltx25 6件を統合列挙。h3 アクティブ中でも ltx25 の URL が 200(静的配信)。delete 1件成功、`..`・`/` 入りファイル名は 400 |
| 7 | prompt/enhance | h3 → 502「LLMサーバに接続できません(H3_LLM_URL=...)」(振り分け疎通OK)。ltx25(未起動)→ 502 起動案内 |
| 8 | 異常系 | 未知 mode 400 / h3 に iclora 400 / 不明 asset_id 400 / params 未知キー 400 / busy 中 generate 409 / auto_load=false 未起動 409 |
| - | Phase 1 回帰 | パススルー /h3/api/status 200、同一構成 load no-op、unload → stopped |
| 9 | 後片付け | 8630/8631/8632 全閉鎖、GPU:0 3.08GB(ベースライン)、GPU:1 19MiB(全工程未使用) |

## 既知の制限(Phase 2 時点)

1. ジョブレジストリは非永続(gateway 再起動で消える。バックエンド側の成果物・
   ltx25 の履歴 DB は残る)。Phase 5 で永続化予定。
2. `GET /api/v1/outputs` は既定の outputs/ ディレクトリ固定
   (OUTPUT_DIR overrides には追従しない)。
3. ltx25 ジョブの実行中にバックエンドを切り替えると、gateway ジョブは
   failed「バックエンドがジョブ完了前に停止しました」になる(busy ガードで通常は
   起きないが、パススルー経由の直接操作では起こりうる)。
4. h3 の progress は /api/progress のグローバル状態を中継するため、パススルー経由で
   別の生成を直接実行すると値が混ざる(統一 API 経由のみなら単一飛行で問題なし)。
5. h3 の negative_prompt / guidance_scale / fps は無視(notes に記録)。
