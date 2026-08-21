# Phase 4: 総合検収(2026-08-21)

GPU:0(RTX PRO 6000 Blackwell 96GB、ベースライン ~2.8〜3.1GB)で、**本番想定の
h3 `96gb` 既定プリセットを含む**実運用シナリオを gateway(8630)経由の通しで実行した。
GPU:1 は全工程 19MiB(未使用)。コード変更なし(検証で実バグは発見されなかった)。

## 実運用シナリオ実測

| # | ステップ | 所要時間 | VRAM / 結果 |
|---|---|---|---|
| 1 | gateway 起動 → `POST /api/v1/backend/load {"backend":"h3"}`(既定 `96gb`)→ ヘルスOK | **45s** | 常駐 86.6GB(nvidia-smi)。「数分想定」より大幅に速い(prequant TE キャッシュ + OS ページキャッシュが温かいため) |
| 2 | 統一API h3 t2v(768²・5秒 → 124フレーム、seed=42) | **155.6s**(denoise 103.7s / decode 7.3s / 0.216→3.58s/step×30) | **peak_vram_gb 91.93**(nvidia-smi 観測ピーク 92.4GB)。completed、video_url GET 200(864KB mp4) |
| 3 | `backend/load {"backend":"ltx25"}`(h3 停止 + ltx25 nf4 起動) | **9.1s** | 8631 閉鎖・8632 開放 |
| 4 | 統一API ltx25 t2v(512²・num_frames=121、seed=42) | **97.5s**(generation_seconds 94.6s) | backend 報告 peak 20.94GB、**nvidia-smi 観測ピーク 31.5GB**(2段アップスケール/デコード中)。completed、video_url GET 200(2.6MB mp4) |
| 5 | h3(96gb)へ戻す切替(2回目) | **64.1s**(ltx25 停止 + h3 プリロード込み) | 常駐 86.6GB へ復帰 |
| 6 | `backend/unload` | **4.1s** | GPU:0 **2.8GB(ベースライン)復帰**、8631/8632 閉鎖、active_backend=null |

追加実測(手順5と6の間): 統一API h3 t2i(512²・seed=42)= completed **54.1s**・
peak 87.7GB(Phase 0 単独スモークの 53.4s/87.7GB と一致 — gateway 経由のオーバーヘッドは無視できる)。

## 全体回帰(簡易)

| 項目 | 結果 |
|---|---|
| パススルー `/h3/api/status`(h3 稼働中) | 200 |
| パススルー `/ltx25/api/health`(未起動) | **502** + 日本語の起動案内 |
| GUI `GET /` | 200(3.4KB シェルページ) |
| `GET /api/v1/backends` | 200、プリセットカタログ h3×4 / ltx25×3 が期待どおり |
| busy 中の `backend/load` | **409**「アクティブなバックエンド h3 が生成中(busy)です…」 |
| busy 中の `generate` | **409** |
| 未知 preset(`"nope"`) | **400** + 有効プリセット一覧入りメッセージ |
| 未知 backend(`"zzz"`) | **400** |
| 成果物静的配信(`/h3/outputs/*.png` / `/ltx25/outputs/*.mp4`) | 200(バックエンド停止後も配信されることを手順6後に確認) |

## 発見事項・注意(コード変更なし、運用知見として記録)

1. **h3 `96gb` の t2v 768²×5秒は peak 91.93GB**(プリセットの vram_hint「~92GB」どおり)。
   ベースライン ~2.8GB との合算で 94.7GB/97.9GB とかなり際どい。他プロセスが数 GB 以上
   VRAM を使う環境では `48gb-lowvram`(t2v peak ~38.9GB)を使うこと。
2. **h3 `96gb` の起動は実測 45〜64s**(prequant TE キャッシュ済み前提)。ヘルスタイム
   アウト 600s は初回 TE ダウンロード時のみ意味を持つ。切替(手順5)の 64.1s は
   旧バックエンド停止(~5s)+ プリロード再実行分。
3. **ltx25 nf4 の「peak ~17.2GB」ヒントは t2i 実測値**。長尺 t2v(121f、既定の2段
   アップスケール込み)では nvidia-smi 観測で 31.5GB に達した(backend 報告の
   torch allocated peak は 20.9GB — reserved キャッシュ+アップスケール段の差)。
   24GB 級カードで長尺 t2v を使う場合は `extra.upscale` 無効化等の調整が必要になりうる。
4. h3 の progress は統一 API 経由の単一飛行なら正しく中間値(0.17→0.95)を返す
   (Phase 2 既知の制限どおり、パススルー併用時は混線しうる)。

## 後片付け

gateway・バックエンド全停止、8630/8631/8632 閉鎖、GPU:0 2.8GB / GPU:1 19MiB
(ベースライン)復帰を確認済み。
