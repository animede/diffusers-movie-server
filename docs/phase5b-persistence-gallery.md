# Phase 5b: 統一ジョブの永続化と統一ギャラリー(2026-08-21)

Phase 2 の統一ジョブ(メモリ内のみ・gateway 再起動で消える)を SQLite へ永続化し、
両バックエンドの成果物を横断表示する「ギャラリー」タブをシェル GUI に追加した。
Phase 5a の resident 実装(procman.py)には一切手を入れていない。

## 1. ジョブ永続化(gateway/jobs.py)

### スキーマ(gateway/data/jobs.sqlite3、標準ライブラリ sqlite3)

```sql
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    backend TEXT NOT NULL,
    mode TEXT NOT NULL,
    status TEXT NOT NULL,        -- queued|running|completed|failed|interrupted
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    result TEXT,                 -- JSON(UnifiedJob.result)
    notes TEXT,                  -- JSON 配列
    created_at REAL NOT NULL,
    finished_at REAL,
    backend_job_id TEXT          -- ltx25 のみ
);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
```

### 設計

- **書き込みは状態遷移時のみ**: submit 受理確定(running)/ h3 worker の終端
  (completed/failed)/ ltx25 refresh での終端遷移。progress の毎ポーリング更新は
  書き込まない(復元時の progress は最後に永続化した値。終端は 1.0 で確定)。
  書き込みは `INSERT OR REPLACE` + `_db_lock`、接続は操作ごとに開閉。
  永続化失敗はログのみ(ジョブ実行は止めない)。
- **復元(JobRegistry.__init__)**: 新しい順に最大 1000 件(`RESTORE_LIMIT`)を
  メモリへ復元。running/queued のまま残っていたジョブは:
  - **h3 系 → 即 `interrupted`**(終端状態、失敗扱い)。同期呼び出しの worker
    スレッドが gateway 再起動で失われ、結果を追跡できないため。
  - **ltx25 系 → `resync_restored()` で同期**(app の startup イベント、
    `manager.adopt_orphans()` の**後**に呼ぶ — バックエンド生存の adopt が先)。
    `backend_job_id` で `GET /api/jobs/{id}` を照会し、応答があれば統一ステータス
    へ変換(running のままなら以後は従来の `_refresh_ltx25` が追跡)、
    照会失敗・404(バックエンド未起動/再起動でジョブ消失)は `interrupted`。
- 統一ステータスに終端状態 **`interrupted`** を追加(`TERMINAL` に含む)。
  GUI のバッジは warn 色。
- **`DELETE /api/v1/jobs/{id}`**: メモリ + DB からレコード削除。
  **成果物ファイルは消さない**。不明 ID は 404。
- API レスポンス形式は従来踏襲(フィールド追加なし。status の値域だけ拡張)。

## 2. 統一ギャラリー

### API(app.py、既存 `GET /api/v1/outputs` の拡張)

- `kind` を各 item に追加: `image`(.png/.jpg/.jpeg/.webp)/ `video`(.mp4)/
  `audio`(.wav)/ `other`。
- ページング `offset`(既定 0)/ `limit`(既定 100、最大 500)。レスポンスに
  `total` / `offset` / `limit` を追加(items は従来どおり新しい順)。
- 動画サムネイルは生成しない(GUI が `<video preload="metadata">` で先頭
  フレームを表示する方式)。

### GUI(static/index.html + shell.js + shell.css、4つ目のタブ)

- タイルグリッド(`repeat(auto-fill, minmax(230px, 1fr))`)。画像は
  `<img loading="lazy">`、動画は `<video controls preload="metadata">`、音声は
  `<audio controls preload="none">`。ファイル名リンク・サイズ・日時・削除ボタン付き。
- **ポーリングなし**: 取得はタブ表示時と「更新」ボタンのみ(idle 20 秒間の
  ネットワーク監視で `/api/v1/outputs` リクエスト 0 件を CDP で確認済み)。
- **DOM は差分更新**: タイルは `backend/filename` キーの Map で一度だけ構築し、
  更新時は並び替え(appendChild 移動)・消えたファイルのタイル除去のみ。
  フィルタ(バックエンド/種類)は `hidden` 切替のクライアント側適用
  (57番の「DOM 再構築 + キャッシュバスターはちらつきの原因」の教訓を踏襲)。
- 削除は confirm 付きで既存 `POST /api/v1/outputs/delete` を呼び、成功時に
  タイルを除去。ジョブ一覧の成果物リンクと同じ `/{backend}/outputs/...` URL 形式。

## 実機検証(GPU:0 RTX PRO 6000 Blackwell 96GB、2026-08-21)

| # | 項目 | 結果 |
|---|---|---|
| 1 | ltx25(nf4、auto_load)で統一 t2i | completed 65.3s / 生成 59.8s / peak 17.2GB(Phase 5a 実測と一致) |
| 2 | gateway だけ再起動 → `GET /api/v1/jobs` | 直前ジョブが **completed のまま復元**(result/notes/elapsed_s も保持)。ltx25 プロセスは adopt |
| 3 | t2v running 中に gateway 再起動 | 復元時 `resync_restored()` がバックエンド照会 → **running のまま同期**(progress 0.78)→ ポーリング続行で **completed**(video_url 取得)。ジョブはバックエンド側で走り続けた |
| 3b | interrupted 経路(running 行を注入して再起動) | h3 行 → 即 interrupted、ltx25 行(backend_job_id 不明=404)→ interrupted(エラーメッセージ付き) |
| 4 | `DELETE /api/v1/jobs/{id}` | レコード消滅(GET 404)・**成果物 PNG は残存**。不明 ID は 404 |
| 5 | ギャラリー(headless Chrome + CDP) | タブ描画 OK・29 タイル(h3 18 + ltx25 11、image 15 + video 14)・`loading=lazy` / `preload=metadata` を DOM 検査で確認・先頭画像の実ロード成功。フィルタ(backend/kind)正動作。**更新ボタン後も 29/29 タイルが同一 DOM ノード**(マーク保持=再構築なし)。削除 1 件: confirm ダイアログ表示 → accept → タイル除去+実ファイル削除(12→11) |
| 6 | 回帰 | `/api/v1/generate`(t2i/t2v)従来どおり 202+追跡。管理タブのジョブ一覧・状態カード(adopted 表示)従来どおり。ギャラリータブ表示中 20 秒で outputs へのリクエスト 0 件(status/jobs の既存ポーリングのみ) |
| 7 | 後片付け | `unload strategy=process` で全プロセス停止 → 8630/8631/8632 全て down、GPU:0 2866MB(ベースライン ~2.85GB 復帰)、GPU:1 19MB |

## 既知の制限・注意

1. 復元件数のサマリログ(「ジョブ履歴を N 件復元しました」)は `JobRegistry` が
   モジュール import 時に走るため、`logging.basicConfig`(app.py)より先に実行され
   出力されないことがある(resync 側のログは startup イベント内のため出る)。実害なし。
2. 復元上限は新しい順 1000 件(`RESTORE_LIMIT`)。それより古い履歴は DB には残るが
   一覧には出ない(DELETE は DB 直接削除のため対象にできる)。
3. ギャラリーの一覧取得は先頭 200 件固定(`GALLERY_FETCH_LIMIT`)。ページ送り UI は
   未実装(API 側の offset/limit は実装済み)。
4. running 復元の同期は起動時の1回のみ。起動時にバックエンドが一時的に無応答だと
   interrupted になる(その後バックエンド側でジョブが完走しても統一ジョブは
   interrupted のまま。成果物はギャラリーには出る)。
5. 検証用に gateway/venv へ websocket-client を追加した(CDP テスト用。gateway の
   実行時依存ではない)。
