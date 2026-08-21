# Phase 3: タブ切替 GUI(シェルページ)(2026-08-21)

gateway(8630)の `GET /` にタブ切替シェルを追加し、headless Chrome + CDP で
実機検証済み。フレームワーク不使用の素の HTML/JS/CSS。gateway API・パススルー・
Phase 2 実装は無変更(app.py へのルート追加のみ)。

## 実装ファイル

| ファイル | 内容 |
|---|---|
| `gateway/static/index.html` | シェルページ(タブ3つ: MiniMax-H3 / LTX-2.5 / バックエンド管理) |
| `gateway/static/shell.js` | 全ロジック(状態ポーリング・オーバーレイ・iframe 管理・管理タブ・ジョブ一覧差分更新) |
| `gateway/static/shell.css` | ダークテーマ。`[hidden] { display: none !important; }` を全体に適用(下記「発見した問題点」1) |
| `gateway/app.py`(拡張) | `GET /` = `static/index.html` の FileResponse、`/static` の StaticFiles マウント。どちらもパススルー catch-all より前に登録(パス自体は非衝突) |

## UI 構成の要点

### バックエンドタブ(H3 / LTX)
- **iframe で既存 SPA をそのまま表示**。src は
  `http://<window.location.hostname>:<port>/`(LAN クライアント対応のため
  ホスト名は動的、ポートは `GET /api/v1/backends` のカタログ由来)。
  パススルー(`/h3/...`)経由にしない理由: 既存 SPA は絶対パス `/api/...` を
  叩くため、プレフィックス付き配信では動かない。
- 対象バックエンドが未起動(status ポーリングで判定)ならオーバーレイを表示:
  - プリセット選択(カタログから動的生成。名前・説明・想定 VRAM 表示、既定選択済み)
  - h3 タブには「96gb プリセットは起動に数分かかる」注意書き
  - 別バックエンドがアクティブなら起動ボタンが
    「切替(現在の ○○ を停止して起動)」+ `window.confirm` の確認ダイアログになる
  - busy(生成中)なら起動ボタン無効化 + 理由表示
  - 起動クリック → `POST /api/v1/backend/load` → スピナー → ヘルス OK で iframe 表示
- iframe 表示後もバックエンドが停止したら(status ポーリングで検知)自動で
  オーバーレイへ戻す。iframe は `about:blank` へ戻し、pid が変わった再起動時は
  再ロードする(`state.framePid` で追跡)。

### バックエンド管理タブ
- 状態カード: アクティブバックエンド(port / adopted 表示)・プリセット・
  PID / 稼働時間・busy・GPU 別 VRAM バー(85% 以上で警告色)
- 操作: バックエンド + プリセット選択 → 起動/切替(別バックエンド稼働中は
  切替ラベル + confirm)、アンロード(busy / 停止中は無効化 + 理由表示)
- 統一ジョブ一覧(`GET /api/v1/jobs?limit=50`、5 秒ポーリング):
  ID / backend / mode / 状態バッジ / プログレスバー / 経過秒 / 成果物リンク
  (image_url / video_url / audio_url)。**行は job id ごとに 1 回だけ生成し、
  以降はテキスト・バー幅だけ差分更新**(DOM 再構築によるちらつきなし)。
  簡易生成フォームは設計どおり作らない(生成は各バックエンドタブの既存 UI)。

### ポーリング設計(重要)
- `manager.load()` はヘルス待ちの間ロックを保持するため、その間
  `/api/v1/status` は応答待ちでブロックする。シェルは **in-flight ガード**
  (前回の応答が返るまで次のリクエストを発行しない)で、サーバ側スレッドの
  滞留を防ぐ(status 3 秒 / jobs 5 秒間隔)。
- 起動要求(load POST)がネットワーク都合で切れても、status ポーリングが
  「対象アクティブ + ヘルス OK」を検知した時点で iframe 表示へ切り替える
  (h3 96gb の数分ロードでもブラウザ側タイムアウトに依存しない)。

## ブラウザ実機検証(2026-08-21、headless Chrome + CDP、GPU:0 のみ)

検証方法: `google-chrome --headless=new --remote-debugging-port=9333
--remote-allow-origins='*'` + websocket-client 製の最小 CDP ヘルパー。
スクリーンショット目視に加え、getComputedStyle / DOM 検査で判定。
iframe は同一ホスト別ポート(Chrome の site 判定では同一サイト)のため
別ターゲットにならず、`Page.getFrameTree` + `Page.createIsolatedWorld` で
iframe 内 DOM を直接検査した。

| # | ステップ | 結果 |
|---|---|---|
| 1 | gateway 起動 → `GET /` 200、`/static/*` 200 | タブ3つ描画(computed display 確認)、初期タブ h3・オーバーレイ表示・topbar「バックエンド停止中」 |
| 2 | 未起動状態で LTX タブ | オーバーレイにプリセット nf4(既定・選択済み)/fp8/bf16 が説明・VRAM ヒント付きで列挙。起動ボタン有効 |
| 3 | 「起動」クリック | スピナー表示 → ltx25 起動 → 約 10 秒で iframe 表示に自動切替(src=:8632)。iframe 内 DOM 検査: `title="LTX-2.5 Studio"`・readyState=complete・bodyLen=13570 |
| 4 | H3 タブへ切替 → 48gb-lowvram 選択 → 起動 | ボタンが「切替(現在の LTX-2.5 を停止して起動)」+ 警告文。confirm メッセージ「現在の LTX-2.5 を停止して MiniMax-H3 を起動します。…」を記録・許可 → 8632 閉鎖・8631 開放 → iframe に H3 UI(`title="MiniMax-H3"`・complete・bodyLen=105018)。ltx25 側 iframe は about:blank へ復帰 |
| 5 | 管理タブ + 統一 t2i(curl で `POST /api/v1/generate` h3/512²/seed42) | 状態カード: h3 アクティブ(port 8631)・preset 48gb-lowvram・PID/uptime・VRAM 2GPU 表示。ジョブ行が現れ progress **2%→5%→62%→95%→completed 100%**、生成中に busy「はい(生成中)」表示、成果物リンク(画像/動画)出現。**0.5 秒間隔の DOM 監視で行要素の再構築 0 回**(dataset マーカー存続、リンクの再生成なし) |
| 6 | アンロード(GUI のボタン) | 「なし(全バックエンド停止中)」へ遷移、H3 タブがオーバーレイへ復帰(iframe hidden + about:blank)、8631/8632 閉鎖 |
| 追加 | 外部要因停止の検知 | ltx25 起動 → iframe 表示中に pid を SIGKILL → 数秒後の status ポーリングでオーバーレイへ自動復帰(topbar も「バックエンド停止中」) |
| 回帰 | Phase 1/2 API | `/api/v1/status` 200、`/ltx25/api/health` パススルー 200(稼働中)、`/h3/api/status` パススルー 200(h3 稼働中)/ 502+起動案内(未起動)、load/unload 正常 |
| 7 | 後片付け | Chrome・gateway・バックエンド全停止、8630/8631/8632/9333 閉鎖、GPU:0 3.07GB(ベースライン 3.05GB 相当)、GPU:1 19MiB(全工程未使用) |

## 発見した問題点

1. **`hidden` 属性が `.spinner-row { display: flex }` に負ける実バグ**
   (スクリーンショット目視で発見・修正済み)。author CSS が display を指定した
   要素では UA スタイルシートの `[hidden] { display: none }` が上書きされ、
   起動完了後もスピナーが表示され続けていた。DOM 検査(`el.hidden` 属性の読取)
   だけでは通ってしまい、スクリーンショット + getComputedStyle で捕捉した。
   修正はグローバル `[hidden] { display: none !important; }`。
   **教訓: hidden 属性で出し分ける要素の検証は属性ではなく computed style で行うこと。**
2. **status ポーリングは in-flight ガードが必須**(設計段階で先取り対処)。
   `manager.load()` がロック保持でヘルス待ちする間、`/api/v1/status` は応答待ちに
   なる。素朴な setInterval + fetch だと h3 96gb の数分ロード中にリクエストが
   数百件滞留し、FastAPI のスレッドプール(anyio 既定 40)を食い潰しうる。
3. iframe の検証は同一ホスト別ポートだと CDP の別ターゲットにならない
   (Chrome の site isolation は scheme+eTLD+1 判定でポートを見ないため)。
   `/json` のターゲット列挙や `Target.getTargets` には現れず、
   `Page.getFrameTree` → `Page.createIsolatedWorld(frameId)` → contextId 指定の
   `Runtime.evaluate` で iframe 内 DOM を検査する必要がある。

## 既知の制限(Phase 3 時点)

- ジョブ一覧は gateway 発行分のみ(Phase 2 と同じ。パススルーで直接投げた
  バックエンドジョブは出ない)。gateway 再起動でジョブ履歴は消える(Phase 5)。
- LAN クライアントから使う場合、iframe はバックエンドのポート(8631/8632)へ
  直接接続するため、ファイアウォールで 8630 だけでなく 8631/8632 も開ける必要がある。
- overrides / toggles(turbo 等)の指定 UI は未実装(API では利用可能。
  必要になったら管理タブに追加する)。
