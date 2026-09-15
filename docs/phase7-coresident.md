# Phase 7: 同居モード(coresident)— 重みの再ロードをゼロにする 2026-09-15

LTX-2.5 をリアルタイム用に常駐させたまま MiniMax-H3 も使いたい、という要求に対して
**「VRAM は両方保持・生成だけ排他」**というモードを足した。Phase 6 の宿題
(「2バックエンド同時アクティブ」)の回収でもあるが、当時想定していた per-GPU 化ではなく
**同一GPU上での同居**になった。実測で 96GB に両方載ることが分かったため。

## なぜ per-GPU 化ではないのか

Phase 6 では「GPU が分かれていれば同時アクティブにできる」と考えていた。しかし実際の
要求は「GPU0 に LTX を常駐させ、**同じ GPU0 の残りで** H3 を動かす」だった。
先に VRAM を実測したところ両方載ることが分かり、GPU を分ける理由がなくなった。

## 設計

排他の軸が2つに分かれるので、状態も2軸で持つ。

| 軸 | 実装 |
|---|---|
| **VRAM 保持** | `ProcessManager._loaded: set[str]`。process/resident では常に 0〜1 個、coresident でのみ複数になる |
| **実行(生成)** | 状態を持たず、投入時に `other_busy()` でバックエンドへ問い合わせる。`jobs.registry.submit()` が唯一のゲート |

`_active: str | None` を `_loaded: set[str]` に置き換えたのが変更の本体。
`active_backend_name()` は単一値しか返せず同居時に意味を持たないため廃止し、
`is_loaded(name)` / `loaded_backends()` に置き換えた(呼び出し元は app.py の
パススルー・prompt/enhance、jobs.py の3箇所)。

`/api/v1/status` には **`loaded_backends`** を追加。`active_backend` は後方互換で
残してあるが、同居時は先頭1つしか表せない。GUI(shell.js)は `active_backend` 依存を
やめ、`backends[name]` の2軸(`process_alive` / `weights_loaded`)で判定するようにした
— これを直さないと同居時に LTX タブが「未起動」オーバーレイを出してしまう。

## 戦略ごとの振る舞い

| strategy | 他バックエンド | 対象バックエンド |
|---|---|---|
| `process`(既定) | **プロセス停止** | 再起動 |
| `resident` | プロセス温存で **VRAM 解放** | env 一致なら reactivate |
| **`coresident`** | **一切触らない** | env 一致なら reactivate |

`unload` には `backend` パラメータを足し、同居中に片方だけ降ろせるようにした。

## 実機検証(GPU0 = RTX PRO 6000 96GB、2026-09-15)

| # | 検証 | 結果 |
|---|---|---|
| 1 | LTX を coresident でロード | started、`coresident_with: []` |
| 2 | H3 を coresident でロード | started、**`coresident_with: ["ltx25"]`**(LTX は退避されず) |
| 3 | status | `loaded_backends: ["h3","ltx25"]`、h3 37450MiB / ltx25 33136MiB、GPU0 71326MiB |
| 4 | 同居中の LTX 生成 | **3660ms**(単独 3.46〜3.81秒と同等 = 再ロードなし・劣化なし) |
| 5 | 同居中の H3 生成 | t2i 768² OK |
| 6 | **実行ゲート** | H3 生成中の LTX 投入が **409**、完了後は 202 |
| 7 | 片方だけ unload | h3 のみ解放(残 686MiB = CUDA コンテキスト)、ltx25 は 33136MiB のまま生成可 |
| 8 | parked からの reactivate | **13.7秒・PID 維持**(プロセス再利用)、LTX に影響なし |
| 9 | 退行: `strategy=process` | ltx25 を停止して h3 起動(従来どおり) |
| 10 | 退行: `strategy=resident` | h3 はプロセス温存で VRAM 解放、ltx25 起動(従来どおり) |

## 既知の制約

1. **`auto_load` は coresident にならない**。`/api/v1/generate` の auto_load は既定の
   process 戦略で起動するため、他バックエンドがロード済みならそれを停止する。
   同居させたいときは先に `backend/load` を明示的に呼ぶこと。
2. **VRAM 予算のガードは入れていない**。coresident の load 応答に `vram`(GPU ごとの
   used/total/free)を載せて可視化するにとどめた。プリセットの `vram_hint` は文字列で
   機械判定に向かず、誤った拒否を出すより実測値を見せる方が安全と判断した。
   余裕は実測で 12.4GiB しかないので、構成を変えるときは必ず測り直すこと。
3. **LTX の `upscale=true` ガードは未実装**(次の作業)。同居時に踏むと OOM する。
4. **h3 の同居構成は投影TE が前提**で、細部のプロンプト追従が落ちる(PSNR 22.64dB、
   鮮鋭度 -23%)。MV アプリ(`~/Minimax-H3-lipsync-mv`)は 6セクション記法で細部を
   指定する設計なので、採用するかは品質判断が要る。なお MV は `<d>` 台詞タグを
   使わない固定仕様のため、投影TE の `<d>` 非対応には抵触しない。
5. **MV アプリはプリセット名を照合する**(`app/services/h3.py`、不一致で H3Error)。
   同居構成は `96gb-int8` + overrides で表現しているのでプリセット名は変わらず、
   現状の MV はそのまま動く。名前付きプリセットに昇格させるなら MV 側の
   `H3_PRESET` も合わせて変えること。
