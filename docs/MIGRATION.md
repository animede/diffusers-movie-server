# 旧サーバからの移行手順

対象: 旧 minimax-h3(`/home/animede/minimax-h3`、port 8611)と
旧 diffusers-ltx2_5(`/home/animede/diffusers-ltx2_5`、port 8000)から
diffusers-movie-server(gateway 8630)への移行。

## (a) 旧ディレクトリは削除禁止(最重要)

本リポジトリの `backends/` は Phase 0 方針(`docs/INTEGRATION_PLAN.md`)により
**venv・モデル・量子化済み重みを旧ディレクトリへの symlink で共有**している:

| symlink | 実体(旧ディレクトリ側) | サイズ |
|---|---|---|
| `backends/minimax-h3/venv` | `/home/animede/minimax-h3/venv` | torch2.9+cu128、diffusers PR#14355 コミット固定 |
| `backends/minimax-h3/models` | `/home/animede/minimax-h3/models` | prequant TE キャッシュ 36GB |
| `backends/ltx2_5/.venv` | `/home/animede/diffusers-ltx2_5/.venv` | torch2.11+cu130、NATTEN shadow install |
| `backends/ltx2_5/LTX-2.5-Diffusers-bnb-4bit` | 旧ディレクトリ | 量子化済み 27GB |
| `backends/ltx2_5/loras` | 旧ディレクトリ | IC-LoRA 2本 1.6GB |

**`/home/animede/minimax-h3` と `/home/animede/diffusers-ltx2_5` を削除・改名・
移動すると新サーバは起動できなくなる。** 旧ディレクトリの整理(削除)は
Phase 5 の完全独立化(下記 (d))を完了してからにすること。

コード自体は rsync コピー済みのため、旧ディレクトリ側のコードを編集しても
新サーバには反映されない(逆も同様)。コード修正は本リポジトリ側で行うこと。

## (b) 既存クライアントの移行

### 方法1(推奨): パススルー URL へ置き換え

API 仕様は不変。ベース URL だけ変更する:

- `http://<host>:8611/...` → `http://<host>:8630/h3/...`
- `http://<host>:8000/...` → `http://<host>:8630/ltx25/...`

エンドポイントごとの対応表は README「既存 API 対応表」参照。注意点:

- **対象バックエンドが未起動だと 502**(日本語の起動案内付き)。事前に
  `POST /api/v1/backend/load {"backend": "h3"}` 等で起動するか、
  クライアント側で 502 を検知して load を叩く。
- 同時アクティブは1バックエンドのみ。h3 と ltx25 を同時に使う旧運用は
  そのままでは移行できない(切替運用にするか、下記方法2の併用)。
- 成果物 URL(`/h3/outputs/...` `/ltx25/outputs/...`)は gateway の静的配信のため
  **バックエンド停止後も取得できる**(旧サーバには無かった利点)。

### 方法2: 互換のため旧ポートで起動する

クライアント改修を後回しにしたい場合、バックエンドを旧ポートで直接起動できる:

```bash
# 旧 8611 互換で h3 を直接起動(gateway 非経由)
cd backends/minimax-h3 && H3_PORT=8611 ./run.sh

# 旧 8000 互換で ltx25 を直接起動
cd backends/ltx2_5 && LTX25_PORT=8000 ./run.sh
```

ただしこの方法は gateway の管理外(プリセット・排他切替・統一APIなし)。
env(`H3_LOWVRAM` 等)は自分で設定すること。恒久運用ではなく移行期間の
つなぎとして使う。

### 統一 API への移行(任意)

新規クライアントは `POST /api/v1/generate`(`docs/API_SPEC.md`)の利用を推奨。
バックエンド差(同期/非同期、resolution プリセット差、単位換算)を gateway が吸収する。

## (c) 旧サーバ直接起動と gateway 運用の併用時の注意

移行期間中に旧ディレクトリのサーバ(8611/8000)を直接起動する運用と
gateway 運用を併用する場合:

1. **VRAM 競合**: gateway は自分が起動したバックエンドしか把握しない。
   旧 8611 の h3(既定は常駐 87.3GB)が動いたまま gateway から何かを load すると
   合算で CUDA OOM する。**同時に GPU を使うのはどちらか一方だけ**にすること。
2. **ポート衝突はない**(8611/8000 と 8630/8631/8632 は非衝突)が、
   紛らわしいので併用期間は最小にする。
3. **HF キャッシュ・モデルは共有**(`~/.cache/huggingface`、prequant キャッシュ、
   量子化ディレクトリ)。同時起動での同一ファイル読み込みは問題ないが、
   量子化スクリプトの再実行(書き込み)は片方だけで行うこと。
4. ltx25 の履歴 DB は別(旧: 旧ディレクトリの `outputs/history.sqlite3`、
   新: `backends/ltx2_5/outputs/history.sqlite3`)。セッション履歴は引き継がれない。
   成果物も別ディレクトリに保存される(旧成果物は旧ディレクトリに残る)。
5. gateway は起動時に 8631/8632 の孤児プロセスを検出して adopt する
   (`docs/phase1-gateway.md`「孤児処理」)が、**8611/8000 で動く旧サーバは
   検出対象外**(別ポートのため)。

## (d) 完全独立化(Phase 5)の条件と手順

旧ディレクトリを削除できるようにするには:

1. venv を symlink から実体の再構築へ置き換える。手順は
   `backends/minimax-h3/VENV_REBUILD.md` / `backends/ltx2_5/VENV_REBUILD.md`
   (pip freeze 実測 = `venv-freeze.txt` 付き。h3 の diffusers はコミット固定、
   ltx25 の diffusers/transformers は freeze のコミット付き行を使うこと)。
2. 大容量物(prequant 36GB / 量子化済み 27GB / loras)を symlink から
   実体コピー(または再生成: `backends/ltx2_5/scripts/download_quantize_ltx25.py`)へ。
   ディスク空きに注意(合算 ~65GB)。
3. 再構築後、`docs/phase4-acceptance.md` のシナリオ(96gb ロード → t2v →
   切替 → ltx25 t2v → unload)を再実行して合格を確認。
4. 合格後にはじめて旧ディレクトリを削除してよい。

補足(Phase 0 発見事項): `backends/ltx2_5/.ltx25-component-caches` は symlink の
都合で量子化スクリプト実行時のみ旧ディレクトリ側を参照する。独立化と同時に整理する。
旧リポジトリ由来の `Dockerfile` / `compose.yaml`(port 8000 前提)はゲートウェイ構成では
未使用のまま残置している。
