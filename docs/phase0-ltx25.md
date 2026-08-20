# Phase 0 作業記録: diffusers-ltx2_5 バックエンド移設(2026-08-20)

担当: サブエージェント(非GPUスモークまで)。GPU生成スモークは後続で親が実施。

## 1. コード移設

- `rsync -a` で `/home/animede/diffusers-ltx2_5/` → `backends/ltx2_5/`。
  旧ディレクトリは読み取りのみ(無変更)。
- 除外: `.venv/` `outputs/` `inputs/` `LTX-2.5-Diffusers-bnb-4bit/` `loras/`
  `scratch*/`(scratch_ab / scratch_fp8_probe / scratch_t2i_probe)`.git/`
  `__pycache__/` `*.pyc` `.pytest_cache/` `*.log`(logs_server.log / server.log)
- コピー後サイズ: **948KB**(コード+ドキュメントのみ、除外が効いていることを確認)。
- `.env` はコピーした(中身は `LLM_BASE_URL` / `LLM_MODEL` の2キーのみ。
  **HF_TOKEN は入っていなかった** — HFキャッシュ `~/.cache/huggingface` 共有で
  動作しており、モデルは pinned revision がキャッシュ済みのため実害なし)。
  リポジトリ `.gitignore`(backends/ltx2_5/.gitignore、旧リポジトリ由来)で
  `.env` は除外済み。
- `.ltx25-component-caches/`(空ディレクトリ)もコピーされたが、参照するのは
  `scripts/download_quantize_ltx25.py` のみ(量子化ディレクトリの親基準で解決
  されるため、symlink 経由では旧側のキャッシュが使われる)。無害。

## 2. symlink / ディレクトリ

| エントリ | 種別 | 先 |
|---|---|---|
| `.venv` | symlink | `/home/animede/diffusers-ltx2_5/.venv` |
| `LTX-2.5-Diffusers-bnb-4bit` | symlink | 旧ディレクトリの量子化済み27GB |
| `loras` | symlink | 旧ディレクトリ(1.6GB、IC-LoRA 2本) |
| `outputs/` `inputs/` | 新規空ディレクトリ | — |

**パス解決の検証**: `app/config.py` は pydantic-settings
(`env_file=".env"`, extra="ignore")で、`output_dir` / `input_dir` / `lora_dir` /
`quantized_model_dir` / `history_db` はすべて **cwd 相対の既定値**
(`outputs` / `inputs` / `loras` / `LTX-2.5-Diffusers-bnb-4bit` /
`outputs/history.sqlite3`)。環境変数指定は不要で、**カレントディレクトリを
backends/ltx2_5 にして起動すれば symlink がそのまま機能する**。run.sh が
`cd "$(dirname "$0")"` を行うため .env へのパス明示は不要と判断(追記なし)。

## 3. run.sh

`backends/ltx2_5/run.sh`(実行可能)。`PORT="${LTX25_PORT:-8632}"`、
`exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`。

## 4. スモークテスト(非GPU、生成ジョブなし)

| 項目 | 結果 |
|---|---|
| `bash run.sh` 起動 | OK(モデルロードなし、`loaded: false`) |
| `GET /api/health` | 200 `{"status":"ok","model":"Lightricks/LTX-2.5-Diffusers","loaded":false,"prompt_enhancer":true}` |
| `GET /api/loras` | 200、symlink 経由で **IC-LoRA 2本**(ingredients 1.3GB / motion-track-control 327MB)がメタデータ付きで返る |
| `GET /`(UI) | 200 |
| `POST /api/sessions` | 201 `{"session_number":1}`、**新 outputs/ に history.sqlite3(20KB)が作成された** |
| `POST /api/jobs` | 実行していない(指示どおり) |
| pytest | `.venv/bin/python -m pytest tests/ -x -q` → **7 passed**(1.23s)。tests は FakeGenerator + TestClient でモデルロードなし、GPU不使用 |
| 後片付け | uvicorn プロセスを PID で kill、8632 が閉じたことを確認 |

## 5. ドキュメント

- `backends/ltx2_5/VENV_REBUILD.md`: venv 完全独立再構築手順(torch 2.11.0+cu130
  固定 / NATTEN は kernels>=0.16 経由の shi-labs/natten プリビルト /
  diffusers・transformers の実績 git コミット明記)。
- `backends/ltx2_5/venv-freeze.txt`: 旧 .venv の `pip freeze`(413行、2026-08-20)。

## 6. 発見した問題点・注意

1. **`.env` に HF_TOKEN が無い**(統合計画の想定と相違)。現状は HF キャッシュ
   共有 + pinned revision で問題ないが、キャッシュを消した場合や gated repo を
   使う場合は `.env` に `HF_TOKEN=` の追記が必要になりうる。
2. **requirements.txt の diffusers/transformers はコミット未固定**(`@main` 追従)。
   完全再現には venv-freeze.txt のコミット付き行を使うこと(VENV_REBUILD.md 記載)。
3. `.ltx25-component-caches` の解決先が symlink の都合で旧ディレクトリ側になる
   (量子化スクリプト実行時のみ関係。Phase 5 の venv 独立化と同時に整理推奨)。
4. 旧リポジトリ由来の `Dockerfile` / `compose.yaml` はポート 8000 前提のまま残置
   (ゲートウェイ構成では未使用。Phase 4 で削除または更新を判断)。

## 7. 未実施(後続)

- GPU 生成スモーク(POST /api/jobs)— 親が実施
- git commit — 親がまとめて実施
