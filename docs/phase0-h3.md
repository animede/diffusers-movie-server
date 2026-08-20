# Phase 0: minimax-h3 バックエンド移設・スモーク結果(2026-08-20)

## 実施内容

1. **コード移設**: `/home/animede/minimax-h3` → `backends/minimax-h3` へ rsync コピー
   (除外: `venv/` `models/` `outputs/` `logs/` `.git/` `__pycache__/` `*.pyc`)。
   コピー後サイズ **190MB**(うち `third_party/` 187MB = SageAttention ソース + ビルド済み wheel)。
   旧ディレクトリは読み取りのみ・無変更。
2. **symlink**:
   - `backends/minimax-h3/venv` → `/home/animede/minimax-h3/venv`
   - `backends/minimax-h3/models` → `/home/animede/minimax-h3/models`(prequant TE キャッシュ 36GB 共有)
   - `outputs/` `logs/` は空ディレクトリを新規作成
3. **run.sh**: `PORT="${H3_PORT:-8631}"` で uvicorn を exec。H3_* env は素通し、秘匿値なし。
4. **VENV_REBUILD.md** + **venv-freeze.txt**(旧 venv の pip freeze 実測 343 行)を
   `backends/minimax-h3/` に作成。

## 起動時プリロード挙動(既定 env)

`app.py` の `@app.on_event("startup")` → `runner.preload_all()`。既定
(`H3_TE_QUANT=bnb-4bit` / `H3_TRANSFORMER_QUANT=none` / `H3_LOWVRAM=0`)では:

- vae / audio_vae: 重みを CPU にロード(GPU 0GB)
- transformer(t2va 用、bf16): GPU へ直接ロード、**allocated 66.28GB**(35.6s)
- text_encoder(32B NF4): prequant キャッシュ
  (`models/prequant/te_bnb-4bit_prune0`、symlink 経由で正しく解決)から GPU へ、
  16.9s → 合計 **allocated 87.29GB 常駐**
- attn=sage / cache=fbc(threshold 0.05)/ turbo_lora=off

つまり既定構成は 96GB 級 GPU 前提の全常駐モード(起動完了まで約55秒)。

## スモーク実測(GPU:0 = RTX PRO 6000 Blackwell 96GB)

| 項目 | 結果 |
|---|---|
| `GET /api/status` | 200(busy:false、transformer_loaded:true、te_quant:bnb-4bit) |
| `GET /`(UI) | 200 |
| `POST /api/t2i`(512×512、seed=12345、frames=22 既定) | 200・**total 53.42s**(denoise 6.27s / decode 0.99s / 0.216s/step ×30steps、fbc で 8 steps スキップ) |
| 出力 | `outputs/t2i_1787227903.png`(284KB)+ 超短尺 mp4(70KB)、レスポンスの png_path/mp4_path とも新ディレクトリ配下 |
| ピーク VRAM | **87.7GB**(レスポンス `peak_vram_gb`。常駐 87.29GB + 生成時ピーク) |
| 後片付け | PID kill 後、GPU:0 は 3.1GiB(他プロセスのベースライン)へ復帰、GPU:1 は終始 19MiB(未使用) |

t2i の total 53.42s のうち大半は TE エンコード等の per-request 固定費
(README 記載どおり「固定費 ~55s〜 支配」の想定範囲)。10分超の停滞・エラーはなし。

## 気づいた問題点

1. **既定構成の常駐 87.3GB は 96GB 専有前提**。他プロセスが数 GB 使っているだけで
   ピーク 87.7GB は際どい(今回ベースライン 3.1GB で成功)。gateway からの起動時は
   プリセット(80gb-int8 / 48gb-lowvram 等)で `H3_TRANSFORMER_QUANT` /
   `H3_LOWVRAM` を明示する運用が安全。
2. **起動ログの progress で `phase:"loading_transformer"` が生成リクエスト中に出る**:
   t2i リクエスト時に TE ロード等のフェーズ表示が transformer ロードと表示される
   場面があった(表示上のみ、実害なし。ゲートウェイの進捗表示実装時に注意)。
3. `venv-freeze.txt` より、torchao は `/tmp/torchao_check/...whl` からのローカル
   インストール。再構築時は PyPI の `torchao==0.17.0` で代替可(VENV_REBUILD.md 記載)。
4. スモークのサーバログは `backends/minimax-h3/logs/smoke_phase0.log`、
   t2i レスポンス JSON は同 `logs/t2i_smoke_response.json` に保存済み。

## 未実施(後続フェーズ)

- git commit(親がまとめて実施)
- gateway(8630)からの起動・パススルー(Phase 1)
- venv 完全独立再構築(Phase 5、手順は VENV_REBUILD.md)
