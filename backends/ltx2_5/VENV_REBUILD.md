# venv 完全独立再構築手順(ltx2_5 バックエンド)

現状(Phase 0)の `.venv` は旧ワークスペース
`/home/animede/diffusers-ltx2_5/.venv` への **symlink**(旧環境と共有)。
このドキュメントは、将来(Phase 5)symlink をやめて本ディレクトリ配下に
完全独立の venv を再構築するための手順と注意点。

参考: 実際に動作している環境の完全な `pip freeze` を `venv-freeze.txt`
(同ディレクトリ、2026-08-20 取得)として保存してある。

## 重要な制約(先に読むこと)

1. **torch は 2.11.0+cu130 固定**。torch 2.13 は torchaudio が非追従のため不可
   (torchaudio 2.11.0+cu130 とペアで入れる)。cu130 wheel は sm_120(Blackwell)
   カーネルを含む。実績の組み合わせ:
   `torch==2.11.0+cu130 / torchvision==0.26.0+cu130 / torchaudio==2.11.0+cu130`
2. **NATTEN は pip パッケージとして入れない**。`kernels>=0.16`(実績 0.16.0)経由で
   HF Hub の `shi-labs/natten` プリビルトカーネルを実行時に取得する方式
   (torch 2.11〜2.13 向けの variant が配布されている。cu130 / sm_120 で動作実績あり)。
   ソースビルドは不要・禁止(CLAUDE.md 50番の OOM 事故の教訓により、
   万一ビルドする場合は `MAX_JOBS=4 NVCC_THREADS=2` + MemoryMax 制限必須)。
3. **diffusers / transformers は git 開発版**(リリース版は LTX-2.5 非対応)。
   実績コミット(venv-freeze.txt より):
   - `diffusers @ git+https://github.com/huggingface/diffusers.git@11a82a15fe473ed974ff35111dd629b05fb1b3ed`
   - `transformers @ git+https://github.com/huggingface/transformers.git@f9b76f2dfc44fdc6109468925bf0e1856fd34278`
   requirements.txt はコミット未固定(`@main` 追従)のため、再現性を優先するなら
   上記コミットを明示すること(git 版は日々更新され壊れうる。CLAUDE.md 6番と同じ罠)。
4. その他の主要実績バージョン: `accelerate==1.12.0` / `bitsandbytes==0.49.0` /
   `huggingface_hub==1.28.0` / `kernels==0.16.0`。

## 手順

```bash
cd /home/animede/diffusers-movie-server/backends/ltx2_5

# 1) symlink を外す(旧ディレクトリ側は消さない)
rm .venv

# 2) 新規 venv(python 3.12 実績)
python3.12 -m venv .venv

# 3) torch ファミリーを cu130 index から固定インストール
.venv/bin/pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 4) 残りの依存(requirements.txt。再現性重視なら venv-freeze.txt を使う)
.venv/bin/pip install -r requirements.txt
#   もしくは完全再現: .venv/bin/pip install -r venv-freeze.txt
#   (venv-freeze.txt の torch 行は cu130 index が必要な点に注意。
#    先に手順3を実行しておけば freeze 中の torch 行は satisfied 扱いになる)

# 5) 開発・テスト用
.venv/bin/pip install -r requirements-dev.txt

# 6) 検証(非GPU): pytest はモデルロードなしで走る
.venv/bin/python -m pytest tests/ -x -q

# 7) 検証(GPU): サーバ起動 → 短尺生成1本
bash run.sh   # port 8632
```

## 注意

- NATTEN カーネルは初回の diffusion-decoder 実行時に `kernels` が HF Hub から
  取得する(要ネットワーク)。`app/generator.py` は NATTEN 不可時に
  compiled flex-attention へフォールバックする(遅くなるが動作はする)。
- 量子化済み重み `LTX-2.5-Diffusers-bnb-4bit/`(27GB)と `loras/` は
  旧ディレクトリへの symlink(再構築の対象外。実体を移す場合は
  `scripts/download_quantize_ltx25.py` で再生成も可能)。
- `.env` は cwd 相対で読まれる(pydantic-settings `env_file=".env"`)。
  run.sh がディレクトリ移動を行うため、起動は必ず run.sh 経由にすること。
