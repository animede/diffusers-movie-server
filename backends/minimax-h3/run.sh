#!/usr/bin/env bash
# minimax-h3 バックエンド起動スクリプト (diffusers-movie-server 統合用)
#
# - 内部ポートは既定 8631 (gateway 8630 からのパススルー先)。H3_PORT で上書き可。
# - H3_* 環境変数は呼び出し側 (gateway / 手動起動) から素通しする。
#   秘匿値 (LLM URL 等) はここにハードコードしない。
# - venv は当面 /home/animede/minimax-h3/venv への symlink (Phase 0 方針、
#   完全独立再構築の手順は VENV_REBUILD.md 参照)。
set -euo pipefail

cd "$(dirname "$0")"

PORT="${H3_PORT:-8631}"

exec venv/bin/python -u -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
