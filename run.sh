#!/usr/bin/env bash
# diffusers-movie-server gateway 起動スクリプト(port 8630、GW_PORT で上書き可)
set -euo pipefail
cd "$(dirname "$0")/gateway"

PORT="${GW_PORT:-8630}"
exec venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port "$PORT"
