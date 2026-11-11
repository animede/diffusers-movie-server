#!/usr/bin/env bash
# diffusers-movie-server backend: MiniMax-H3-Image (internal port 8633)
# config.py resolves outputs/ inputs/ cache/ models/ relative to this directory.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${H3IMG_PORT:-8633}"
exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
