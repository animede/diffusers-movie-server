#!/usr/bin/env bash
# diffusers-movie-server backend: LTX-2.5 (internal port 8632)
# config.py resolves outputs/ inputs/ loras/ LTX-2.5-Diffusers-bnb-4bit/ and .env
# relative to the CWD, so always run from this directory.
set -euo pipefail
cd "$(dirname "$0")"

PORT="${LTX25_PORT:-8632}"
exec .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
