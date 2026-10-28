#!/bin/sh
# Start Options Radar at http://localhost:8765
cd "$(dirname "$0")"
# Optional keys live in .env (e.g. ANTHROPIC_API_KEY=sk-ant-...)
[ -f .env ] && { set -a; . ./.env; set +a; }
[ -d .venv ] || { /usr/local/bin/python3.13 -m venv .venv && .venv/bin/pip install -q -r requirements.txt; }
lsof -ti :8765 | xargs kill 2>/dev/null
(sleep 2 && open http://localhost:8765) &
exec .venv/bin/uvicorn server:app --port 8765
