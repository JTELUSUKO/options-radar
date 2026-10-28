# Options Radar

Local dashboard that ranks options trade ideas using institutional-style signals:
dealer gamma (call/put walls, zero-gamma), unusual options activity, implied-vs-realized
volatility, earnings pricing vs history, trend, and news/rumor buzz. It shows the
top 5 ideas for whatever filters you set, with strikes, entry, take-profit, stop and timing.

No broker needed — you place trades yourself. Research tool, not financial advice.

## Run
    ./run.sh            # opens http://localhost:8765

Optional: `export ANTHROPIC_API_KEY=...` before running to have Claude read headlines
(rumor detection + sentiment) instead of keyword matching.

## Files
- `radar/engine.py`   – data (Yahoo Finance, free/delayed) and signal maths
- `radar/scoring.py`  – weights, strategy choice, strikes, trade plan
- `radar/news.py`     – Google News RSS + rumor/sentiment scoring
- `server.py`         – API, scanning, daily snapshots, track record
- `static/index.html` – the dashboard
- `data/`             – latest scan, daily snapshots, history of top-5 picks

Signal weights come from `~/reports/Institutional options trading signals.md`.

## Hosting (Azure, free F1)
- App: see `.azure` (app name + dashboard password; username can be anything)
- Update after changes / after adding keys to `.env`: `./deploy.sh`
- Free tier sleeps after ~20 min idle (first visit is slow) and has 60 CPU-min/day.
- Data (snapshots, history, Schwab token) persists in `/home/data` on the app.

## Schwab (thinkorswim) data
1. developer.schwab.com → create app, callback URL `https://options-radar-16f492.azurewebsites.net/`
2. Put `SCHWAB_APP_KEY` / `SCHWAB_APP_SECRET` in `.env` (then `./deploy.sh` for Azure)
3. Dashboard → Watchlist tab → Connect Schwab (repeat every 7 days)
