"""Options Radar — local dashboard server.

Run:  ./run.sh   then open http://localhost:8765
"""
import asyncio
import base64
import json
import math
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from radar import engine, levels, paper, scoring, sources
from radar.universe import DEFAULT_UNIVERSE, FUTURES, INDEX_SYMBOLS, norm_ticker

ROOT = Path(__file__).parent
DATA = sources.DATA
DATA.mkdir(parents=True, exist_ok=True)
(DATA / "snapshots").mkdir(exist_ok=True)
SETTINGS = DATA / "settings.json"
LATEST = DATA / "latest.json"
HISTORY = DATA / "history.jsonl"

app = FastAPI(title="Options Radar")
PASSWORD = os.environ.get("DASHBOARD_PASSWORD")


@app.middleware("http")
async def require_password(request: Request, call_next):
    """Simple browser password prompt when DASHBOARD_PASSWORD is set (used when hosted)."""
    if PASSWORD:
        ok = False
        hdr = request.headers.get("authorization", "")
        if hdr.startswith("Basic "):
            try:
                _, pw = base64.b64decode(hdr[6:]).decode().split(":", 1)
                ok = secrets.compare_digest(pw, PASSWORD)
            except Exception:
                ok = False
        if not ok:
            return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Options Radar"'})
    return await call_next(request)
state = {"running": False, "done": 0, "total": 0, "errors": [], "started": None, "finished": None}
details: dict = {}        # ticker -> full detail (candles, gamma profile, flow, news)
lock = threading.Lock()


def load_settings():
    if SETTINGS.exists():
        return json.loads(SETTINGS.read_text())
    return {"universe": DEFAULT_UNIVERSE}


def _clean(o):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not k.startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def summary(r):
    i = r["idea"]
    return {
        "ticker": r["ticker"], "sector": r["sector"], "price": r["trend"]["price"],
        "change_1d": r["trend"]["change_1d"], "mom_1m": r["trend"]["mom_1m"],
        "atm_iv": r["vol"]["atm_iv"], "iv_rv": r["vol"]["iv_rv"],
        "regime": r["gamma"]["regime"], "call_wall": r["gamma"]["call_wall"],
        "put_wall": r["gamma"]["put_wall"], "zero_gamma": r["gamma"]["zero_gamma"],
        "earnings": r["earnings"], "days_to_earnings": r["days_to_earnings"],
        "direction": r["direction"], "vol_edge": r["vol_edge"], "signals": r["signals"],
        "rumor_flow_confirmed": r["rumor_flow_confirmed"],
        "rumor_score": r["news"]["rumor_score"], "sentiment": r["news"]["sentiment"],
        "unusual_count": r["flow"]["unusual_count"],
        "liquidity_oi": sum(l["oi"] for l in i["legs"]),
        "idea": i, "as_of": r["as_of"], "data_source": r["data_source"],
    }


def skipped_summary(r):
    """A scanned ticker with no tradeable idea — still listed so it can be searched."""
    t, v, g = r["trend"], r.get("vol") or {}, r.get("gamma") or {}
    return {"ticker": r["ticker"], "sector": r["sector"], "price": t["price"], "change_1d": t["change_1d"],
            "mom_1m": t["mom_1m"], "atm_iv": v.get("atm_iv"), "iv_rv": v.get("iv_rv"), "regime": g.get("regime"),
            "earnings": r["earnings"], "days_to_earnings": r["days_to_earnings"],
            "reason": r.get("no_idea") or "no trade passed the checks", "data_source": r["data_source"]}


_queued_scan: dict = {}   # {"tickers": ...} when a scan was asked for while another was running


def run_scan(tickers: dict) -> bool:
    """Scan the tickers. If a scan is already running, queue this one to run right after it (returns False)."""
    with lock:
        if state["running"]:
            _queued_scan["tickers"] = tickers   # newest request wins
            return False
        state.update(running=True, done=0, total=len(tickers), errors=[], started=datetime.now().isoformat())
    scanned = None
    try:
        scanned = _scan(tickers)
    except Exception as e:  # noqa: BLE001 - report it, but never leave the scan stuck as "running"
        state["errors"].append(f"scan failed: {e}")
    finally:
        with lock:
            state.update(running=False, finished=datetime.now().isoformat())
            queued = _queued_scan.pop("tickers", None)
    if queued is not None:
        threading.Thread(target=run_scan, args=(queued,), daemon=True).start()
        return True
    if scanned:
        _paper_after_scan(*scanned)
        threading.Thread(target=_kronos_precompute, daemon=True).start()
    return True


def _scan(tickers: dict):
    failed = {}

    def work(item):
        t, sector = item
        try:
            return engine.analyze(t, sector)
        except Exception as e:  # noqa: BLE001 - one bad ticker shouldn't stop the scan
            state["errors"].append(f"{t}: {e}")
            failed[t] = {"ticker": t, "sector": sector, "price": None, "reason": str(e)}
            return None
        finally:
            state["done"] += 1

    with ThreadPoolExecutor(max_workers=6) as ex:
        results = [r for r in ex.map(work, tickers.items()) if r]
    scored = [r for r in scoring.score_universe(results) if r.get("idea")]
    rows = sorted((_clean(summary(r)) for r in scored), key=lambda x: -x["idea"]["score"])
    skipped = sorted([_clean(skipped_summary(r)) for r in results if not r.get("idea")] + list(failed.values()),
                     key=lambda x: x["ticker"])
    for r in scored:
        details[r["ticker"]] = _clean({
            "ticker": r["ticker"], "candles": r["candles"], "gamma": r["gamma"], "flow": r["flow"],
            "vol": r["vol"], "news": r["news"], "trend": r["trend"],
            "earnings_history": r["earnings_history"], "earnings_implied": r["earnings_implied"],
        })
    payload = {"generated": datetime.now().isoformat(timespec="seconds"), "rows": rows, "skipped": skipped}
    LATEST.write_text(json.dumps(payload))
    (DATA / "snapshots" / f"{date.today().isoformat()}.json").write_text(json.dumps(payload))
    (DATA / "details.json").write_text(json.dumps(details))
    _record_history(rows[:5])
    return results, rows


def _paper_after_scan(results, rows):
    """Paper account: reuse the scan's chains for marking, apply exit rules, auto-trade if switched on."""
    try:
        wanted = {p["ticker"] for p in paper.load()["positions"]} | {r["ticker"] for r in rows[:5]}
        for r in results:
            if r["ticker"] in wanted:
                paper.seed_quotes(r["ticker"], r.get("_chains"), r["trend"]["price"])
        paper.after_scan(rows)
    except Exception as e:  # noqa: BLE001 - paper trading must never break a scan
        state["errors"].append(f"paper trading: {e}")


def _history_lines() -> list[dict]:
    """history.jsonl rows, skipping blank or half-written lines (e.g. from a crash mid-append)."""
    rows = []
    for line in HISTORY.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def _record_history(top):
    """Append today's top 5 once per day (for the track record)."""
    today = date.today().isoformat()
    seen = set()
    if HISTORY.exists():
        for h in _history_lines():
            if h.get("date") == today:
                seen.add(h.get("ticker"))
    with HISTORY.open("a") as f:
        for r in top:
            if r["ticker"] in seen:
                continue
            i = r["idea"]
            f.write(json.dumps({
                "date": today, "ticker": r["ticker"], "price": r["price"], "label": i["label"],
                "strategy": i["strategy"], "bias": i["bias"], "score": i["score"], "expiry": i["expiry"],
                "breakevens": i["breakevens"], "legs": [{k: l[k] for k in ("action", "type", "strike", "mid")} for l in i["legs"]],
                "cost": i["cost"], "max_loss": i["max_loss"], "max_profit": i["max_profit"],
            }) + "\n")


# ------------------------------------------------------------------ API
class ScanReq(BaseModel):
    tickers: list[str] | None = None


@app.post("/api/scan")
def scan(req: ScanReq | None = None):
    uni = load_settings()["universe"]
    if req and req.tickers:
        uni = {t.upper(): uni.get(t.upper(), "Custom") for t in req.tickers}
    with lock:
        busy = state["running"]
    threading.Thread(target=run_scan, args=(uni,), daemon=True).start()
    return {"started": not busy, "queued": busy, "total": len(uni)}


@app.get("/api/status")
def status():
    return state


@app.get("/api/results")
def results():
    if not LATEST.exists():
        return {"generated": None, "rows": []}
    return json.loads(LATEST.read_text())


_details_loaded = {"done": False}


def _load_details():
    """Fill `details` from the last scan saved on disk, once per process (scans keep it current after that)."""
    if _details_loaded["done"]:
        return
    _details_loaded["done"] = True
    if (DATA / "details.json").exists():
        for k, v in json.loads((DATA / "details.json").read_text()).items():
            details.setdefault(k, v)


@app.get("/api/ticker/{t}")
def ticker(t: str):
    t = t.upper()
    _load_details()
    if t not in details:
        raise HTTPException(404, "Ticker not in last scan")
    return details[t]


class Settings(BaseModel):
    universe: dict[str, str]


@app.get("/api/settings")
def get_settings():
    return load_settings()


@app.post("/api/settings")
def save_settings(s: Settings):
    uni = {k.upper().strip(): v for k, v in s.universe.items() if k.strip()}
    if not uni:  # empty list = reset to the default watchlist
        SETTINGS.unlink(missing_ok=True)
        return {"ok": True, "count": len(DEFAULT_UNIVERSE)}
    SETTINGS.write_text(json.dumps({"universe": uni}, indent=2))
    return {"ok": True, "count": len(uni)}


@app.get("/api/history")
def history():
    """Past top-5 picks, marked against the latest stock price."""
    if not HISTORY.exists():
        return {"picks": [], "stats": {}}
    picks = _history_lines()
    tickers = sorted({p["ticker"] for p in picks})
    prices = {}
    try:
        ysyms = [sources.yahoo_symbol(t) for t in tickers]
        data = yf.download(ysyms, period="5d", progress=False, auto_adjust=False)["Close"]
        last = data.ffill().iloc[-1]
        prices = {t: float(last[y]) if len(tickers) > 1 else float(last) for t, y in zip(tickers, ysyms)}
        prices = {t: v for t, v in prices.items() if math.isfinite(v)}   # no data (e.g. delisted) → unknown, not NaN
    except Exception:
        pass
    wins = graded = 0
    for p in picks:
        now = prices.get(p["ticker"])
        p["now"] = now
        if now is None:
            p["status"] = "?"
            continue
        p["move"] = now / p["price"] - 1
        be = p["breakevens"]
        if p["bias"] == "bullish":
            good = now > p["price"] if p["strategy"] != "credit_spread" else now > be[0]
        elif p["bias"] == "bearish":
            good = now < p["price"] if p["strategy"] != "credit_spread" else now < be[0]
        elif p["strategy"] == "iron_condor":
            good = be[0] < now < be[1]
        else:
            good = now < be[0] or now > be[1]
        p["status"] = "on track" if good else "against"
        if p["date"] != date.today().isoformat():
            graded += 1
            wins += good
    return _clean({"picks": picks[::-1], "stats": {"graded": graded, "on_track": wins,
                                                   "hit_rate": wins / graded if graded else None}})


@app.get("/api/schwab/status")
def schwab_status():
    return sources.schwab_status()


@app.post("/api/schwab/start")
def schwab_start():
    if not sources.schwab_configured():
        raise HTTPException(400, "Add SCHWAB_APP_KEY and SCHWAB_APP_SECRET first.")
    return {"url": sources.schwab_login_url(), "callback": sources.CALLBACK_URL}


class SchwabFinish(BaseModel):
    url: str


@app.post("/api/schwab/finish")
def schwab_finish(req: SchwabFinish):
    try:
        sources.schwab_finish_login(req.url)
    except Exception as e:  # noqa: BLE001 - surface the reason to the user
        raise HTTPException(400, f"Schwab login failed: {e}")
    return sources.schwab_status()


# ------------------------------------------------------------------ paper trading (fake money, no broker orders)
class PaperOpen(BaseModel):
    ticker: str
    contracts: int = 1


class PaperClose(BaseModel):
    id: str


class PaperReset(BaseModel):
    cash: float = paper.START_CASH


class PaperSettings(BaseModel):
    auto_trade: bool | None = None
    top_n: int | None = None
    auto_close: bool | None = None


def _paper_payload(acct):
    return _clean({**paper.summary(acct), "positions": acct["positions"], "closed": acct["closed"][::-1],
                   "equity_history": acct["equity_history"], "settings": acct["settings"],
                   "log": acct.get("log", [])[::-1][:15], "data_source": sources.schwab_status()["source"]})


@app.get("/api/paper")
def paper_account():
    return _paper_payload(paper.mark_all())


@app.post("/api/paper/open")
def paper_open(req: PaperOpen):
    t = req.ticker.upper().strip()
    rows = json.loads(LATEST.read_text())["rows"] if LATEST.exists() else []
    row = next((r for r in rows if r["ticker"] == t), None)
    if row is None:
        raise HTTPException(404, f"{t} has no trade idea in the current scan.")
    if req.contracts < 1 or req.contracts > 50:
        raise HTTPException(400, "Contracts must be between 1 and 50.")
    try:
        pos = paper.open_idea(row, req.contracts)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _clean(pos)


@app.post("/api/paper/close")
def paper_close(req: PaperClose):
    try:
        return _clean(paper.close(req.id))
    except KeyError as e:
        raise HTTPException(404, e.args[0])
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/paper/reset")
def paper_reset(req: PaperReset | None = None):
    cash = req.cash if req else paper.START_CASH
    if not 100 <= cash <= 10_000_000:
        raise HTTPException(400, "Starting cash must be between $100 and $10,000,000.")
    return _paper_payload(paper.reset(cash))


@app.get("/api/paper/settings")
def paper_get_settings():
    return paper.load()["settings"]


@app.post("/api/paper/settings")
def paper_save_settings(req: PaperSettings):
    return paper.update_settings({k: v for k, v in req.model_dump().items() if v is not None})


# ------------------------------------------------------------------ charts tab
# Kronos needs PyTorch. On the Mac it lives in its own .venv-kronos; on Azure (Linux) the server installs it
# once into persistent storage (DATA/kronos-pkgs) in the background, so deploys stay small and fast.
KRONOS_PY = ROOT / ".venv-kronos" / "bin" / "python"
KRONOS_PKGS = DATA / "kronos-pkgs"
kronos_install = {"state": "ready" if KRONOS_PY.exists() else "missing", "error": None}


def _kronos_ready() -> bool:
    return kronos_install["state"] == "ready"


def _install_kronos():
    """Linux only: pip-install CPU PyTorch + Kronos deps into DATA/kronos-pkgs (≈1 GB, once)."""
    global KRONOS_PY
    if (KRONOS_PKGS / ".done").exists():
        KRONOS_PY = Path(sys.executable); kronos_install["state"] = "ready"
        return
    kronos_install["state"] = "installing"
    pip = [sys.executable, "-m", "pip", "install", "-q", "--target", str(KRONOS_PKGS), "--upgrade"]
    try:
        subprocess.run(pip + ["--index-url", "https://download.pytorch.org/whl/cpu", "torch"], check=True,
                       capture_output=True, text=True, timeout=3600)
        subprocess.run(pip + ["einops", "huggingface_hub", "safetensors", "tqdm"], check=True,
                       capture_output=True, text=True, timeout=1800)
        (KRONOS_PKGS / ".done").write_text(datetime.now().isoformat())
        KRONOS_PY = Path(sys.executable); kronos_install["state"] = "ready"
    except Exception as e:  # noqa: BLE001
        kronos_install.update(state="failed", error=(getattr(e, "stderr", "") or str(e))[-300:])
KRONOS_DIR = DATA / "kronos"
chart_cache: dict = {}    # (ticker, timeframe) -> (fetched at, payload)
TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "1d"]
TF_LABEL = {"1m": "1-minute", "5m": "5-minute", "15m": "15-minute", "30m": "30-minute", "1h": "1-hour", "1d": "daily"}
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}
KRONOS_BARS = {"1m": 30, "5m": 39, "15m": 26, "30m": 26, "1h": 28, "1d": 20}   # forecast length per timeframe


def _tf(tf: str) -> str:
    if tf not in TIMEFRAMES:
        raise HTTPException(400, f"Timeframe must be one of {', '.join(TIMEFRAMES)}")
    return tf


def _chart_time(ts, tf: str):
    """Chart x value: a date for daily bars; for intraday, New York wall-clock time as epoch seconds
    (the chart library shows UTC, so this makes it display NY times like TradingView)."""
    if tf == "1d":
        return ts.strftime("%Y-%m-%d")
    return int(pd.Timestamp(ts.replace(tzinfo=None)).tz_localize("UTC").timestamp())


def _stamp(ts, tf: str) -> str:
    return ts.strftime("%Y-%m-%d" if tf == "1d" else "%b %d %H:%M")


def _chart_data(t: str, tf: str = "1d") -> dict:
    hit = chart_cache.get((t, tf))
    # short cache so the chart's live refresh (every 10–60 s) sees new bars
    if hit and time.time() - hit[0] < {"1m": 8, "5m": 12, "15m": 25, "30m": 25, "1h": 50}.get(tf, 50):
        return hit[1]
    hist = sources.history(t) if tf == "1d" else sources.intraday_history(t, tf)
    if hist is None or len(hist) < 60:
        raise HTTPException(404, f"No {TF_LABEL[tf]} price history for {t}")
    hist = hist.tail(400)
    candles = [{"time": _chart_time(idx, tf), "iso": idx.isoformat(), "open": round(float(r.Open), 2),
                "high": round(float(r.High), 2), "low": round(float(r.Low), 2), "close": round(float(r.Close), 2),
                "volume": float(r.Volume)} for idx, r in hist.iterrows()]
    _load_details()
    g = details.get(t, {}).get("gamma")
    zones = levels.find_zones(hist, stamp=lambda x: _stamp(x, tf))
    for z in zones["demand"] + zones["supply"]:
        z["time"] = _chart_time(z.pop("ts"), tf)
    payload = _clean({"ticker": t, "tf": tf, "tf_label": TF_LABEL[tf], "candles": candles, "futures": sources.is_future(t),
                      "levels": levels.find_levels(hist, stamp=lambda x: _stamp(x, tf)), "zones": zones,
                      "gamma": {k: g.get(k) for k in ("call_wall", "put_wall", "zero_gamma", "regime")} if g else None,
                      "claude_available": bool(os.environ.get("ANTHROPIC_API_KEY"))})
    chart_cache[(t, tf)] = (time.time(), payload)
    return payload


def _kronos_file(t: str, tf: str) -> Path:
    t = t.replace("/", "_")   # futures: /ES -> _ES
    return KRONOS_DIR / (f"{t}-{date.today().isoformat()}.json" if tf == "1d" else f"{t}-{tf}.json")


def _kronos_cached(t: str, tf: str = "1d") -> dict | None:
    f = _kronos_file(t, tf)
    if not f.exists():
        return None
    if tf != "1d" and time.time() - f.stat().st_mtime > 2 * 60 * TF_MINUTES[tf] + 300:  # intraday forecasts go stale fast
        return None
    return json.loads(f.read_text())


def _future_bars(last: pd.Timestamp, tf: str, n: int, ticker: str = "") -> list[pd.Timestamp]:
    """Next n bar times inside trading hours: 9:30–16:00 New York weekdays for stocks, the ~23h session for futures."""
    if tf == "1d":
        return list(pd.bdate_range(last.date(), periods=n + 1)[1:])
    step, out, t = pd.Timedelta(minutes=TF_MINUTES[tf]), [], last
    if sources.is_future(ticker):   # step through, skipping the 17:00–18:00 break and the weekend
        while len(out) < n:
            t = t + step
            if sources.in_session(ticker, t):
                out.append(t)
        return out
    while len(out) < n:
        t = t + step
        if t.weekday() >= 5 or t.hour * 60 + t.minute >= 16 * 60:
            t = (t + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=9, minutes=30)
            while t.weekday() >= 5:
                t += pd.Timedelta(days=1)
        elif t.hour * 60 + t.minute < 9 * 60 + 30:
            t = t.normalize() + pd.Timedelta(hours=9, minutes=30)
        out.append(t)
    return out


@app.get("/api/chart/{t}")
def chart(t: str, tf: str = "1d"):
    t, tf = norm_ticker(t), _tf(tf)
    # Kronos status is added per request (not cached with the chart) so it flips to ready right away
    return {**_chart_data(t, tf), "forecast": _kronos_cached(t, tf), "market_open": sources.in_session(t),
            "kronos_available": _kronos_ready(), "kronos_state": kronos_install["state"]}


kronos_jobs: dict = {}    # (ticker, tf) -> {"state": running|done|failed, "error": str, "started": ts}
# the web server has one CPU core, so it runs fewer sample paths than the Mac
KRONOS_PATHS = int(os.environ.get("KRONOS_PATHS", "4" if sys.platform.startswith("linux") else "8"))
# bars of history Kronos reads; less history = faster on the single-core web server
KRONOS_CONTEXT = int(os.environ.get("KRONOS_CONTEXT", "256" if sys.platform.startswith("linux") else "400"))


@app.post("/api/chart/{t}/kronos")
def kronos_start(t: str, tf: str = "1d"):
    """Start a Kronos forecast in the background (takes 1–4 min); poll GET for the result."""
    t, tf = norm_ticker(t), _tf(tf)
    if cached := _kronos_cached(t, tf):
        return {"state": "done", "forecast": cached}
    job = kronos_jobs.get((t, tf))
    if not job or job["state"] != "running":
        if not _kronos_ready():
            kronos(t, tf)  # raises the "not installed / installing" error
        kronos_jobs[(t, tf)] = {"state": "running", "error": None, "started": time.time()}

        def work():
            _user_waiting["n"] += 1
            try:
                kronos(t, tf)
                kronos_jobs[(t, tf)]["state"] = "done"
            except HTTPException as e:
                kronos_jobs[(t, tf)].update(state="failed", error=e.detail)
            except Exception as e:  # noqa: BLE001
                kronos_jobs[(t, tf)].update(state="failed", error=str(e))
            finally:
                _user_waiting["n"] -= 1
        threading.Thread(target=work, daemon=True).start()
    return kronos_status(t, tf)


@app.get("/api/chart/{t}/kronos")
def kronos_status(t: str, tf: str = "1d"):
    t, tf = norm_ticker(t), _tf(tf)
    if cached := _kronos_cached(t, tf):
        return {"state": "done", "forecast": cached}
    job = kronos_jobs.get((t, tf))
    if not job:
        return {"state": "none"}
    return {"state": job["state"], "error": job["error"], "elapsed": round(time.time() - job["started"])}


_kworker = {"proc": None}
_kworker_lock = threading.Lock()   # one forecast at a time (the web server has one CPU core)
_user_waiting = {"n": 0}           # forecasts someone clicked for; background pre-compute steps aside for these


KRONOS_START_TIMEOUT = 900   # first start may download the model
KRONOS_CALL_TIMEOUT = 300


def _read_line(p: subprocess.Popen, timeout: float) -> str:
    """readline() that kills the process (so the read returns "") if no line arrives in time."""
    timer = threading.Timer(timeout, p.kill)
    timer.start()
    try:
        return p.stdout.readline()
    finally:
        timer.cancel()


def _kronos_worker_call(req: dict) -> dict:
    """Send one forecast to the long-running Kronos process (started on first use, restarted if it dies)."""
    with _kworker_lock:
        p = _kworker["proc"]
        if p is None or p.poll() is not None:
            env = {**os.environ, "HF_HOME": str(DATA / "hf"), "KRONOS_PKGS": str(KRONOS_PKGS)}  # model cache survives restarts
            p = subprocess.Popen([str(KRONOS_PY), str(ROOT / "radar" / "kronos_run.py"), "--serve"], env=env, text=True,
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1)
            _kworker["proc"] = p
            if not _read_line(p, KRONOS_START_TIMEOUT):   # waits for {"ready": true} while torch + the model load
                p.kill()
                _kworker["proc"] = None
                raise HTTPException(500, "Kronos failed to start")
        p.stdin.write(json.dumps(req) + "\n")
        p.stdin.flush()
        line = _read_line(p, KRONOS_CALL_TIMEOUT)
        try:
            out = json.loads(line) if line else None
        except ValueError:
            out = None
        if not isinstance(out, dict):   # died, timed out, or out of step: start a fresh worker next time
            p.kill()
            _kworker["proc"] = None
            raise HTTPException(500, "Kronos stopped unexpectedly — try again")
        if "error" in out:
            raise HTTPException(500, "Kronos failed: " + out["error"])
        return out


def kronos(t: str, tf: str = "1d"):
    """Run Kronos now (blocking) and cache the forecast."""
    if not _kronos_ready():
        msg = {"installing": "Kronos is still being set up on the server (one-time, ~10–15 min) — try again soon.",
               "failed": f"Kronos setup failed: {kronos_install['error']}"}.get(kronos_install["state"], "Kronos isn't installed here.")
        raise HTTPException(503 if kronos_install["state"] == "installing" else 501, msg)
    if cached := _kronos_cached(t, tf):
        return cached
    d = _chart_data(t, tf)
    last = pd.Timestamp(d["candles"][-1]["iso"])
    future_ts = _future_bars(last, tf, KRONOS_BARS[tf], t)
    # Kronos reads the bar timestamps (minute/hour/weekday), so give it New York wall-clock times
    kcandles = [{**c, "time": c["iso"][:19]} for c in d["candles"]]
    future = [x.strftime("%Y-%m-%dT%H:%M:%S") for x in future_ts]
    out = _kronos_worker_call({"candles": kcandles[-KRONOS_CONTEXT:], "future": future, "paths": KRONOS_PATHS})
    out.update(tf=tf, labels=[_stamp(x, tf) for x in future_ts], dates=[_chart_time(x, tf) for x in future_ts])
    KRONOS_DIR.mkdir(exist_ok=True)
    _kronos_file(t, tf).write_text(json.dumps(out))
    return out


def _ticker_context(t: str, tf: str = "1d") -> str:
    d = _chart_data(t, tf)
    return levels.chart_context(t, d["levels"], d["zones"], d["gamma"], _kronos_cached(t, tf), d["candles"], TF_LABEL[tf])


NOT_TICKERS = {"OK", "IV", "AI", "EMA", "ATM", "OTM", "ITM", "DTE", "USD", "ETF", "PUT", "CALL", "THE", "AND", "WHAT", "HOW", "WHY"}


# futures by name in chat ("how is gold doing?")
FUTURES_WORDS = {"GOLD": "/GC", "CRUDE": "/CL", "OIL": "/CL", "WTI": "/CL"}


def _known_tickers() -> set:
    return set(load_settings()["universe"]) | set(INDEX_SYMBOLS) | {f[1:] for f in FUTURES} | set(FUTURES_WORDS)


class ChatReq(BaseModel):
    messages: list[dict]          # [{role: user|assistant, content: str}, ...]
    ticker: str | None = None     # the chart the viewer has open, if any
    tf: str = "1d"                # and its timeframe


CHAT_SYSTEM = (
    "You are the assistant inside Options Radar, an options-trading research dashboard. The user is a retail "
    "options trader whose English is informal; answer in plain, simple English with short sentences and bullet "
    "points, and keep answers under ~200 words unless asked for more. Explain terms (support/resistance, "
    "supply/demand zones, gamma walls, IV, spreads, Kronos forecast, Ripster EMA clouds) simply when they come up. "
    "Base answers on the market data below (from Schwab when connected, otherwise Yahoo) and say when something "
    "isn't in it. When asked for a prediction, give one clearly: a lean (bullish / bearish / range-bound), a likely "
    "price range for the next 1-4 weeks, the levels that would prove it wrong, and how confident you are (low / "
    "medium / high) with the reasons from the data. Suggest an options idea that fits if useful. You can't place "
    "trades. Predictions are educated guesses, not financial advice — mention risk and position size briefly.")


@app.post("/api/chat")
def chat(req: ChatReq):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(400, "Add ANTHROPIC_API_KEY to .env to use the chat.")
    ctx = []
    if LATEST.exists():
        rows = json.loads(LATEST.read_text())["rows"][:5]
        ctx.append("Today's top 5 ideas:\n" + "\n".join(
            f"- {r['ticker']} ${r['price']:.2f}: {r['idea']['label']} ({r['idea']['bias']}), score {r['idea']['score']:.0f}, "
            f"expiry {r['idea']['expiry']}, max loss ${r['idea']['max_loss']}, why: {'; '.join(r['idea']['why'][:2])}"
            for r in rows))
    src = sources.schwab_status()["source"]
    ctx.insert(0, f"Market data source right now: {'Schwab (live)' if src == 'schwab' else 'Yahoo (delayed ~15 min)'}. Today is {date.today().isoformat()}.")
    # the open chart, plus up to 2 tickers named in the question (e.g. "predict TSLA")
    asked = req.messages[-1].get("content", "") if req.messages else ""
    known = _known_tickers()
    named = [w for dollar, w in re.findall(r"(\$?)\b([A-Za-z]{1,5})\b", asked)
             if (dollar and w.upper() in known) or w.upper() in FUTURES_WORDS
             or (w.isupper() and w not in NOT_TICKERS and (w in known or len(w) > 1))]
    named = [FUTURES_WORDS.get(w.upper(), norm_ticker(w)) for w in named]
    req.ticker = norm_ticker(req.ticker) if req.ticker else None
    wanted = list(dict.fromkeys(([req.ticker] if req.ticker else []) + named))[:3]
    # a prediction question starts Kronos and waits for it up to ~2.5 min (the web host cuts requests at ~4 min),
    # so Claude can weigh it; if it isn't done in time Claude answers without it
    kronos_note = None
    if wanted and _kronos_ready() and re.search(r"predict|forecast|kronos|target|next (few )?(day|week|month)", asked, re.I):
        kt, ktf = wanted[0], req.tf if wanted[0] == (req.ticker or "").upper() else "1d"
        try:
            kronos_start(kt, ktf)
            for _ in range(75):
                if kronos_status(kt, ktf)["state"] != "running":
                    break
                time.sleep(2)
            if kronos_status(kt, ktf)["state"] == "running":
                kronos_note = f"The Kronos forecast for {kt} is still running — tell the user to ask again in a minute to include it."
        except Exception:  # noqa: BLE001 - answer without it rather than fail
            pass
    if kronos_note:
        ctx.append(kronos_note)
    for t in wanted:
        try:
            on_chart = t == (req.ticker or "").upper()
            tf = _tf(req.tf) if on_chart else "1d"
            ctx.append(f"{t} {TF_LABEL[tf]} chart data{' (chart the user has open)' if on_chart else ''}:\n{_ticker_context(t, tf)}")
        except Exception:  # noqa: BLE001 - not a real ticker / no data
            pass
    msgs = [{"role": m["role"], "content": str(m["content"])[:4000]} for m in req.messages[-12:]
            if m.get("role") in ("user", "assistant") and m.get("content")]
    if not msgs or msgs[-1]["role"] != "user":
        raise HTTPException(400, "Send a question.")
    text = _ask_claude(msgs, CHAT_SYSTEM + "\n\n--- Dashboard data ---\n" + "\n\n".join(ctx))
    return {"text": text}


def _ask_claude(msgs, system):
    text = levels.news.ask_claude(messages=msgs, system=system, max_tokens=1500)
    if not text:
        raise HTTPException(502, f"Claude didn't answer: {levels.news.last_error or 'try again in a minute.'}")
    return text


@app.post("/api/chart/{t}/explain")
def explain_chart(t: str, tf: str = "1d"):
    t, tf = norm_ticker(t), _tf(tf)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(400, "Add ANTHROPIC_API_KEY to .env to have Claude explain charts.")
    text = levels.explain(t, _ticker_context(t, tf))
    if not text:
        raise HTTPException(502, f"Claude didn't answer: {levels.news.last_error or 'try again in a minute.'}")
    return {"text": text}


# ------------------------------------------------------------------ Schwab streaming (tick-by-tick quotes)
# One background thread runs an asyncio Schwab stream, subscribed to the tickers browsers have open right now.
# Browsers listen on GET /api/stream/{ticker} (Server-Sent Events); without Schwab they fall back to polling.
STREAM_MAX = 10
STREAM_FIELDS = ["SYMBOL", "LAST_PRICE", "TOTAL_VOLUME", "TRADE_TIME_MILLIS"]
hub = {"thread": None, "state": "off", "error": None, "want": {}, "ticks": {}, "idle_since": None}
hub_lock = threading.Lock()


def _stream_symbol(t: str) -> str:
    """Schwab streaming symbol: $SPX for indexes; the active contract (e.g. /ESZ26) for futures."""
    if not sources.is_future(t):
        return sources.schwab_symbol(t)
    sym = sources.schwab_symbol(t)
    try:
        q = sources.schwab_client().get_quote(sym).json()
        return next(iter(q.values()))["quote"].get("futureActiveSymbol") or sym
    except Exception:  # noqa: BLE001 - root symbol usually works too
        return sym


def stream_add(t: str) -> bool:
    """A browser opened t's stream; False = can't stream it (no Schwab / too many tickers)."""
    if not sources.schwab_status()["connected"] or sources.schwab_client() is None:
        return False
    with hub_lock:
        if t not in hub["want"] and len(hub["want"]) >= STREAM_MAX:
            return False
        hub["want"][t] = hub["want"].get(t, 0) + 1
        hub["idle_since"] = None
        if hub["thread"] is None or not hub["thread"].is_alive():
            hub["thread"] = threading.Thread(target=lambda: asyncio.run(_stream_main()), daemon=True)
            hub["thread"].start()
    return True


def stream_remove(t: str):
    with hub_lock:
        n = hub["want"].get(t, 0) - 1
        if n > 0:
            hub["want"][t] = n
        else:
            hub["want"].pop(t, None)
            if not hub["want"]:
                hub["idle_since"] = time.time()


def _sync_token(c, force=False):
    """Stream login sends token_metadata's access token, which can lag the one REST calls use; align them
    (force = refresh it first, after Schwab answered token_invalid)."""
    if force:
        c.session.refresh_token("https://api.schwabapi.com/v1/oauth/token", refresh_token=c.session.token["refresh_token"])
    if (c.session.token or {}).get("access_token"):
        c.token_metadata.token = c.session.token


async def _stream_main():
    """Keep one stream logged in; resubscribe when the wanted tickers change; reconnect with backoff."""
    from schwab.streaming import StreamClient
    backoff, bad_token = 2, False
    while True:
        with hub_lock:   # nobody watching for 2 min -> stop (restarts on the next viewer)
            if not hub["want"] and hub["idle_since"] and time.time() - hub["idle_since"] > 120:
                hub.update(thread=None, state="off")
                return
        c = sources.schwab_client()
        if c is None:
            hub.update(thread=None, state="off", error="Schwab not connected")
            return
        sc, subs = StreamClient(c), {"eq": {}, "fut": {}}   # kind -> {stream symbol: ticker}
        try:
            hub["state"] = "connecting"
            await asyncio.to_thread(c.get_user_preferences)   # refreshes an expired access token
            await asyncio.to_thread(_sync_token, c, bad_token)
            hub["attempts"] = hub.get("attempts", 0) + 1; hub["forced"] = hub.get("forced", 0) + bool(bad_token)
            await sc.login()
            hub.update(state="streaming", error=None)
            backoff, bad_token = 2, False

            def on_quote(msg):
                book = {**subs["eq"], **subs["fut"]}
                for q in msg.get("content", []):
                    t = book.get(q.get("key"))
                    if not t or "LAST_PRICE" not in q and "TOTAL_VOLUME" not in q:
                        continue
                    old = hub["ticks"].get(t, {})   # Schwab only sends fields that changed
                    hub["ticks"][t] = {"price": q.get("LAST_PRICE", old.get("price")),
                                       "volume": q.get("TOTAL_VOLUME", old.get("volume")),
                                       "ms": q.get("TRADE_TIME_MILLIS", old.get("ms")) or int(time.time() * 1000),
                                       "seq": old.get("seq", 0) + 1}
            sc.add_level_one_equity_handler(on_quote)
            sc.add_level_one_futures_handler(on_quote)
            while True:
                with hub_lock:
                    want = set(hub["want"])
                    if not want and hub["idle_since"] and time.time() - hub["idle_since"] > 120:
                        break
                for kind, subs_fn, unsubs_fn, fields in (
                        ("eq", sc.level_one_equity_subs, sc.level_one_equity_unsubs, sc.LevelOneEquityFields),
                        ("fut", sc.level_one_futures_subs, sc.level_one_futures_unsubs, sc.LevelOneFuturesFields)):
                    mine = {t for t in want if sources.is_future(t) == (kind == "fut")}
                    if mine == set(subs[kind].values()):
                        continue
                    new = {_stream_symbol(t): t for t in mine}
                    if new:   # SUBS replaces the whole list for that service
                        await subs_fn(list(new), fields=[getattr(fields, f) for f in STREAM_FIELDS])
                    elif subs[kind]:
                        await unsubs_fn(list(subs[kind]))
                    subs[kind] = new
                try:
                    await asyncio.wait_for(sc.handle_message(), timeout=1)
                except asyncio.TimeoutError:
                    pass
            await sc.logout()
        except Exception as e:  # noqa: BLE001 - disconnects, expired token…: retry with backoff
            hub.update(state="reconnecting", error=str(e)[:200])
            bad_token = "token_invalid" in str(e)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


def _tick_time(ms: int) -> int:
    """Trade time as New York wall-clock epoch seconds (same x axis as the chart)."""
    ts = pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert("America/New_York")
    return int(pd.Timestamp(ts.replace(tzinfo=None)).tz_localize("UTC").timestamp())


@app.get("/api/stream/{t}")
async def stream(t: str, request: Request):
    """Server-Sent Events: {price, volume (day total), time} on every trade; 204 = no stream, use polling."""
    t = norm_ticker(t)
    if not await asyncio.to_thread(stream_add, t):
        return Response(status_code=204)

    async def events():
        seq, beat = None, time.time()
        try:
            yield "retry: 5000\n\n"
            while not await request.is_disconnected():
                k = hub["ticks"].get(t)
                if k and k["seq"] != seq and k["price"] is not None:
                    seq = k["seq"]
                    when = pd.Timestamp(k["ms"], unit="ms", tz="UTC").tz_convert("America/New_York")
                    if sources.in_session(t, when):   # outside the chart's session (pre/post market) is skipped
                        yield f"data: {json.dumps({'price': k['price'], 'volume': k['volume'], 'time': _tick_time(k['ms'])})}\n\n"
                        beat = time.time()
                if time.time() - beat >= 15:
                    yield ": heartbeat\n\n"
                    beat = time.time()
                await asyncio.sleep(0.25)
        finally:
            stream_remove(t)
    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/stream-status")
def stream_status():
    rest = None
    if c := sources.schwab_client():   # is Schwab's REST API answering? (status code only)
        try:
            rest = c.get_quote("SPY").status_code
        except Exception as e:  # noqa: BLE001
            rest = type(e).__name__
    return {"state": hub["state"], "error": hub["error"], "tickers": sorted(hub["want"]), "rest_quote": rest,
            "attempts": hub.get("attempts", 0), "forced_refreshes": hub.get("forced", 0),
            "last_ticks": {t: k.get("ms") for t, k in hub["ticks"].items()}}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


KRONOS_PRECOMPUTE = ["SPY", "QQQ", "SPX"]
_precompute_lock = threading.Lock()


def _kronos_precompute():
    """After a scan: forecast today's top 5 + the main indexes on 1D, so those clicks are instant.
    Runs one at a time and waits whenever someone has clicked for a forecast, so it never makes them queue."""
    if not _kronos_ready() or not LATEST.exists() or not _precompute_lock.acquire(blocking=False):
        return
    try:
        top = [r["ticker"] for r in json.loads(LATEST.read_text())["rows"][:5]]
        for t in dict.fromkeys(top + KRONOS_PRECOMPUTE):
            while _user_waiting["n"] > 0:
                time.sleep(5)
            if _kronos_cached(t, "1d") or kronos_jobs.get((t, "1d"), {}).get("state") == "running":
                continue
            _precompute_one(t)
    finally:
        _precompute_lock.release()


def _precompute_one(t: str):
    try:
        kronos(t, "1d")
    except Exception:  # noqa: BLE001 - skip this one, keep going
        pass


@app.on_event("startup")
def startup():
    if sys.platform.startswith("linux") and not KRONOS_PY.exists():
        threading.Thread(target=_install_kronos, daemon=True).start()
    _load_details()
    stale = not LATEST.exists() or time.time() - LATEST.stat().st_mtime > 3600
    if stale:
        threading.Thread(target=run_scan, args=(load_settings()["universe"],), daemon=True).start()

