"""Market data sources: Schwab (thinkorswim) API when configured, Yahoo Finance otherwise.

Schwab needs SCHWAB_APP_KEY / SCHWAB_APP_SECRET (from developer.schwab.com) plus a
token created by logging in once (Watchlist tab -> "Connect Schwab").
Schwab refresh tokens expire after 7 days.
"""
import json
import os
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from radar.universe import FUTURES, INDEX_SYMBOLS

DATA = Path(os.environ.get("RADAR_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
TOKEN_PATH = DATA / "schwab_token.json"
CALLBACK_URL = os.environ.get("SCHWAB_CALLBACK_URL", "https://options-radar-16f492.azurewebsites.net/")
MAX_DTE = 60

_client = None
_client_lock = threading.Lock()
_pending_auth = {}


# ---------------------------------------------------------------- Schwab auth
def schwab_configured() -> bool:
    return bool(os.environ.get("SCHWAB_APP_KEY") and os.environ.get("SCHWAB_APP_SECRET"))


def _write_token(token, *args, **kwargs):
    TOKEN_PATH.write_text(json.dumps(token))
    os.chmod(TOKEN_PATH, 0o600)


def schwab_client():
    """Cached client from the saved token, or None if not connected / expired."""
    global _client
    if not schwab_configured() or not TOKEN_PATH.exists():
        return None
    with _client_lock:
        if _client is None:
            from schwab import auth
            try:
                _client = auth.client_from_token_file(
                    str(TOKEN_PATH), os.environ["SCHWAB_APP_KEY"], os.environ["SCHWAB_APP_SECRET"])
            except Exception:
                _client = None
        return _client


def schwab_status() -> dict:
    out = {"configured": schwab_configured(), "connected": False, "expires": None, "source": "yahoo"}
    if TOKEN_PATH.exists():
        try:
            saved = json.loads(TOKEN_PATH.read_text())
            created, tok = saved.get("creation_timestamp"), saved.get("token") or {}
            if created:
                exp = datetime.fromtimestamp(created) + timedelta(days=7)
                out["expires"] = exp.isoformat(timespec="minutes")
                # usable = can refresh, or the access token itself is still valid
                usable = bool(tok.get("refresh_token")) or tok.get("expires_at", 0) > time.time()
                out["connected"] = schwab_configured() and exp > datetime.now() and usable
        except Exception:
            pass
    if out["connected"]:
        out["source"] = "schwab"
    return out


def schwab_login_url() -> str:
    """Step 1 of the browser login: the Schwab URL to open."""
    from schwab import auth
    ctx = auth.get_auth_context(os.environ["SCHWAB_APP_KEY"], CALLBACK_URL)
    _pending_auth["ctx"] = ctx
    return ctx.authorization_url


def schwab_finish_login(received_url: str):
    """Step 2: the URL Schwab redirected to (starts with the callback URL, contains ?code=)."""
    global _client
    from schwab import auth
    ctx = _pending_auth.get("ctx")
    if ctx is None:
        raise ValueError("Start the login again — the session expired.")
    client = auth.client_from_received_url(
        os.environ["SCHWAB_APP_KEY"], os.environ["SCHWAB_APP_SECRET"], ctx, received_url.strip(), _write_token)
    with _client_lock:
        _client = client
    _pending_auth.clear()


def yahoo_symbol(ticker: str) -> str:
    return {**INDEX_SYMBOLS, **FUTURES}.get(ticker, {}).get("yahoo", ticker)


def schwab_symbol(ticker: str) -> str:
    return {**INDEX_SYMBOLS, **FUTURES}.get(ticker, {}).get("schwab", ticker)


def is_future(ticker: str) -> bool:
    return ticker in FUTURES


def in_session(ticker: str, ts=None) -> bool:
    """Is the market open at ts (New York time; default now)? Stocks 9:30–16:00 weekdays (holidays ignored);
    futures Sun 18:00 – Fri 17:00 with a daily 17:00–18:00 break."""
    ts = pd.Timestamp.now(tz="America/New_York") if ts is None else ts
    wd, m = ts.weekday(), ts.hour * 60 + ts.minute
    if is_future(ticker):
        return not (wd == 5 or (wd == 4 and m >= 17 * 60) or (wd == 6 and m < 18 * 60) or 17 * 60 <= m < 18 * 60)
    return wd < 5 and 9 * 60 + 30 <= m < 16 * 60


# ---------------------------------------------------------------- price history
def history(ticker: str) -> pd.DataFrame:
    """~2 years of daily OHLCV. Schwab first, Yahoo fallback (Yahoo only for futures)."""
    c = None if is_future(ticker) else schwab_client()
    if c is not None:
        try:
            r = c.get_price_history_every_day(
                schwab_symbol(ticker), start_datetime=datetime.now() - timedelta(days=730), end_datetime=datetime.now())
            r.raise_for_status()
            candles = r.json().get("candles", [])
            if len(candles) > 60:
                df = pd.DataFrame(candles)
                df.index = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_convert("America/New_York")
                df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            pass
    return yf.Ticker(yahoo_symbol(ticker)).history(period="2y", auto_adjust=False)


# timeframe -> (Schwab method, days of history to ask for, Yahoo interval, Yahoo period, resample rule)
INTRADAY = {
    "1m": ("get_price_history_every_minute", 5, "1m", "5d", None),
    "5m": ("get_price_history_every_five_minutes", 20, "5m", "30d", None),
    "15m": ("get_price_history_every_fifteen_minutes", 45, "15m", "60d", None),
    "30m": ("get_price_history_every_thirty_minutes", 90, "30m", "60d", None),
    "1h": ("get_price_history_every_thirty_minutes", 180, "60m", "180d", "60min"),
}


def intraday_history(ticker: str, tf: str) -> pd.DataFrame:
    """Intraday OHLCV (New York time) for a timeframe in INTRADAY: regular hours for stocks, the full ~23h session
    for futures. Schwab first, Yahoo fallback (Yahoo only for futures)."""
    method, days, y_int, y_per, rule = INTRADAY[tf]
    df = None
    c = None if is_future(ticker) else schwab_client()
    if c is not None:
        try:
            r = getattr(c, method)(schwab_symbol(ticker), start_datetime=datetime.now() - timedelta(days=days),
                                   end_datetime=datetime.now(), need_extended_hours_data=False)
            r.raise_for_status()
            candles = r.json().get("candles", [])
            if len(candles) > 60:
                df = pd.DataFrame(candles)
                df.index = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_convert("America/New_York")
                df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
                df = df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            df = None
    if df is None:
        df = yf.Ticker(yahoo_symbol(ticker)).history(period=y_per, interval=y_int, auto_adjust=False)
        df.index = df.index.tz_convert("America/New_York")
        rule = None  # Yahoo serves 60m bars directly
    if rule:  # hourly bars anchored on the 9:30 open, like TradingView
        df = df.resample(rule, offset="30min").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna()
    return df


# ---------------------------------------------------------------- option chains
CHAIN_COLUMNS = ["contractSymbol", "strike", "bid", "ask", "lastPrice", "volume", "openInterest",
                 "impliedVolatility", "type", "expiry", "dte"]


def chains(ticker: str, spot: float) -> tuple[pd.DataFrame, str]:
    """Raw option rows (calls+puts) for near expirations, plus which source served them."""
    c = schwab_client()
    if c is not None:
        try:
            df = _schwab_chains(c, ticker, spot)
            if not df.empty:
                return df, "schwab"
        except Exception:
            pass
    return _yahoo_chains(ticker), "yahoo"


def _pick_expiries(dated):
    """First few expirations (near-term gamma/flow) plus the ones closest to 21/30/45 DTE."""
    pick = {e for e, _ in dated[:5]}
    for target in (21, 30, 45):
        if dated:
            pick.add(min(dated, key=lambda x: abs(x[1] - target))[0])
    return pick


def _yahoo_chains(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(yahoo_symbol(ticker))
    today = date.today()
    dated = []
    for exp in tk.options:
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - today).days
        if 0 <= dte <= MAX_DTE:
            dated.append((exp, dte))
    pick = _pick_expiries(dated)
    frames = []
    for exp, dte in dated:
        if exp not in pick:
            continue
        try:
            ch = tk.option_chain(exp)
        except Exception:
            continue
        for kind, df in (("C", ch.calls), ("P", ch.puts)):
            if df is None or df.empty:
                continue
            d = df.copy()
            d["type"], d["expiry"], d["dte"] = kind, exp, dte
            frames.append(d[CHAIN_COLUMNS])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CHAIN_COLUMNS)


def _schwab_chains(c, ticker: str, spot: float) -> pd.DataFrame:
    today = date.today()
    sym = schwab_symbol(ticker)
    if ticker in INDEX_SYMBOLS:
        # Cash indexes (SPX…) have daily expiries and thousands of strikes — a 60-day request is too big
        # (Schwab answers 502), so fetch only the expiries we use, one at a time, nearest the money
        r = c.get_option_expiration_chain(sym)
        r.raise_for_status()
        dated = [(e["expirationDate"], int(e["daysToExpiration"])) for e in r.json().get("expirationList", [])
                 if 0 <= int(e["daysToExpiration"]) <= MAX_DTE]
        windows = [(date.fromisoformat(e), date.fromisoformat(e)) for e in sorted(_pick_expiries(dated))]
        strikes = {"strike_count": 300}
    else:
        windows = [(today, today + timedelta(days=MAX_DTE))]
        strikes = {"strike_range": c.Options.StrikeRange.ALL}
    rows = []
    for start, end in windows:
        r = c.get_option_chain(sym, from_date=start, to_date=end, **strikes)
        r.raise_for_status()
        rows += _schwab_rows(r.json(), today)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    dated = sorted({(e, d) for e, d in zip(df.expiry, df.dte)}, key=lambda x: x[1])
    return df[df.expiry.isin(_pick_expiries(dated))].reset_index(drop=True)


def _schwab_rows(js: dict, today: date) -> list[dict]:
    rows = []
    for kind, key in (("C", "callExpDateMap"), ("P", "putExpDateMap")):
        for exp_key, strikes in (js.get(key) or {}).items():
            exp = exp_key.split(":")[0]
            dte = int(exp_key.split(":")[1]) if ":" in exp_key else (date.fromisoformat(exp) - today).days
            for contracts in strikes.values():
                for o in contracts:
                    iv = o.get("volatility")
                    rows.append({
                        "contractSymbol": o.get("symbol", "").replace(" ", ""),
                        "strike": float(o.get("strikePrice", 0)),
                        "bid": float(o.get("bid") or 0), "ask": float(o.get("ask") or 0),
                        "lastPrice": float(o.get("last") or 0),
                        "volume": float(o.get("totalVolume") or 0),
                        "openInterest": float(o.get("openInterest") or 0),
                        # Schwab reports IV in percent and uses -999 / NaN for "no value"
                        "impliedVolatility": float(iv) / 100 if iv not in (None, "NaN") and float(iv) > 0 else 0.0,
                        "type": kind, "expiry": exp, "dte": dte,
                        "schwab_delta": _num(o.get("delta")),
                    })
    return rows


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) and f > -900 else np.nan
    except (TypeError, ValueError):
        return np.nan
