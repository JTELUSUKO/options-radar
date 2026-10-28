"""Paper trading: try the radar's ideas with fake money. Never sends broker orders.

The account lives in DATA/paper.json. Fills are realistic: opening BUY legs pay the ask, SELL legs get the
bid (and the reverse when closing), falling back to the mid when a side is 0. Open positions are marked at
the mid from sources.chains (Schwab when connected, else Yahoo), cached ~5 min per ticker.

Exit rules (same as the trade plan in scoring._plan), applied when "auto_close" is on:
  credit trades: take profit at +50% of the credit; stop when the spread costs 2x the credit to buy back
  buy call/put, straddle: take profit at +100%; stop at -50% of what you paid
  debit spreads: take profit at +60% of max profit; stop at -50% of what you paid
  every trade: closed on expiry day (or settled at intrinsic value once expired)
"""
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from radar import sources

PAPER = sources.DATA / "paper.json"
START_CASH = 10_000.0
CACHE_SECONDS = 300
NY = ZoneInfo("America/New_York")
DEFAULT_SETTINGS = {"auto_trade": False, "top_n": 3, "auto_close": True}

_lock = threading.RLock()
_quotes: dict = {}        # ticker -> (fetched at, chain DataFrame, spot)


# ------------------------------------------------------------------ storage
def _new_account(cash: float = START_CASH) -> dict:
    return {"start_cash": cash, "cash": cash, "created": _now(), "positions": [], "closed": [],
            "equity_history": [{"date": _ny_today().isoformat(), "equity": cash}],
            "settings": dict(DEFAULT_SETTINGS), "last_auto_date": None, "log": []}


def load() -> dict:
    with _lock:
        if not PAPER.exists():
            return _new_account()
        acct = json.loads(PAPER.read_text())
        acct["settings"] = {**DEFAULT_SETTINGS, **acct.get("settings", {})}
        return acct


def save(acct: dict):
    with _lock:
        tmp = PAPER.with_suffix(".tmp")
        tmp.write_text(json.dumps(acct, indent=1))
        tmp.replace(PAPER)


def _now() -> str:
    return datetime.now(NY).isoformat(timespec="seconds")


def _ny_today() -> date:
    return datetime.now(NY).date()


def _log(acct: dict, msg: str):
    acct["log"] = (acct.get("log", []) + [{"time": _now(), "msg": msg}])[-50:]


# ------------------------------------------------------------------ quotes
def seed_quotes(ticker: str, chain: pd.DataFrame, spot: float):
    """Reuse a scan's freshly downloaded chain so marking right after a scan needs no extra download."""
    if chain is not None and not chain.empty:
        _quotes[ticker] = (time.time(), chain, spot)


def _spot(ticker: str, on_or_before: str | None = None) -> float | None:
    """Last underlying close (or the close on/before a date, for settling expired options)."""
    try:
        h = yf.Ticker(sources.yahoo_symbol(ticker)).history(period="3mo", auto_adjust=False)["Close"].dropna()
        if on_or_before:
            h = h[h.index.date <= date.fromisoformat(on_or_before)]
        return float(h.iloc[-1]) if len(h) else None
    except Exception:  # noqa: BLE001 - no price = keep the last mark
        return None


def _chain(ticker: str, fallback_spot: float | None) -> tuple[pd.DataFrame | None, float | None]:
    hit = _quotes.get(ticker)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1], hit[2]
    spot = _spot(ticker) or fallback_spot
    try:
        df, _ = sources.chains(ticker, spot or 0)
    except Exception:  # noqa: BLE001
        df = None
    _quotes[ticker] = (time.time(), df, spot)
    return df, spot


def _expiry_rows(ticker: str, expiry: str) -> pd.DataFrame | None:
    """One expiry straight from Yahoo — for legs whose expiry the near-term chain no longer includes."""
    key = f"{ticker}|{expiry}"
    hit = _quotes.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    df = None
    try:
        ch = yf.Ticker(sources.yahoo_symbol(ticker)).option_chain(expiry)
        df = pd.concat([ch.calls.assign(type="C"), ch.puts.assign(type="P")], ignore_index=True)
        df["expiry"] = expiry
    except Exception:  # noqa: BLE001
        pass
    _quotes[key] = (time.time(), df, None)
    return df


def _find(df: pd.DataFrame | None, leg: dict):
    if df is None or df.empty:
        return None
    m = df[(df["type"] == leg["type"][0].upper()) & (df["expiry"] == leg["expiry"])
           & ((df["strike"].astype(float) - float(leg["strike"])).abs() < 1e-6)]
    return None if m.empty else m.iloc[0]


def _quote(ticker: str, leg: dict, chain: pd.DataFrame | None) -> dict | None:
    """{bid, ask} for a leg, or None when nobody is quoting it (e.g. weekend / no data)."""
    row = _find(chain, leg)
    if row is None:
        row = _find(_expiry_rows(ticker, leg["expiry"]), leg)
    if row is None:
        return None
    bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
    if bid != bid or ask != ask:   # NaN
        return None
    return {"bid": bid, "ask": ask} if bid > 0 or ask > 0 else None


def _mid(q: dict) -> float:
    return (q["bid"] + q["ask"]) / 2 if q["bid"] > 0 and q["ask"] > 0 else max(q["bid"], q["ask"])


def _fill(q: dict, buying: bool) -> float:
    """Realistic fill: buy at the ask, sell at the bid; the mid if that side is 0."""
    px = q["ask"] if buying else q["bid"]
    return px if px > 0 else _mid(q)


# ------------------------------------------------------------------ maths
def _sign(leg) -> int:
    return 1 if leg["action"] == "BUY" else -1


def _intrinsic(leg, spot: float) -> float:
    return max(spot - leg["strike"], 0.0) if leg["type"] == "call" else max(leg["strike"] - spot, 0.0)


def _payoff_range(legs, net_debit: float) -> tuple[float, float | None]:
    """Max loss and max profit at expiry per 1 contract (x100); max profit None = unlimited."""
    ks = sorted({l["strike"] for l in legs})
    pts = [0.0] + ks + [ks[-1] * 10]
    pay = [100 * (sum(_sign(l) * _intrinsic(l, s) for l in legs) - net_debit) for s in pts]
    far = 100 * (sum(_sign(l) * _intrinsic(l, ks[-1] * 20) for l in legs) - net_debit)
    return max(-min(pay), 0.0), (None if far > pay[-1] + 1e-6 else max(pay))


def _value(p: dict, prices: list[float]) -> float:
    """Market value of the position at these per-leg prices (long legs +, short legs −)."""
    return sum(_sign(l) * px for l, px in zip(p["legs"], prices)) * 100 * p["contracts"]


def _targets(p: dict) -> tuple[float, float]:
    """(take-profit P&L $, stop P&L $) from the trade-plan rules."""
    basis = abs(p["entry_cost"])
    if p["credit"]:
        return 0.5 * basis, -basis                       # buy back at half the credit / spread price doubles
    if p["strategy"] == "debit_spread" and p.get("max_profit"):
        return 0.6 * p["max_profit"], -0.5 * basis
    return basis, -0.5 * basis                           # +100% / −50%


def _expired(expiry: str) -> bool:
    now = datetime.now(NY)
    return now.date() > date.fromisoformat(expiry) or (
        now.date() == date.fromisoformat(expiry) and now.hour >= 16)


# ------------------------------------------------------------------ actions
def open_idea(row: dict, contracts: int = 1, fresh: bool = True, source: str = "manual") -> dict:
    """Open a scan row's idea. fresh=True re-quotes the legs now; otherwise uses the scan's bid/ask."""
    i, t = row["idea"], row["ticker"]
    contracts = max(1, int(contracts))
    chain, spot = _chain(t, row.get("price")) if fresh else (None, row.get("price"))
    legs = []
    for l in i["legs"]:
        q = (_quote(t, l, chain) if fresh else None) or (
            {"bid": l.get("bid") or 0, "ask": l.get("ask") or 0} if (l.get("bid") or l.get("ask")) else None)
        if q is None:
            raise ValueError(f"No live quote for the {l['strike']:g} {l['type']} — markets may be closed; try during market hours.")
        px = _fill(q, l["action"] == "BUY")
        if px <= 0:
            raise ValueError(f"No price for the {l['strike']:g} {l['type']}.")
        legs.append({k: l[k] for k in ("action", "type", "strike", "expiry")} | {
            "contract": l.get("contract"), "entry": round(px, 2), "entry_bid": q["bid"], "entry_ask": q["ask"],
            "mark": round(_mid(q), 2), "mark_time": _now()})
    net = sum(_sign(l) * l["entry"] for l in legs)       # + = debit paid, − = credit received (per share)
    max_loss, max_profit = _payoff_range(legs, net)
    pos = {"id": uuid.uuid4().hex[:8], "ticker": t, "label": i["label"], "strategy": i["strategy"],
           "bias": i["bias"], "score": i["score"], "expiry": i["expiry"], "credit": net < 0,
           "contracts": contracts, "opened": _now(), "entry_spot": spot or row.get("price"), "legs": legs,
           "entry_cost": round(net * 100 * contracts, 2), "max_loss": round(max_loss * contracts, 2),
           "max_profit": None if max_profit is None else round(max_profit * contracts, 2),
           "source": source, "plan": {k: i.get("plan", {}).get(k) for k in ("take_profit", "stop", "timing")}}
    with _lock:
        acct = load()
        avail = _buying_power(acct)
        if pos["max_loss"] > avail:
            raise ValueError(f"Max loss ${pos['max_loss']:,.0f} is more than your available cash ${avail:,.0f}.")
        acct["cash"] = round(acct["cash"] - pos["entry_cost"], 2)
        acct["positions"].append(pos)
        _log(acct, f"Opened {contracts}x {t} {i['label']} ({source})")
        save(acct)
    return pos


def _close_prices(p: dict, chain, spot) -> tuple[list[float], str] | None:
    """Exit price per leg: intrinsic once expired, else realistic fills (sell longs at bid, buy shorts at ask)."""
    if _expired(p["expiry"]):
        s = _spot(p["ticker"], p["expiry"]) or spot
        return ([_intrinsic(l, s) for l in p["legs"]], f"expired (stock ${s:,.2f})") if s else None
    out = []
    for l in p["legs"]:
        q = _quote(p["ticker"], l, chain)
        out.append(_fill(q, l["action"] == "SELL") if q else l["mark"])   # no quote: last mark
    return out, "closed"


def close(pid: str, reason: str | None = None) -> dict:
    with _lock:
        acct = load()
        p = next((x for x in acct["positions"] if x["id"] == pid), None)
        if p is None:
            raise KeyError("No open position with that id.")
        chain, spot = _chain(p["ticker"], p.get("entry_spot"))
        res = _close_prices(p, chain, spot)
        if res is None:
            raise ValueError("Couldn't get a price to close at — try again later.")
        _settle(acct, p, *(res if reason is None else (res[0], reason)))
        save(acct)
        return acct["closed"][-1]


def _settle(acct: dict, p: dict, prices: list[float], reason: str):
    exit_value = _value(p, prices)
    pnl = exit_value - p["entry_cost"]
    for l, px in zip(p["legs"], prices):
        l["exit"] = round(px, 2)
    p.update(closed=_now(), exit_value=round(exit_value, 2), pnl=round(pnl, 2),
             pnl_pct=pnl / abs(p["entry_cost"]) if p["entry_cost"] else None, reason=reason)
    acct["cash"] = round(acct["cash"] + exit_value, 2)
    acct["positions"] = [x for x in acct["positions"] if x["id"] != p["id"]]
    acct["closed"].append(p)
    _log(acct, f"Closed {p['ticker']} {p['label']}: {reason}, P&L {'+' if pnl >= 0 else '−'}${abs(pnl):,.0f}")


def reset(cash: float = START_CASH) -> dict:
    with _lock:
        settings = load()["settings"]
        acct = _new_account(float(cash))
        acct["settings"] = settings
        save(acct)
        return acct


def update_settings(new: dict) -> dict:
    with _lock:
        acct = load()
        s = acct["settings"]
        if "auto_trade" in new:
            s["auto_trade"] = bool(new["auto_trade"])
        if "auto_close" in new:
            s["auto_close"] = bool(new["auto_close"])
        if "top_n" in new:
            s["top_n"] = max(1, min(5, int(new["top_n"])))
        save(acct)
        return s


# ------------------------------------------------------------------ marking
def _buying_power(acct: dict) -> float:
    """Cash minus collateral held for credit trades (their max loss)."""
    return acct["cash"] - sum(p["max_loss"] for p in acct["positions"] if p["credit"])


def _mark_position(p: dict) -> tuple[list[float], str] | None:
    """Re-price each leg at the mid; keep the last mark when a leg has no quote. Returns an exit if a rule hit."""
    chain, spot = _chain(p["ticker"], p.get("entry_spot"))
    p["spot"] = spot
    p.pop("error", None)
    if _expired(p["expiry"]):
        return _close_prices(p, chain, spot)
    fresh = True
    for l in p["legs"]:
        q = _quote(p["ticker"], l, chain)
        if q and _mid(q) > 0:
            l["mark"], l["mark_time"] = round(_mid(q), 2), _now()
        else:
            fresh = False
    p["stale"] = not fresh
    p["value"] = round(_value(p, [l["mark"] for l in p["legs"]]), 2)
    p["pnl"] = round(p["value"] - p["entry_cost"], 2)
    p["pnl_pct"] = p["pnl"] / abs(p["entry_cost"]) if p["entry_cost"] else None
    take, stop = _targets(p)
    p["take_profit_at"], p["stop_at"] = round(take, 2), round(stop, 2)
    if not fresh:      # never trigger exits on missing quotes
        return None
    if p["pnl"] >= take:
        return _close_prices(p, chain, spot)[0], "take profit hit"
    if p["pnl"] <= stop:
        return _close_prices(p, chain, spot)[0], "stop hit"
    if _ny_today().isoformat() == p["expiry"]:
        return _close_prices(p, chain, spot)[0], "expiry day"
    return None


def mark_all() -> dict:
    """Mark every open position, apply exit rules, snapshot today's equity. Returns the account."""
    acct = load()
    if acct["positions"]:
        with ThreadPoolExecutor(max_workers=4) as ex:      # download chains in parallel
            list(ex.map(lambda t: _chain(t, None), {p["ticker"] for p in acct["positions"]}))
    with _lock:
        acct = load()
        for p in list(acct["positions"]):
            try:
                hit = _mark_position(p)
            except Exception as e:  # noqa: BLE001 - one bad quote shouldn't break the page
                p["stale"], p["error"] = True, str(e)
                continue
            if hit and (acct["settings"]["auto_close"] or _expired(p["expiry"])):
                _settle(acct, p, *hit)
        _snapshot(acct)
        save(acct)
    return acct


def _snapshot(acct: dict):
    eq = round(acct["cash"] + sum(_value(p, [l["mark"] for l in p["legs"]]) for p in acct["positions"]), 2)
    hist = [h for h in acct.get("equity_history", []) if h["date"] != _ny_today().isoformat()]
    acct["equity_history"] = (hist + [{"date": _ny_today().isoformat(), "equity": eq}])[-400:]


def summary(acct: dict) -> dict:
    open_value = sum(_value(p, [l["mark"] for l in p["legs"]]) for p in acct["positions"])
    open_pnl = open_value - sum(p["entry_cost"] for p in acct["positions"])
    realized = sum(c["pnl"] for c in acct["closed"])
    wins = sum(1 for c in acct["closed"] if c["pnl"] > 0)
    equity = acct["cash"] + open_value
    return {"start_cash": acct["start_cash"], "cash": round(acct["cash"], 2),
            "buying_power": round(_buying_power(acct), 2), "open_value": round(open_value, 2),
            "open_pnl": round(open_pnl, 2), "realized_pnl": round(realized, 2), "equity": round(equity, 2),
            "total_return": equity / acct["start_cash"] - 1 if acct["start_cash"] else None,
            "win_rate": wins / len(acct["closed"]) if acct["closed"] else None, "wins": wins,
            "closed_count": len(acct["closed"]), "open_count": len(acct["positions"])}


# ------------------------------------------------------------------ auto mode (called from run_scan)
def after_scan(rows: list[dict]):
    """Exit rules on open positions, then (auto mode) open the top N ideas once per scan day."""
    acct = mark_all()
    s, today = acct["settings"], _ny_today().isoformat()
    if not s["auto_trade"] or acct.get("last_auto_date") == today:
        return
    held = {p["ticker"] for p in acct["positions"]}
    for r in [r for r in rows if r.get("idea")][: s["top_n"]]:
        if r["ticker"] in held:
            continue
        try:
            open_idea(r, 1, fresh=False, source="auto")     # the scan's quotes are seconds old
        except Exception as e:  # noqa: BLE001 - e.g. not enough cash; note it and move on
            acct2 = load(); _log(acct2, f"Auto-trade skipped {r['ticker']}: {e}"); save(acct2)
    with _lock:
        acct = load()
        acct["last_auto_date"] = today
        save(acct)
