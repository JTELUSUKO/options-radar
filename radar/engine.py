"""Per-ticker analytics: dealer gamma, options flow, volatility, trend, news.

All market data comes from Yahoo Finance via yfinance (free, ~15 min delayed,
open interest updated once per day, no buyer/seller side on trades).
"""
import math
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

from . import news, sources

RISK_FREE = 0.04
STRIKE_BAND = 0.25        # keep strikes within +/-25% of spot


# ---------------------------------------------------------------- Black-Scholes
def bs_d1(S, K, T, sigma, r=RISK_FREE):
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_gamma(S, K, T, sigma):
    d1 = bs_d1(S, K, T, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_delta(S, K, T, sigma, is_call):
    d1 = bs_d1(S, K, T, sigma)
    return np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)


# ---------------------------------------------------------------- helpers
def _mid(row):
    b, a = row.get("bid", 0) or 0, row.get("ask", 0) or 0
    if b > 0 and a > 0:
        return (b + a) / 2
    return float(row.get("lastPrice", 0) or 0)


def third_friday(year, month):
    d = date(year, month, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def next_monthly_opex(today=None):
    today = today or date.today()
    tf = third_friday(today.year, today.month)
    if tf < today:
        m = today.month % 12 + 1
        y = today.year + (1 if m == 1 else 0)
        tf = third_friday(y, m)
    return tf


def load_chains(ticker: str, spot: float):
    """Return a tidy DataFrame of calls+puts for near expirations, and its data source."""
    df, source = sources.chains(ticker, spot)
    if df.empty:
        return df, source
    df = df[(df["strike"] > spot * (1 - STRIKE_BAND)) & (df["strike"] < spot * (1 + STRIKE_BAND))]
    for c in ("volume", "openInterest", "bid", "ask", "lastPrice", "impliedVolatility"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df = df[df.impliedVolatility > 0.005]  # no usable IV (illiquid / bad quote)
    df["mid"] = np.where((df.bid > 0) & (df.ask > 0), (df.bid + df.ask) / 2, df.lastPrice)
    # Time to expiry in years; same-day options get a few hours
    df["T"] = np.maximum(df["dte"], 0.25) / 365.0
    df["iv"] = df["impliedVolatility"].clip(lower=0.01, upper=5.0)
    df["is_call"] = df["type"] == "C"
    df["delta"] = bs_delta(spot, df.strike.values, df["T"].values, df.iv.values, df.is_call.values)
    if "schwab_delta" in df:  # prefer the broker's own greeks when available
        df["delta"] = df["schwab_delta"].fillna(df["delta"])
    return df, source


# ---------------------------------------------------------------- signals
def gamma_profile(df: pd.DataFrame, spot: float):
    """Dealer gamma exposure (SqueezeMetrics convention: dealers long calls, short puts).

    GEX per strike = gamma * OI * 100 * S^2 * 0.01  (dollars per 1% move), puts negative.
    The sign convention is an assumption -- real dealer positioning is unknown.
    """
    g = df[(df.openInterest > 0) & (df.iv > 0.02)].copy()
    if g.empty:
        return None
    sign = np.where(g.is_call, 1.0, -1.0)

    def total_gex(S):
        gam = bs_gamma(S, g.strike.values, g["T"].values, g.iv.values)
        return float(np.sum(sign * gam * g.openInterest.values * 100 * S * S * 0.01))

    gam_now = bs_gamma(spot, g.strike.values, g["T"].values, g.iv.values)
    g["gex"] = sign * gam_now * g.openInterest.values * 100 * spot * spot * 0.01
    by_strike = g.groupby("strike")["gex"].sum()
    calls = g[g.is_call].groupby("strike")["gex"].sum()
    puts = g[~g.is_call].groupby("strike")["gex"].sum()

    net = float(by_strike.sum())
    call_wall = float(calls.idxmax()) if not calls.empty else None
    put_wall = float(puts.idxmin()) if not puts.empty else None

    # Zero-gamma flip: scan hypothetical spot levels and find the sign change nearest spot
    grid = np.linspace(spot * 0.85, spot * 1.15, 61)
    vals = np.array([total_gex(s) for s in grid])
    flip = None
    crossings = np.where(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    if len(crossings):
        best = min(crossings, key=lambda i: abs(grid[i] - spot))
        x0, x1, y0, y1 = grid[best], grid[best + 1], vals[best], vals[best + 1]
        flip = float(x0 - y0 * (x1 - x0) / (y1 - y0))

    near = by_strike[(by_strike.index > spot * 0.9) & (by_strike.index < spot * 1.1)]
    return {
        "net_gex": net,
        "regime": "positive" if net > 0 else "negative",
        "call_wall": call_wall,
        "put_wall": put_wall,
        "zero_gamma": flip,
        "profile": [{"strike": float(k), "gex": float(v)} for k, v in near.items()],
    }


def flow_signals(df: pd.DataFrame, spot: float, stock_volume: float):
    """Unusual activity from the latest session's volume vs open interest.

    Evidence: unusually large OTM, short-dated call buying before M&A (Augustin et al.);
    only calls that are unusual relative to the stock's normal option volume predict
    returns (Jiang & Strong). Yahoo gives no trade side, so direction is inferred.
    """
    d = df.copy()
    d["premium"] = d.volume * d.mid * 100
    call_prem = float(d[d.is_call].premium.sum())
    put_prem = float(d[~d.is_call].premium.sum())
    tot_vol = float(d.volume.sum())
    tot_oi = float(d.openInterest.sum())
    call_vol = float(d[d.is_call].volume.sum())
    put_vol = float(d[~d.is_call].volume.sum())

    otm_short_calls = d[d.is_call & (d.strike > spot * 1.02) & (d.dte <= 30)]
    otm_short_puts = d[~d.is_call & (d.strike < spot * 0.98) & (d.dte <= 30)]
    otm_call_share = float(otm_short_calls.volume.sum() / tot_vol) if tot_vol else 0.0
    otm_put_share = float(otm_short_puts.volume.sum() / tot_vol) if tot_vol else 0.0

    unusual = d[(d.volume > d.openInterest.clip(lower=1) * 1.2) & (d.volume >= 250) & (d.premium >= 50_000)]
    unusual = unusual.sort_values("premium", ascending=False)
    u_call_prem = float(unusual[unusual.is_call].premium.sum())
    u_put_prem = float(unusual[~unusual.is_call].premium.sum())

    trades = []
    for _, r in unusual.head(12).iterrows():
        side = "ask?" if r.lastPrice >= r.mid else "bid?"
        trades.append({
            "contract": r.contractSymbol, "type": r.type, "strike": float(r.strike),
            "expiry": r.expiry, "dte": int(r.dte), "volume": int(r.volume),
            "oi": int(r.openInterest), "premium": float(r.premium), "iv": float(r.iv),
            "vol_oi": float(r.volume / max(r.openInterest, 1)), "side_guess": side,
        })
    return {
        "call_premium": call_prem, "put_premium": put_prem,
        "pc_volume_ratio": put_vol / call_vol if call_vol else None,
        "vol_oi_ratio": tot_vol / tot_oi if tot_oi else 0.0,
        "otm_short_call_share": otm_call_share,
        "otm_short_put_share": otm_put_share,
        "unusual_call_premium": u_call_prem,
        "unusual_put_premium": u_put_prem,
        "unusual_count": int(len(unusual)),
        "option_stock_ratio": (tot_vol * 100) / stock_volume if stock_volume else None,
        "unusual_trades": trades,
    }


def vol_signals(df: pd.DataFrame, spot: float, closes: pd.Series):
    """ATM IV ~30d, realized vol, IV/RV (volatility risk premium), skew, put-call IV spread."""
    rets = np.log(closes).diff().dropna()
    rv20 = float(rets.tail(20).std() * math.sqrt(252))
    rv60 = float(rets.tail(60).std() * math.sqrt(252))

    exps = df[df.dte >= 14].groupby("expiry")["dte"].first()
    if exps.empty:
        exps = df.groupby("expiry")["dte"].first()
    target = exps.index[np.argmin(np.abs(exps.values - 30))]
    e = df[(df.expiry == target) & (df.bid > 0)]
    calls, puts = e[e.is_call], e[~e.is_call]
    if calls.empty or puts.empty:
        return None
    atm_k = float(e.strike.iloc[np.argmin(np.abs(e.strike.values - spot))])
    c_atm = calls.iloc[np.argmin(np.abs(calls.strike.values - spot))]
    p_atm = puts.iloc[np.argmin(np.abs(puts.strike.values - spot))]
    atm_iv = float((c_atm.iv + p_atm.iv) / 2)

    # Cremers-Weinbaum: call IV minus put IV at matched strikes near the money (bullish if > 0)
    m = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    m = m[(m.strike > spot * 0.95) & (m.strike < spot * 1.05)]
    cw = float((m.iv_c - m.iv_p).mean()) if not m.empty else 0.0

    # Xing-Zhang-Zhao skew: OTM put IV (~0.95 moneyness) minus ATM call IV (steep = bearish)
    otm_p = puts.iloc[np.argmin(np.abs(puts.strike.values / spot - 0.95))]
    skew = float(otm_p.iv - c_atm.iv)

    dte = int(e.dte.iloc[0])
    straddle = float(_mid(c_atm) + _mid(p_atm))
    return {
        "atm_iv": atm_iv, "rv20": rv20, "rv60": rv60,
        "iv_rv": atm_iv / (0.5 * rv20 + 0.5 * rv60) if rv20 > 0 else None,
        "cw_spread": cw, "skew": skew,
        "ref_expiry": target, "ref_dte": dte, "atm_strike": atm_k,
        "implied_move_pct": straddle / spot if spot else None,
        "atm_spread_pct": float((c_atm.ask - c_atm.bid) / max(_mid(c_atm), 0.01)),
    }


def trend_signals(hist: pd.DataFrame):
    c = hist["Close"]
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    dn = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    tr = pd.concat([hist.High - hist.Low, (hist.High - c.shift()).abs(), (hist.Low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    last = float(c.iloc[-1])
    score = 0.0
    score += 0.35 if last > sma50.iloc[-1] else -0.35
    score += 0.25 if sma20.iloc[-1] > sma50.iloc[-1] else -0.25
    if not math.isnan(sma200.iloc[-1]):
        score += 0.2 if last > sma200.iloc[-1] else -0.2
    mom = float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else 0.0
    score += float(np.clip(mom * 4, -0.2, 0.2))
    return {
        "price": last,
        "change_1d": float(c.iloc[-1] / c.iloc[-2] - 1),
        "mom_1m": mom,
        "mom_3m": float(c.iloc[-1] / c.iloc[-63] - 1) if len(c) > 63 else 0.0,
        "sma20": float(sma20.iloc[-1]), "sma50": float(sma50.iloc[-1]),
        "sma200": None if math.isnan(sma200.iloc[-1]) else float(sma200.iloc[-1]),
        "rsi14": float(rsi.iloc[-1]),
        "atr": float(atr.iloc[-1]),
        "support": float(hist.Low.tail(20).min()),
        "resistance": float(hist.High.tail(20).max()),
        "trend_score": float(np.clip(score, -1, 1)),
    }


def next_earnings(tk: yf.Ticker):
    try:
        ed = (tk.calendar or {}).get("Earnings Date")
        if ed:
            ed = ed[0] if isinstance(ed, list) else ed
            if ed >= date.today():
                return ed.isoformat()
    except Exception:
        pass
    return None


def earnings_history(tk: yf.Ticker, hist: pd.DataFrame, n: int = 8):
    """Absolute one-day stock moves on the last n earnings reactions."""
    try:
        ed = tk.get_earnings_dates(limit=n + 6)
    except Exception:
        return None
    if ed is None or ed.empty:
        return None
    closes = hist["Close"].copy()
    closes.index = closes.index.tz_localize(None).normalize()
    moves = []
    for ts in ed.index:
        ts = pd.Timestamp(ts)
        day = ts.tz_localize(None).normalize()
        if day >= closes.index[-1]:
            continue
        pos = closes.index.searchsorted(day)
        if pos >= len(closes) or pos == 0:
            continue
        if ts.hour >= 12:      # reported after the close -> next session reacts
            if pos + 1 >= len(closes):
                continue
            mv = closes.iloc[pos + 1] / closes.iloc[pos] - 1
        else:                  # before the open -> same session reacts
            mv = closes.iloc[pos] / closes.iloc[pos - 1] - 1
        moves.append({"date": day.strftime("%Y-%m-%d"), "move": float(mv)})
        if len(moves) >= n:
            break
    if not moves:
        return None
    return {"moves": moves, "avg_abs_move": float(np.mean([abs(m["move"]) for m in moves]))}


def earnings_implied_move(chains: pd.DataFrame, spot: float, earnings: str):
    """ATM straddle of the first expiry after earnings, as a % of spot."""
    e = chains[chains.expiry > earnings]  # strictly after, in case the report is after the close
    if e.empty:
        return None
    exp = sorted(e.expiry.unique())[0]
    e = e[(e.expiry == exp) & (e.bid > 0)]
    c, p = e[e.is_call], e[~e.is_call]
    if c.empty or p.empty:
        return None
    k = float(c.strike.iloc[np.argmin(np.abs(c.strike.values - spot))])
    cm, pm = c[c.strike == k], p[p.strike == k]
    if cm.empty or pm.empty:
        return None
    return {"expiry": exp, "dte": int(e.dte.iloc[0]), "strike": k,
            "straddle_pct": float((cm["mid"].iloc[0] + pm["mid"].iloc[0]) / spot)}


# Yahoo reports ~zero open interest on index options; use the matching ETF's positioning
GAMMA_PROXY = {"SPX": "SPY", "NDX": "QQQ", "RUT": "IWM"}


def _gamma_with_proxy(ticker: str, chains: pd.DataFrame, spot: float):
    proxy = GAMMA_PROXY.get(ticker)
    if not proxy or chains.openInterest.sum() > 10_000:
        return gamma_profile(chains, spot)
    p_spot = float(sources.history(proxy)["Close"].iloc[-1])
    p_chains, _ = load_chains(proxy, p_spot)
    g = gamma_profile(p_chains, p_spot)
    if not g:
        return gamma_profile(chains, spot)
    k = spot / p_spot  # rescale ETF strike levels to index points
    for f in ("call_wall", "put_wall", "zero_gamma"):
        if g[f] is not None:
            g[f] = g[f] * k
    g["profile"] = [{"strike": p["strike"] * k, "gex": p["gex"]} for p in g["profile"]]
    g["proxy"] = proxy
    return g


def analyze(ticker: str, sector: str = "") -> dict:
    tk = yf.Ticker(ticker)   # earnings calendar still comes from Yahoo
    hist = sources.history(ticker)
    if hist.empty or len(hist) < 60:
        raise ValueError("not enough price history")
    trend = trend_signals(hist)
    spot = trend["price"]
    chains, data_source = load_chains(ticker, spot)
    if chains.empty:
        raise ValueError("no option chains")

    earnings = None if sector.startswith("Index") else next_earnings(tk)

    earn_hist = earnings_history(tk, hist) if earnings else None
    earn_implied = earnings_implied_move(chains, spot, earnings) if earnings else None
    if earn_implied:
        # Strip the normal (non-earnings) variance out of the straddle to isolate the event move.
        # Straddle ~= 0.8 * sigma * sqrt(T)  =>  total sd = straddle / 0.8
        rets = np.log(hist["Close"]).diff().dropna()
        rv = float(rets.tail(60).std() * math.sqrt(252))
        total_sd = earn_implied["straddle_pct"] / 0.8
        normal_sd = rv * math.sqrt(max(earn_implied["dte"] - 1, 0) / 252)
        event_sd = math.sqrt(max(total_sd ** 2 - normal_sd ** 2, 0.0))
        earn_implied["move_pct"] = 0.8 * event_sd

    heads = news.fetch_headlines(ticker)
    candles = [
        {"time": idx.strftime("%Y-%m-%d"), "open": round(float(r.Open), 2), "high": round(float(r.High), 2),
         "low": round(float(r.Low), 2), "close": round(float(r.Close), 2), "volume": float(r.Volume)}
        for idx, r in hist.tail(180).iterrows()
    ]
    return {
        "ticker": ticker,
        "sector": sector,
        "trend": trend,
        "gamma": _gamma_with_proxy(ticker, chains, spot),
        "flow": flow_signals(chains, spot, float(hist.Volume.iloc[-1])),
        "vol": vol_signals(chains, spot, hist["Close"]),
        "news": news.score_headlines(ticker, heads),
        "earnings": earnings,
        "days_to_earnings": (date.fromisoformat(earnings) - date.today()).days if earnings else None,
        "earnings_history": earn_hist,
        "earnings_implied": earn_implied,
        "candles": candles,
        "as_of": hist.index[-1].strftime("%Y-%m-%d"),
        "data_source": data_source,
        "_chains": chains,   # kept in memory for strategy building, stripped before JSON
    }
