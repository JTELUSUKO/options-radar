"""Support / resistance levels from price swings, plus the Claude chart explanation.

A swing high/low is a bar whose high/low is the extreme of the bars around it.
Nearby swings are merged into one level (within ~half a day's average range), and
each level is scored by how many times price touched it, how recently, and on what volume.
"""
import numpy as np
import pandas as pd

from . import news

SWING = 5          # bars each side that a swing high/low must beat
LOOKBACK = 250     # bars (~1 year of daily bars)


def atr(df: pd.DataFrame, n: int = 14) -> float:
    prev = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    return float(tr.tail(n).mean())


def _day(ts) -> str:
    return ts.strftime("%Y-%m-%d")


def find_levels(hist: pd.DataFrame, max_each: int = 4, stamp=_day) -> dict:
    df = hist.tail(LOOKBACK).dropna()
    price = float(df["Close"].iloc[-1])
    a = atr(df)
    hi, lo, vol = df["High"].values, df["Low"].values, df["Volume"].values
    n = len(df)
    vol_avg = float(np.mean(vol)) or 1.0

    swings = []  # (price, bar index, volume ratio)
    for i in range(SWING, n - SWING):
        win = slice(i - SWING, i + SWING + 1)
        if hi[i] == hi[win].max():
            swings.append((hi[i], i, vol[i] / vol_avg))
        if lo[i] == lo[win].min():
            swings.append((lo[i], i, vol[i] / vol_avg))
    # the last few bars can't be confirmed swings, but the recent range still matters
    swings.append((float(hi[-SWING:].max()), n - 1, 1.0))
    swings.append((float(lo[-SWING:].min()), n - 1, 1.0))

    clusters = []
    for p, i, v in sorted(swings):
        if clusters and p - clusters[-1]["prices"][-1] <= 0.5 * a:
            c = clusters[-1]
        else:
            c = {"prices": [], "bars": [], "vols": []}
            clusters.append(c)
        c["prices"].append(p); c["bars"].append(i); c["vols"].append(v)

    levels = []
    for c in clusters:
        lvl = float(np.mean(c["prices"]))
        # touches: bars whose range came within a quarter-ATR of the level
        touches = int(((lo <= lvl + 0.25 * a) & (hi >= lvl - 0.25 * a)).sum())
        recency = max(c["bars"]) / n
        score = len(c["prices"]) * 2 + min(touches, 15) * 0.5 + recency * 3 + min(np.mean(c["vols"]), 3)
        last = df.index[max(c["bars"])]
        levels.append({"price": round(lvl, 2), "swings": len(c["prices"]), "touches": touches,
                       "last_tested": stamp(last), "score": round(float(score), 1),
                       "distance": lvl / price - 1})

    def pick(side):
        # strongest levels, but always keep the nearest one so S1/R1 are actually close by
        near = min(side, key=lambda l: abs(l["distance"]), default=None)
        side = sorted(side, key=lambda l: -l["score"])[:max_each]
        if near and near not in side:
            side = side[:max_each - 1] + [near]
        top = max((l["score"] for l in side), default=1)
        for l in side:
            l["strength"] = max(1, round(5 * l["score"] / top))
        return sorted(side, key=lambda l: abs(l["distance"]))

    supports = pick([l for l in levels if l["price"] < price - 0.1 * a])
    resistances = pick([l for l in levels if l["price"] > price + 0.1 * a])
    for k, l in enumerate(supports, 1):
        l["label"] = f"S{k}"
    for k, l in enumerate(resistances, 1):
        l["label"] = f"R{k}"
    return {"price": round(price, 2), "atr": round(a, 2), "supports": supports, "resistances": resistances}


def find_zones(hist: pd.DataFrame, max_each: int = 3, stamp=_day) -> dict:
    """Supply / demand zones: a quiet "base" candle right before a strong move away.

    Demand = base before a sharp rally (buyers overwhelmed sellers there); supply = base before
    a sharp drop. A zone is dropped once price closes through it (the orders there are used up).
    """
    df = hist.tail(LOOKBACK).dropna()
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    prev = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(), (df["Low"] - prev).abs()], axis=1).max(axis=1)
    atr_s = tr.rolling(14, min_periods=5).mean().values
    n, price = len(df), float(c[-1])
    zones = []
    for i in range(14, n - 2):
        a = atr_s[i]
        if not a or h[i] - l[i] > 0.8 * a:      # base candle must be quiet
            continue
        up = max(c[i + 1], c[i + 2]) - h[i]
        down = l[i] - min(c[i + 1], c[i + 2])
        if up > 1.2 * a:
            kind, lo_, hi_, move = "demand", l[i], max(o[i], c[i]), up / a
        elif down > 1.2 * a:
            kind, lo_, hi_, move = "supply", min(o[i], c[i]), h[i], down / a
        else:
            continue
        later_c, later_h, later_l = c[i + 3:], h[i + 3:], l[i + 3:]
        if kind == "demand":
            if (later_c < lo_).any():
                continue
            tests = int((later_l <= hi_).sum())
        else:
            if (later_c > hi_).any():
                continue
            tests = int((later_h >= lo_).sum())
        zones.append({"kind": kind, "low": round(float(lo_), 2), "high": round(float(hi_), 2),
                      "start": stamp(df.index[i]), "ts": df.index[i], "move_atr": round(float(move), 1), "tests": tests,
                      "fresh": tests == 0, "distance": ((lo_ + hi_) / 2) / price - 1})
    # merge overlapping zones of the same kind (keep the stronger move)
    a_last = float(np.nanmean(atr_s[-50:]))
    merged = []
    for z in sorted(zones, key=lambda z: (z["kind"], z["low"])):
        m = merged[-1] if merged else None
        # merge overlaps, but don't let merging grow a zone into a huge box over the chart
        if m and m["kind"] == z["kind"] and z["low"] <= m["high"] and max(m["high"], z["high"]) - m["low"] <= 1.2 * a_last:
            m["high"] = max(m["high"], z["high"]); m["tests"] = max(m["tests"], z["tests"])
            m["move_atr"] = max(m["move_atr"], z["move_atr"]); m["start"], m["ts"] = min((m["ts"], m["start"]), (z["ts"], z["start"]))[::-1]
            m["fresh"] = m["tests"] == 0; m["distance"] = ((m["low"] + m["high"]) / 2) / price - 1
        else:
            merged.append(dict(z))
    demand = sorted([z for z in merged if z["kind"] == "demand" and z["low"] <= price], key=lambda z: abs(z["distance"]))[:max_each]
    supply = sorted([z for z in merged if z["kind"] == "supply" and z["high"] >= price], key=lambda z: abs(z["distance"]))[:max_each]
    return {"demand": demand, "supply": supply}


def chart_context(ticker: str, levels: dict, zones: dict | None, gamma: dict | None, forecast: dict | None,
                  candles: list[dict], tf_label: str = "daily") -> str:
    """The chart's numbers as text, so Claude can reason about them."""
    fmt = lambda ls: "\n".join(f"  {l['label']} ${l['price']} (strength {l['strength']}/5, {l['touches']} touches, "
                               f"last tested {l['last_tested']}, {l['distance'] * 100:+.1f}% away)" for l in ls) or "  none"
    zfmt = lambda zs: "\n".join(f"  ${z['low']}–${z['high']} (formed {z['start']}, move {z['move_atr']}x avg range, "
                                f"{'fresh/untested' if z['fresh'] else str(z['tests']) + ' retests'}, {z['distance'] * 100:+.1f}% away)" for z in zs) or "  none"
    closes = ", ".join(f"{c['close']}" for c in candles[-30:])
    parts = [
        f"Stock: {ticker}, {tf_label} chart. Last price ${levels['price']}, average {tf_label} bar range ${levels['atr']}.",
        f"Last 30 {tf_label} closes (oldest first): {closes}",
        f"Support levels found from price swings:\n{fmt(levels['supports'])}",
        f"Resistance levels found from price swings:\n{fmt(levels['resistances'])}",
    ]
    if zones:
        parts.append(f"Demand zones (base before a sharp rally, not yet broken):\n{zfmt(zones['demand'])}")
        parts.append(f"Supply zones (base before a sharp drop, not yet broken):\n{zfmt(zones['supply'])}")
    if gamma:
        parts.append(f"Options levels: put wall ${gamma.get('put_wall')}, call wall ${gamma.get('call_wall')}, "
                     f"zero gamma ${gamma.get('zero_gamma')}, dealer gamma regime {gamma.get('regime')}.")
    if forecast:
        parts.append(f"Kronos AI model forecast for the next {len(forecast['dates'])} {tf_label} bars (to {(forecast.get('labels') or forecast['dates'])[-1]}): "
                     f"median close ${forecast['mid'][-1]:.2f} (10–90% range ${forecast['low'][-1]:.2f}–${forecast['high'][-1]:.2f}).")
    return "\n\n".join(parts)


def explain(ticker: str, context: str) -> str | None:
    """Ask Claude to read the chart in plain English (needs ANTHROPIC_API_KEY)."""
    return news.ask_claude(context + "\n\n" + (
        "Explain this chart to a beginner options trader in plain, simple English (short sentences). "
        "Cover: 1) where the strongest support/resistance and supply/demand zones are and why they matter, "
        "2) where options levels agree with price levels (stronger), "
        "3) what would happen on a break above resistance or below support, "
        "4) one practical idea for using these levels to pick option strikes or stops. "
        "Max 200 words, use short bullet points, no headings. Not financial advice."), max_tokens=1200)
