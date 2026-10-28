"""Turn raw signals into ranked, concrete trade ideas.

Weights follow "Signal design for our dashboard" in
reports/Institutional options trading signals.md:

  volatility risk premium (IV vs realized) ....... 30%   vol edge
  earnings implied move vs own history ........... 25%   vol edge
  flow anomaly (unsigned, free data) .............  7%   direction
  call-put IV spread & skew tilt ................. 10%   direction
  news / rumor (counts fully only if flow agrees)  10%   direction
  trend / timing ................................. 10%   direction
  dealer gamma regime ............................ multiplier only (halves short-vol below the flip)
  call / put walls, zero-gamma .................... displayed + used for strikes, not scored

Raw put/call ratio and option-to-stock volume get zero weight.
"""
import math
from datetime import date

import numpy as np
import pandas as pd

from .engine import next_monthly_opex

W_VRP, W_EARN = 0.30, 0.25
W_FLOW, W_IVTILT, W_NEWS, W_TREND = 0.07, 0.10, 0.10, 0.10
W_VOL = W_VRP + W_EARN
W_DIR = W_FLOW + W_IVTILT + W_NEWS + W_TREND


def _z(values):
    a = np.array([np.nan if v is None else v for v in values], dtype=float)
    mu, sd = np.nanmean(a), np.nanstd(a)
    if not sd or np.isnan(sd):
        return np.zeros_like(a)
    return np.nan_to_num((a - mu) / sd)


def _clip1(x):
    return float(max(-1.0, min(1.0, x)))


def score_universe(results: list[dict]) -> list[dict]:
    """Cross-sectional scoring, then pick the best strategy per ticker."""
    ok = [r for r in results if r.get("vol") and r.get("gamma")]
    for r in results:
        if not r.get("vol"):
            r["no_idea"] = "no live option quotes (bid/ask) near 30 days to expiry"
        elif not r.get("gamma"):
            r["no_idea"] = "no open interest to estimate dealer gamma"
    if not ok:
        return []

    flow_raw = [r["flow"]["otm_short_call_share"] - r["flow"]["otm_short_put_share"]
                + _ratio(r["flow"]["unusual_call_premium"], r["flow"]["unusual_put_premium"]) * 0.5
                for r in ok]
    flow_z = _z(flow_raw)
    cw_z = _z([r["vol"]["cw_spread"] for r in ok])
    skew_z = _z([r["vol"]["skew"] for r in ok])
    activity_z = _z([r["flow"]["vol_oi_ratio"] for r in ok])

    for i, r in enumerate(ok):
        v, f, n, t = r["vol"], r["flow"], r["news"], r["trend"]
        s = {}
        # --- volatility edge (+ = options rich -> sell premium, - = cheap -> buy)
        iv_rv = v["iv_rv"] or 1.0
        earnings_in_window = r["days_to_earnings"] is not None and r["days_to_earnings"] <= v["ref_dte"]
        s["vrp"] = math.tanh((iv_rv - 1.15) / 0.25) * (0.5 if earnings_in_window else 1.0)
        s["earn"] = 0.0
        ei, eh = r.get("earnings_implied"), r.get("earnings_history")
        if ei and eh and eh["avg_abs_move"] > 0 and r["days_to_earnings"] is not None and r["days_to_earnings"] <= 10:
            s["earn"] = math.tanh((ei["move_pct"] / eh["avg_abs_move"] - 1.0) / 0.3)
        vol_edge = (W_VRP * s["vrp"] + W_EARN * s["earn"]) / W_VOL

        # --- direction (+ bullish / - bearish)
        s["trend"] = t["trend_score"]
        s["flow"] = _clip1(math.tanh(flow_z[i] / 1.5) * (1.0 if activity_z[i] > 0 else 0.6))
        s["ivtilt"] = _clip1(math.tanh((cw_z[i] - skew_z[i]) / 2.0))
        flow_confirms = (n["sentiment"] > 0 and s["flow"] > 0.2) or (n["sentiment"] < 0 and s["flow"] < -0.2)
        rumor_boost = 1.0 if (n["rumor_score"] > 0.5 and flow_confirms) else 0.5
        s["news"] = _clip1(n["sentiment"] * rumor_boost)
        direction = (W_TREND * s["trend"] + W_FLOW * s["flow"] + W_IVTILT * s["ivtilt"]
                     + W_NEWS * s["news"]) / W_DIR

        r["signals"] = {k: round(val, 3) for k, val in s.items()}
        r["vol_edge"] = round(vol_edge, 3)
        r["direction"] = round(direction, 3)
        r["rumor_flow_confirmed"] = bool(n["rumor_score"] > 0.5 and flow_confirms)
        r["idea"] = build_idea(r, vol_edge, direction)
    return ok


def _ratio(a, b):
    return (a - b) / (a + b) if (a + b) > 0 else 0.0


# ------------------------------------------------------------------ strategies
def build_idea(r: dict, vol_edge: float, direction: float):
    """The best trade for this ticker, or None with the reason left in r["no_idea"]."""
    chains: pd.DataFrame = r.get("_chains")
    if chains is None or chains.empty:
        r["no_idea"] = "no option chain"
        return None
    spot = r["trend"]["price"]
    g = r["gamma"]
    neg_gamma = g["regime"] == "negative"
    d_abs = abs(direction)
    bull = direction > 0

    # Expiry: earnings plays use the first expiry after earnings, others ~30 DTE
    earnings_play = abs(r["signals"]["earn"]) > 0.35 and r.get("earnings_implied")
    exp = r["earnings_implied"]["expiry"] if earnings_play else _pick_expiry(chains, 30)
    if exp is None:
        r["no_idea"] = "no expiry 14+ days out"
        return None
    ch = chains[(chains.expiry == exp) & (chains.bid > 0) & (chains.ask > 0)]
    if ch.empty:
        r["no_idea"] = f"no live bid/ask on the {exp} expiry"
        return None

    short_vol_mult = 0.5 if neg_gamma else 1.0
    cands = []
    # Scores are on a 0..1 scale using the same weights as the signal model
    if d_abs >= 0.15:
        cands.append(("credit_spread", (W_VOL * max(vol_edge, 0) + W_DIR * d_abs) * short_vol_mult))
        cands.append(("long_option", W_VOL * max(-vol_edge, 0) + W_DIR * d_abs * (1.15 if neg_gamma else 1.0)))
        cands.append(("debit_spread", W_VOL * 0.25 * (1 - abs(vol_edge)) + W_DIR * d_abs))
    cands.append(("iron_condor", (W_VOL * max(vol_edge, 0) + W_DIR * 0.5 * max(0, 1 - d_abs * 2)) * short_vol_mult))
    cands.append(("long_straddle", W_VOL * max(-vol_edge, 0) + W_DIR * 0.3 * max(0, 1 - d_abs * 2)
                  + (0.05 if neg_gamma else 0)))
    kind, raw = max(cands, key=lambda c: c[1])
    legs = _legs(kind, ch, spot, bull, g)
    if not legs:
        r["no_idea"] = f"couldn't find strikes for a {_label(kind, bull).lower()}"
        return None
    econ = _economics(kind, legs, spot)
    if econ is None:
        r["no_idea"] = f"{_label(kind, bull).lower()} prices don't make sense (credit/debit vs width)"
        return None

    # Liquidity gate: wide markets or thin open interest cut the score
    liq = 1.0
    spread = r["vol"]["atm_spread_pct"]
    if spread > 0.15:
        liq *= 0.5
    elif spread > 0.08:
        liq *= 0.8
    if sum(l["oi"] for l in legs) < 500:
        liq *= 0.7
    score = round(100 * raw / (W_VOL + W_DIR) * liq * 1.6, 1)
    score = min(score, 99.0)

    dte = int(ch.dte.iloc[0])
    return {
        "strategy": kind,
        "label": _label(kind, bull),
        "bias": "bullish" if (kind in ("credit_spread", "long_option", "debit_spread") and bull)
                else "bearish" if kind in ("credit_spread", "long_option", "debit_spread")
                else "neutral (range)" if kind == "iron_condor" else "big move either way",
        "expiry": exp, "dte": dte,
        "legs": legs, **econ,
        "score": score,
        "plan": _plan(kind, r, legs, econ, bull, dte, exp),
        "why": _why(r, kind, vol_edge, direction),
        "warnings": _warnings(r, kind, exp),
    }


def _pick_expiry(chains, target):
    exps = chains.groupby("expiry")["dte"].first()
    exps = exps[exps >= 14]
    if exps.empty:
        return None
    return exps.index[np.argmin(np.abs(exps.values - target))]


def _nearest_delta(ch, is_call, target):
    side = ch[ch.is_call == is_call]
    if side.empty:
        return None
    return side.iloc[np.argmin(np.abs(np.abs(side.delta.values) - target))]


def _row_at(ch, is_call, strike):
    side = ch[(ch.is_call == is_call)]
    if side.empty:
        return None
    return side.iloc[np.argmin(np.abs(side.strike.values - strike))]


def _leg(row, action):
    return {"action": action, "type": "call" if row.is_call else "put", "strike": float(row.strike),
            "expiry": row.expiry, "mid": round(float(row.mid), 2), "bid": float(row.bid),
            "ask": float(row.ask), "delta": round(float(row.delta), 2), "iv": round(float(row.iv), 3),
            "oi": int(row.openInterest), "contract": row.contractSymbol}


def _wing(ch, short, is_call, spot):
    """Long protective strike ~4-6% of spot beyond the short strike."""
    width = max(spot * 0.05, 1.0)
    target = short.strike + width if is_call else short.strike - width
    w = _row_at(ch, is_call, target)
    if w is None or w.strike == short.strike:
        return None
    return w


def _legs(kind, ch, spot, bull, g):
    if kind == "long_option":
        row = _nearest_delta(ch, bull, 0.40)
        return [_leg(row, "BUY")] if row is not None else None
    if kind == "debit_spread":
        long_ = _nearest_delta(ch, bull, 0.50)
        short = _nearest_delta(ch, bull, 0.25)
        if long_ is None or short is None or long_.strike == short.strike:
            return None
        return [_leg(long_, "BUY"), _leg(short, "SELL")]
    if kind == "credit_spread":
        # Sell the put (bull) / call (bear) at ~0.25 delta, preferring beyond the gamma wall
        short = _nearest_delta(ch, not bull, 0.25)
        wall = g["put_wall"] if bull else g["call_wall"]
        if short is not None and wall:
            at_wall = _row_at(ch, not bull, wall)
            if at_wall is not None and 0.12 <= abs(at_wall.delta) <= 0.35:
                short = at_wall
        if short is None:
            return None
        wing = _wing(ch, short, not bull, spot)
        return [_leg(short, "SELL"), _leg(wing, "BUY")] if wing is not None else None
    if kind == "iron_condor":
        sp = _nearest_delta(ch, False, 0.18)
        sc = _nearest_delta(ch, True, 0.18)
        if sp is None or sc is None:
            return None
        lp, lc = _wing(ch, sp, False, spot), _wing(ch, sc, True, spot)
        if lp is None or lc is None:
            return None
        return [_leg(sp, "SELL"), _leg(lp, "BUY"), _leg(sc, "SELL"), _leg(lc, "BUY")]
    if kind == "long_straddle":
        c = _nearest_delta(ch, True, 0.50)
        if c is None:
            return None
        p = _row_at(ch, False, c.strike)
        return [_leg(c, "BUY"), _leg(p, "BUY")] if p is not None else None
    return None


def _economics(kind, legs, spot):
    """Per 1 contract (x100), at mid prices."""
    net = sum((-l["mid"] if l["action"] == "BUY" else l["mid"]) for l in legs)  # + = credit
    if kind == "long_option":
        l = legs[0]
        be = l["strike"] + l["mid"] if l["type"] == "call" else l["strike"] - l["mid"]
        return {"cost": round(l["mid"] * 100), "max_loss": round(l["mid"] * 100),
                "max_profit": None, "breakevens": [round(be, 2)], "credit": False}
    if kind == "long_straddle":
        debit = -net
        k = legs[0]["strike"]
        return {"cost": round(debit * 100), "max_loss": round(debit * 100), "max_profit": None,
                "breakevens": [round(k - debit, 2), round(k + debit, 2)], "credit": False}
    if kind == "debit_spread":
        debit = -net
        width = abs(legs[1]["strike"] - legs[0]["strike"])
        if debit <= 0 or debit >= width:
            return None
        be = legs[0]["strike"] + debit if legs[0]["type"] == "call" else legs[0]["strike"] - debit
        return {"cost": round(debit * 100), "max_loss": round(debit * 100),
                "max_profit": round((width - debit) * 100), "breakevens": [round(be, 2)], "credit": False}
    if kind == "credit_spread":
        credit = net
        width = abs(legs[0]["strike"] - legs[1]["strike"])
        if credit <= 0 or credit >= width:
            return None
        be = legs[0]["strike"] - credit if legs[0]["type"] == "put" else legs[0]["strike"] + credit
        return {"cost": -round(credit * 100), "max_loss": round((width - credit) * 100),
                "max_profit": round(credit * 100), "breakevens": [round(be, 2)], "credit": True,
                "pop_est": round(1 - abs(legs[0]["delta"]), 2)}
    if kind == "iron_condor":
        credit = net
        width = max(abs(legs[0]["strike"] - legs[1]["strike"]), abs(legs[2]["strike"] - legs[3]["strike"]))
        if credit <= 0 or credit >= width:
            return None
        return {"cost": -round(credit * 100), "max_loss": round((width - credit) * 100),
                "max_profit": round(credit * 100),
                "breakevens": [round(legs[0]["strike"] - credit, 2), round(legs[2]["strike"] + credit, 2)],
                "credit": True, "pop_est": round(1 - abs(legs[0]["delta"]) - abs(legs[2]["delta"]), 2)}
    return None


def _label(kind, bull):
    return {
        "long_option": "Buy call" if bull else "Buy put",
        "debit_spread": "Bull call spread" if bull else "Bear put spread",
        "credit_spread": "Bull put credit spread" if bull else "Bear call credit spread",
        "iron_condor": "Iron condor",
        "long_straddle": "Long straddle",
    }[kind]


def _plan(kind, r, legs, econ, bull, dte, exp):
    t, g = r["trend"], r["gamma"]
    spot = t["price"]
    opex = next_monthly_opex()
    exit_day = max(dte - 7, 1)
    if econ["credit"]:
        take = f"Close when you can buy it back for 50% of the credit (≈${econ['max_profit'] * 0.5:,.0f} profit)."
        stop = f"Close if the spread's price doubles (loss ≈${econ['max_profit']:,.0f}) or price closes beyond your short strike."
    else:
        tgt = "+100%" if kind in ("long_option", "long_straddle") else "+60% of max profit"
        take = f"Take profit at {tgt}; for single options consider selling half at +50%."
        stop = "Cut the trade at -50% of what you paid."
    if kind in ("long_option", "debit_spread", "credit_spread"):
        if bull:
            entry_lvl = max([x for x in (t["support"], g["put_wall"]) if x and x < spot] or [t["support"]])
            entry = (f"Best entry on a dip toward ${entry_lvl:,.2f} (support / put wall) while price holds above "
                     f"${t['sma50']:,.2f} (50-day avg). Avoid chasing above ${t['resistance']:,.2f}.")
            invalid = f"Idea is wrong if the stock closes below ${min(t['support'], t['sma50']):,.2f}."
        else:
            entry_lvl = min([x for x in (t["resistance"], g["call_wall"]) if x and x > spot] or [t["resistance"]])
            entry = (f"Best entry on a bounce toward ${entry_lvl:,.2f} (resistance / call wall) while price stays below "
                     f"${t['sma50']:,.2f}. Avoid shorting into ${t['support']:,.2f} support.")
            invalid = f"Idea is wrong if the stock closes above ${max(t['resistance'], t['sma50']):,.2f}."
    elif kind == "iron_condor":
        entry = (f"Enter while price (${spot:,.2f}) sits well inside your short strikes "
                 f"(${legs[0]['strike']:g}–${legs[2]['strike']:g}); ideally after a calm open, not on a big gap day.")
        invalid = f"Exit if price breaks either short strike (${legs[0]['strike']:g} / ${legs[2]['strike']:g})."
    else:
        entry = ("Enter 1–5 days before earnings while implied volatility is still low; don't buy after a spike."
                 if r.get("earnings") else
                 "Options are cheap and dealers are short gamma — enter on a quiet day while price is coiling; "
                 "don't buy right after a big move has already happened.")
        invalid = "Exit if implied volatility jumps without the stock moving (take the vol gain) or 2 days before expiry."
    if r.get("days_to_earnings") is not None and r["signals"]["earn"] and abs(r["signals"]["earn"]) > 0.35:
        timing = (f"Earnings on {r['earnings']}. Enter the day of / before the report; "
                  "close the morning after — this is a one-event volatility trade.")
    else:
        timing = f"Hold up to ~{exit_day} days; close by {exit_day} days from now so you're out a week before expiry ({exp})."
    return {"entry": entry, "take_profit": take, "stop": stop, "invalidation": invalid,
            "timing": timing, "opex": opex.isoformat(),
            "size": "Risk no more than 1–2% of your account on this idea (max loss above, per contract)."}


def _why(r, kind, vol_edge, direction):
    v, t, g, f, n, s = r["vol"], r["trend"], r["gamma"], r["flow"], r["news"], r["signals"]
    out = []
    if abs(s["vrp"]) > 0.2:
        out.append(f"Options are {'expensive' if s['vrp'] > 0 else 'cheap'}: 30-day implied vol {v['atm_iv']:.0%} vs "
                   f"realized {0.5 * v['rv20'] + 0.5 * v['rv60']:.0%} (ratio {v['iv_rv']:.2f}).")
    if abs(s["earn"]) > 0.2:
        out.append(f"Earnings: market prices a ±{r['earnings_implied']['move_pct']:.1%} move vs "
                   f"±{r['earnings_history']['avg_abs_move']:.1%} average over the last "
                   f"{len(r['earnings_history']['moves'])} reports.")
    if abs(s["trend"]) > 0.3:
        out.append(f"Trend is {'up' if s['trend'] > 0 else 'down'}: price ${t['price']:,.2f} vs 50-day ${t['sma50']:,.2f}, "
                   f"1-month {t['mom_1m']:+.1%}, RSI {t['rsi14']:.0f}.")
    if abs(s["flow"]) > 0.3:
        side = "calls" if s["flow"] > 0 else "puts"
        out.append(f"Unusual activity leans to short-dated out-of-the-money {side} "
                   f"({f['unusual_count']} unusual contracts; side of trade unknown on free data).")
    if r.get("rumor_flow_confirmed"):
        out.append("Rumor/news chatter is confirmed by options activity in the same direction.")
    elif abs(s["news"]) > 0.25:
        out.append(f"News tone is {'positive' if s['news'] > 0 else 'negative'} ({n['buzz']} headlines this week).")
    if abs(s["ivtilt"]) > 0.4:
        out.append(f"Options pricing tilt (call-put IV spread / put skew) is {'bullish' if s['ivtilt'] > 0 else 'bearish'}.")
    out.append(f"Dealer gamma is {g['regime']}" +
               (" — dealers dampen moves; ranges tend to hold." if g["regime"] == "positive"
                else " — dealers amplify moves; expect bigger swings, size smaller."))
    return out


def _warnings(r, kind, exp):
    w = []
    if r.get("earnings") and r["earnings"] <= exp and kind not in ("long_straddle",) and abs(r["signals"]["earn"]) <= 0.35:
        w.append(f"Earnings ({r['earnings']}) falls before expiry — a gap can blow through your strikes.")
    if r["gamma"]["regime"] == "negative" and kind in ("credit_spread", "iron_condor"):
        w.append("Selling premium in negative gamma: score halved; use smaller size.")
    if r["vol"]["atm_spread_pct"] > 0.08:
        w.append("Wide bid/ask — use limit orders at mid, expect worse fills.")
    opex = next_monthly_opex()
    if (opex - date.today()).days <= 2:
        w.append(f"Monthly options expiration {opex} — prices often pin to big strikes, then move after.")
    return w
