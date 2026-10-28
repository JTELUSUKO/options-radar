"""Rumor & sentiment from free news RSS, with optional Claude classification.

Evidence notes (see reports/Institutional options trading signals.md):
- Press takeover rumors pay off only ~1/3 of the time and prices often reverse,
  so rumor buzz alone is weak; it is strongest when options flow confirms it.
- LLM headline sentiment predicts short-horizon returns, mostly in small caps.
"""
import json
import os
import re
import time
import urllib.parse
from datetime import datetime, timezone

import feedparser
import requests

RUMOR_WORDS = [
    "rumor", "rumour", "chatter", "talks", "in talks", "considering", "explores",
    "exploring", "takeover", "buyout", "acquire", "acquisition", "bid for",
    "approach", "stake", "activist", "reportedly", "people familiar", "sources say",
    "weighs", "mulls", "merger", "deal", "short report", "probe",
]
POS_WORDS = [
    "upgrade", "upgraded", "beat", "beats", "raises", "raised", "surge", "soar",
    "record", "approval", "approved", "wins", "partnership", "buyback", "outperform",
    "bullish", "rally", "jumps", "strong demand", "price target raised", "expands",
]
NEG_WORDS = [
    "downgrade", "downgraded", "miss", "misses", "cuts", "cut", "plunge", "tumble",
    "probe", "lawsuit", "recall", "investigation", "delay", "warns", "warning",
    "bearish", "falls", "slumps", "layoffs", "fraud", "short seller", "halt",
    "price target cut", "underperform", "sell rating",
]

_CACHE: dict = {}
_CLAUDE_CACHE: dict = {}   # (ticker, headlines) -> Claude's read, so a rescan doesn't pay to re-read the same headlines
_CLAUDE_CACHE_MAX = 500


def _count(text: str, words) -> int:
    t = text.lower()
    return sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", t))


def fetch_headlines(ticker: str, name_hint: str = "") -> list[dict]:
    """Last ~7 days of headlines from Google News RSS (free, no key)."""
    key = ticker
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < 1800:
        return hit[1]
    q = f'"{ticker}" stock when:7d'
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=en-US&gl=US&ceid=US:en"
    items = []
    try:
        feed = feedparser.parse(url)
        for e in feed.entries[:40]:
            try:
                ts = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
            items.append({
                "title": e.title,
                "link": e.link,
                "source": e.get("source", {}).get("title", ""),
                "published": ts.isoformat(),
            })
    except Exception:
        pass
    _CACHE[key] = (time.time(), items)
    return items


def score_headlines(ticker: str, items: list[dict]) -> dict:
    """Keyword sentiment + rumor detection, recency-weighted (half-life ~2 days)."""
    now = datetime.now(timezone.utc)
    pos = neg = rumor = 0.0
    rumor_heads = []
    recent_24h = 0
    for it in items:
        age_days = max(0.0, (now - datetime.fromisoformat(it["published"])).total_seconds() / 86400)
        w = 0.5 ** (age_days / 2.0)
        if age_days <= 1:
            recent_24h += 1
        title = it["title"]
        p, n, r = _count(title, POS_WORDS), _count(title, NEG_WORDS), _count(title, RUMOR_WORDS)
        pos += w * p
        neg += w * n
        if r:
            rumor += w * r
            rumor_heads.append(it)
        it["tone"] = "pos" if p > n else ("neg" if n > p else "neu")
        it["rumor"] = bool(r)
    total = pos + neg
    sentiment = 0.0 if total == 0 else (pos - neg) / (total + 1.0)
    out = {
        "sentiment": round(sentiment, 3),
        "rumor_score": round(rumor, 2),
        "buzz": len(items),
        "buzz_24h": recent_24h,
        "rumor_headlines": rumor_heads[:5],
        "headlines": items[:12],
        "ai_summary": None,
        "method": "keywords",
    }
    llm = _claude_classify(ticker, items[:20])
    if llm:
        out.update({
            "sentiment": float(llm.get("sentiment", out["sentiment"])),
            "rumor_score": max(out["rumor_score"], 2.0 if llm.get("rumor") else 0.0),
            "ai_summary": llm.get("summary"),
            "method": "claude",
        })
    return out


def _claude_classify(ticker: str, items: list[dict]):
    """Optional: if ANTHROPIC_API_KEY is set, ask Claude for sentiment/rumor JSON."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not items:
        return None
    heads = "\n".join(f"- {i['published'][:10]} {i['title']}" for i in items)
    key = (ticker, heads)
    if key in _CLAUDE_CACHE:
        return _CLAUDE_CACHE[key]
    prompt = (
        f"Headlines about {ticker} from the last 7 days:\n{heads}\n\n"
        "Return ONLY JSON: {\"sentiment\": number from -1 (very bearish) to 1 (very bullish) "
        "for the stock over the next 1-2 weeks, \"rumor\": true if there is an unconfirmed "
        "rumor/report (M&A, deal, activist, regulatory) not yet officially announced, "
        "\"summary\": one sentence on what the market is talking about}."
    )
    text = ask_claude(prompt, max_tokens=1500)
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        out = json.loads(m.group(0)) if m else None
    except Exception:
        return None
    if out is not None:   # failures aren't cached, so they're retried next scan
        if len(_CLAUDE_CACHE) >= _CLAUDE_CACHE_MAX:
            _CLAUDE_CACHE.pop(next(iter(_CLAUDE_CACHE)))
        _CLAUDE_CACHE[key] = out
    return out


last_error = None  # why the most recent ask_claude call failed, shown in the UI


def ask_claude(prompt: str | None = None, max_tokens: int = 1500, messages: list | None = None,
               system: str | None = None) -> str | None:
    """Claude call (one prompt, or a chat history); None when no key or the call fails (see last_error)."""
    global last_error
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01",
               "content-type": "application/json"}
    if os.environ.get("ANTHROPIC_WORKSPACE_ID"):  # needed for keys not scoped to a workspace
        headers["anthropic-workspace-id"] = os.environ["ANTHROPIC_WORKSPACE_ID"]
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json={"model": os.environ.get("RADAR_MODEL", "claude-haiku-5-5"),
                  "max_tokens": max_tokens,
                  **({"system": system} if system else {}),
                  "messages": messages or [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        if not r.ok:
            last_error = r.json().get("error", {}).get("message", r.text[:200])
            return None
        last_error = None
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")
    except Exception as e:  # noqa: BLE001
        last_error = str(e)
        return None
