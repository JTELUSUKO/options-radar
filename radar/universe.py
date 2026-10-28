"""Default scan universe: liquid, optionable US names with a sector label."""

DEFAULT_UNIVERSE = {
    # Index / broad ETFs
    "SPX": "Index", "SPY": "Index ETF", "QQQ": "Index ETF", "IWM": "Index ETF", "DIA": "Index ETF",
    # Tech
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "AMD": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "INTC": "Technology", "MU": "Technology", "QCOM": "Technology", "PLTR": "Technology",
    "SMCI": "Technology", "ARM": "Technology", "SNOW": "Technology", "CRWD": "Technology",
    # Communication / consumer internet
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
    "DIS": "Communication", "UBER": "Communication",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "NKE": "Consumer", "SBUX": "Consumer",
    "WMT": "Consumer", "COST": "Consumer", "HD": "Consumer", "MCD": "Consumer",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "V": "Financials",
    "PYPL": "Financials", "COIN": "Financials", "HOOD": "Financials", "SOFI": "Financials",
    # Health
    "LLY": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "MRNA": "Healthcare",
    # Energy / industrials
    "XOM": "Energy", "CVX": "Energy", "BA": "Industrials", "CAT": "Industrials",
    "GE": "Industrials",
}

INDEX_ETFS = {"SPX", "SPY", "QQQ", "IWM", "DIA"}

# Cash-settled indexes use a prefix on each data source
INDEX_SYMBOLS = {"SPX": {"yahoo": "^SPX", "schwab": "$SPX"},
                 "NDX": {"yahoo": "^NDX", "schwab": "$NDX"},
                 "RUT": {"yahoo": "^RUT", "schwab": "$RUT"},
                 "VIX": {"yahoo": "^VIX", "schwab": "$VIX"}}

# CME futures (continuous front month): charted as "/ES" etc.; trade ~23h Sun 18:00 – Fri 17:00 New York
FUTURES = {"/ES": {"yahoo": "ES=F", "schwab": "/ES", "name": "E-mini S&P 500"},
           "/NQ": {"yahoo": "NQ=F", "schwab": "/NQ", "name": "E-mini Nasdaq-100"},
           "/CL": {"yahoo": "CL=F", "schwab": "/CL", "name": "Crude oil"},
           "/GC": {"yahoo": "GC=F", "schwab": "/GC", "name": "Gold"}}


def norm_ticker(t: str) -> str:
    """Upper-case; futures spelled ES, /ES, ES=F, /ESZ26… all become "/ES"."""
    t = (t or "").upper().strip()
    root = t.lstrip("/").removesuffix("=F")
    if root in {f[1:] for f in FUTURES}:
        return "/" + root
    if t.startswith("/") and root[:-3] in {f[1:] for f in FUTURES}:   # dated contract like /ESZ26
        return "/" + root[:-3]
    return t
