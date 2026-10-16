"""
London Strategic Edge "Databank" client — a free, clean market-data source used
as the PRIMARY price feed (the loader falls back to yfinance when LSE lacks a
symbol). Candles reach back to 2003 (US stocks, split-adjusted), 2009 (forex),
2017 (crypto), at resolutions down to 1 second.

HTTP API base https://api.londonstrategicedge.com/vault, authenticated with the
x-api-key header (LSE_API_KEY). Also serves macro series and government bond
yields, which we expose for the macro layer.
"""
from __future__ import annotations

import os
import re

import pandas as pd
import requests

BASE = "https://api.londonstrategicedge.com/vault"

# fund-symbol -> LSE-symbol for the non-passthrough cases
_CRYPTO = {"BTCUSD": "BTC/USD", "ETHUSD": "ETH/USD", "SOLUSD": "SOL/USD",
           "XRPUSD": "XRP/USD", "ADAUSD": "ADA/USD", "DOGEUSD": "DOGE/USD",
           "AVAXUSD": "AVAX/USD", "BNBUSD": "BNB/USD", "DOTUSD": "DOT/USD",
           "LINKUSD": "LINK/USD", "LTCUSD": "LTC/USD"}
_METALS = {"XAUUSD": "XAU/USD", "XAGUSD": "XAG/USD", "XPTUSD": "XPT/USD", "XPDUSD": "XPD/USD"}
_FX_QUOTES = ("USD", "JPY", "EUR", "GBP", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK")


def to_lse_symbol(instrument: str) -> str | None:
    """Map a fund symbol to its LSE form, or None if LSE will not have it
    (indices, futures, '=X'/'^'/'!' style), so the caller falls back to yfinance."""
    s = (instrument or "").upper().strip()
    if not s:
        return None
    if s in _CRYPTO:
        return _CRYPTO[s]
    if s in _METALS:
        return _METALS[s]
    if len(s) == 6 and s.isalpha() and s[3:] in _FX_QUOTES and s[:3] in _FX_QUOTES + ("XAU", "XAG"):
        return s[:3] + "/" + s[3:]
    # plain US stock / ETF ticker (letters, optional . or - like BRK-B)
    if re.fullmatch(r"[A-Z]{1,6}([.\-][A-Z]{1,3})?", s):
        return s
    return None


def _key():
    return os.environ.get("LSE_API_KEY")


def candles(lse_symbol: str, timeframe: str = "1d", limit: int = 5000) -> pd.DataFrame | None:
    """OHLCV DataFrame indexed by UTC date, or None on any miss."""
    key = _key()
    if not key:
        return None
    try:
        r = requests.get(f"{BASE}/candles",
                         params={"symbol": lse_symbol, "timeframe": timeframe,
                                 "limit": limit, "order": "desc"},
                         headers={"x-api-key": key}, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        df = pd.DataFrame(data)
        if not {"ts", "open", "high", "low", "close"} <= set(df.columns):
            return None
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").sort_index()
        if "volume" not in df.columns:          # FX has no consolidated volume
            df["volume"] = 0.0
        out = df[["open", "high", "low", "close", "volume"]].astype(float)
        out.index.name = "date"
        return out
    except Exception:
        return None


def load(instrument: str, timeframe: str = "1d", bars: int = 3000) -> pd.DataFrame | None:
    """Loader entry point: returns the full available OHLCV (up to the 5000-row
    page) for a fund symbol from LSE, or None so the caller falls back to yfinance.
    The caller caches and slices to `bars`, so we must NOT pre-slice here (doing so
    would poison the cache if a small `bars` was requested first)."""
    ls = to_lse_symbol(instrument)
    if not ls:
        return None
    df = candles(ls, timeframe=timeframe)
    if df is None or len(df) < 60:
        return None
    return df


def series(symbol: str, limit: int = 5000) -> pd.Series | None:
    """A macro series or bond yield (date,value) as a pd.Series, or None."""
    key = _key()
    if not key:
        return None
    try:
        r = requests.get(f"{BASE}/series", params={"symbol": symbol, "order": "asc", "limit": limit},
                         headers={"x-api-key": key}, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or not data:
            return None
        s = pd.Series({d["date"]: float(d["value"]) for d in data if d.get("value") is not None})
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return None
