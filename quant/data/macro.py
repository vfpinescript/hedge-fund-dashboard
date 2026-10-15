"""
Macro & energy data access: FRED (Federal Reserve) and EIA (Energy Information
Administration). Free government APIs; keys come from the environment
(FRED_API_KEY, EIA_API_KEY) and are never hard-coded.

Everything is cached in-process (6h TTL) so a dashboard refresh does not hammer
the APIs. Series come back as pandas Series indexed by date, so they align
cleanly with the price data elsewhere in the engine.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import requests

FRED_KEY = os.environ.get("FRED_API_KEY")
EIA_KEY = os.environ.get("EIA_API_KEY")

_CACHE: dict = {}
_TTL = 6 * 3600


def _cached(key, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


# --------------------------------------------------------------------------- #
# FRED
# --------------------------------------------------------------------------- #
def fred_series(series_id: str, start: str | None = None) -> pd.Series:
    """Full daily FRED series as a date-indexed Series (missing values dropped)."""
    def _fetch():
        if not FRED_KEY:
            raise RuntimeError("FRED_API_KEY not set")
        params = {"series_id": series_id, "api_key": FRED_KEY, "file_type": "json"}
        if start:
            params["observation_start"] = start
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params=params, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        vals = {}
        for o in obs:
            v = o.get("value")
            if v not in (".", "", None):
                try:
                    vals[o["date"]] = float(v)
                except ValueError:
                    pass
        s = pd.Series(vals)
        if len(s):
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
        return s
    return _cached(f"fred:{series_id}:{start}", _fetch)


def fred_latest(series_id: str):
    """(date, value) of the most recent observation, or (None, None)."""
    try:
        s = fred_series(series_id)
        if len(s):
            return str(s.index[-1].date()), float(s.iloc[-1])
    except Exception:
        pass
    return None, None


# --------------------------------------------------------------------------- #
# EIA (v2 seriesid endpoint)
# --------------------------------------------------------------------------- #
def eia_series(series_id: str, length: int = 5000) -> pd.Series:
    """EIA v2 series (legacy series id) as a date-indexed Series. Always fetches
    the full history (cache key is length-independent) so `eia_latest` and the
    chart series share one cached copy."""
    def _fetch():
        if not EIA_KEY:
            raise RuntimeError("EIA_API_KEY not set")
        r = requests.get(f"https://api.eia.gov/v2/seriesid/{series_id}",
                         params={"api_key": EIA_KEY, "length": 5000}, timeout=30)
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        vals = {}
        for d in data:
            v = d.get("value")
            if v is not None:
                try:
                    vals[d["period"]] = float(v)
                except (ValueError, TypeError):
                    pass
        s = pd.Series(vals)
        if len(s):
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
        return s
    return _cached(f"eia:{series_id}", _fetch)


def eia_latest(series_id: str):
    try:
        s = eia_series(series_id)
        if len(s):
            return str(s.index[-1].date()), float(s.iloc[-1])
    except Exception:
        pass
    return None, None


# --------------------------------------------------------------------------- #
# Risk-free rate helper (used by the backtest/live financing assumptions)
# --------------------------------------------------------------------------- #
def riskfree_series(start: str | None = None) -> pd.Series:
    """3-month T-bill (DGS3MO) as a DAILY DECIMAL rate (e.g. 0.0426), the honest
    time-varying risk-free / cash rate, replacing a hard-coded constant. Falls
    back to an empty series if FRED is unavailable (callers use a default)."""
    try:
        s = fred_series("DGS3MO", start=start) / 100.0
        return s
    except Exception:
        return pd.Series(dtype=float)


def riskfree_latest(default: float = 0.04) -> float:
    d, v = fred_latest("DGS3MO")
    return v / 100.0 if v is not None else default
