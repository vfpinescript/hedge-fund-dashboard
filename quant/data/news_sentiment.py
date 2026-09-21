"""
Historical news-sentiment loader (Alpha Vantage NEWS_SENTIMENT).

Builds a REAL dated daily-sentiment series so news can be a backtestable stream
(no look-ahead: each day's sentiment uses only articles published that day).
One API call returns the most recent ~1000 articles in a window, so we paginate
backward with `time_to`. Free tier = 25 calls/day, 5/min, so calls are capped
and throttled, and results are cached to parquet.

Instrument -> Alpha Vantage ticker: crypto and equities are well covered; FX and
commodities are not, so those simply have no news stream (the agent abstains).
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import certifi
import pandas as pd

_SSL = ssl.create_default_context(cafile=certifi.where())
_CACHE = Path(__file__).resolve().parents[2] / "data_cache"

AV_TICKER = {
    "BTCUSD": "CRYPTO:BTC", "ETHUSD": "CRYPTO:ETH", "SOLUSD": "CRYPTO:SOL",
    "ES1!": "SPY", "NQ1!": "QQQ", "SPX": "SPY", "NDX": "QQQ",
}


def _key():
    env = Path(__file__).resolve().parents[2] / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("ALPHAVANTAGE_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


def _cache_path(av_ticker):
    return _CACHE / f"news_{av_ticker.replace(':', '_')}.parquet"


def load_av_sentiment(av_ticker, max_calls=12, throttle=13.0, use_cache=True) -> pd.Series:
    """Daily mean news-sentiment score for one ticker (paginated backward)."""
    cache = _cache_path(av_ticker)
    if use_cache and cache.exists():
        return pd.read_parquet(cache)["sentiment"]

    daily = _fetch_sentiment(av_ticker, max_calls, throttle)
    daily.to_frame().to_parquet(cache)
    return daily


def refresh_sentiment(av_ticker, max_calls=2, throttle=13.0) -> pd.Series:
    """Fetch the latest sentiment and UNION-merge it into the cache, so daily
    runs accumulate a growing multi-month/year series (never overwrite)."""
    cache = _cache_path(av_ticker)
    fresh = _fetch_sentiment(av_ticker, max_calls, throttle)
    if cache.exists():
        old = pd.read_parquet(cache)["sentiment"]
        merged = pd.concat([old, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = fresh
    merged.name = "sentiment"
    _CACHE.mkdir(exist_ok=True)
    merged.to_frame().to_parquet(cache)
    return merged


def _fetch_sentiment(av_ticker, max_calls=12, throttle=13.0) -> pd.Series:
    """Paginate Alpha Vantage NEWS_SENTIMENT backward; return a daily series."""
    key = _key()
    if not key:
        raise RuntimeError("no Alpha Vantage key")
    _CACHE.mkdir(exist_ok=True)
    rows = []
    time_to = None
    for _ in range(max_calls):
        params = f"function=NEWS_SENTIMENT&tickers={av_ticker}&limit=1000&sort=LATEST&apikey={key}"
        if time_to:
            params += f"&time_to={time_to}"
        url = "https://www.alphavantage.co/query?" + params
        try:
            r = json.loads(urllib.request.urlopen(url, timeout=45, context=_SSL).read())
        except Exception:
            break
        feed = r.get("feed", [])
        if not feed:
            break
        for a in feed:
            tp = a.get("time_published", "")
            score = None
            for t in a.get("ticker_sentiment", []):
                if t.get("ticker") == av_ticker:
                    score = float(t.get("ticker_sentiment_score", 0))
                    break
            if score is None:
                score = float(a.get("overall_sentiment_score", 0) or 0)
            if tp:
                rows.append((tp[:8], score))
        oldest = min(a["time_published"] for a in feed)
        new_to = (datetime.strptime(oldest[:8], "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%dT2359")
        if new_to == time_to:
            break
        time_to = new_to
        time.sleep(throttle)

    if not rows:
        raise RuntimeError(f"no news for {av_ticker}")
    df = pd.DataFrame(rows, columns=["date", "score"])
    daily = df.groupby("date")["score"].mean()
    daily.index = pd.to_datetime(daily.index, format="%Y%m%d")
    daily = daily.sort_index()
    daily.name = "sentiment"
    return daily
