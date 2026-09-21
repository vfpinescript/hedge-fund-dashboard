"""
Alternative (non-price) streams for the combination pool: news sentiment and
Kalshi arbitrage.

Honest status: the combination search blends HISTORICAL daily return streams,
and free data gives only ~6-12 weeks of dated news sentiment (Alpha Vantage
free tier) and no Kalshi lock history at all. So these streams are WIRED into
the pool but GATED by a minimum-history requirement — today they are too short
to blend without look-ahead, and they activate automatically once enough
forward history accrues (via `record_today`) or a paid historical backfill is
added. This is the only honest way to add them: never fabricate a past series.

Kalshi note: arbitrage is market-neutral, not a directional signal on our
instruments, so it does not belong in the SIGNAL blend — it is a separate
capital sleeve. Its opportunities are recorded forward to build a real track
record.
"""
from __future__ import annotations

from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

from ..data.loader import load_ohlcv
from ..data.news_sentiment import load_av_sentiment, AV_TICKER

MIN_HISTORY = 252          # ~1 trading year before a stream may enter the blend
ALT_INSTRUMENTS = ["BTCUSD", "ETHUSD"]
_LOG = Path(__file__).resolve().parents[2] / "data_cache" / "stream_log.csv"


def news_stream(instrument, cost_bps=1.0) -> pd.Series | None:
    """Daily return stream from news-sentiment timing on one instrument.
    Position = sign of sentiment vs its recent mean (relative optimism)."""
    av = AV_TICKER.get(instrument.upper())
    if not av:
        return None
    try:
        sent = load_av_sentiment(av, use_cache=True)
    except Exception:
        return None
    try:
        df = load_ohlcv(instrument, bars=3000)
    except Exception:
        return None
    ret = df["close"].pct_change()
    s = sent.reindex(df.index).ffill(limit=3)
    pos = np.sign(s - s.rolling(10, min_periods=3).mean())
    strat = pos.shift(1) * ret - (pos - pos.shift(1)).abs() * (cost_bps / 1e4)
    return strat.dropna()


def build_alt_streams():
    """All available alt streams with their history length and gate status."""
    out = {}
    for inst in ALT_INSTRUMENTS:
        s = news_stream(inst)
        if s is not None and len(s):
            out[f"news_{inst}"] = s
    return out


def stream_status():
    streams = build_alt_streams()
    rows = []
    for name, s in streams.items():
        rows.append({"stream": name, "days": len(s),
                     "gate": MIN_HISTORY, "eligible": len(s) >= MIN_HISTORY,
                     "status": "eligible" if len(s) >= MIN_HISTORY
                     else f"gated: {len(s)}/{MIN_HISTORY} days of history"})
    return {"min_history": MIN_HISTORY, "streams": rows,
            "note": "News streams enter the blend automatically once history >= gate. "
                    "Free data gives only weeks today; grows via record_today or a paid backfill."}


def eligible_alt_streams():
    """Only streams that clear the history gate — safe to blend."""
    return {n: s for n, s in build_alt_streams().items() if len(s) >= MIN_HISTORY}


def record_today(instruments=ALT_INSTRUMENTS):
    """Daily job: (1) UNION-merge the latest Alpha Vantage sentiment into each
    ticker's cache so the backtestable news_stream grows over time, and (2) log
    the Kalshi arbitrage snapshot to build the arb sleeve's forward track record.
    Idempotent per day. Designed to run once daily from cron."""
    from ..data.news_sentiment import refresh_sentiment
    from ..agents.kalshi_arb import KalshiArbitrageAgent
    today = date.today().isoformat()
    result = {"date": today, "news": {}, "kalshi": None}

    # 1) grow the AV sentiment caches (the news_stream data source)
    for inst in instruments:
        av = AV_TICKER.get(inst.upper())
        if not av:
            continue
        try:
            merged = refresh_sentiment(av, max_calls=2)
            result["news"][inst] = {"days": int(len(merged)),
                                    "last": round(float(merged.iloc[-1]), 4)}
        except Exception as e:
            result["news"][inst] = {"error": str(e)[:60]}

    # 2) log the Kalshi arb snapshot (sleeve track record)
    row = {"date": today}
    try:
        k = KalshiArbitrageAgent(min_edge_cents=0.3).scan(max_markets=1500)
        row["kalshi_locks"] = k.get("n_buy_both", 0)
        row["kalshi_best_net_c"] = max([x["net_profit_cents"] for x in k.get("buy_both", [])], default=0)
        row["markets_priced"] = k.get("priced", 0)
        result["kalshi"] = {"locks": row["kalshi_locks"], "priced": row["markets_priced"]}
    except Exception as e:
        row["kalshi_locks"] = None
        result["kalshi"] = {"error": str(e)[:60]}

    df = pd.DataFrame([row])
    if _LOG.exists():
        old = pd.read_csv(_LOG)
        old = old[old["date"].astype(str) != today]
        df = pd.concat([old, df], ignore_index=True)
    _LOG.parent.mkdir(exist_ok=True)
    df.to_csv(_LOG, index=False)
    result["log"] = str(_LOG)
    return result
