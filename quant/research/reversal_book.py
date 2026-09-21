"""
Multi-asset reversal book — the reversal_7 signal extended beyond FX to metals,
gas, and index futures (the instruments that showed positive reversal IC).
Crude oil is EXCLUDED: it trends (negative IC), so reversal loses there.

Each instrument is a sub-strategy: vol-scaled position = sign(reversal_7),
rebalanced every 7 days, held between. Returns are net of a per-side cost.
The portfolio is the equal-risk (gross-normalised) combination.

This is the vectorised engine — uniform across asset classes and validated to
agree with the nautilus FX result (~0.5 Sharpe). Use nautilus for FX
execution-realism; use this for multi-asset breadth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import load_ohlcv
from .signals import reversal
from .metrics import perf_metrics

# Instruments with positive reversal IC (crude excluded).
BOOK = {
    "FX": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"],
    "Metals": ["XAUUSD", "XAGUSD"],
    "Energy": ["NATGAS"],
    "Equity": ["ES1!", "NQ1!"],
}


def _instrument_series(sym, lookback=7, vol_win=63, cost_bps=1.0):
    df = load_ohlcv(sym, bars=3000)
    ret = df["close"].pct_change()
    sig = reversal(df, window=lookback)
    vol = ret.rolling(vol_win).std()
    raw = np.sign(sig) / vol                        # vol-scaled (risk parity)
    # rebalance every `lookback` days, hold between
    reb = pd.Series(np.nan, index=df.index)
    reb.iloc[::lookback] = raw.iloc[::lookback]
    pos = reb.ffill()
    pos = pos / pos.abs().rolling(252, min_periods=20).median()   # keep unit-ish scale
    strat = pos.shift(1) * ret
    turnover = (pos - pos.shift(1)).abs()
    strat_net = strat - turnover * (cost_bps / 1e4)
    return pos, strat_net.dropna(), ret


def build_book(cost_bps=1.0):
    """Return {strategy_name: metrics} for each instrument, each asset class,
    and the full portfolio."""
    results = {}
    daily = {}
    positions = {}
    for cls, syms in BOOK.items():
        for sym in syms:
            try:
                pos, strat, _ = _instrument_series(sym, cost_bps=cost_bps)
            except Exception as e:
                results[sym] = {"error": str(e)}
                continue
            daily[sym] = strat
            positions[sym] = pos
            results[sym] = perf_metrics(strat, pos)

    # Asset-class portfolios + full portfolio (equal-risk daily normalisation).
    def combine(symlist, name):
        cols = [daily[s] for s in symlist if s in daily]
        if not cols:
            return
        D = pd.concat(cols, axis=1).dropna(how="all")
        P = pd.concat([positions[s] for s in symlist if s in positions], axis=1).reindex(D.index)
        w = P.div(P.abs().sum(axis=1), axis=0).fillna(0)     # equal-risk weights
        # portfolio daily return = weighted mean of instrument strat returns
        port = (D.fillna(0) * (1.0 / D.notna().sum(axis=1).clip(lower=1)).values[:, None]).sum(axis=1)
        results[name] = perf_metrics(port)
        daily[name] = port

    for cls, syms in BOOK.items():
        combine(syms, f"[{cls}]")
    combine([s for syms in BOOK.values() for s in syms], "[PORTFOLIO]")
    return results, daily
