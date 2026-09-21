"""
Strategy library — the registry behind the dashboard's Strategy Builder page.

Each entry is a named strategy (a signal function + parameters). The page lets
you point any strategy at any instrument (stock, ETF, bond, FX, crypto,
commodity, index) and backtests it on the spot, TradingView-style. When you
describe a new strategy, it gets added here as one more registry entry and
immediately shows up on the page.

`run_strategy(id, instrument)` backtests one strategy on one instrument and
returns the full metric suite, the equity curve vs buy-and-hold, and the
Deflated Sharpe Ratio so you can see at a glance whether a result is real or
likely noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import load_ohlcv, SYMBOL_MAP
from .signals import reversal, tsmom, trend, vol_scaled_mom
from .metrics import perf_metrics
from .overfitting import deflated_sharpe_ratio

# --- the strategy registry (extended as the user describes new strategies) ---
STRATEGIES = {
    "reversal_7": {"name": "Short-Term Reversal (7d)", "family": "mean-reversion",
                   "desc": "Buy recent 7-day losers, short recent winners — fade the move.",
                   "fn": reversal, "params": {"window": 7}, "hold": 7},
    "reversal_3": {"name": "Short-Term Reversal (3d)", "family": "mean-reversion",
                   "desc": "Faster 3-day reversal. Higher turnover.",
                   "fn": reversal, "params": {"window": 3}, "hold": 3},
    "tsmom_12m": {"name": "Time-Series Momentum (12m)", "family": "momentum",
                  "desc": "Go with the 12-month trend — the managed-futures premium.",
                  "fn": tsmom, "params": {"lookback": 252}, "hold": 21},
    "tsmom_3m": {"name": "Time-Series Momentum (3m)", "family": "momentum",
                 "desc": "Shorter 3-month momentum.",
                 "fn": tsmom, "params": {"lookback": 63}, "hold": 21},
    "trend_100": {"name": "Trend Following (100d MA)", "family": "trend",
                  "desc": "Long above the 100-day average, short below.",
                  "fn": trend, "params": {"window": 100}, "hold": 21},
    "trend_200": {"name": "Trend Following (200d MA)", "family": "trend",
                  "desc": "Slower 200-day trend filter.",
                  "fn": trend, "params": {"window": 200}, "hold": 21},
    "vol_mom": {"name": "Volatility-Scaled Momentum", "family": "momentum",
                "desc": "12-month momentum normalised by recent volatility.",
                "fn": vol_scaled_mom, "params": {}, "hold": 21},
}

# instrument universe, grouped for the picker (+ any yfinance ticker works)
INSTRUMENT_GROUPS = {
    "FX": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD",
           "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF"],
    "Commodities": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "USOIL", "UKOIL",
                    "NATGAS", "CORN", "WHEAT", "SOYBEAN", "COFFEE", "SUGAR"],
    "Crypto": ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD"],
    "Indices": ["ES1!", "NQ1!", "SPX", "NDX", "RTY1!", "DAX", "NIKKEI", "FTSE", "ESTX50", "HSI"],
    "Rates/Bonds": ["US10Y", "US02Y", "US30Y", "ZN1!", "ZB1!"],
}


def _strategy_returns(spec, instrument, cost_bps=1.0, bars=3000, vol_win=63):
    """One strategy on one instrument: vol-scaled, rebalanced at `hold`."""
    df = load_ohlcv(instrument, bars=bars)
    ret = df["close"].pct_change()
    sig = spec["fn"](df, **spec["params"])
    vol = ret.rolling(vol_win).std()
    hold = spec["hold"]
    raw = np.sign(sig) / vol
    reb = pd.Series(np.nan, index=df.index)
    reb.iloc[::hold] = raw.iloc[::hold]
    pos = reb.ffill()
    pos = pos / pos.abs().rolling(252, min_periods=20).median()
    strat = pos.shift(1) * ret - (pos - pos.shift(1)).abs() * (cost_bps / 1e4)
    return strat.dropna(), ret


def run_strategy(strategy_id, instrument, cost_bps=1.0) -> dict:
    spec = STRATEGIES.get(strategy_id)
    if not spec:
        return {"error": f"unknown strategy {strategy_id}"}
    try:
        strat, ret = _strategy_returns(spec, instrument, cost_bps)
    except Exception as e:
        return {"error": f"could not load {instrument}: {e}"}
    if len(strat) < 250:
        return {"error": f"insufficient history for {instrument}"}

    m = perf_metrics(strat)
    dsr = deflated_sharpe_ratio(strat.values, n_trials=len(STRATEGIES))
    eq = (1 + strat).cumprod()
    bh = (1 + ret.reindex(strat.index).fillna(0)).cumprod()
    return {
        "strategy": spec["name"], "instrument": instrument,
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "dsr": round(dsr["dsr"], 3), "dsr_verdict": dsr["verdict"],
        "equity": {"dates": [d.strftime("%Y-%m-%d") for d in eq.index],
                   "strategy": (eq.values * 100 - 100).round(2).tolist(),
                   "buyhold": (bh.values * 100 - 100).round(2).tolist()},
    }


def library():
    return {
        "strategies": [{"id": k, **{f: v[f] for f in ("name", "family", "desc")}}
                       for k, v in STRATEGIES.items()],
        "instrument_groups": INSTRUMENT_GROUPS,
    }
