"""
Signal-combination engine — combine uncorrelated signals into one book.

The diversification math: if two signals each have Sharpe s and correlation ρ,
an equal blend has Sharpe  s·sqrt(2 / (1+ρ)).  At ρ=0 that's a sqrt(2)≈1.41×
lift; at ρ=1 no benefit. So the whole game is finding signals with real edge
that are *uncorrelated with each other* — this module measures that correlation
explicitly and blends the return streams accordingly.

Each signal becomes a cross-instrument equal-risk return stream (via
candidate_returns). Streams are then combined equal-weight, inverse-vol, or
risk-parity. The report shows the correlation matrix (proof they're
uncorrelated), the per-signal Sharpe, and the combined Sharpe + DSR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import reversal, tsmom, trend, vol_scaled_mom
from .strategy_search import candidate_returns
from .metrics import perf_metrics
from .overfitting import deflated_sharpe_ratio

# A default set of economically-distinct signals (different mechanisms).
DEFAULT_SIGNALS = [
    ("reversal_7", reversal, {"window": 7}, 7),
    ("reversal_3", reversal, {"window": 3}, 3),
    ("tsmom_12m", tsmom, {"lookback": 252}, 21),
    ("trend_100", trend, {"window": 100}, 21),
    ("vol_mom", vol_scaled_mom, {}, 21),
]


def signal_streams(specs, universe, cost_bps=1.0) -> pd.DataFrame:
    """One daily-return column per signal (its equal-risk portfolio)."""
    cols = {}
    for name, fn, params, hold in specs:
        try:
            cols[name] = candidate_returns(fn, params, universe, hold, cost_bps)
        except Exception:
            continue
    return pd.DataFrame(cols)


def combine(streams: pd.DataFrame, method="inverse_vol") -> tuple[pd.Series, pd.Series]:
    """Combine signal streams. Returns (combined daily series, weights)."""
    S = streams.dropna(how="any")                     # common period, all signals live
    if S.shape[1] == 0 or len(S) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if method == "equal":
        w = pd.Series(1.0 / S.shape[1], index=S.columns)
    elif method == "risk_parity":
        cov = S.cov()
        w = pd.Series(1.0, index=S.columns)
        for _ in range(50):                           # simple iterative risk parity
            mrc = cov.values @ w.values
            w = w * (1.0 / mrc)
            w = w / w.sum()
        w = pd.Series(w.values, index=S.columns)
    else:  # inverse_vol
        vol = S.std()
        w = (1.0 / vol)
        w = w / w.sum()
    combined = (S * w).sum(axis=1)
    return combined, w


def blend_report(specs=DEFAULT_SIGNALS, universe=None, method="inverse_vol",
                 n_trials=1, cost_bps=1.0) -> dict:
    if universe is None:
        universe = ["EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "USDCAD",
                    "XAUUSD", "XAGUSD", "NATGAS"]
    streams = signal_streams(specs, universe, cost_bps)
    if streams.shape[1] == 0:
        return {"error": "no signal streams built"}
    combined, w = combine(streams, method)
    corr = streams.corr()
    per_signal = {c: round(perf_metrics(streams[c].dropna()).get("sharpe", 0), 3)
                  for c in streams.columns}
    m = perf_metrics(combined)
    dsr = deflated_sharpe_ratio(combined.values, n_trials)
    avg_corr = float(corr.where(~np.eye(len(corr), dtype=bool)).stack().mean())
    return {
        "method": method, "universe_size": len(universe),
        "signals": list(streams.columns),
        "weights": {c: round(float(w.get(c, 0)), 3) for c in streams.columns},
        "per_signal_sharpe": per_signal,
        "avg_pairwise_corr": round(avg_corr, 3),
        "corr_matrix": {c: {d: round(float(corr.loc[c, d]), 2) for d in corr.columns}
                        for c in corr.index},
        "combined_sharpe": round(m.get("sharpe", 0), 3),
        "combined_total_return": round(m.get("total_return", 0) * 100, 1),
        "combined_max_dd": round(m.get("max_drawdown", 0) * 100, 1),
        "combined_sortino": round(m.get("sortino", 0), 3),
        "dsr": round(dsr["dsr"], 3), "dsr_verdict": dsr["verdict"],
        "sr_star": round(dsr["sr_star_ann"], 2),
        "_combined_returns": combined,     # kept for downstream use (not JSON)
    }
