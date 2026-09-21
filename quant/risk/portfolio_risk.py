"""
Portfolio-level risk aggregation.

The per-trade risk engine (risk.js) sizes and vets one position. This aggregates
risk across the WHOLE book — the thing that actually blows accounts up when
positions are correlated. Given the current signed weights, it computes:

  - annualised portfolio volatility (accounting for the correlation matrix)
  - historical & Gaussian Value-at-Risk / Conditional VaR (95%, 99%)
  - marginal risk contribution per position (who is driving the risk)
  - diversification ratio (weighted avg vol / portfolio vol; 1 = no diversification)
  - the naive (sum of |risk|) vs true (correlation-adjusted) heat

Weights are signed (long +, short −) and are normalised to gross exposure 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import load_ohlcv

ANN = np.sqrt(252)


def _returns_panel(symbols, bars=500):
    cols = {}
    for s in symbols:
        try:
            cols[s] = load_ohlcv(s, bars=bars)["close"].pct_change()
        except Exception:
            continue
    return pd.DataFrame(cols).dropna()


def portfolio_risk(weights: dict, bars: int = 500, returns: pd.DataFrame | None = None) -> dict:
    """weights: {symbol: signed_weight}. Returns aggregated portfolio risk.
    `returns` (T x N return panel) may be supplied to skip the data fetch."""
    syms = [s for s, w in weights.items() if w]
    if len(syms) < 1:
        return {"error": "no positions"}
    R = returns[[s for s in syms if s in returns.columns]].dropna() if returns is not None \
        else _returns_panel(syms, bars)
    if R.shape[0] < 60 or R.shape[1] < 1:
        return {"error": "insufficient return history"}
    syms = list(R.columns)
    w = np.array([weights[s] for s in syms], dtype=float)
    gross = np.abs(w).sum()
    if gross == 0:
        return {"error": "zero gross exposure"}
    w = w / gross                                        # normalise to gross 1

    cov = R.cov().values * 252                           # annualised covariance
    vol_i = np.sqrt(np.diag(cov))                        # per-instrument ann vol
    port_var = float(w @ cov @ w)
    port_vol = float(np.sqrt(max(port_var, 0)))

    # portfolio daily P&L series (for historical VaR) — signed weighted returns
    port_ret = (R.values @ w)
    def hvar(conf):
        return float(-np.quantile(port_ret, 1 - conf))
    def cvar(conf):
        q = np.quantile(port_ret, 1 - conf)
        tail = port_ret[port_ret <= q]
        return float(-tail.mean()) if len(tail) else float("nan")
    from scipy.stats import norm
    gauss_var99 = float(-(port_ret.mean() + norm.ppf(0.01) * port_ret.std()))

    # marginal & component risk contributions
    mrc = (cov @ w) / port_vol if port_vol > 0 else np.zeros_like(w)   # d sigma / d w_i
    crc = w * mrc                                        # component contribution to vol
    crc_pct = crc / crc.sum() if crc.sum() != 0 else crc

    weighted_avg_vol = float(np.abs(w) @ vol_i)
    diversification_ratio = weighted_avg_vol / port_vol if port_vol > 0 else float("nan")

    contributions = sorted(
        [{"symbol": s, "weight": round(float(w[i]), 3),
          "vol": round(float(vol_i[i]), 3),
          "risk_contribution_pct": round(float(crc_pct[i]) * 100, 1)}
         for i, s in enumerate(syms)],
        key=lambda x: -abs(x["risk_contribution_pct"]))

    return {
        "n_positions": len(syms),
        "portfolio_vol_annual": round(port_vol * 100, 2),
        "var_95_daily": round(hvar(0.95) * 100, 3),
        "var_99_daily": round(hvar(0.99) * 100, 3),
        "cvar_99_daily": round(cvar(0.99) * 100, 3),
        "gaussian_var_99_daily": round(gauss_var99 * 100, 3),
        "diversification_ratio": round(diversification_ratio, 2),
        "naive_heat": round(weighted_avg_vol * 100, 2),
        "true_heat": round(port_vol * 100, 2),
        "contributions": contributions,
    }
