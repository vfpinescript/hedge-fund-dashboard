"""
3D surface generators — the visualisations from the original Quant_Analysis
screenshots, computed from real data / analytics (for Plotly Surface plots).

  - tail_exponent_surface : Hill alpha over (rolling window end x order-statistic k)
  - gamma_surface         : Black-Scholes gamma over (spot x time-to-maturity)
  - charm_surface         : Black-Scholes charm over (spot x time-to-maturity)
  - vol_surface           : implied vol over (strike x expiry) from a real chain
  - return_distribution   : return densities across horizons
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..agents.hill_tail import hill_alpha
from ..pricing.black_scholes import BSInputs, bs_price_greeks
from ..pricing.greeks_book import _vanna_charm


def tail_exponent_surface(returns: np.ndarray, n_windows=40, win=750):
    """Hill tail index alpha over (window-end time, order statistic k).
    Mirrors the screenshot's 'Tail exponent surface'."""
    losses_all = returns
    n = len(losses_all)
    ends = np.linspace(win, n, n_windows).astype(int)
    ks = np.arange(20, 220, 8)
    Z = np.full((len(ends), len(ks)), np.nan)
    for i, e in enumerate(ends):
        seg = losses_all[max(0, e - win):e]
        loss = -seg[seg < 0]
        if len(loss) < ks[-1] + 5:
            continue
        for j, k in enumerate(ks):
            Z[i, j] = hill_alpha(loss, int(k))
    Z = np.clip(Z, 1.5, 6.0)
    return {"x": ks.tolist(), "y": list(range(len(ends))), "z": np.nan_to_num(Z, nan=3.0).tolist(),
            "xlabel": "order statistic k", "ylabel": "window", "zlabel": "tail index a"}


def _greek_surface(which, S0=100.0, r=0.02, sigma=0.25, kind="call"):
    spots = np.linspace(S0 * 0.7, S0 * 1.3, 45)
    ttms = np.linspace(0.02, 1.5, 45)
    Z = np.zeros((len(ttms), len(spots)))
    for i, T in enumerate(ttms):
        for j, S in enumerate(spots):
            p = BSInputs(S, S0, T, r, sigma, kind)
            if which == "gamma":
                Z[i, j] = bs_price_greeks(p)["gamma"]
            elif which == "charm":
                _, charm = _vanna_charm(p)
                Z[i, j] = charm
    return {"x": spots.round(2).tolist(), "y": ttms.round(3).tolist(), "z": Z.tolist(),
            "xlabel": "spot", "ylabel": "time to maturity", "zlabel": which}


def gamma_surface(**kw):
    return _greek_surface("gamma", **kw)


def charm_surface(**kw):
    return _greek_surface("charm", **kw)


def vol_surface(underlying="SPY", n_expiries=6):
    """Implied-vol surface (strike x expiry) from a real option chain (yfinance).
    Returns None if no chain is available (agent-style abstain)."""
    import yfinance as yf
    from ..data.loader import to_yahoo
    tk = yf.Ticker(to_yahoo(underlying))
    exps = tk.options
    if not exps:
        return None
    exps = exps[:n_expiries]
    spot = float(tk.history(period="1d")["Close"].iloc[-1])
    rows = []
    for ei, exp in enumerate(exps):
        try:
            calls = tk.option_chain(exp).calls
        except Exception:
            continue
        calls = calls[(calls["impliedVolatility"] > 0.01) & (calls["impliedVolatility"] < 3)]
        # keep strikes within +/-25% of spot
        calls = calls[(calls["strike"] > spot * 0.75) & (calls["strike"] < spot * 1.25)]
        for _, row in calls.iterrows():
            rows.append((ei, float(row["strike"]), float(row["impliedVolatility"]) * 100))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["ei", "strike", "iv"])
    strikes = np.sort(df["strike"].unique())
    Z = np.full((len(exps), len(strikes)), np.nan)
    for ei in range(len(exps)):
        sub = df[df["ei"] == ei]
        for _, r in sub.iterrows():
            j = int(np.searchsorted(strikes, r["strike"]))
            if j < len(strikes):
                Z[ei, j] = r["iv"]
    # forward/back fill across strikes for a smooth surface
    Zdf = pd.DataFrame(Z).ffill(axis=1).bfill(axis=1).ffill(axis=0).bfill(axis=0)
    return {"x": strikes.round(1).tolist(), "y": list(exps), "z": Zdf.values.tolist(),
            "xlabel": "strike", "ylabel": "expiry", "zlabel": "implied vol %",
            "spot": spot, "underlying": underlying}


def return_distribution(returns: np.ndarray, horizons=(1, 5, 10, 20, 60)):
    """Return densities across horizons (screenshot 'Return distribution across
    horizons'). Returns histogram data per horizon in units of std."""
    out = []
    for h in horizons:
        if h >= len(returns):
            continue
        agg = pd.Series(returns).rolling(h).sum().dropna().to_numpy()
        z = (agg - agg.mean()) / (agg.std() + 1e-12)
        hist, edges = np.histogram(z, bins=60, range=(-6, 6), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        out.append({"horizon": h, "x": centers.round(3).tolist(), "y": hist.round(4).tolist()})
    return out
