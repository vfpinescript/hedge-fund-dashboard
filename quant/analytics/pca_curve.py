"""
Yield-curve PCA  (Quant_Analysis: "4. PCA").

Run PCA on daily CHANGES of the Treasury curve (never levels — the screenshots'
explicit rule). The first three components arrive as level, slope and curvature
and together explain almost all daily variation, so a fixed-income desk hedges
three risks instead of ten correlated tenors. Role: ANALYTICS / fixed-income
relative value — not a directional committee vote.

Free data: yfinance tenors (13wk, 5y, 10y, 30y). Add FiscalData/FRED later for
the full tenor set (3m/6m/1y/2y/3y/7y/20y).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def curve_pca(n_components: int = 3, bars: int = 1000) -> dict:
    from ..data.loader import load_yield_curve
    curve = load_yield_curve(bars=bars)
    if curve.shape[1] < n_components:
        n_components = curve.shape[1]
    changes = curve.diff().dropna()
    pca = PCA(n_components=n_components).fit(changes.values)

    names = ["level", "slope", "curvature", "pc4", "pc5"][:n_components]
    loadings = {names[i]: dict(zip([f"{t:g}y" for t in curve.columns],
                                   np.round(pca.components_[i], 3)))
                for i in range(n_components)}
    latest_scores = pca.transform(changes.values[-1:])[0]
    return {
        "tenors_years": [float(t) for t in curve.columns],
        "explained_variance_ratio": [round(float(x), 4) for x in pca.explained_variance_ratio_],
        "factor_names": names,
        "loadings": loadings,
        "latest_daily_scores": {names[i]: float(latest_scores[i]) for i in range(n_components)},
        "current_curve_bps": {f"{t:g}y": round(float(curve.iloc[-1][t]) * 100, 1) for t in curve.columns},
        "n_obs": int(len(changes)),
    }
