"""
Jump-model regime detection (Princeton 2024 style).

k-means on a few risk/return features, plus a fixed penalty `lam` on every state
switch. At lam=0 it is plain k-means (noisy, flips often); raising lam makes each
switch earn its cost, so the state sequence is stable and flips about once a year.
The optimal state path for a given set of cluster means is solved exactly by
dynamic programming (a Viterbi with a switching cost).

Strictly causal: `current_regime` infers today's state from data up to today only,
so there is no look-ahead. The turbulent state scales book exposure down.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = np.sqrt(252)


def _features(close: pd.Series) -> pd.DataFrame:
    r = close.pct_change()
    feats = pd.DataFrame({
        "ret20": r.rolling(20).mean() * 252,            # trailing drift
        "vol20": r.rolling(20).std() * ANN,             # trailing volatility
        "dd": close / close.cummax() - 1,               # drawdown depth
    }).dropna()
    return feats


def _dp_states(X: np.ndarray, mu: np.ndarray, lam: float) -> np.ndarray:
    """Optimal state sequence minimising sum ||x-mu_s||^2 + lam*#switches."""
    T, K = len(X), len(mu)
    cost = np.sum((X[:, None, :] - mu[None, :, :]) ** 2, axis=2)   # T x K
    V = np.empty((T, K)); back = np.zeros((T, K), dtype=int)
    V[0] = cost[0]
    for t in range(1, T):
        for s in range(K):
            trans = V[t - 1] + lam * (np.arange(K) != s)
            back[t, s] = int(np.argmin(trans))
            V[t, s] = cost[t, s] + trans[back[t, s]]
    s = np.empty(T, dtype=int); s[-1] = int(np.argmin(V[-1]))
    for t in range(T - 2, -1, -1):
        s[t] = back[t + 1, s[t + 1]]
    return s


def fit_jump_model(feats: pd.DataFrame, k: int = 2, lam: float = 20.0, iters: int = 30):
    """Standardise features, then alternate (means | optimal path) to convergence.
    Returns the state series and which state label is 'turbulent' (high vol)."""
    Z = (feats - feats.mean()) / feats.std()
    X = Z.values
    # init means by vol quantiles so labels are stable
    order = np.argsort(X[:, 1])
    mu = np.array([X[order[: len(X) // k]].mean(axis=0) if i == 0
                   else X[order[-len(X) // k:]].mean(axis=0) for i in range(k)])
    s = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        s_new = _dp_states(X, mu, lam)
        if np.array_equal(s_new, s):
            s = s_new; break
        s = s_new
        for j in range(k):
            if (s == j).any():
                mu[j] = X[s == j].mean(axis=0)
    turbulent = int(np.argmax([feats.values[s == j][:, 1].mean() if (s == j).any() else -1e9
                               for j in range(k)]))   # highest-vol cluster
    return pd.Series(s, index=feats.index), turbulent


def current_regime(close: pd.Series, lam: float = 20.0,
                   calm_scale: float = 1.0, turbulent_scale: float = 0.4) -> dict:
    feats = _features(close)
    if len(feats) < 120:
        return {"state": "unknown", "exposure_scale": calm_scale, "n_switches": None}
    states, turb = fit_jump_model(feats, lam=lam)
    switches = int((states.diff() != 0).sum())
    yrs = len(states) / 252.0
    cur = int(states.iloc[-1])
    is_turb = cur == turb
    return {
        "state": "turbulent" if is_turb else "calm",
        "exposure_scale": turbulent_scale if is_turb else calm_scale,
        "switches_per_year": round(switches / yrs, 2),
        "as_of_vol": round(float(feats["vol20"].iloc[-1]) * 100, 1),
        "as_of_drawdown": round(float(feats["dd"].iloc[-1]) * 100, 1),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, __file__.rsplit("/quant/", 1)[0])
    from quant.data.loader import load_ohlcv
    for sym in ["SPX", "NDX"]:
        df = load_ohlcv(sym, bars=3000)
        r = current_regime(df["close"])
        # count how many days were turbulent historically
        feats = _features(df["close"]); states, turb = fit_jump_model(feats)
        pct_turb = round(100 * (states == turb).mean(), 1)
        print(f"{sym}: now {r['state']}  scale x{r['exposure_scale']}  "
              f"{r['switches_per_year']} switches/yr  {pct_turb}% of days turbulent  "
              f"(vol {r['as_of_vol']}%, dd {r['as_of_drawdown']}%)")
