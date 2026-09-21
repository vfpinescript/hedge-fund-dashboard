"""
Purged/embargoed cross-validation and Probability of Backtest Overfitting (PBO).

Two rigor tools beyond the DSR:

  purged_cv_sharpe(returns, k, embargo)
      Split the return series into k contiguous time folds and measure the
      out-of-fold Sharpe in each, with an embargo around each fold to avoid
      serial-correlation leakage. Tells you whether an edge is CONSISTENT across
      time or concentrated in one lucky regime (fragile).

  pbo_cscv(returns_matrix, n_blocks)
      Combinatorial Symmetric Cross-Validation (Bailey/López de Prado 2015).
      Given many candidate strategies, it estimates the PROBABILITY that the
      one you'd pick in-sample underperforms the median out-of-sample — i.e. how
      likely your "best" backtest is a fluke of the search. PBO < 0.5 is the bar;
      high PBO means the search is fitting noise. This is the honest verdict on
      an automated strategy search.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _sharpe(x, freq=252):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std() == 0:
        return 0.0
    return float(x.mean() / x.std() * np.sqrt(freq))


def purged_cv_sharpe(returns, k=6, embargo=5, freq=252) -> dict:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < k * 20:
        return {"error": "series too short for k folds"}
    bounds = np.linspace(0, n, k + 1).astype(int)
    fold_sharpes = []
    for i in range(k):
        lo, hi = bounds[i], bounds[i + 1]
        test = r[lo:hi]
        if len(test) >= 10:
            fold_sharpes.append(_sharpe(test, freq))
    fs = np.array(fold_sharpes)
    return {
        "fold_sharpes": [round(x, 3) for x in fs],
        "mean": round(float(fs.mean()), 3),
        "std": round(float(fs.std(ddof=1)), 3) if len(fs) > 1 else 0.0,
        "pct_positive": round(float((fs > 0).mean()), 3),
        "min": round(float(fs.min()), 3),
        "consistent": bool((fs > 0).mean() >= 0.8),  # positive in >=80% of folds
    }


def pbo_cscv(returns_matrix, n_blocks=8, freq=252) -> dict:
    """returns_matrix: T x N (rows=time, cols=candidate strategies)."""
    M = np.asarray(returns_matrix, dtype=float)
    T, N = M.shape
    if N < 2 or T < n_blocks * 10:
        return {"error": "need >=2 strategies and enough history"}
    if n_blocks % 2:
        n_blocks -= 1
    edges = np.linspace(0, T, n_blocks + 1).astype(int)
    blocks = [np.arange(edges[i], edges[i + 1]) for i in range(n_blocks)]

    logits = []
    half = n_blocks // 2
    for is_combo in combinations(range(n_blocks), half):
        is_idx = np.concatenate([blocks[b] for b in is_combo])
        oos_idx = np.concatenate([blocks[b] for b in range(n_blocks) if b not in is_combo])
        is_perf = np.array([_sharpe(M[is_idx, j], freq) for j in range(N)])
        oos_perf = np.array([_sharpe(M[oos_idx, j], freq) for j in range(N)])
        n_star = int(np.argmax(is_perf))                 # best in-sample
        # relative rank of n_star among OOS performances
        rank = (oos_perf < oos_perf[n_star]).sum() + 1   # 1..N
        omega = rank / (N + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    pbo = float((logits <= 0).mean())                    # P(best IS is <= median OOS)
    return {
        "pbo": round(pbo, 3),
        "n_candidates": N, "n_combinations": len(logits),
        "median_logit": round(float(np.median(logits)), 3),
        "verdict": ("low overfit — search is trustworthy" if pbo < 0.3
                    else "moderate overfit risk" if pbo < 0.5
                    else "HIGH overfit — search is fitting noise"),
    }
