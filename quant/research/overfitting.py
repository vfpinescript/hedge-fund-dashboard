"""
Overfitting diagnostics — is a backtest real, or did we torture the data?

Implements the three checks from the spec:

  ΔPerf  = Metric_IS − Metric_OOS         performance decay in/out of sample
  ΔError = Error_OOS / Error_IS           error inflation out of sample
  DSR    = Z[ (SR̂ − SR*) / σ̂_SR ]        Deflated Sharpe Ratio

The Deflated Sharpe Ratio (Bailey & López de Prado, 2014) is the important one:
it is the probability that the TRUE Sharpe is positive, after (a) correcting for
how many strategy variants were tried — SR* is the Sharpe you'd expect as the
MAXIMUM of N random trials — and (b) the non-normality (skew/kurtosis) of the
returns. DSR > 0.95 ≈ statistically robust; low DSR ≈ likely a false discovery
from data mining. This is exactly the guardrail an automated strategy-search
agent needs, so its "best" strategy is not just the luckiest of many trials.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, skew as _skew, kurtosis as _kurtosis

EULER = 0.5772156649015329


def perf_degradation(metric_is: float, metric_oos: float) -> float:
    """ΔPerf — how much the metric dropped out of sample (higher = worse)."""
    return float(metric_is - metric_oos)


def error_inflation(err_oos: float, err_is: float) -> float:
    """ΔError — out-of-sample error relative to in-sample (>1 = worse; ~1 = healthy)."""
    return float(err_oos / err_is) if err_is else float("inf")


def probabilistic_sharpe_ratio(sr, T, sr_benchmark=0.0, skew=0.0, kurt=3.0) -> float:
    """PSR — probability the true (per-observation) Sharpe exceeds sr_benchmark."""
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2)
    if denom <= 0 or T < 2:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(T - 1.0) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(sr_std: float, n_trials: int) -> float:
    """SR* — expected MAXIMUM per-observation Sharpe across N independent trials
    under the null of zero true skill (the bar a real strategy must clear)."""
    N = max(2, int(n_trials))
    return sr_std * ((1 - EULER) * norm.ppf(1 - 1.0 / N)
                     + EULER * norm.ppf(1 - 1.0 / (N * np.e)))


def deflated_sharpe_ratio(returns, n_trials: int, freq: int = 252,
                          sr_trials_std: float | None = None) -> dict:
    """Full DSR for a return series, deflated for `n_trials` strategy variants."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    T = len(r)
    if T < 8 or r.std() == 0:
        return {"dsr": float("nan"), "verdict": "insufficient data", "n_trials": n_trials, "T": T}

    sr = r.mean() / r.std()                       # per-observation Sharpe
    sk = float(_skew(r))
    ku = float(_kurtosis(r, fisher=False))        # non-excess kurtosis

    # If cross-trial SR dispersion is unknown, use the SR estimator's own
    # standard error (standard PSR/DSR fallback).
    if sr_trials_std is None:
        sr_trials_std = np.sqrt(max(1e-12, (1 - sk * sr + (ku - 1) / 4 * sr ** 2)) / (T - 1))

    sr_star = expected_max_sharpe(sr_trials_std, n_trials)
    dsr = probabilistic_sharpe_ratio(sr, T, sr_benchmark=sr_star, skew=sk, kurt=ku)
    verdict = ("robust" if dsr > 0.95 else
               "inconclusive" if dsr > 0.60 else "likely overfit")
    return {"dsr": float(dsr), "verdict": verdict,
            "sharpe_ann": float(sr * np.sqrt(freq)),
            "sr_star_ann": float(sr_star * np.sqrt(freq)),
            "n_trials": int(n_trials), "skew": sk, "kurtosis": ku, "T": T}
