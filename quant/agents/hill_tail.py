"""
Hill-estimator tail-risk agent  (Quant_Analysis: "Hill estimator / break Gaussian VaR").

Fits a power law to the far left tail of the return distribution and reports the
tail index alpha. Role: RISK. It never picks a direction — it scales size down
when tails are fat and vetoes when variance is effectively infinite (alpha < 2).

alpha_hat_k = [ (1/k) * sum_{i=1..k} ln( X_(i) / X_(k+1) ) ]^-1
where X_(1) >= X_(2) >= ... are the sorted loss magnitudes.

Equity tail indices usually land in 3..5 (variance finite, kurtosis not).
alpha < 3  -> heavy: 4th moment blows up -> shrink size.
alpha < 2  -> variance itself is infinite -> VETO.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import RiskAgent, RiskSignal


def hill_alpha(losses: np.ndarray, k: int) -> float:
    """Hill estimator of the tail index from the largest k losses (>0)."""
    x = np.sort(losses)[::-1]          # descending
    x = x[x > 0]
    if len(x) <= k or k < 2:
        return float("nan")
    logs = np.log(x[:k]) - np.log(x[k])
    return 1.0 / logs.mean()


def stable_alpha(losses: np.ndarray, k_lo_frac=0.02, k_hi_frac=0.12) -> tuple[float, dict]:
    """Estimate alpha over a range of k and take the median where the Hill plot
    is flattest — the screenshots' point that choosing k is the real work."""
    n = len(losses)
    k_lo = max(10, int(n * k_lo_frac))
    k_hi = max(k_lo + 5, int(n * k_hi_frac))
    ks = np.arange(k_lo, min(k_hi, n - 1))
    alphas = np.array([hill_alpha(losses, int(k)) for k in ks])
    alphas = alphas[np.isfinite(alphas)]
    if len(alphas) == 0:
        return float("nan"), {}
    return float(np.median(alphas)), {
        "k_range": [int(ks[0]), int(ks[-1])],
        "alpha_min": float(np.nanmin(alphas)),
        "alpha_max": float(np.nanmax(alphas)),
    }


def gaussian_var(returns: np.ndarray, conf: float = 0.99) -> float:
    from scipy.stats import norm
    mu, sd = returns.mean(), returns.std()
    return float(-(mu + norm.ppf(1 - conf) * sd))   # positive loss number


def historical_var(returns: np.ndarray, conf: float = 0.99) -> float:
    return float(-np.quantile(returns, 1 - conf))


class HillTailAgent(RiskAgent):
    name = "hill_tail"

    def __init__(self, min_bars: int = 500):
        self.min_bars = min_bars

    def evaluate(self, instrument: str, ohlcv: pd.DataFrame, **kwargs) -> RiskSignal:
        if ohlcv is None or len(ohlcv) < self.min_bars:
            return self._abstain(instrument, f"need >= {self.min_bars} bars")

        rets = ohlcv["close"].pct_change().dropna().to_numpy()
        losses = -rets[rets < 0]                    # magnitudes of down moves
        if len(losses) < 100:
            return self._abstain(instrument, "too few loss observations")

        alpha, diag = stable_alpha(losses)
        if not np.isfinite(alpha):
            return self._abstain(instrument, "alpha not estimable")

        g_var = gaussian_var(rets)
        h_var = historical_var(rets)
        var_ratio = h_var / g_var if g_var > 0 else float("nan")

        # Map alpha -> risk_scale / veto.
        veto = alpha < 2.0
        if alpha >= 4.0:
            scale = 1.0
        elif alpha >= 3.0:
            scale = 0.75
        elif alpha >= 2.0:
            scale = 0.5
        else:
            scale = 0.0

        why = (
            f"tail index alpha={alpha:.2f} (k={diag.get('k_range')}); "
            f"99% VaR hist/gauss={var_ratio:.2f}x"
            + ("  -> VETO: variance effectively infinite" if veto else "")
        )
        return RiskSignal(
            self.name, instrument, risk_scale=scale, veto=veto, rationale=why,
            diagnostics={"alpha": alpha, "gaussian_var": g_var,
                         "historical_var": h_var, "var_ratio": var_ratio, **diag},
        )
