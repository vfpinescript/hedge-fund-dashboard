"""
Robust data-quality & anomaly detection.

These are the outlier detectors from the "market chaos" screenshots, implemented
honestly for what they actually are — data cleaning and anomaly flagging, NOT
alpha generation:

  1. GARCH(1,1) conditional-vol filter   sigma_t^2 = w + a*eps_{t-1}^2 + b*sigma_{t-1}^2
     Eliminate / winsorize a return if  |R_t| > kappa * sigma_t.
     Catches bad ticks and extreme prints relative to the *current* vol regime,
     not a flat full-sample stdev.

  2. MCD Mahalanobis distance            D^2(x) = (x - mu_mcd)^T Sigma_mcd^{-1} (x - mu_mcd)
     Robust (Minimum Covariance Determinant) multivariate outlier distance across
     a cross-section of assets. High distance = anomalous *joint* move.

  3. Local Outlier Factor                LOF_k(p) = mean_{o in N_k(p)} lrd_k(o)/lrd_k(p)
     Density-based: flags points in locally sparse regions of feature space.

A consensus combines the three so no single detector's quirk dominates. GARCH is
fit directly with scipy (no `arch` dependency); it falls back to RiskMetrics EWMA
vol if the MLE does not converge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZE = np.sqrt(252)


# --------------------------------------------------------------------------- #
# 1. GARCH(1,1) conditional volatility + vol-scaled outlier filter
# --------------------------------------------------------------------------- #
def _garch_filter(eps2, w, a, b):
    """Run the GARCH(1,1) variance recursion for given params. Returns sigma^2_t."""
    n = len(eps2)
    s2 = np.empty(n)
    s2[0] = eps2.mean() if eps2.mean() > 0 else 1e-8
    for t in range(1, n):
        s2[t] = w + a * eps2[t - 1] + b * s2[t - 1]
    return s2


def _garch_negloglik(theta, eps2):
    w, a, b = theta
    if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
        return 1e12
    s2 = _garch_filter(eps2, w, a, b)
    s2 = np.clip(s2, 1e-12, None)
    return 0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + eps2 / s2)


def conditional_vol(returns, method="garch"):
    """Conditional (time-varying) volatility series for a return series.

    method="garch": fit GARCH(1,1) by MLE (scipy), fall back to EWMA on failure.
    method="ewma":  RiskMetrics EWMA (lambda=0.94), cheap and robust.
    Returns a pd.Series of sigma_t aligned to `returns` (same index).
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 30:
        # too short to estimate: fall back to trailing stdev (or NaN)
        s = r.rolling(10, min_periods=3).std().bfill()
        return s.reindex(pd.Series(returns).index)

    eps = (r - r.mean()).values

    if method == "garch":
        try:
            from scipy.optimize import minimize
            scale = 100.0 / (np.std(eps) + 1e-12)      # condition the optimiser
            e2 = (eps * scale) ** 2
            var0 = e2.mean()
            x0 = [var0 * 0.05, 0.08, 0.90]
            res = minimize(_garch_negloglik, x0, args=(e2,), method="Nelder-Mead",
                           options={"maxiter": 800, "xatol": 1e-4, "fatol": 1e-4})
            w, a, b = res.x
            if res.success or (w > 0 and a >= 0 and b >= 0 and a + b < 0.999):
                s2 = _garch_filter(e2, w, a, b) / (scale ** 2)
                sig = pd.Series(np.sqrt(np.clip(s2, 1e-16, None)), index=r.index)
                return sig.reindex(pd.Series(returns).index)
        except Exception:
            pass  # fall through to EWMA

    # EWMA (RiskMetrics) fallback
    lam = 0.94
    s2 = np.empty(len(eps))
    s2[0] = np.var(eps)
    for t in range(1, len(eps)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * eps[t - 1] ** 2
    sig = pd.Series(np.sqrt(np.clip(s2, 1e-16, None)), index=r.index)
    return sig.reindex(pd.Series(returns).index)


def garch_outliers(returns, kappa=4.0, method="garch"):
    """Boolean mask of returns exceeding kappa conditional sigmas: |R_t| > kappa*sigma_t.
    Returns (mask, sigma) aligned to `returns`."""
    r = pd.Series(returns).astype(float)
    sig = conditional_vol(r, method=method)
    mask = (r.abs() > kappa * sig) & sig.notna()
    return mask.fillna(False), sig


def clean_series(close, kappa=8.0, method="ewma"):
    """Repair isolated bad ticks in a PRICE series by neighbour interpolation.

    A single corrupt print (e.g. SPY 738 -> 420 -> 738) is an ISOLATED spike:
    the price jumps far from BOTH its neighbours and reverts. We detect such a
    point by its deviation from the mean of its two neighbours in log space, and
    when that deviation exceeds `kappa` conditional sigmas we replace the point
    with the neighbour mean. Deliberately conservative (kappa=8 ~ a 1-in-10^15
    event under normality) so it only removes data errors and leaves real moves,
    including crashes and trends, intact.

    This replaces the earlier return-winsorizing version, which had a serious
    bug: a bad down-tick inflated the EWMA vol estimate so the mirror-image
    recovery tick slipped under the bound; only the down-move was capped and the
    recovery compounded, permanently DOUBLING every subsequent price. Repairing
    the single point instead leaves both neighbours' returns intact, so the level
    can never shift. Returns the cleaned price series plus the count of points
    touched.
    """
    s = pd.Series(close).astype(float)
    if len(s) < 40:
        return s, 0
    logp = np.log(s)
    logret = logp.diff()
    sig = conditional_vol(logret, method=method)
    neigh = 0.5 * (logp.shift(1) + logp.shift(-1))     # local linear interpolation
    dev = (logp - neigh).abs()                         # isolated-spike deviation
    bound = kappa * sig
    bad = ((dev > bound) & bound.notna() & (bound > 0)
           & logp.shift(1).notna() & logp.shift(-1).notna())   # interior points only
    if not bad.any():
        return s, 0
    repaired = logp.copy()
    repaired[bad] = neigh[bad]
    return np.exp(repaired), int(bad.sum())


# --------------------------------------------------------------------------- #
# 2. MCD (robust) Mahalanobis distance
# --------------------------------------------------------------------------- #
def mcd_mahalanobis(X, support_fraction=None):
    """Robust Mahalanobis distance of each row of X using a Minimum Covariance
    Determinant estimate of location/scatter. Returns (distances, threshold),
    where threshold is the sqrt of the 97.5% chi-square quantile with p dof
    (the conventional outlier cutoff). Falls back to the classical estimate if
    MCD fails (e.g. singular support)."""
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    n, p = X.shape
    from scipy.stats import chi2
    thresh = float(np.sqrt(chi2.ppf(0.975, p)))
    if n < max(2 * p + 2, 20):
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(p) * 1e-8
        d = np.sqrt(np.einsum("ij,jk,ik->i", X - mu, np.linalg.pinv(cov), X - mu))
        return d, thresh
    try:
        from sklearn.covariance import MinCovDet
        mcd = MinCovDet(support_fraction=support_fraction, random_state=0).fit(X)
        d = np.sqrt(mcd.mahalanobis(X))
        return d, thresh
    except Exception:
        mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(p) * 1e-8
        d = np.sqrt(np.einsum("ij,jk,ik->i", X - mu, np.linalg.pinv(cov), X - mu))
        return d, thresh


# --------------------------------------------------------------------------- #
# 3. Local Outlier Factor
# --------------------------------------------------------------------------- #
def lof_scores(X, k=20):
    """Local Outlier Factor per row. Returns (lof, is_outlier). lof ~1 is inlier,
    >1 is increasingly outlying; sklearn's negative_outlier_factor_ is -lof."""
    X = np.asarray(X, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    n = len(X)
    if n < k + 2:
        return np.ones(n), np.zeros(n, dtype=bool)
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=min(k, n - 1))
    pred = lof.fit_predict(X)                 # -1 outlier, 1 inlier
    scores = -lof.negative_outlier_factor_    # LOF_k(p)
    return scores, (pred == -1)


# --------------------------------------------------------------------------- #
# Consensus + market-level anomaly score
# --------------------------------------------------------------------------- #
def consensus_outliers(X, kappa=4.0, k=20):
    """Combine the three detectors on a feature matrix X (rows = observations).
    Column 0 is treated as the primary return for the GARCH filter. Returns a
    DataFrame with each detector's flag and a 'votes' count (0..3)."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    garch_mask, _ = garch_outliers(X[:, 0], kappa=kappa)
    garch_mask = np.asarray(garch_mask)[:n]
    d, dthr = mcd_mahalanobis(X)
    mcd_mask = d > dthr if len(d) == n else np.zeros(n, dtype=bool)
    lof, lof_mask = lof_scores(X, k=k)
    lof_mask = lof_mask if len(lof_mask) == n else np.zeros(n, dtype=bool)
    votes = garch_mask.astype(int) + mcd_mask.astype(int) + lof_mask.astype(int)
    return pd.DataFrame({"garch": garch_mask, "mcd": mcd_mask,
                         "lof": lof_mask, "votes": votes})


def market_anomaly(returns_panel, window=252):
    """Cross-sectional market anomaly score from the robust Mahalanobis distance
    of the most recent joint return vector against a trailing window.

    Returns a dict with the latest robust distance, the chi-square outlier
    threshold, a 0..1 `score` (chi-square CDF of D^2 — how extreme today's joint
    move is), and a recent time series of the distance. Report-only: a high score
    means 'today is statistically unusual across the book', a de-risk signal that
    complements the jump-model regime.
    """
    df = pd.DataFrame(returns_panel).dropna(how="all")
    df = df.dropna(axis=1, thresh=int(len(df) * 0.6)).dropna()
    if df.shape[0] < 60 or df.shape[1] < 2:
        return {"ok": False, "reason": "need >=60 aligned rows and >=2 assets"}
    X = df.iloc[-window:].values
    p = X.shape[1]
    d, thr = mcd_mahalanobis(X)
    from scipy.stats import chi2
    dates = df.index[-len(d):]
    series = [{"date": str(pd.Timestamp(t).date()), "dist": round(float(x), 2)}
              for t, x in zip(dates, d)]
    latest = float(d[-1])
    return {
        "ok": True, "assets": list(df.columns), "p": p,
        "latest_dist": round(latest, 2), "threshold": round(float(thr), 2),
        "score": round(float(chi2.cdf(latest ** 2, p)), 3),
        "is_anomalous": bool(latest > thr),
        "series": series[-window:],
    }
