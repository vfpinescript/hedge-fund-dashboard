"""
Covariance hygiene: repair and shrink before any matrix is inverted or weighted.

- nearest_corr: Higham's nearest valid correlation matrix via Dykstra's
  alternating projections. A single negative eigenvalue means the matrix was
  never a real correlation matrix (pairs estimated on different samples), and
  inverting it turns tiny noisy eigenvalues into huge weights.
- shrunk_cov: Ledoit-Wolf shrinkage toward a scaled-identity target, which pulls
  the noisiest eigenvalues in so the optimiser does not pile into whatever got
  lucky in-sample. Always returned positive-semidefinite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def nearest_corr(C: np.ndarray, n_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """Nearest valid (PSD, unit-diagonal) correlation matrix to C (Higham 2002)."""
    C = np.asarray(C, dtype=float)
    n = C.shape[0]
    Y = C.copy()
    dS = np.zeros_like(C)
    for _ in range(n_iter):
        R = Y - dS
        # project onto PSD cone
        w, V = np.linalg.eigh((R + R.T) / 2)
        X = (V * np.clip(w, 0, None)) @ V.T
        dS = X - R
        Y_new = X.copy()
        np.fill_diagonal(Y_new, 1.0)            # project onto unit diagonal
        if np.linalg.norm(Y_new - Y, "fro") / max(np.linalg.norm(Y, "fro"), 1e-12) < tol:
            Y = Y_new; break
        Y = Y_new
    return Y


def shrunk_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance from a returns frame; PSD-guaranteed."""
    R = np.asarray(returns.dropna(), dtype=float)
    if R.shape[0] < 5 or R.shape[1] == 0:
        return np.cov(R.T) if R.size else np.zeros((R.shape[1], R.shape[1]))
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(R).covariance_
    except Exception:
        cov = np.cov(R.T)
    # PSD guard: clip any residual tiny-negative eigenvalues
    w, V = np.linalg.eigh((cov + cov.T) / 2)
    w = np.clip(w, 1e-12, None)
    return (V * w) @ V.T


def repair_and_shrink(returns: pd.DataFrame):
    """Shrunk covariance plus its repaired correlation matrix and diagnostics."""
    cov = shrunk_cov(returns)
    d = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    corr = cov / np.outer(d, d)
    raw_corr = returns.corr().values if returns.shape[1] > 1 else np.array([[1.0]])
    min_eig_raw = float(np.linalg.eigvalsh((raw_corr + raw_corr.T) / 2).min()) if raw_corr.size else 1.0
    fixed = nearest_corr(corr)
    return {"cov": cov, "corr": fixed,
            "was_valid": bool(min_eig_raw >= -1e-10),
            "min_eig_raw": round(min_eig_raw, 4)}
