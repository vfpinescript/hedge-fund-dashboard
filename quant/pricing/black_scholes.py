"""
Black-Scholes pricing agent  (Quant_Analysis: "Black Scholes model").

Prices European options and returns analytic Greeks. Role: PRICING — no
direction. Used to fair-value the option the committee wants to trade and to
feed the Greeks book.

  C = S*N(d1) - K*e^{-rT}*N(d2)
  P = K*e^{-rT}*N(-d2) - S*N(-d1)
  d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T),   d2 = d1 - σ√T
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm

from ..base import PricingAgent, PricingResult


@dataclass
class BSInputs:
    S: float      # spot
    K: float      # strike
    T: float      # years to expiry
    r: float      # risk-free rate (annual, cont.)
    sigma: float  # volatility (annual)
    kind: str = "call"   # "call" | "put"
    q: float = 0.0       # continuous dividend / carry yield


def _d1_d2(p: BSInputs):
    vol_t = p.sigma * math.sqrt(p.T)
    d1 = (math.log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma ** 2) * p.T) / vol_t
    return d1, d1 - vol_t


def bs_price_greeks(p: BSInputs) -> dict:
    if p.T <= 0 or p.sigma <= 0:
        # Degenerate: intrinsic value, no time value.
        intrinsic = max(0.0, (p.S - p.K) if p.kind == "call" else (p.K - p.S))
        return {"price": intrinsic, "delta": float("nan"), "gamma": 0.0,
                "vega": 0.0, "theta": 0.0, "rho": 0.0}
    d1, d2 = _d1_d2(p)
    disc = math.exp(-p.r * p.T)
    dq = math.exp(-p.q * p.T)
    pdf = norm.pdf(d1)
    if p.kind == "call":
        price = p.S * dq * norm.cdf(d1) - p.K * disc * norm.cdf(d2)
        delta = dq * norm.cdf(d1)
        theta = (-p.S * dq * pdf * p.sigma / (2 * math.sqrt(p.T))
                 - p.r * p.K * disc * norm.cdf(d2)
                 + p.q * p.S * dq * norm.cdf(d1))
        rho = p.K * p.T * disc * norm.cdf(d2)
    else:
        price = p.K * disc * norm.cdf(-d2) - p.S * dq * norm.cdf(-d1)
        delta = -dq * norm.cdf(-d1)
        theta = (-p.S * dq * pdf * p.sigma / (2 * math.sqrt(p.T))
                 + p.r * p.K * disc * norm.cdf(-d2)
                 - p.q * p.S * dq * norm.cdf(-d1))
        rho = -p.K * p.T * disc * norm.cdf(-d2)
    gamma = dq * pdf / (p.S * p.sigma * math.sqrt(p.T))
    vega = p.S * dq * pdf * math.sqrt(p.T)
    return {"price": price, "delta": delta, "gamma": gamma,
            "vega": vega, "theta": theta, "rho": rho, "d1": d1, "d2": d2}


def implied_vol(target_price: float, p: BSInputs, tol=1e-6, max_iter=100) -> float:
    """Newton solve for implied vol; falls back to bisection on failure."""
    lo, hi = 1e-4, 5.0
    sigma = 0.2
    for _ in range(max_iter):
        trial = BSInputs(p.S, p.K, p.T, p.r, sigma, p.kind, p.q)
        g = bs_price_greeks(trial)
        diff = g["price"] - target_price
        if abs(diff) < tol:
            return sigma
        vega = g["vega"]
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if sigma <= lo or sigma >= hi:
            break
    # Bisection fallback
    lo, hi = 1e-4, 5.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        pm = bs_price_greeks(BSInputs(p.S, p.K, p.T, p.r, mid, p.kind, p.q))["price"]
        if abs(pm - target_price) < tol:
            return mid
        if pm > target_price:
            hi = mid
        else:
            lo = mid
    return float("nan")


class BlackScholesAgent(PricingAgent):
    name = "black_scholes"

    def price(self, instrument: str, **kwargs) -> PricingResult:
        p = BSInputs(**kwargs)
        g = bs_price_greeks(p)
        price = g.pop("price")
        return PricingResult(self.name, instrument, price=price, greeks=g,
                             diagnostics={"inputs": p.__dict__})
