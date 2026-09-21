"""
Greeks book  (Quant_Analysis: "options greeks" reference table).

Aggregates first- and second-order Greeks across a portfolio of options, each
priced with Black-Scholes, plus the cross-Greeks the table lists: vanna
(dVega/dSpot) and charm (dDelta/dTime). Role: PRICING / risk aggregation — it
tells you the book's net exposure to spot, vol, time and rates so you know what
you are actually long or short. No direction.

A position = {S,K,T,r,sigma,kind,q, qty}. qty>0 long, <0 short.
"""
from __future__ import annotations

import math
from scipy.stats import norm

from .black_scholes import BSInputs, bs_price_greeks


def _vanna_charm(p: BSInputs):
    """Analytic vanna and charm (per calendar day for charm)."""
    if p.T <= 0 or p.sigma <= 0:
        return 0.0, 0.0
    vt = p.sigma * math.sqrt(p.T)
    d1 = (math.log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma ** 2) * p.T) / vt
    d2 = d1 - vt
    dq = math.exp(-p.q * p.T)
    pdf = norm.pdf(d1)
    vanna = -dq * pdf * d2 / p.sigma
    charm_annual = -dq * pdf * (2 * (p.r - p.q) * p.T - d2 * vt) / (2 * p.T * vt)
    if p.q > 0:  # dividend term differs by option type
        charm_annual += (p.q * dq * norm.cdf(d1)) if p.kind == "call" else (-p.q * dq * norm.cdf(-d1))
    return vanna, charm_annual / 365.0


def aggregate_greeks(positions: list[dict]) -> dict:
    book = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0,
            "rho": 0.0, "vanna": 0.0, "charm": 0.0, "value": 0.0}
    for pos in positions:
        qty = pos.get("qty", 1)
        p = BSInputs(pos["S"], pos["K"], pos["T"], pos["r"], pos["sigma"],
                     pos.get("kind", "call"), pos.get("q", 0.0))
        g = bs_price_greeks(p)
        vanna, charm = _vanna_charm(p)
        book["value"] += qty * g["price"]
        for k in ("delta", "gamma", "vega", "theta", "rho"):
            book[k] += qty * g[k]
        book["vanna"] += qty * vanna
        book["charm"] += qty * charm
    return {k: float(v) for k, v in book.items()}
