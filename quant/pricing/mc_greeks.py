"""
Monte-Carlo pricer + Greeks  (Quant_Analysis: "Monte Carlo sensitivities").

Simulate terminal prices under GBM, discount the payoff, average. Greeks by
finite difference with COMMON RANDOM NUMBERS — the screenshots' point is that
bumping with shared draws lets MC noise cancel between the two runs, which is
what makes a bump delta usable. (True autodiff would need JAX, absent here; CRN
finite-difference is the robust no-JAX substitute and we note it as such.)

Role: PRICING. Validated in tests against Black-Scholes closed form.
"""
from __future__ import annotations

import numpy as np

from ..base import PricingAgent, PricingResult


def _terminal_prices(S0, r, sigma, T, Z, q=0.0):
    return S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)


def _payoff(ST, K, kind):
    return np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)


def mc_price_greeks(S0, K, T, r, sigma, kind="call", q=0.0,
                    n_paths=200_000, seed=0, hS=None, hSig=1e-4):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)          # fixed draws = common random numbers
    disc = np.exp(-r * T)
    hS = hS or S0 * 1e-3

    def price_at(S, sig):
        ST = _terminal_prices(S, r, sig, T, Z, q)
        return disc * _payoff(ST, K, kind).mean()

    price = price_at(S0, sigma)
    # Central differences with shared Z (noise cancels).
    delta = (price_at(S0 + hS, sigma) - price_at(S0 - hS, sigma)) / (2 * hS)
    gamma = (price_at(S0 + hS, sigma) - 2 * price + price_at(S0 - hS, sigma)) / (hS ** 2)
    vega = (price_at(S0, sigma + hSig) - price_at(S0, sigma - hSig)) / (2 * hSig)
    stderr = disc * _payoff(_terminal_prices(S0, r, sigma, T, Z, q), K, kind).std() / np.sqrt(n_paths)
    return {"price": float(price), "delta": float(delta), "gamma": float(gamma),
            "vega": float(vega), "std_error": float(stderr)}


class MCGreeksAgent(PricingAgent):
    name = "mc_greeks"

    def price(self, instrument: str, **kw) -> PricingResult:
        g = mc_price_greeks(kw["S"], kw["K"], kw["T"], kw["r"], kw["sigma"],
                            kind=kw.get("kind", "call"), q=kw.get("q", 0.0),
                            n_paths=kw.get("n_paths", 200_000), seed=kw.get("seed", 0))
        price = g.pop("price")
        return PricingResult(self.name, instrument, price=price, greeks=g,
                             diagnostics={"method": "MC + common-random-number finite diff"})
