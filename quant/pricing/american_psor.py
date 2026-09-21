"""
American option pricing  (Quant_Analysis: "American put free boundary by
Crank-Nicolson and projected SOR").

Discretise the Black-Scholes PDE on an S-grid, step backward from the payoff
with Crank-Nicolson, and at each step enforce the early-exercise constraint
V >= payoff via Projected SOR. What falls out is the early-exercise boundary.
Role: PRICING (no direction).

Sanity check baked into tests: an American CALL on a non-dividend underlying is
never exercised early, so it must equal the European (Black-Scholes) call.
"""
from __future__ import annotations

import numpy as np

from ..base import PricingAgent, PricingResult


def american_price(S0, K, T, r, sigma, kind="put", q=0.0,
                   M=400, N=400, omega=1.2, tol=1e-6, max_sor=10000):
    """Price an American option via CN + PSOR. Returns (price, early_ex_boundary)."""
    Smax = max(S0, K) * 4.0
    ds = Smax / M
    dt = T / N
    j = np.arange(M + 1)
    S = j * ds

    payoff = np.maximum(K - S, 0.0) if kind == "put" else np.maximum(S - K, 0.0)
    V = payoff.copy()

    # Operator L bands on interior nodes j=1..M-1.
    ji = j[1:-1]
    A = 0.5 * (sigma ** 2 * ji ** 2 - (r - q) * ji)
    B = -(sigma ** 2 * ji ** 2 + r)
    C = 0.5 * (sigma ** 2 * ji ** 2 + (r - q) * ji)
    theta = 0.5
    # Left (implicit) tridiagonal bands.
    lo = -theta * dt * A
    di = 1.0 - theta * dt * B
    up = -theta * dt * C
    # Right (explicit) bands.
    rlo = (1 - theta) * dt * A
    rdi = 1.0 + (1 - theta) * dt * B
    rup = (1 - theta) * dt * C

    boundary = np.full(N + 1, np.nan)

    for n in range(N, 0, -1):
        tau = (N - n + 1) * dt   # time-to-maturity at the earlier layer
        # Dirichlet boundaries at the earlier time layer.
        if kind == "put":
            lowB, highB = K * np.exp(-r * tau), 0.0
        else:
            lowB, highB = 0.0, Smax - K * np.exp(-r * tau)

        Vin = V[1:-1]
        rhs = rlo * V[:-2] + rdi * Vin + rup * V[2:]
        rhs[0] += theta * dt * A[0] * lowB      # move known boundary to RHS
        rhs[-1] += theta * dt * C[-1] * highB

        pay_in = payoff[1:-1]
        x = np.maximum(Vin, pay_in)             # warm start, feasible
        for _ in range(max_sor):
            x_old = x.copy()
            for i in range(len(x)):
                left = x[i - 1] if i > 0 else lowB
                right = x[i + 1] if i < len(x) - 1 else highB
                y = (rhs[i] - lo[i] * left - up[i] * right) / di[i]
                x[i] = max(pay_in[i], x[i] + omega * (y - x[i]))
            if np.linalg.norm(x - x_old, np.inf) < tol:
                break

        V = np.empty(M + 1)
        V[0], V[-1] = lowB, highB
        V[1:-1] = x

        # Early-exercise boundary: highest S where holder should exercise (put).
        exercised = np.where(x <= pay_in + 1e-8)[0]
        if kind == "put" and len(exercised):
            boundary[n - 1] = S[1:-1][exercised[-1]]
        elif kind == "call" and len(exercised):
            boundary[n - 1] = S[1:-1][exercised[0]]

    price = float(np.interp(S0, S, V))
    return price, boundary


class AmericanPSORAgent(PricingAgent):
    name = "american_psor"

    def price(self, instrument: str, **kw) -> PricingResult:
        p, boundary = american_price(
            kw["S"], kw["K"], kw["T"], kw["r"], kw["sigma"],
            kind=kw.get("kind", "put"), q=kw.get("q", 0.0),
            M=kw.get("M", 400), N=kw.get("N", 400))
        b = boundary[np.isfinite(boundary)]
        return PricingResult(
            self.name, instrument, price=p,
            diagnostics={"early_exercise_boundary_now": float(b[0]) if len(b) else None,
                         "inputs": {k: kw[k] for k in ("S", "K", "T", "r", "sigma")}})
