"""
Pairs / statistical-arbitrage research.

Tests the classic "buy the laggard, short the leader, bet on convergence" trade
across a universe of economically-linked pairs (Coke/Pepsi, Visa/Mastercard, ...).

HONESTY FIRST. The viral "46,000% since 1972" figure is survivorship + cherry
picking: KO/PEP is the one textbook pair that worked, and the claim ignores
shorting costs and the fact that pairs trading's edge was largely arbitraged away
after the early 2000s (Gatev-Goetzmann-Rouwenhorst, 2006). So here every pair is
backtested net of transaction AND short-borrow costs, and the Sharpe is deflated
(DSR) by the NUMBER OF PAIRS tried, so picking the best one out of ~50 is
penalised exactly as multiple testing demands. The table then shows the sober
reality, not the highlight reel.

Spread = log(P_A) - log(P_B) (equal-dollar ratio). Enter when the rolling z-score
exceeds +/- Z_ENTRY (long the cheap leg, short the rich leg), exit near z=0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data.loader import load_ohlcv
from quant.research.overfitting import deflated_sharpe_ratio

# Economically-linked pairs (RB.L dropped: no clean Yahoo history).
PAIRS = [
    ("KO", "PEP"), ("V", "MA"), ("XOM", "CVX"), ("HD", "LOW"), ("JPM", "BAC"),
    ("T", "VZ"), ("F", "GM"), ("WMT", "TGT"), ("MCD", "WEN"), ("UPS", "FDX"),
    ("BA", "AIR.PA"), ("NKE", "ADDYY"), ("PFE", "MRK"), ("BMY", "LLY"),
    ("CVX", "SHEL"), ("COP", "EOG"), ("SLB", "HAL"), ("MS", "GS"), ("C", "WFC"),
    ("SCHW", "IBKR"), ("SPGI", "MCO"), ("AXP", "DFS"), ("BRK-B", "MKL"),
    ("PG", "CL"), ("JNJ", "ABT"), ("CVS", "WBA"), ("DG", "DLTR"), ("COST", "BJ"),
    ("AMZN", "EBAY"), ("GOOGL", "META"), ("MSFT", "AAPL"), ("INTC", "AMD"),
    ("NVDA", "AVGO"), ("CSCO", "JNPR"), ("HPQ", "DELL"), ("CRM", "WDAY"),
    ("NOW", "TEAM"), ("LMT", "NOC"), ("GD", "RTX"), ("CAT", "DE"), ("UNP", "NSC"),
    ("DAL", "UAL"), ("AAL", "LUV"), ("MAR", "HLT"), ("RCL", "CCL"), ("BHP", "RIO"),
    ("FCX", "SCCO"), ("NEM", "GOLD"), ("DUK", "SO"),
]

WINDOW = 60
Z_ENTRY = 2.0
Z_EXIT = 0.5
COST_BPS = 5.0          # per unit turnover, per side
BORROW_ANNUAL = 0.02    # 2%/yr borrow on the short leg (liquid large caps)


def _positions(z: np.ndarray) -> np.ndarray:
    """State machine: +1 long-spread (long A / short B), -1 short-spread, 0 flat."""
    pos = np.zeros(len(z))
    cur = 0.0
    for i in range(len(z)):
        if np.isnan(z[i]):
            pos[i] = cur
            continue
        if cur == 0.0:
            if z[i] > Z_ENTRY:
                cur = -1.0
            elif z[i] < -Z_ENTRY:
                cur = 1.0
        elif abs(z[i]) < Z_EXIT:
            cur = 0.0
        pos[i] = cur
    return pos


def _half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    if len(s) < 60:
        return float("nan")
    lag = s.shift(1).dropna()
    dsp = (s - s.shift(1)).dropna()
    x = lag.loc[dsp.index].values
    y = dsp.values
    try:
        b = np.polyfit(x, y, 1)[0]
        return float(-np.log(2) / b) if b < 0 else float("inf")
    except Exception:
        return float("nan")


def backtest_pair(a: str, b: str, bars: int = 8000) -> dict:
    try:
        pa = load_ohlcv(a, bars=bars)["close"]
        pb = load_ohlcv(b, bars=bars)["close"]
    except Exception as e:
        return {"a": a, "b": b, "error": f"load failed: {str(e)[:40]}"}
    df = pd.concat([pa.rename("A"), pb.rename("B")], axis=1).dropna()
    if len(df) < 300:
        return {"a": a, "b": b, "error": "insufficient overlapping history"}
    spread = np.log(df["A"]) - np.log(df["B"])
    mu = spread.rolling(WINDOW).mean()
    sd = spread.rolling(WINDOW).std()
    z = (spread - mu) / sd
    pos = pd.Series(_positions(z.values), index=df.index)
    retA = df["A"].pct_change()
    retB = df["B"].pct_change()
    gross = pos.shift(1) * (retA - retB) / 2.0                 # $0.5 long + $0.5 short
    turnover = pos.diff().abs().fillna(0.0)
    cost = turnover * (COST_BPS / 1e4)
    borrow = pos.abs() * 0.5 * (BORROW_ANNUAL / 252.0)         # borrow on the short leg
    strat = (gross - cost - borrow).dropna()
    if len(strat) < 250 or strat.std() == 0:
        return {"a": a, "b": b, "error": "degenerate"}
    eq = (1 + strat).cumprod()
    cagr = float(eq.iloc[-1] ** (252 / len(strat)) - 1)
    sd_d = strat.std()
    sharpe = float(strat.mean() / sd_d * np.sqrt(252)) if sd_d > 0 else float("nan")
    mdd = float((eq / eq.cummax() - 1).min())
    n_trades = int((pos.diff().abs() > 0).sum())
    dsr = deflated_sharpe_ratio(strat.values, n_trials=len(PAIRS))
    corr = float(retA.corr(retB))
    cz = float(z.dropna().iloc[-1]) if z.notna().any() else float("nan")
    # which leg is the laggard right now (buy) vs leader (short)
    if np.isnan(cz) or abs(cz) < Z_ENTRY:
        sig = "no trade (within band)"
    elif cz > 0:  # spread rich: A rich vs B -> short A, long B
        sig = f"diverged: short {a}, long {b}"
    else:
        sig = f"diverged: long {a}, short {b}"
    return {
        "a": a, "b": b, "years": round(len(strat) / 252, 1),
        "corr": round(corr, 2), "half_life": round(_half_life(spread), 0) if np.isfinite(_half_life(spread)) else None,
        "cagr": round(cagr * 100, 1), "sharpe": round(sharpe, 2) if np.isfinite(sharpe) else None,
        "max_dd": round(mdd * 100, 1), "n_trades": n_trades,
        "dsr": round(float(dsr["dsr"]), 3) if np.isfinite(dsr["dsr"]) else None,
        "verdict": dsr["verdict"], "z_now": round(cz, 2) if np.isfinite(cz) else None,
        "signal": sig,
    }


def run_all() -> dict:
    rows = [backtest_pair(a, b) for a, b in PAIRS]
    ok = [r for r in rows if "error" not in r]
    ok.sort(key=lambda r: -(r["sharpe"] if r.get("sharpe") is not None else -9))
    robust = sum(1 for r in ok if r.get("dsr") is not None and r["dsr"] > 0.95)
    incon = sum(1 for r in ok if r.get("dsr") is not None and 0.6 < r["dsr"] <= 0.95)
    med_sharpe = float(np.median([r["sharpe"] for r in ok if r.get("sharpe") is not None])) if ok else None
    return {
        "pairs": ok,
        "n": len(ok), "n_robust": robust, "n_inconclusive": incon,
        "median_sharpe": round(med_sharpe, 2) if med_sharpe is not None else None,
        "cost_bps": COST_BPS, "borrow_annual": BORROW_ANNUAL,
        "window": WINDOW, "z_entry": Z_ENTRY,
    }
