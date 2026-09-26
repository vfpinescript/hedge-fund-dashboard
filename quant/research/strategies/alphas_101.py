"""
A representative subset of the "101 Formulaic Alphas" (Kakushadze 2015),
tested cross-sectionally, dollar-neutral, on a liquid US large-cap set.

Each alpha produces a per-name score each day; we cross-sectionally demean it
(dollar-neutral long/short), hold one day, and measure the portfolio's
Deflated-Sharpe-gated performance. Note: these alphas were designed for a
universe of thousands of names; on a ~34-name set the cross-section is thin, so
treat this as indicative, not a full replication.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from quant.data.loader import load_ohlcv                     # noqa: E402
from quant.research.metrics import perf_metrics              # noqa: E402
from quant.research.overfitting import deflated_sharpe_ratio  # noqa: E402

UNIVERSE = ["AAPL","MSFT","JNJ","PG","KO","PEP","WMT","XOM","CVX","JPM","BAC",
            "WFC","C","T","VZ","MRK","PFE","HD","MCD","DIS","CSCO","INTC","ORCL",
            "IBM","MMM","CAT","BA","GS","AXP","UNH","ABT","COST","TGT","LOW"]


def _panels():
    o, h, l, c, v = {}, {}, {}, {}, {}
    for s in UNIVERSE:
        try:
            df = load_ohlcv(s, bars=6000)
            if df is None or len(df) < 800:
                continue
            o[s], h[s], l[s], c[s], v[s] = df["open"], df["high"], df["low"], df["close"], df["volume"]
        except Exception:
            continue
    O, H, L, C, V = (pd.DataFrame(x).ffill() for x in (o, h, l, c, v))
    common = C.dropna(how="all").index
    return O.loc[common], H.loc[common], L.loc[common], C.loc[common], V.loc[common]


def _rank(df):   # cross-sectional rank in [0,1]
    return df.rank(axis=1, pct=True)


def alphas(O, H, L, C, V):
    ret = C.pct_change()
    vwap = (H + L + C) / 3.0
    a = {}
    # 101: (close-open)/(high-low)  -> intraday momentum
    a["alpha101"] = (C - O) / ((H - L) + 1e-3)
    # 54: -((low-close)*open^5)/((low-high)*close^5)
    a["alpha054"] = -((L - C) * (O ** 5)) / (((L - H) * (C ** 5)) + 1e-9)
    # 12: sign(delta(volume,1)) * (-delta(close,1))
    a["alpha012"] = np.sign(V.diff(1)) * (-C.diff(1))
    # 41: sqrt(high*low) - vwap
    a["alpha041"] = np.sqrt(H * L) - vwap
    # short-term reversal (rank of -5d return) — alpha-family mean reversion
    a["reversal5"] = -_rank(C / C.shift(5) - 1.0)
    # 1-day reversal
    a["reversal1"] = -_rank(ret)
    return a


def score_alpha(sig: pd.DataFrame, ret: pd.DataFrame, n_trials: int) -> dict:
    z = sig.sub(sig.mean(axis=1), axis=0)             # cross-sectional demean
    w = z.div(z.abs().sum(axis=1).replace(0, np.nan), axis=0)  # dollar-neutral, gross=1
    port = (w.shift(1) * ret).sum(axis=1).dropna()
    port = port[port.ne(0).cumsum() > 0]
    if len(port) < 300:
        return {"error": "too short"}
    m = perf_metrics(port)
    dsr = deflated_sharpe_ratio(port.values, n_trials)
    return {"sharpe": round(m["sharpe"], 3), "ann_pct": round(port.mean() * 252 * 100, 2),
            "max_dd_pct": round(m["max_drawdown"] * 100, 1),
            "dsr": round(dsr["dsr"], 3), "verdict": dsr["verdict"]}


def run():
    O, H, L, C, V = _panels()
    ret = C.pct_change()
    A = alphas(O, H, L, C, V)
    n_trials = len(A) * 4     # deflate for the family we scanned
    rows = []
    for name, sig in A.items():
        r = score_alpha(sig.replace([np.inf, -np.inf], np.nan), ret, n_trials)
        if "error" not in r:
            rows.append({"alpha": name, **r})
    rows.sort(key=lambda x: -x["sharpe"])
    return {"n_names": C.shape[1], "n_trials": n_trials, "alphas": rows}


if __name__ == "__main__":
    import json
    out = run()
    print(f"names={out['n_names']}  DSR n_trials={out['n_trials']}\n")
    print(f"{'alpha':12}{'Sharpe':>8}{'Ann%':>8}{'MaxDD%':>8}{'DSR':>7}  verdict")
    for r in out["alphas"]:
        print(f"{r['alpha']:12}{r['sharpe']:>8}{r['ann_pct']:>8}{r['max_dd_pct']:>8}{r['dsr']:>7}  {r['verdict']}")
