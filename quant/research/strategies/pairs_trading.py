"""
Pairs Trading (Gatev, Goetzmann, Rouwenhorst 2006) on a liquid US large-cap set.

Method: rolling 12-month FORMATION window normalises each stock to a cumulative
return index (start = 1); the 20 pairs with the smallest sum-of-squared distance
are selected. In the next 6-month TRADING window, a pair opens when its spread
diverges by 2 formation-standard-deviations (long the loser, short the winner),
and closes when the spread crosses back through zero (or the window ends).
Returns are on committed capital (averaged over all selected pairs), net of a
round-trip cost per leg. Gated by the Deflated Sharpe Ratio.
"""
from __future__ import annotations
import sys
from itertools import combinations
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
FORM = 252          # 12-month formation
TRADE = 126         # 6-month trading
N_PAIRS = 20
ENTRY_SD = 2.0
COST_BPS = 5.0      # per leg per transaction


def _load_prices() -> pd.DataFrame:
    cols = {}
    for s in UNIVERSE:
        try:
            df = load_ohlcv(s, bars=7000)
            if df is not None and len(df) > FORM + TRADE:
                cols[s] = df["close"].astype(float)
        except Exception:
            continue
    px = pd.DataFrame(cols).dropna(how="all").ffill()
    return px.dropna(axis=1, thresh=int(len(px) * 0.9))


def backtest() -> pd.Series:
    px = _load_prices()
    rets = px.pct_change()
    idx = px.index
    daily = pd.Series(0.0, index=idx)
    c = COST_BPS / 1e4
    start = FORM
    while start + TRADE <= len(idx):
        fwin = px.iloc[start - FORM:start]
        norm = fwin / fwin.iloc[0]
        # select pairs by smallest sum of squared distance
        dists = []
        syms = [s for s in norm.columns if norm[s].notna().all()]
        for a, b in combinations(syms, 2):
            d = float(((norm[a] - norm[b]) ** 2).sum())
            dists.append((d, a, b))
        dists.sort(key=lambda x: x[0])
        pairs = dists[:N_PAIRS]
        # trading window
        twin_slice = slice(start, start + TRADE)
        tdates = idx[twin_slice]
        pair_daily = []
        for _, a, b in pairs:
            sd = float((norm[a] - norm[b]).std()) or 1e-9
            pa = px[a].iloc[twin_slice] / px[a].iloc[start - 1]
            pb = px[b].iloc[twin_slice] / px[b].iloc[start - 1]
            spread = (pa - pb).to_numpy()
            ra = rets[a].iloc[twin_slice].to_numpy()
            rb = rets[b].iloc[twin_slice].to_numpy()
            pos = 0            # +1 => long a short b ; -1 => short a long b
            pr = np.zeros(len(tdates))
            for k in range(len(tdates)):
                if pos != 0:
                    pr[k] = 0.5 * pos * (np.nan_to_num(ra[k]) - np.nan_to_num(rb[k]))
                    if (pos == 1 and spread[k] >= 0) or (pos == -1 and spread[k] <= 0):
                        pr[k] -= 2 * c            # closing cost (both legs)
                        pos = 0
                elif abs(spread[k]) > ENTRY_SD * sd:
                    pos = -1 if spread[k] > 0 else 1   # bet on convergence
                    pr[k] -= 2 * c                      # opening cost
            pair_daily.append(pd.Series(pr, index=tdates))
        if pair_daily:
            daily.loc[tdates] = pd.concat(pair_daily, axis=1).mean(axis=1)
        start += TRADE
    return daily.loc[daily.ne(0).cumsum() > 0]   # from first traded day


def run(n_trials: int = 30) -> dict:
    d = backtest().dropna()
    m = perf_metrics(d)
    dsr = deflated_sharpe_ratio(d.values, n_trials)
    eq = (1 + d).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(d)) - 1
    return {"n_days": len(d), "cagr_pct": round(cagr * 100, 2),
            "ann_excess_pct": round(d.mean() * 252 * 100, 2),
            "sharpe": round(m["sharpe"], 3), "sortino": round(m.get("sortino", 0), 2),
            "max_dd_pct": round(m["max_drawdown"] * 100, 2),
            "dsr": round(dsr["dsr"], 3), "verdict": dsr["verdict"],
            "universe": len([c for c in _load_prices().columns])}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
