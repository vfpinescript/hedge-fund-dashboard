"""
DSR-gated strategy-search agent.

Searches a space of (signal family × parameter × instrument universe), backtests
each as a vol-targeted equal-risk portfolio, and ranks candidates to MAXIMISE
Sharpe — but every candidate's Sharpe is deflated by the Deflated Sharpe Ratio
using N = the number of candidates tried. So the more the agent searches, the
higher the bar it must clear, and it can only crown a winner that survives that
correction. Without this gate a Sharpe-maximiser just finds the luckiest of many
trials; with it, the search is honest.

Objective: among candidates whose DSR clears the gate (default 0.90), pick the
highest Sharpe. If none clear it, report the best DSR and say so plainly —
"no robust strategy found" is a valid, honest result.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from ..data.loader import load_ohlcv
from .signals import reversal, tsmom, trend, vol_scaled_mom
from .metrics import perf_metrics
from .overfitting import deflated_sharpe_ratio, perf_degradation

UNIVERSES = {
    "FX": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD"],
    "Diversified": ["EURUSD", "GBPUSD", "USDCHF", "AUDUSD", "XAUUSD", "XAGUSD", "NATGAS"],
}

# (family name, signal fn, params, holding period in days)
SPACE = (
    [("reversal", reversal, {"window": lb}, lb) for lb in (3, 5, 7, 10)]
    + [("tsmom", tsmom, {"lookback": lb}, 21) for lb in (63, 126, 252)]
    + [("trend", trend, {"window": w}, 21) for w in (50, 100, 200)]
    + [("vol_scaled_mom", vol_scaled_mom, {}, 21)]
)


def _sharpe(r, freq=252):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(freq)) if r.std() > 0 else 0.0


def candidate_returns(signal_fn, kwargs, instruments, hold, cost_bps=1.0,
                      bars=3000, vol_win=63, _cache={}):
    cols = []
    for sym in instruments:
        if sym not in _cache:
            _cache[sym] = load_ohlcv(sym, bars=bars)
        df = _cache[sym]
        ret = df["close"].pct_change()
        sig = signal_fn(df, **kwargs)
        vol = ret.rolling(vol_win).std()
        raw = np.sign(sig) / vol
        reb = pd.Series(np.nan, index=df.index)
        reb.iloc[::hold] = raw.iloc[::hold]
        pos = reb.ffill()
        pos = pos / pos.abs().rolling(252, min_periods=20).median()
        strat = pos.shift(1) * ret - (pos - pos.shift(1)).abs() * (cost_bps / 1e4)
        cols.append(strat.rename(sym))
    D = pd.concat(cols, axis=1).dropna(how="all")
    return D.mean(axis=1).dropna()


class StrategySearchAgent:
    name = "strategy_search"

    def __init__(self, dsr_gate: float = 0.90, cost_bps: float = 1.0):
        self.dsr_gate = dsr_gate
        self.cost_bps = cost_bps

    def search(self) -> dict:
        candidates = list(product(SPACE, UNIVERSES.items()))
        n_trials = len(candidates)                 # <-- what the DSR deflates against
        rows = []
        for (fam, fn, params, hold), (uni_name, insts) in candidates:
            try:
                r = candidate_returns(fn, params, insts, hold, self.cost_bps)
            except Exception:
                continue
            if len(r) < 300:
                continue
            m = perf_metrics(r)
            dsr = deflated_sharpe_ratio(r.values, n_trials)
            # in/out-of-sample Sharpe decay (ΔPerf)
            half = len(r) // 2
            sr_is, sr_oos = _sharpe(r.iloc[:half]), _sharpe(r.iloc[half:])
            rows.append({
                "strategy": f"{fam}({list(params.values()) or ['-']}) · {uni_name}",
                "family": fam, "universe": uni_name, "params": params, "hold": hold,
                "sharpe": round(m["sharpe"], 3),
                "total_return": round(m["total_return"] * 100, 1),
                "max_dd": round(m["max_drawdown"] * 100, 1),
                "profit_factor": round(m["profit_factor"], 2),
                "dsr": round(dsr["dsr"], 3), "verdict": dsr["verdict"],
                "sr_star": round(dsr["sr_star_ann"], 2),
                "sharpe_is": round(sr_is, 2), "sharpe_oos": round(sr_oos, 2),
                "perf_decay": round(perf_degradation(sr_is, sr_oos), 2),
            })
        rows.sort(key=lambda x: -x["dsr"])          # most robust first
        passing = [r for r in rows if r["dsr"] >= self.dsr_gate and r["sharpe_oos"] > 0]
        # objective: maximise Sharpe AMONG those that clear the DSR gate
        winner = max(passing, key=lambda x: x["sharpe"]) if passing else None
        return {
            "n_trials": n_trials, "dsr_gate": self.dsr_gate,
            "winner": winner,
            "robust_count": len(passing),
            "leaderboard": rows,
            "note": ("Winner = highest Sharpe among DSR-robust candidates."
                     if winner else
                     "NO candidate cleared the DSR gate — no robust edge found. "
                     "This is an honest negative result, not a failure."),
        }


# ---------------------------------------------------------------------------
# Combination search: search over BLENDS of signals, not just single signals.
# ---------------------------------------------------------------------------
from itertools import combinations as _combinations  # noqa: E402

# Pool of economically-distinct signals to blend.
COMBO_POOL = [
    ("reversal_7", reversal, {"window": 7}, 7),
    ("reversal_3", reversal, {"window": 3}, 3),
    ("tsmom_12m", tsmom, {"lookback": 252}, 21),
    ("tsmom_3m", tsmom, {"lookback": 63}, 21),
    ("trend_100", trend, {"window": 100}, 21),
]


class CombinationSearchAgent:
    """Search over subsets (blends) of signals for the best DSR-robust Sharpe.

    Combining uncorrelated edges is the path over the DSR bar, so this searches
    which 2- and 3-signal blends actually clear it. Every blend's Sharpe is
    deflated by the total number of blends tried, keeping the search honest.
    """
    name = "combination_search"

    def __init__(self, dsr_gate: float = 0.90, sizes=(2, 3), cost_bps: float = 1.0):
        self.dsr_gate = dsr_gate
        self.sizes = sizes
        self.cost_bps = cost_bps

    def search(self) -> dict:
        from .combination import blend_report, signal_streams, combine
        from .metrics import perf_metrics
        from .overfitting import deflated_sharpe_ratio
        from .alt_streams import eligible_alt_streams, stream_status
        combos = [c for size in self.sizes for c in _combinations(COMBO_POOL, size)]
        cands = [(c, u) for c in combos for u in UNIVERSES.items()]
        alt = eligible_alt_streams()          # news/Kalshi streams past the history gate
        n_trials = len(cands) + len(alt)
        rows = []
        for combo, (uname, insts) in cands:
            rep = blend_report(list(combo), insts, method="inverse_vol",
                               n_trials=n_trials, cost_bps=self.cost_bps)
            if "error" in rep:
                continue
            rows.append({
                "blend": " + ".join(rep["signals"]) + f"  · {uname}",
                "signals": rep["signals"], "universe": uname,
                "sharpe": rep["combined_sharpe"], "sortino": rep["combined_sortino"],
                "max_dd": rep["combined_max_dd"], "total_return": rep["combined_total_return"],
                "avg_corr": rep["avg_pairwise_corr"],
                "dsr": rep["dsr"], "verdict": rep["dsr_verdict"], "sr_star": rep["sr_star"],
            })
        # Blend eligible alt streams (news) with the top price blend, when any
        # have cleared the history gate. Today the gate holds them out.
        if alt:
            top = max(rows, key=lambda x: x["sharpe"]) if rows else None
            base_streams = signal_streams(list(COMBO_POOL[:2]),
                                          UNIVERSES["Diversified"], self.cost_bps)
            for name, s in alt.items():
                merged = pd.concat([base_streams.mean(axis=1).rename("price"),
                                    s.rename(name)], axis=1)
                combined, _ = combine(merged, "inverse_vol")
                m = perf_metrics(combined)
                dsr = deflated_sharpe_ratio(combined.values, n_trials)
                corr = float(merged.corr().iloc[0, 1])
                rows.append({
                    "blend": f"price + {name}", "signals": ["price", name],
                    "universe": "alt", "sharpe": round(m.get("sharpe", 0), 3),
                    "sortino": round(m.get("sortino", 0), 3),
                    "max_dd": round(m.get("max_drawdown", 0) * 100, 1),
                    "total_return": round(m.get("total_return", 0) * 100, 1),
                    "avg_corr": round(corr, 3), "dsr": round(dsr["dsr"], 3),
                    "verdict": dsr["verdict"], "sr_star": round(dsr["sr_star_ann"], 2),
                })

        rows.sort(key=lambda x: -x["dsr"])
        passing = [r for r in rows if r["dsr"] >= self.dsr_gate]
        winner = max(passing, key=lambda x: x["sharpe"]) if passing else None
        return {
            "n_trials": n_trials, "dsr_gate": self.dsr_gate,
            "winner": winner, "robust_count": len(passing),
            "leaderboard": rows, "alt_status": stream_status(),
            "note": ("Winner = highest-Sharpe DSR-robust blend."
                     if winner else
                     "No blend cleared the DSR gate yet — need more uncorrelated "
                     "edge (e.g. Kalshi arbitrage, news diffusion)."),
        }
