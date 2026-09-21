"""
Market Intraday Momentum (Gao, Han, Li & Zhou, JFE 2018) — SPY, backtested on
real 30-minute bars in OUR engine, gated by the Deflated Sharpe Ratio.

Rule (faithful):
  r1 = P[10:00 ET] / P[prev close] - 1        (first-half-hour return)
  last-half trade = long 15:30->16:00 when r1 > 0 (long-only version), and
  the paper's long/short version = sign(r1) on the same window.

Bars: timestamp = bar OPEN; ET session 09:30..15:30 (13 bars/day). The 09:30
bar closes at 10:00 (=> its close is P[10:00]); the 15:30 bar spans the closing
half-hour (open=P[15:30], close=P[16:00]).
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from quant.research.overfitting import deflated_sharpe_ratio  # noqa: E402

HERE = Path(__file__).resolve().parent


def load() -> pd.DataFrame:
    df = pd.read_csv(HERE / "spy_30m.csv")
    df["t"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df["day"] = df["t"].dt.date
    df["et"] = df["t"].dt.strftime("%H:%M")
    return df


def build_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One row per trading day with the two quantities the strategy needs."""
    rows = []
    prev_close = None
    for day, g in df.groupby("day"):
        g = g.sort_values("t")
        first = g[g["et"] == "09:30"]      # covers 9:30-10:00 -> close is P[10:00]
        last = g[g["et"] == "15:30"]       # covers 15:30-16:00 (the closing half-hour)
        day_close = g["close"].iloc[-1]
        if not first.empty and not last.empty and prev_close is not None:
            p_1000 = float(first["close"].iloc[0])
            r1 = p_1000 / prev_close - 1.0
            open_1530 = float(last["open"].iloc[0])
            close_1600 = float(last["close"].iloc[0])
            last_half = close_1600 / open_1530 - 1.0
            rows.append({"day": day, "r1": r1, "last_half": last_half})
        prev_close = float(day_close)
    return pd.DataFrame(rows).set_index("day")


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


def metrics(r: pd.Series, name: str, cost_bps_per_side: float, n_trials: int) -> dict:
    r = r.dropna()
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    eq = (1 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    wins = (r > 0).sum()
    dsr = deflated_sharpe_ratio(r.values, n_trials)
    return {
        "strategy": name,
        "cost_bps_side": cost_bps_per_side,
        "n_days": len(r),
        "ann_return_pct": round(ann_ret * 100, 2),
        "sharpe": round(sharpe(r), 3),
        "max_dd_pct": round(mdd * 100, 2),
        "hit_rate_pct": round(100 * wins / len(r), 1),
        "avg_bps_per_day": round(r.mean() * 1e4, 2),
        "dsr": round(dsr["dsr"], 3),
        "dsr_verdict": dsr["verdict"],
    }


def run(cost_bps_per_side: float = 1.0, n_trials: int = 10) -> pd.DataFrame:
    d = build_daily(load())
    c = cost_bps_per_side / 1e4
    traded_long = (d["r1"] > 0).astype(float)          # long-only: 1 or 0
    traded_ls = np.sign(d["r1"])                        # long/short: +1 or -1
    # costs: pay round-trip (enter+exit) only on days we hold a position
    long_ret = traded_long * d["last_half"] - traded_long.abs() * 2 * c
    ls_ret = traded_ls * d["last_half"] - (traded_ls != 0) * 2 * c
    always = d["last_half"] - 2 * c                     # benchmark: always long the close
    out = [
        metrics(long_ret, "Intraday Momentum (long-only)", cost_bps_per_side, n_trials),
        metrics(ls_ret, "Intraday Momentum (long/short, paper)", cost_bps_per_side, n_trials),
        metrics(always, "Benchmark: always long last half-hour", cost_bps_per_side, n_trials),
    ]
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 160, "display.max_columns", 20)
    for cbs in (0.0, 1.0, 2.0):
        print(f"\n=== cost = {cbs} bps/side (round-trip {2*cbs} bps) ===")
        print(run(cost_bps_per_side=cbs).to_string(index=False))
