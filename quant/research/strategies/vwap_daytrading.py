"""
VWAP Trend day-trading (Zarattini & Aziz 2023) on QQQ, intraday minute bars.

Rule: each session, accumulate the anchored VWAP from the open
(VWAP_t = sum(HLC*Vol)/sum(Vol)). Go LONG when price > VWAP, SHORT when
price < VWAP; hold intraday, flat at the close. Aggregated to daily returns and
gated by the Deflated Sharpe Ratio.

Caveats: Alpaca's free IEX feed sees only a slice of volume, so this VWAP is a
proxy for the consolidated VWAP the paper used; history starts 2020-07. Treat
as indicative, not an exact replication.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from alpaca.client import Alpaca                              # noqa: E402
from quant.research.metrics import perf_metrics              # noqa: E402
from quant.research.overfitting import deflated_sharpe_ratio  # noqa: E402

SYMBOL = "QQQ"
START = "2020-07-27"
COST_BPS_PER_SWITCH = 1.0        # ~0.5bp/side round-trip on a switch


def fetch_minutes(symbol=SYMBOL, start=START) -> pd.DataFrame:
    a = Alpaca()
    tok, rows = None, []
    while True:
        p = {"timeframe": "1Min", "start": start + "T00:00:00Z", "limit": 10000, "feed": "iex"}
        if tok:
            p["page_token"] = tok
        r = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                         headers=a.h, params=p, timeout=45).json()
        rows.extend(r.get("bars") or [])
        tok = r.get("next_page_token")
        if not tok:
            break
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True).dt.tz_convert("America/New_York")
    df["day"] = df["t"].dt.date
    df["hm"] = df["t"].dt.strftime("%H:%M")
    df = df[(df["hm"] >= "09:30") & (df["hm"] < "16:00")].copy()   # regular session
    return df


def backtest(df: pd.DataFrame, cost_bps=COST_BPS_PER_SWITCH) -> pd.Series:
    c = cost_bps / 1e4
    daily = {}
    for day, g in df.groupby("day"):
        g = g.sort_values("t")
        price = g["c"].to_numpy(float)
        hlc = (g["h"] + g["l"] + g["c"]).to_numpy(float) / 3.0
        vol = g["v"].to_numpy(float)
        cum_pv = np.cumsum(hlc * vol)
        cum_v = np.cumsum(vol)
        vwap = np.where(cum_v > 0, cum_pv / cum_v, price)
        sig = np.sign(price - vwap)                 # +1 long, -1 short
        ret = np.zeros(len(price))
        ret[1:] = price[1:] / price[:-1] - 1.0      # minute returns
        pos = np.concatenate([[0.0], sig[:-1]])     # act next minute (no look-ahead)
        pos[-1] = 0.0                               # flat at the close
        switches = np.abs(np.diff(np.concatenate([[0.0], pos])))
        pnl = pos * ret - switches * c
        daily[pd.Timestamp(day)] = float(np.nansum(pnl))
    return pd.Series(daily).sort_index()


def run(cost_bps=COST_BPS_PER_SWITCH, n_trials=10) -> dict:
    df = fetch_minutes()
    d = backtest(df, cost_bps).dropna()
    m = perf_metrics(d)
    dsr = deflated_sharpe_ratio(d.values, n_trials)
    eq = (1 + d).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(d)) - 1
    # buy & hold QQQ over the same days (close-to-close)
    closes = df.groupby("day")["c"].last()
    bh = closes.pct_change().dropna()
    bh_cagr = (1 + bh).prod() ** (252 / len(bh)) - 1
    return {"symbol": SYMBOL, "days": len(d), "cost_bps_per_switch": cost_bps,
            "total_return_pct": round((eq.iloc[-1] - 1) * 100, 1),
            "cagr_pct": round(cagr * 100, 2), "sharpe": round(m["sharpe"], 3),
            "max_dd_pct": round(m["max_drawdown"] * 100, 2),
            "dsr": round(dsr["dsr"], 3), "verdict": dsr["verdict"],
            "buyhold_cagr_pct": round(bh_cagr * 100, 2)}


if __name__ == "__main__":
    import json
    for cb in (0.0, 1.0, 2.0):
        print(json.dumps(run(cost_bps=cb), indent=None))
