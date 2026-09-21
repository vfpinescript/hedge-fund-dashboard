"""
Signal research — hunt for edge with PARAMETER-FREE, economically-motivated
signals instead of fitted ML. Nothing here is trained, so nothing can overfit;
each signal is a causal rolling transform (uses only past data at each bar).

Signals tested are the classic, academically-documented premia:
  - time-series momentum (Moskowitz-Ooi-Pedersen): trend persists ~1-12 months
  - trend-following (price vs long MA): the CTA/managed-futures workhorse
  - short-term reversal: 1-4 week overreaction mean-reverts
  - volatility-scaled momentum: momentum normalized by recent vol

Edge metric = Information Coefficient (IC): the Spearman rank correlation
between the signal at time t and the forward return over the next `horizon`.
We sample NON-OVERLAPPING (step = horizon) so observations are independent, and
report IC per instrument plus a t-stat across instruments — an honest test of
whether edge is distinguishable from luck across a whole universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --- signal library (each returns a causal Series aligned to the index) ----
def tsmom(ohlcv, lookback=252, skip=0):
    c = ohlcv["close"]
    return c.shift(skip) / c.shift(lookback) - 1.0

def trend(ohlcv, window=200):
    c = ohlcv["close"]
    return c / c.rolling(window).mean() - 1.0

def reversal(ohlcv, window=5):
    c = ohlcv["close"]
    return -(c / c.shift(window) - 1.0)

def vol_scaled_mom(ohlcv, lookback=252, vol_win=63):
    c = ohlcv["close"]
    vol = c.pct_change().rolling(vol_win).std()
    return (c / c.shift(lookback) - 1.0) / vol.replace(0, np.nan)


SIGNALS = {
    "tsmom_12m":        (tsmom, {"lookback": 252}, 21),
    "tsmom_12m_skip1m": (tsmom, {"lookback": 252, "skip": 21}, 21),
    "tsmom_3m":         (tsmom, {"lookback": 63}, 21),
    "trend_200":        (trend, {"window": 200}, 21),
    "trend_50":         (trend, {"window": 50}, 21),
    "reversal_5":       (reversal, {"window": 5}, 5),
    "reversal_21":      (reversal, {"window": 21}, 21),
    "vol_scaled_mom":   (vol_scaled_mom, {}, 21),
}


def forward_return(ohlcv, horizon):
    c = ohlcv["close"]
    return c.shift(-horizon) / c - 1.0


def signal_ic(ohlcv, fn, kwargs, horizon):
    """Non-overlapping IC for one signal on one instrument."""
    s = fn(ohlcv, **kwargs)
    f = forward_return(ohlcv, horizon)
    df = pd.DataFrame({"s": s, "f": f}).dropna()
    if len(df) < 4 * horizon:
        return None
    df = df.iloc[::horizon]                      # non-overlapping samples
    if len(df) < 12 or df["s"].std() == 0:
        return None
    ic, _ = stats.spearmanr(df["s"], df["f"])
    # directional strategy: position = sign(signal), return = sign*fwd
    strat = np.sign(df["s"]) * df["f"]
    hit = float((np.sign(df["s"]) == np.sign(df["f"])).mean())
    ppy = 252.0 / horizon
    sharpe = float(strat.mean() / strat.std() * np.sqrt(ppy)) if strat.std() > 0 else np.nan
    return {"n": len(df), "IC": float(ic), "hit_rate": hit,
            "strat_mean": float(strat.mean()), "strat_sharpe": sharpe}


def research_universe(instruments, loader, signals=SIGNALS, bars=3000):
    """Run every signal over every instrument; aggregate IC per signal."""
    rows = []
    data = {}
    for inst in instruments:
        try:
            data[inst] = loader(inst, bars=bars)
        except Exception:
            continue
    for name, (fn, kw, h) in signals.items():
        for inst, df in data.items():
            r = signal_ic(df, fn, kw, h)
            if r is None:
                continue
            r.update({"signal": name, "instrument": inst, "horizon": h})
            rows.append(r)
    res = pd.DataFrame(rows)
    agg = []
    for name in signals:
        sub = res[res["signal"] == name]
        if len(sub) < 2:
            continue
        ics = sub["IC"].to_numpy()
        t = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))) if ics.std() > 0 else np.nan
        agg.append({"signal": name,
                    "mean_IC": float(ics.mean()),
                    "IC_t_stat": t,
                    "pct_positive": float((ics > 0).mean()),
                    "mean_hit": float(sub["hit_rate"].mean()),
                    "mean_sharpe": float(sub["strat_sharpe"].mean()),
                    "n_instruments": len(sub)})
    agg = pd.DataFrame(agg).sort_values("mean_IC", ascending=False) if agg else pd.DataFrame()
    return res, agg
