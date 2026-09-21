"""
Per-agent attribution — which agents actually have edge?

For each (agent, instrument) it walks forward point-in-time (df.iloc[:i+1], no
lookahead), records the agent's signed conviction and the realized forward
return, and computes:

  IC (information coefficient) = corr(signed_conviction, forward_return)
      The standard quant edge metric. Positive & consistent across instruments
      => real signal. Near zero => noise. Negative => the agent is backwards.
  hit_rate  = fraction of directional calls whose sign matched the move.
  n_signals / n_directional = how often it actually spoke vs abstained/neutral.

Aggregating IC across instruments (mean and t-stat) is how a real desk decides
whether to keep, down-weight, or fire a signal — instead of trusting a
backtest that got lucky on one asset.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .base import Direction
from .data.loader import load_ohlcv

warnings.filterwarnings("ignore")


def agent_ic(agent, instrument, ohlcv, horizon=5, warmup=700, step=15):
    close = ohlcv["close"].to_numpy()
    n = len(ohlcv)
    scores, fwds, n_dir = [], [], 0
    for i in range(warmup, n - horizon, step):
        sub = ohlcv.iloc[:i + 1]
        sig = agent.evaluate(instrument, sub)
        if sig.direction == Direction.ABSTAIN:
            continue
        s = sig.signed_score
        fwd = close[i + horizon] / close[i] - 1.0
        scores.append(s)
        fwds.append(fwd)
        if sig.direction != Direction.NEUTRAL:
            n_dir += 1
    scores, fwds = np.array(scores), np.array(fwds)
    if len(scores) < 8 or np.std(scores) == 0:
        return None
    directional = scores != 0
    hit = float((np.sign(scores[directional]) == np.sign(fwds[directional])).mean()) \
        if directional.sum() else float("nan")
    ic = float(np.corrcoef(scores, fwds)[0, 1]) if np.std(scores) > 0 else float("nan")
    return {"instrument": instrument, "n": len(scores), "n_directional": int(n_dir),
            "hit_rate": hit, "IC": ic,
            "mean_ret_dir": float(np.mean(np.sign(scores[directional]) * fwds[directional]))
            if directional.sum() else float("nan")}


def run_attribution(agents, instruments, horizon=5, warmup=700, step=15, bars=2000):
    rows = []
    per_agent = {}
    for agent in agents:
        ics = []
        for inst in instruments:
            try:
                df = load_ohlcv(inst, bars=bars)
            except Exception:
                continue
            r = agent_ic(agent, inst, df, horizon, warmup, step)
            if r is None:
                continue
            r["agent"] = agent.name
            rows.append(r)
            ics.append(r["IC"])
        if ics:
            ics = np.array(ics)
            # t-stat of mean IC across instruments (is edge distinguishable from 0?)
            t = float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))) if len(ics) > 1 and ics.std() > 0 else float("nan")
            per_agent[agent.name] = {"mean_IC": float(ics.mean()),
                                     "IC_t_stat": t, "n_instruments": len(ics)}
    return pd.DataFrame(rows), per_agent
