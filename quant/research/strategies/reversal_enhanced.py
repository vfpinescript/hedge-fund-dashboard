"""
Agent-enhanced short-term reversal.

Base = the 7-day reversal (already vol-targeted). We overlay two of the fund's
RISK agents, vectorised over the whole series:

  * vol-regime veto  — go flat when realised vol is at the extreme (98th pct),
    the VolRegimeAgent's veto condition. Removes exposure in the worst regimes.
  * conviction gate  — only take a position when the reversal signal is strong
    (|past move| above a rolling quantile), which drops the marginal trades.
  * tail scale        — shrink size when the left-tail Hill alpha is heavy (<3),
    the HillTailAgent's scaler.

Goal: raise Sortino and cut the number of trades while barely moving total
return, on every tradeable asset.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from quant.data.loader import load_ohlcv                     # noqa: E402
from quant.research.signals import reversal                  # noqa: E402
from quant.research.metrics import perf_metrics              # noqa: E402
from quant.agents.hill_tail import stable_alpha              # noqa: E402

VOL_WIN, HOLD, VOL_TARGET_WIN = 63, 7, 252


def _base_position(df, window=7):
    ret = df["close"].pct_change()
    sig = reversal(df, window=window)
    vol = ret.rolling(VOL_WIN).std()
    raw = np.sign(sig) / vol
    reb = pd.Series(np.nan, index=df.index)
    reb.iloc[::HOLD] = raw.iloc[::HOLD]
    pos = reb.ffill()
    pos = pos / pos.abs().rolling(VOL_TARGET_WIN, min_periods=20).median()
    return pos, ret, sig


def _risk_overlay(df, pos, sig, conviction_q=0.40, vol_veto_pct=0.98,
                  vol_win=20, look=500):
    ret = df["close"].pct_change()
    # (1) conviction gate: skip weak reversal signals (fewer trades)
    conv = sig.abs()
    thr = conv.rolling(look, min_periods=60).quantile(conviction_q)
    keep = (conv >= thr).astype(float)
    # (2) vol-regime veto: flat when realised vol at 98th percentile of history
    rv = ret.rolling(vol_win).std()
    hi = rv.rolling(look, min_periods=120).quantile(vol_veto_pct)
    veto = (rv >= hi).astype(float)             # 1 = extreme regime
    # (3) tail scale: shrink when left-tail alpha < 3 (rolling, cheap cadence)
    tail_scale = pd.Series(1.0, index=df.index)
    losses = (-ret).to_numpy()
    step = 21
    for i in range(look, len(ret), step):
        window = losses[i - look:i]
        window = window[np.isfinite(window)]
        a, _ = stable_alpha(window[window > 0]) if (window > 0).sum() > 60 else (np.nan, {})
        s = 1.0 if not np.isfinite(a) or a >= 3 else (0.5 if a >= 2 else 0.0)
        tail_scale.iloc[i:i + step] = s
    overlay = keep * (1 - veto) * tail_scale
    # apply overlay only at rebalances so we don't add turnover
    ov = pd.Series(np.nan, index=df.index)
    ov.iloc[::HOLD] = overlay.iloc[::HOLD]
    ov = ov.ffill().fillna(1.0)
    return pos * ov


def _stats(pos, ret, cost_bps=1.0):
    strat = pos.shift(1) * ret - (pos - pos.shift(1)).abs() * (cost_bps / 1e4)
    strat = strat.dropna()
    m = perf_metrics(strat)
    # trade count = number of rebalances where the position sign actually changes
    p = np.sign(pos.reindex(strat.index).fillna(0).to_numpy())
    trades = int((np.diff(p) != 0).sum())
    return {"total_return_pct": round(m["total_return"] * 100, 2),
            "sharpe": round(m["sharpe"], 3), "sortino": round(m.get("sortino", 0), 3),
            "max_dd_pct": round(m["max_drawdown"] * 100, 2), "trades": trades}, strat


def compare(instrument, **ov_kw):
    df = load_ohlcv(instrument, bars=3000)
    pos, ret, sig = _base_position(df)
    base, _ = _stats(pos, ret)
    enh_pos = _risk_overlay(df, pos, sig, **ov_kw)
    enh, _ = _stats(enh_pos, ret)
    return {"instrument": instrument, "base": base, "enhanced": enh}


if __name__ == "__main__":
    import json
    for inst in ["AUDUSD", "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "XAUUSD"]:
        try:
            r = compare(inst)
            b, e = r["base"], r["enhanced"]
            print(f"{inst}: return {b['total_return_pct']:>7}->{e['total_return_pct']:>7}  "
                  f"sortino {b['sortino']:>5}->{e['sortino']:>5}  trades {b['trades']:>4}->{e['trades']:>4}")
        except Exception as ex:
            print(inst, "ERR", str(ex)[:60])
