"""
Performance metrics — the full stat suite a strategy dashboard shows.

perf_metrics() takes a daily strategy-return series (and, optionally, the daily
position series so per-TRADE stats can be derived) and returns:
  total_return, cagr, ann_return, ann_vol, sharpe, sortino,
  max_drawdown, time_in_dd, n_trades, win_rate, profit_factor,
  avg_win, avg_loss, risk_reward, max_win_streak, max_loss_streak.

A "trade" = one holding segment where the position sign stays constant (entry to
flip/flat). Trade return = the compounded daily strategy returns over that
segment. Daily-based metrics (Sharpe, Sortino, drawdown) use the daily series;
count/streak metrics use the trades.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _streaks(wins: np.ndarray):
    max_w = max_l = cur_w = cur_l = 0
    for w in wins:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_w = max(max_w, cur_w); max_l = max(max_l, cur_l)
    return max_w, max_l


def derive_trades(daily_ret: pd.Series, position: pd.Series) -> np.ndarray:
    """Compound daily returns within each constant-sign holding segment."""
    pos = np.sign(position.reindex(daily_ret.index).fillna(0).to_numpy())
    r = daily_ret.to_numpy()
    trades, i, n = [], 0, len(r)
    while i < n:
        if pos[i] == 0:
            i += 1; continue
        j = i
        eq = 1.0
        s = pos[i]
        while j < n and pos[j] == s:
            eq *= (1.0 + r[j]); j += 1
        trades.append(eq - 1.0)
        i = j
    return np.array(trades)


def perf_metrics(daily_ret: pd.Series, position: pd.Series | None = None,
                 periods_per_year: int = 252) -> dict:
    r = daily_ret.dropna()
    if len(r) < 5:
        return {}
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = eq / peak - 1.0
    ppy = periods_per_year

    ann_ret = r.mean() * ppy
    ann_vol = r.std() * np.sqrt(ppy)
    downside = r[r < 0].std() * np.sqrt(ppy)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    sortino = ann_ret / downside if downside > 0 else np.nan

    m = {
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": float(eq.iloc[-1] ** (ppy / len(r)) - 1.0),
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd.min()),
        "time_in_dd": float((dd < 0).mean()),
    }

    # Trade-based stats.
    trades = derive_trades(r, position) if position is not None else r.to_numpy()
    if len(trades):
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        gross_win = wins.sum()
        gross_loss = -losses.sum()
        mw, ml = _streaks(trades > 0)
        m.update({
            "n_trades": int(len(trades)),
            "win_rate": float((trades > 0).mean()),
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else np.inf,
            "avg_win": float(wins.mean()) if len(wins) else 0.0,
            "avg_loss": float(losses.mean()) if len(losses) else 0.0,
            "risk_reward": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else np.nan,
            "max_win_streak": int(mw),
            "max_loss_streak": int(ml),
        })
    return m
