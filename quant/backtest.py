"""
Walk-forward backtest harness — the honesty check.

For each step, the committee sees ONLY data up to that bar (df.iloc[:i+1]); the
agents train on that history and predict the next move. The realized forward
return over `horizon` is then compared to the call. No future data ever enters a
decision, so the hit-rate and PnL are genuinely out-of-sample.

Steps are spaced `horizon` bars apart by default so forward windows do not
overlap (independent observations). This measures whether the committee has
edge BEFORE any capital is risked. It does not place orders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .committee import Committee
from .base import Direction


@dataclass
class BacktestResult:
    instrument: str
    n_decisions: int
    n_directional: int
    hit_rate: float
    avg_ret_when_bull: float
    avg_ret_when_bear: float
    strategy_total_return: float
    strategy_sharpe: float
    buy_hold_return: float
    periods_per_year: float
    records: list = field(default_factory=list)

    def summary(self) -> dict:
        d = self.__dict__.copy()
        d.pop("records")
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def walk_forward(committee: Committee, instrument: str, ohlcv: pd.DataFrame,
                 horizon: int = 5, warmup: int = 500, step: int | None = None,
                 verbose: bool = False) -> BacktestResult:
    step = step or horizon
    close = ohlcv["close"].to_numpy()
    n = len(ohlcv)
    recs = []

    for i in range(warmup, n - horizon, step):
        sub = ohlcv.iloc[:i + 1]
        dec = committee.decide(instrument, sub)
        fwd = close[i + horizon] / close[i] - 1.0
        if dec.direction == Direction.BULL:
            signed = +dec.conviction * dec.risk_scale
        elif dec.direction == Direction.BEAR:
            signed = -dec.conviction * dec.risk_scale
        else:
            signed = 0.0
        recs.append({"idx": i, "date": str(ohlcv.index[i].date()),
                     "dir": dec.direction.value, "exposure": signed, "fwd_ret": fwd})
        if verbose:
            print(f"  {ohlcv.index[i].date()}  {dec.direction.value:7} "
                  f"exp={signed:+.3f}  fwd={fwd*100:+.2f}%")

    df = pd.DataFrame(recs)
    directional = df[df["dir"] != "neutral"]
    if len(directional):
        hits = (np.sign(directional["exposure"]) == np.sign(directional["fwd_ret"]))
        hit_rate = float(hits.mean())
    else:
        hit_rate = float("nan")

    strat = df["exposure"] * df["fwd_ret"]           # position-sized period returns
    ppy = 252.0 / horizon                            # non-overlapping periods/yr
    sharpe = float(strat.mean() / strat.std() * np.sqrt(ppy)) if strat.std() > 0 else float("nan")
    bull = df[df["dir"] == "bull"]["fwd_ret"]
    bear = df[df["dir"] == "bear"]["fwd_ret"]

    # Buy & hold over the same tested span.
    bh = close[df["idx"].iloc[-1] + horizon] / close[df["idx"].iloc[0]] - 1.0 if len(df) else 0.0

    return BacktestResult(
        instrument=instrument,
        n_decisions=len(df),
        n_directional=len(directional),
        hit_rate=hit_rate,
        avg_ret_when_bull=float(bull.mean()) if len(bull) else float("nan"),
        avg_ret_when_bear=float(bear.mean()) if len(bear) else float("nan"),
        strategy_total_return=float(strat.sum()),
        strategy_sharpe=sharpe,
        buy_hold_return=float(bh),
        periods_per_year=ppy,
        records=recs,
    )
