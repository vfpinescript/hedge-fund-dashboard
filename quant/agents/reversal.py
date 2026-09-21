"""
Short-term reversal directional agent — THE validated edge.

This is the only signal in the book with a proven, out-of-sample information
coefficient (reversal_7: mean IC +0.027, t=+4.4 across 12/13 instruments;
FX reversal book Sharpe 0.72 net of costs, verified distinct from buy-and-hold).
Everything else in the directional committee had ~0 IC, so this agent carries
the real weight.

Signal = -(close/close[-lookback] - 1): long recent losers, short recent
winners. Conviction scales with the vol-normalised signal magnitude. The edge
is FX/metals/gas-specific and FAILS on trending assets (crude, equity indices),
so the agent down-weights conviction on instruments known to trend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import DirectionalAgent, DirectionalSignal, Direction

# Instruments where reversal was shown NOT to work (they trend) -> low weight.
TRENDING = {"USOIL", "UKOIL", "ES1!", "NQ1!", "SPX", "NDX", "BTCUSD", "ETHUSD"}


class ReversalAgent(DirectionalAgent):
    name = "reversal"
    weight = 1.6   # the validated edge — weighted above the no-edge ML agents

    def __init__(self, lookback: int = 7, vol_window: int = 63, min_bars: int = 120):
        self.lookback = lookback
        self.vol_window = vol_window
        self.min_bars = min_bars

    def evaluate(self, instrument: str, ohlcv: pd.DataFrame, **kwargs) -> DirectionalSignal:
        if ohlcv is None or len(ohlcv) < self.min_bars:
            return self._abstain(instrument, f"need >= {self.min_bars} bars")

        c = ohlcv["close"]
        ret = c.pct_change()
        sig = -(c.iloc[-1] / c.iloc[-1 - self.lookback] - 1.0)   # reversal signal
        vol = float(ret.iloc[-self.vol_window:].std())
        if not np.isfinite(vol) or vol == 0:
            return self._abstain(instrument, "vol not estimable")

        # Vol-normalised signal (z-score over the holding horizon).
        z = sig / (vol * np.sqrt(self.lookback))
        direction = (Direction.BULL if z > 0 else
                     Direction.BEAR if z < 0 else Direction.NEUTRAL)
        conviction = min(1.0, abs(z) / 1.5)          # |z|~1.5 => full conviction
        if instrument.upper() in TRENDING:
            conviction *= 0.3                        # reversal fails on trenders

        past = (c.iloc[-1] / c.iloc[-1 - self.lookback] - 1.0) * 100
        return DirectionalSignal(
            self.name, instrument, direction, conviction,
            rationale=(f"{self.lookback}d return {past:+.2f}% -> "
                       f"{'buy the dip' if z>0 else 'fade the rally'} "
                       f"(z={z:+.2f}, vol {vol*100:.2f}%)"
                       + ("; down-weighted: trending asset" if instrument.upper() in TRENDING else "")),
            diagnostics={"signal_z": float(z), "past_return": float(past/100), "daily_vol": vol},
        )
