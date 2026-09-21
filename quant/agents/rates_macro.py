"""
Rates-macro directional agent  (Quant_Analysis: "stocks and interest rates").

Role: DIRECTIONAL, but heuristic and deliberately low-weight. The screenshots
show the equity/rates relationship: rising yields pressure equities, falling
yields support them, with the strength varying by regime. This agent reads the
10Y yield trend and tilts equity-index instruments accordingly. It ABSTAINS for
non-equity instruments (FX/commodities), where the relationship does not apply
cleanly, rather than guess.

This is a macro *tilt*, not an alpha model — hence weight 0.5 and modest
conviction. It fetches its own yield series (free, no key).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import DirectionalAgent, DirectionalSignal, Direction

# Instruments for which the equity/rates tilt is meaningful.
EQUITY_LIKE = {"ES1!", "NQ1!", "SPX", "NDX", "^GSPC", "^NDX", "ES=F", "NQ=F"}


class RatesMacroAgent(DirectionalAgent):
    name = "rates_macro"
    weight = 0.5

    def __init__(self, mom_window: int = 20):
        self.mom_window = mom_window

    def evaluate(self, instrument: str, ohlcv: pd.DataFrame, **kwargs) -> DirectionalSignal:
        if instrument.upper() not in EQUITY_LIKE:
            return self._abstain(instrument, "rates tilt only applies to equity indices")

        # Fetch the 10Y yield series (own data pull; cached).
        try:
            from ..data.loader import load_ohlcv
            y10 = load_ohlcv("US10Y", bars=self.mom_window * 4)["close"]
        except Exception as e:
            return self._abstain(instrument, f"yield data unavailable: {e}")
        if len(y10) < self.mom_window + 1:
            return self._abstain(instrument, "insufficient yield history")

        # Yield momentum: change over the window, in basis points.
        dchg_bps = float((y10.iloc[-1] - y10.iloc[-1 - self.mom_window]) * 100)
        # Rising yields -> bearish equities; falling -> bullish.
        direction = (Direction.BEAR if dchg_bps > 0 else
                     Direction.BULL if dchg_bps < 0 else Direction.NEUTRAL)
        # Conviction from magnitude: ~50bps move over a month is a strong signal.
        conviction = float(min(0.6, abs(dchg_bps) / 50.0 * 0.6))

        return DirectionalSignal(
            self.name, instrument, direction, conviction,
            rationale=f"10Y yield {'+' if dchg_bps>=0 else ''}{dchg_bps:.0f}bps over "
                      f"{self.mom_window}d -> {'headwind' if dchg_bps>0 else 'tailwind'} for equities",
            diagnostics={"yield_change_bps": dchg_bps, "y10_last": float(y10.iloc[-1])},
        )
