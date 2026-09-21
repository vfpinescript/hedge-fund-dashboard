"""
Core contracts for the signal committee.

Three agent roles (see ARCHITECTURE.md):
  - DirectionalAgent -> DirectionalSignal (bull/bear/neutral + conviction)
  - RiskAgent        -> RiskSignal (risk_scale in [0,1], optional hard veto)
  - PricingAgent     -> PricingResult (fair value + greeks, no direction)

Every agent MUST be able to `abstain` when its data is missing or insufficient.
An abstaining agent contributes nothing to the committee — it never guesses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import pandas as pd


class Direction(str, Enum):
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    ABSTAIN = "abstain"  # agent had no usable data / no opinion


@dataclass
class DirectionalSignal:
    """Output of a directional agent for one instrument."""
    agent: str
    instrument: str
    direction: Direction
    conviction: float = 0.0          # 0..1; 0 when ABSTAIN
    rationale: str = ""
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.conviction = float(max(0.0, min(1.0, self.conviction)))
        if self.direction == Direction.ABSTAIN:
            self.conviction = 0.0

    @property
    def signed_score(self) -> float:
        """+conviction for bull, -conviction for bear, 0 otherwise."""
        if self.direction == Direction.BULL:
            return self.conviction
        if self.direction == Direction.BEAR:
            return -self.conviction
        return 0.0


@dataclass
class RiskSignal:
    """Output of a risk agent. Never picks a direction."""
    agent: str
    instrument: str
    risk_scale: float = 1.0          # multiply position size by this (0..1)
    veto: bool = False               # if True, force the trade flat
    rationale: str = ""
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.risk_scale = float(max(0.0, min(1.0, self.risk_scale)))


@dataclass
class PricingResult:
    """Output of a pricing agent — fair value of a specific derivative."""
    agent: str
    instrument: str
    price: float
    greeks: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


class DirectionalAgent:
    """Base class. Subclasses implement `evaluate` and set `name`/`weight`."""
    name: str = "directional_agent"
    weight: float = 1.0              # committee weight

    def evaluate(self, instrument: str, ohlcv: pd.DataFrame, **kwargs) -> DirectionalSignal:
        raise NotImplementedError

    def _abstain(self, instrument: str, why: str) -> DirectionalSignal:
        return DirectionalSignal(self.name, instrument, Direction.ABSTAIN, 0.0, why)


class RiskAgent:
    name: str = "risk_agent"

    def evaluate(self, instrument: str, ohlcv: pd.DataFrame, **kwargs) -> RiskSignal:
        raise NotImplementedError

    def _abstain(self, instrument: str, why: str) -> RiskSignal:
        # Abstaining risk agent = no effect (scale 1, no veto).
        return RiskSignal(self.name, instrument, 1.0, False, "abstain: " + why)


class PricingAgent:
    name: str = "pricing_agent"

    def price(self, instrument: str, **kwargs) -> PricingResult:
        raise NotImplementedError
