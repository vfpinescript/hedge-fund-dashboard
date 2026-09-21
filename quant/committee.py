"""
The CIO: aggregate agent outputs into one decision per instrument.

Pipeline (see ARCHITECTURE.md section 2d):
  1. Directional agents vote; weight = conviction * agent.weight.
  2. Net signed score -> bull / bear / neutral + aggregate conviction.
  3. Risk agents multiply a risk_scale and may hard-veto (force flat).
  4. Emit a Decision. NO ORDER IS PLACED — this only decides direction & scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence
import pandas as pd

from .base import (
    DirectionalAgent, RiskAgent, Direction,
    DirectionalSignal, RiskSignal,
)


@dataclass
class Decision:
    instrument: str
    direction: Direction
    conviction: float                 # 0..1 aggregate
    risk_scale: float                 # 0..1 from risk agents
    vetoed: bool
    rationale: list = field(default_factory=list)
    votes: list = field(default_factory=list)      # DirectionalSignal[]
    risk_signals: list = field(default_factory=list)  # RiskSignal[]

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "direction": self.direction.value,
            "conviction": round(self.conviction, 4),
            "risk_scale": round(self.risk_scale, 4),
            "vetoed": self.vetoed,
            "rationale": self.rationale,
            "votes": [
                {"agent": v.agent, "dir": v.direction.value,
                 "conviction": round(v.conviction, 3), "why": v.rationale}
                for v in self.votes
            ],
            "risk": [
                {"agent": r.agent, "scale": round(r.risk_scale, 3),
                 "veto": r.veto, "why": r.rationale}
                for r in self.risk_signals
            ],
        }


class Committee:
    def __init__(
        self,
        directional: Sequence[DirectionalAgent],
        risk: Sequence[RiskAgent] = (),
        neutral_band: float = 0.15,   # |net score| below this => neutral
    ):
        self.directional = list(directional)
        self.risk = list(risk)
        self.neutral_band = neutral_band

    def decide(self, instrument: str, ohlcv: pd.DataFrame, **kwargs) -> Decision:
        votes = [a.evaluate(instrument, ohlcv, **kwargs) for a in self.directional]
        risk_signals = [r.evaluate(instrument, ohlcv, **kwargs) for r in self.risk]

        # --- weighted directional aggregation ---
        num = 0.0   # sum of weight * signed_score
        den = 0.0   # sum of weights of agents that actually voted (not abstain)
        for agent, v in zip(self.directional, votes):
            if v.direction == Direction.ABSTAIN:
                continue
            num += agent.weight * v.signed_score
            den += agent.weight
        net = (num / den) if den > 0 else 0.0   # in [-1, 1]

        if den == 0:
            direction = Direction.NEUTRAL
            conviction = 0.0
        elif net > self.neutral_band:
            direction = Direction.BULL
            conviction = abs(net)
        elif net < -self.neutral_band:
            direction = Direction.BEAR
            conviction = abs(net)
        else:
            direction = Direction.NEUTRAL
            conviction = abs(net)

        # --- risk overlay ---
        risk_scale = 1.0
        vetoed = False
        rationale = []
        for r in risk_signals:
            risk_scale *= r.risk_scale
            if r.veto:
                vetoed = True
            if r.risk_scale < 1.0 or r.veto:
                rationale.append(f"[risk] {r.agent}: {r.rationale}")

        if vetoed:
            rationale.insert(0, "VETOED by a risk agent -> forced flat")
            direction = Direction.NEUTRAL
            risk_scale = 0.0

        voters = [v for v in votes if v.direction != Direction.ABSTAIN]
        rationale.append(
            f"net directional score {net:+.3f} from {len(voters)}/{len(votes)} voting agents"
        )

        return Decision(
            instrument=instrument,
            direction=direction,
            conviction=conviction,
            risk_scale=risk_scale,
            vetoed=vetoed,
            rationale=rationale,
            votes=votes,
            risk_signals=risk_signals,
        )
