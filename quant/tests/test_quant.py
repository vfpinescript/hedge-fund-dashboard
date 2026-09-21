"""
Tests for the quant framework. Run:
    ./venv/bin/python -m pytest quant/tests -q
or (no pytest dependency):
    ./venv/bin/python -m unittest quant.tests.test_quant -v
"""
import math
import unittest

import numpy as np

from quant.base import Direction, DirectionalSignal, RiskSignal
from quant.committee import Committee
from quant.data.loader import (
    synthetic_ohlcv, make_features, label_forward, temporal_split,
)
from quant.agents.hill_tail import HillTailAgent, hill_alpha
from quant.agents.decision_tree import DecisionTreeAgent
from quant.pricing.black_scholes import BSInputs, bs_price_greeks, implied_vol, BlackScholesAgent


class TestBlackScholes(unittest.TestCase):
    def test_atm_call_known_value(self):
        # S=K=100, r=0, T=1, sigma=0.2 -> C ≈ 7.9656 (textbook).
        g = bs_price_greeks(BSInputs(100, 100, 1.0, 0.0, 0.2, "call"))
        self.assertAlmostEqual(g["price"], 7.9656, places=3)

    def test_put_call_parity(self):
        S, K, T, r, sig = 100, 95, 0.75, 0.03, 0.25
        c = bs_price_greeks(BSInputs(S, K, T, r, sig, "call"))["price"]
        p = bs_price_greeks(BSInputs(S, K, T, r, sig, "put"))["price"]
        # C - P = S - K e^{-rT}
        self.assertAlmostEqual(c - p, S - K * math.exp(-r * T), places=6)

    def test_call_delta_bounds_and_gamma_positive(self):
        g = bs_price_greeks(BSInputs(100, 100, 1.0, 0.01, 0.2, "call"))
        self.assertTrue(0 < g["delta"] < 1)
        self.assertGreater(g["gamma"], 0)
        self.assertGreater(g["vega"], 0)

    def test_implied_vol_roundtrip(self):
        p = BSInputs(100, 110, 0.5, 0.02, 0.3, "call")
        price = bs_price_greeks(p)["price"]
        iv = implied_vol(price, BSInputs(100, 110, 0.5, 0.02, 0.2, "call"))
        self.assertAlmostEqual(iv, 0.3, places=4)

    def test_agent_returns_pricing_result(self):
        r = BlackScholesAgent().price("SPX-C", S=100, K=100, T=1, r=0.0, sigma=0.2, kind="call")
        self.assertAlmostEqual(r.price, 7.9656, places=3)
        self.assertIn("delta", r.greeks)


class TestHill(unittest.TestCase):
    def test_recovers_known_tail_index(self):
        # Pareto(alpha=3): Hill estimator should recover ~3.
        rng = np.random.default_rng(0)
        x = (1.0 / rng.random(200_000) ** (1 / 3.0))     # Pareto tail alpha=3
        losses = x
        a = hill_alpha(losses, k=2000)
        self.assertTrue(2.6 < a < 3.4, f"alpha={a}")

    def test_agent_scales_and_may_veto(self):
        ohlcv = synthetic_ohlcv(bars=1500, seed=1)
        sig = HillTailAgent().evaluate("TEST", ohlcv)
        self.assertIsInstance(sig, RiskSignal)
        self.assertTrue(0.0 <= sig.risk_scale <= 1.0)
        self.assertIn("alpha", sig.diagnostics)

    def test_agent_abstains_on_short_series(self):
        ohlcv = synthetic_ohlcv(bars=100)
        sig = HillTailAgent(min_bars=500).evaluate("TEST", ohlcv)
        self.assertEqual(sig.risk_scale, 1.0)  # abstain = no effect
        self.assertFalse(sig.veto)


class TestFeatures(unittest.TestCase):
    def test_feature_panel_columns(self):
        f = make_features(synthetic_ohlcv(300))
        for col in ["ret_1d", "ret_5d", "vol_20d", "rsi_14", "dist_sma50", "volume_z"]:
            self.assertIn(col, f.columns)

    def test_temporal_split_is_chronological(self):
        ohlcv = synthetic_ohlcv(500)
        X, y = make_features(ohlcv), label_forward(ohlcv, 5)
        Xtr, ytr, Xte, yte = temporal_split(X, y, 0.7)
        self.assertTrue(Xtr.index.max() <= Xte.index.min())


class TestDecisionTree(unittest.TestCase):
    def test_returns_valid_directional_signal(self):
        sig = DecisionTreeAgent().evaluate("TEST", synthetic_ohlcv(1200, seed=2))
        self.assertIsInstance(sig, DirectionalSignal)
        self.assertIn(sig.direction, list(Direction))
        self.assertTrue(0.0 <= sig.conviction <= 1.0)

    def test_abstains_when_too_short(self):
        sig = DecisionTreeAgent(min_bars=400).evaluate("TEST", synthetic_ohlcv(200))
        self.assertEqual(sig.direction, Direction.ABSTAIN)


class TestCommittee(unittest.TestCase):
    def test_weighted_vote_and_neutral_band(self):
        # Two fake agents: build directly via a tiny stub.
        class Stub(DecisionTreeAgent):
            def __init__(self, d, conv, w=1.0):
                self._d, self._c, self.weight = d, conv, w
                self.name = f"stub_{d.value}"
            def evaluate(self, instrument, ohlcv, **k):
                return DirectionalSignal(self.name, instrument, self._d, self._c)

        bull = Stub(Direction.BULL, 0.9, 1.0)
        bear = Stub(Direction.BEAR, 0.3, 1.0)
        c = Committee([bull, bear])
        dec = c.decide("TEST", None)
        self.assertEqual(dec.direction, Direction.BULL)   # net +0.3 > band
        self.assertGreater(dec.conviction, 0)

    def test_risk_veto_forces_flat(self):
        class BullStub(DecisionTreeAgent):
            name = "bull"
            def evaluate(self, instrument, ohlcv, **k):
                return DirectionalSignal("bull", instrument, Direction.BULL, 1.0)

        class VetoRisk(HillTailAgent):
            name = "veto"
            def evaluate(self, instrument, ohlcv, **k):
                return RiskSignal("veto", instrument, risk_scale=0.0, veto=True, rationale="test veto")

        dec = Committee([BullStub()], [VetoRisk()]).decide("TEST", None)
        self.assertTrue(dec.vetoed)
        self.assertEqual(dec.direction, Direction.NEUTRAL)
        self.assertEqual(dec.risk_scale, 0.0)

    def test_all_abstain_is_neutral(self):
        class AbstainStub(DecisionTreeAgent):
            name = "ab"
            def evaluate(self, instrument, ohlcv, **k):
                return self._abstain(instrument, "no data")
        dec = Committee([AbstainStub()]).decide("TEST", None)
        self.assertEqual(dec.direction, Direction.NEUTRAL)
        self.assertEqual(dec.conviction, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
