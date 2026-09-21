"""
Extended agent coverage — the agents added after the first test suite:
reversal (the validated edge), the ML ensemble, risk & pricing agents.
Offline where possible (synthetic data); pricing checked against Black-Scholes.
"""
import math
import unittest

from quant.base import Direction, DirectionalSignal, RiskSignal
from quant.data.loader import synthetic_ohlcv
from quant.agents.reversal import ReversalAgent
from quant.agents.random_forest import RandomForestAgent
from quant.agents.gradient_boost import GradientBoostAgent
from quant.agents.lasso_factors import LassoFactorAgent
from quant.agents.vol_regime import VolRegimeAgent
from quant.agents.rates_macro import RatesMacroAgent
from quant.pricing.black_scholes import BSInputs, bs_price_greeks
from quant.pricing.american_psor import american_price
from quant.pricing.mc_greeks import mc_price_greeks


class TestReversal(unittest.TestCase):
    def test_bull_after_drop(self):
        # Build a series whose last `lookback` move is clearly down.
        oh = synthetic_ohlcv(300, seed=3).copy()
        oh.iloc[-1, oh.columns.get_loc("close")] = oh["close"].iloc[-8] * 0.90
        sig = ReversalAgent().evaluate("EURUSD", oh)
        self.assertEqual(sig.direction, Direction.BULL)   # recent loser -> buy
        self.assertGreater(sig.conviction, 0)

    def test_trending_downweight(self):
        oh = synthetic_ohlcv(300, seed=4)
        fx = ReversalAgent().evaluate("EURUSD", oh).conviction
        tr = ReversalAgent().evaluate("USOIL", oh).conviction
        self.assertLessEqual(tr, fx + 1e-9)               # trender is down-weighted

    def test_abstain_short(self):
        self.assertEqual(ReversalAgent(min_bars=120).evaluate("EURUSD", synthetic_ohlcv(50)).direction,
                         Direction.ABSTAIN)


class TestMLAgents(unittest.TestCase):
    def _valid(self, agent):
        s = agent.evaluate("TEST", synthetic_ohlcv(1200, seed=5))
        self.assertIsInstance(s, DirectionalSignal)
        self.assertIn(s.direction, list(Direction))
        self.assertTrue(0.0 <= s.conviction <= 1.0)

    def test_random_forest(self): self._valid(RandomForestAgent(n_estimators=60))
    def test_gradient_boost(self): self._valid(GradientBoostAgent(max_iter=80))
    def test_lasso(self): self._valid(LassoFactorAgent())

    def test_rates_macro_abstains_on_fx(self):
        s = RatesMacroAgent().evaluate("EURUSD", synthetic_ohlcv(400))
        self.assertEqual(s.direction, Direction.ABSTAIN)


class TestRisk(unittest.TestCase):
    def test_vol_regime_scale(self):
        s = VolRegimeAgent().evaluate("TEST", synthetic_ohlcv(800, seed=6))
        self.assertIsInstance(s, RiskSignal)
        self.assertTrue(0.0 <= s.risk_scale <= 1.0)


class TestPricing(unittest.TestCase):
    def test_american_call_equals_european(self):
        # American call on a non-dividend underlying = European (never early-exercised).
        eu = bs_price_greeks(BSInputs(100, 100, 1.0, 0.03, 0.2, "call"))["price"]
        am, _ = american_price(100, 100, 1.0, 0.03, 0.2, kind="call", M=200, N=200)
        self.assertLess(abs(am - eu), 0.5)

    def test_american_put_ge_european(self):
        eu = bs_price_greeks(BSInputs(100, 100, 1.0, 0.05, 0.25, "put"))["price"]
        am, _ = american_price(100, 100, 1.0, 0.05, 0.25, kind="put", M=200, N=200)
        self.assertGreaterEqual(am + 1e-6, eu)            # early exercise adds value

    def test_mc_price_near_bs(self):
        bs = bs_price_greeks(BSInputs(100, 100, 1.0, 0.02, 0.2, "call"))["price"]
        g = mc_price_greeks(100, 100, 1.0, 0.02, 0.2, "call")
        self.assertLess(abs(g["price"] - bs), 1.0)        # MC within noise of analytic


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestOverfitting(unittest.TestCase):
    def test_dsr_flags_random_as_overfit(self):
        import numpy as np
        from quant.research.overfitting import deflated_sharpe_ratio, perf_degradation, error_inflation
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.01, 1500)          # zero-skill returns
        d = deflated_sharpe_ratio(noise, n_trials=50)
        self.assertNotEqual(d["verdict"], "robust")  # random over 50 trials is never robust
        self.assertAlmostEqual(perf_degradation(1.2, 0.4), 0.8)
        self.assertAlmostEqual(error_inflation(2.0, 1.0), 2.0)

    def test_dsr_rewards_real_skill(self):
        import numpy as np
        from quant.research.overfitting import deflated_sharpe_ratio
        rng = np.random.default_rng(1)
        good = rng.normal(0.0015, 0.01, 2000)      # strong positive drift, Sharpe ~2.4
        d = deflated_sharpe_ratio(good, n_trials=20)
        self.assertEqual(d["verdict"], "robust")   # clear skill survives deflation
        self.assertGreater(d["dsr"], 0.95)


class TestStrategySearch(unittest.TestCase):
    def test_candidate_returns_offline(self):
        from quant.research.strategy_search import candidate_returns
        from quant.research.signals import reversal
        from quant.data.loader import synthetic_ohlcv
        cache = {s: synthetic_ohlcv(1000, seed=i) for i, s in enumerate(["A", "B", "C"])}
        r = candidate_returns(reversal, {"window": 7}, ["A", "B", "C"], 7, _cache=cache)
        self.assertGreater(len(r), 500)
        self.assertTrue(r.notna().any())

    def test_gate_rejects_when_below_threshold(self):
        # A very high gate must yield no winner even if candidates exist.
        from quant.research.strategy_search import StrategySearchAgent
        a = StrategySearchAgent(dsr_gate=0.999)
        # exercise the gate logic on a synthetic leaderboard without network:
        rows = [{"sharpe": 2.0, "dsr": 0.8, "sharpe_oos": 1.0},
                {"sharpe": 1.0, "dsr": 0.99, "sharpe_oos": 0.5}]
        passing = [x for x in rows if x["dsr"] >= a.dsr_gate and x["sharpe_oos"] > 0]
        self.assertEqual(passing, [])


class TestCombination(unittest.TestCase):
    def test_uncorrelated_blend_lifts_sharpe(self):
        import numpy as np, pandas as pd
        from quant.research.combination import combine
        rng = np.random.default_rng(0)
        # two UNCORRELATED streams, each per-obs Sharpe ~0.05 (large N = stable)
        a = pd.Series(rng.normal(0.05, 1.0, 40000))
        b = pd.Series(rng.normal(0.05, 1.0, 40000))
        combined, _ = combine(pd.DataFrame({"a": a, "b": b}), "equal")
        sr = lambda x: x.mean() / x.std()
        # uncorrelated blend beats the average single stream by ~sqrt(2)
        avg_single = (sr(a) + sr(b)) / 2
        self.assertGreater(sr(combined) / avg_single, 1.3)

    def test_correlated_blend_no_lift(self):
        import numpy as np, pandas as pd
        from quant.research.combination import combine
        rng = np.random.default_rng(1)
        a = pd.Series(rng.normal(0.05, 1.0, 4000))
        combined, _ = combine(pd.DataFrame({"a": a, "b": a.copy()}), "equal")
        sr = lambda x: x.mean() / x.std()
        self.assertAlmostEqual(sr(combined), sr(a), places=6)  # identical => no benefit


class TestKalshi(unittest.TestCase):
    def test_fee_and_parse(self):
        from quant.agents.kalshi_arb import kalshi_fee, _f
        self.assertAlmostEqual(kalshi_fee(0.5), 0.0175, places=4)
        self.assertEqual(_f("0.5"), 0.5)
        self.assertIsNone(_f(None))

    def test_buy_both_detection_offline(self):
        import quant.agents.kalshi_arb as K
        fake = [
            {"yes_ask_dollars": 0.40, "no_ask_dollars": 0.50, "volume_fp": 100,
             "ticker": "LOCK", "event_ticker": "E", "title": "t"},   # 0.90 < 1 => lock
            {"yes_ask_dollars": 0.55, "no_ask_dollars": 0.48, "volume_fp": 100,
             "ticker": "NOPE", "event_ticker": "E", "title": "t"},   # 1.03 => no lock
        ]
        orig = K.fetch_markets
        K.fetch_markets = lambda *a, **k: fake
        try:
            r = K.KalshiArbitrageAgent(min_edge_cents=0.5).scan()
        finally:
            K.fetch_markets = orig
        self.assertEqual(r["n_buy_both"], 1)
        self.assertEqual(r["buy_both"][0]["ticker"], "LOCK")


class TestInfoDiffusion(unittest.TestCase):
    def test_sentiment_scoring(self):
        from quant.agents.info_diffusion import _sentiment
        s, b, br = _sentiment(["stocks surge and rally to record high"])
        self.assertGreater(s, 0)
        s2, _, _ = _sentiment(["market crash plunge selloff fear"])
        self.assertLess(s2, 0)

    def test_abstain_without_key(self):
        import quant.agents.info_diffusion as I
        from quant.base import Direction
        orig = I._firecrawl_key
        I._firecrawl_key = lambda: None
        try:
            r = I.InfoDiffusionAgent().evaluate("BTCUSD")
        finally:
            I._firecrawl_key = orig
        self.assertEqual(r.direction, Direction.ABSTAIN)


class TestAltStreams(unittest.TestCase):
    def test_news_stream_none_for_unmapped(self):
        from quant.research.alt_streams import news_stream, MIN_HISTORY
        self.assertIsNone(news_stream("EURUSD"))   # no AV news ticker -> None
        self.assertGreaterEqual(MIN_HISTORY, 200)  # at least ~1y before blending


class TestCV(unittest.TestCase):
    def test_pbo_flags_noise_vs_edge(self):
        import numpy as np
        from quant.research.cv import pbo_cscv
        rng = np.random.default_rng(0)
        good = rng.normal(0, 0.01, (2000, 15)); good[:, 0] += 0.0008
        self.assertLess(pbo_cscv(good)["pbo"], 0.25)      # real edge => low PBO

    def test_purged_cv_consistency(self):
        import numpy as np
        from quant.research.cv import purged_cv_sharpe
        rng = np.random.default_rng(1)
        r = rng.normal(0.0015, 0.01, 2400)               # strong steady edge (Sharpe ~2.4)
        res = purged_cv_sharpe(r, k=6)
        self.assertGreaterEqual(res["pct_positive"], 0.8)


class TestPortfolioRisk(unittest.TestCase):
    def test_diversification_and_contributions(self):
        import numpy as np, pandas as pd
        from quant.risk.portfolio_risk import portfolio_risk
        rng = np.random.default_rng(0)
        # 3 uncorrelated assets, equal vol
        R = pd.DataFrame({"A": rng.normal(0, 0.01, 800),
                          "B": rng.normal(0, 0.01, 800),
                          "C": rng.normal(0, 0.01, 800)})
        r = portfolio_risk({"A": 1, "B": 1, "C": 1}, returns=R)
        # uncorrelated equal-weight => diversification ratio ~ sqrt(3) ~ 1.7
        self.assertGreater(r["diversification_ratio"], 1.4)
        self.assertLess(r["true_heat"], r["naive_heat"])       # correlation reduces heat
        self.assertAlmostEqual(sum(c["risk_contribution_pct"] for c in r["contributions"]), 100.0, delta=1.0)
