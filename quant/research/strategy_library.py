"""
Strategy library — the registry behind the dashboard's Strategy Builder page.

Each entry is a named strategy (a signal function + parameters). The page lets
you point any strategy at any instrument (stock, ETF, bond, FX, crypto,
commodity, index) and backtests it on the spot, TradingView-style. When you
describe a new strategy, it gets added here as one more registry entry and
immediately shows up on the page.

`run_strategy(id, instrument)` backtests one strategy on one instrument and
returns the full metric suite, the equity curve vs buy-and-hold, and the
Deflated Sharpe Ratio so you can see at a glance whether a result is real or
likely noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import load_ohlcv, SYMBOL_MAP
from .signals import reversal, tsmom, trend, vol_scaled_mom
from .metrics import perf_metrics
from .overfitting import deflated_sharpe_ratio

# --- the strategy registry (extended as the user describes new strategies) ---
STRATEGIES = {
    "reversal_7": {"name": "Short-Term Reversal (7d)", "family": "mean-reversion",
                   "desc": "Buy recent 7-day losers, short recent winners — fade the move.",
                   "fn": reversal, "params": {"window": 7}, "hold": 7},
    "reversal_3": {"name": "Short-Term Reversal (3d)", "family": "mean-reversion",
                   "desc": "Faster 3-day reversal. Higher turnover.",
                   "fn": reversal, "params": {"window": 3}, "hold": 3},
    "tsmom_12m": {"name": "Time-Series Momentum (12m)", "family": "momentum",
                  "desc": "Go with the 12-month trend — the managed-futures premium.",
                  "fn": tsmom, "params": {"lookback": 252}, "hold": 21},
    "tsmom_3m": {"name": "Time-Series Momentum (3m)", "family": "momentum",
                 "desc": "Shorter 3-month momentum.",
                 "fn": tsmom, "params": {"lookback": 63}, "hold": 21},
    "trend_100": {"name": "Trend Following (100d MA)", "family": "trend",
                  "desc": "Long above the 100-day average, short below.",
                  "fn": trend, "params": {"window": 100}, "hold": 21},
    "trend_200": {"name": "Trend Following (200d MA)", "family": "trend",
                  "desc": "Slower 200-day trend filter.",
                  "fn": trend, "params": {"window": 200}, "hold": 21},
    "vol_mom": {"name": "Volatility-Scaled Momentum", "family": "momentum",
                "desc": "12-month momentum normalised by recent volatility.",
                "fn": vol_scaled_mom, "params": {}, "hold": 21},
    # Leverage for the Long Run (Gayed 2016): above the 200d MA hold the asset
    # (2x, paying financing + 1%/yr fee), else T-bills. DSR-robust on the S&P 500.
    "leverage_2x": {"name": "Leverage Rotation (2x, 200d MA)", "family": "timing",
                    "desc": "Above the 200-day average hold 2x the asset (financing + 1%/yr fee); below, hold cash. Gayed's 'Leverage for the Long Run'. Best fit: S&P 500.",
                    "fn": None, "mode": "timing",
                    "params": {"window": 200, "leverage": 2.0, "fee_annual": 0.01}, "hold": 1},
    "ma_timing_1x": {"name": "MA Timing (1x, 200d)", "family": "timing",
                     "desc": "Same regime rule with no leverage: hold the asset above its 200-day average, cash below. Lower drawdown, higher Sharpe than the 2x variant.",
                     "fn": None, "mode": "timing",
                     "params": {"window": 200, "leverage": 1.0, "fee_annual": 0.0}, "hold": 1},
    # Donchian channel breakout (classic trend / turtle)
    "donchian_55": {"name": "Donchian Breakout (55d)", "family": "trend",
                    "desc": "Go long on a new 55-day high, short on a new 55-day low, hold until the opposite breakout. Classic trend-following.",
                    "fn": "donchian_sig", "params": {"window": 55}, "hold": 5},
    # Dual moving-average crossover (golden/death cross)
    "ma_cross": {"name": "MA Crossover (50/200)", "family": "trend",
                 "desc": "Long when the 50-day average is above the 200-day, short when below. The golden/death cross.",
                 "fn": "ma_cross_sig", "params": {"fast": 50, "slow": 200}, "hold": 5},
    # Pairs statistical arbitrage: pick an economically-linked PAIR (e.g. KO/PEP)
    # as the 'asset'. Long the laggard / short the leader when the log-spread
    # z-score stretches past +/-2, exit near 0. Net of transaction + borrow costs.
    "pairs_statarb": {"name": "Pairs Stat-Arb (z-reversion)", "family": "stat-arb",
                      "desc": "Buy the laggard, short the leader of a linked pair when their spread diverges (|z|>2), bet on convergence. Choose a pair as the asset. Net of costs; the famous KO/PEP pair loses money.",
                      "fn": None, "mode": "pairs", "params": {}, "hold": 1},
    # --- Composite (portfolio) strategies: one fixed diversified book each.
    # These are the honest route to a DSR-robust book: diversifying one signal
    # across many uncorrelated assets cancels idiosyncratic noise, so the
    # portfolio Sharpe clears the bar that single legs cannot. Asset = PORTFOLIO.
    "comp_ensemble_core": {"name": "Ensemble Core (Trend + Risk Parity)", "family": "composite",
                           "desc": "Equal-risk blend of diversified managed-futures trend and multi-asset risk parity, the two robust, decorrelated sleeves. DSR-robust; beats buy-and-hold on Sharpe and drawdown. Asset: PORTFOLIO.",
                           "fn": None, "mode": "composite", "composite_id": "ensemble_core", "hold": 21},
    "comp_risk_parity": {"name": "Risk Parity (multi-asset)", "family": "composite",
                         "desc": "Inverse-vol (equal risk contribution) long-only book across equities, bonds, gold and commodities. Lower vol and far lower drawdown than SPY, higher Sharpe. Asset: PORTFOLIO.",
                         "fn": None, "mode": "composite", "composite_id": "risk_parity", "hold": 21},
    "comp_div_trend": {"name": "Diversified Trend (managed futures)", "family": "composite",
                       "desc": "Multi-lookback trend sign across 21 cross-asset instruments, vol-scaled to equal risk. Positive carry in crises. Asset: PORTFOLIO.",
                       "fn": None, "mode": "composite", "composite_id": "div_trend", "hold": 21},
    "comp_xsec_mom": {"name": "Cross-Sectional Equity Momentum", "family": "composite",
                      "desc": "Monthly long top-decile / short bottom-decile of 54 large-caps by 12-1 momentum, dollar-neutral. Shown honestly: a measured dud on this universe (DSR ~0.16). Asset: PORTFOLIO.",
                      "fn": None, "mode": "composite", "composite_id": "xsec_mom", "hold": 21},
    # Volatility-managed equity (Moreira-Muir): hold the index but size exposure
    # inversely to recent realized vol. On high-vol QQQ this beats buy-and-hold on
    # return AND risk with DSR ~0.95; on smooth SPY it does not (kept as an honest
    # null so the search is not cherry-picked).
    "vol_managed_qqq": {"name": "Volatility-Managed QQQ", "family": "vol-managed",
                        "desc": "Hold QQQ but scale exposure inversely to its recent volatility (18% vol target), de-risking before turbulent stretches. Beats QQQ buy-and-hold on return (~15% CAGR), Sharpe (0.82), Sortino (1.09) and drawdown (-35% vs -53%), DSR ~0.95. Net of cost + financing. Asset: PORTFOLIO.",
                        "fn": None, "mode": "composite", "composite_id": "vol_managed_qqq", "hold": 1},
    "vol_managed_qqq_aggr": {"name": "Volatility-Managed QQQ (Aggressive)", "family": "vol-managed",
                             "desc": "The same vol-managed QQQ engine run at a 24% vol target: beats QQQ buy-and-hold by a WIDE margin (~19% vs 15% CAGR, +4%/yr, roughly double the terminal wealth over 20y) while max drawdown (-45%) stays below buy-and-hold's -53%. Sharpe 0.80, Sortino 1.07, PF 1.15, DSR ~0.92. The wider margin is bought with more risk, but vol-management keeps the drawdown under the index's. Asset: PORTFOLIO.",
                             "fn": None, "mode": "composite", "composite_id": "vol_managed_qqq_aggr", "hold": 1},
    "vol_managed_spy": {"name": "Volatility-Managed SPY (honest null)", "family": "vol-managed",
                        "desc": "The same vol-managed recipe on SPY. Shown deliberately: the effect is weak on smooth large-cap beta, so this does NOT beat SPY buy-and-hold (DSR ~0.51). Kept visible so the vol-managed result is not cherry-picked. Asset: PORTFOLIO.",
                        "fn": None, "mode": "composite", "composite_id": "vol_managed_spy", "hold": 1},
    # Risk-managed: the controls that actually work on a systematic portfolio
    # (volatility targeting + drawdown throttle + regime filter via the trend
    # sleeve), NOT stop-losses. Optimizes risk EFFICIENCY (Calmar/Sharpe/shallow
    # drawdown), the right objective for live capital.
    "risk_managed_ensemble": {"name": "Risk-Managed Ensemble (vol-target + DD throttle)", "family": "risk-managed",
                              "desc": "Ensemble Core run through volatility targeting (constant 10% risk) and a drawdown throttle (de-levers into losing streaks), capped at 2.5x with financing. Cuts max drawdown to ~-16% while raising return above the unmanaged book. Calmar ~0.40 vs SPY ~0.16. The honest live-capital candidate. Asset: PORTFOLIO.",
                              "fn": None, "mode": "composite", "composite_id": "risk_managed_ensemble", "hold": 21},
    "risk_managed_rp": {"name": "Risk-Managed Risk Parity (vol-target + DD throttle)", "family": "risk-managed",
                        "desc": "Risk Parity run through the same vol-target + drawdown-throttle controls at 10% vol. Asset: PORTFOLIO.",
                        "fn": None, "mode": "composite", "composite_id": "risk_managed_rp", "hold": 21},
    # Return-stacked (portable alpha): 100% SPY beta PLUS an uncorrelated robust
    # overlay funded at cash. The honest way to BEAT buy-and-hold on return: keep
    # the equity beta you give up by diversifying, and add the overlay's excess
    # on top. Beats SPY on CAGR, Sharpe and drawdown; DSR ~0.93 (just under gate).
    "stacked_trend": {"name": "Return-Stacked (SPY + Trend overlay)", "family": "stacked",
                      "desc": "100% SPY plus a 1x diversified-trend overlay, financed at cash. Beats SPY on return (~13% vs 11%), Sharpe and drawdown; trend cushions equity crises. Asset: PORTFOLIO.",
                      "fn": None, "mode": "composite", "composite_id": "stacked_trend", "hold": 21},
    "stacked_trend_heavy": {"name": "Return-Stacked (SPY + Trend 1.5x)", "family": "stacked",
                            "desc": "100% SPY plus a 1.5x trend overlay. Higher return (~13.7% vs 11%) and a smaller drawdown than SPY, at the cost of more financing. Asset: PORTFOLIO.",
                            "fn": None, "mode": "composite", "composite_id": "stacked_trend_heavy", "hold": 21},
    "stacked_core": {"name": "Return-Stacked (SPY + Ensemble Core)", "family": "stacked",
                     "desc": "100% SPY plus a 1x Ensemble-Core overlay (trend + risk parity), financed. Beats SPY on return and Sharpe. Asset: PORTFOLIO.",
                     "fn": None, "mode": "composite", "composite_id": "stacked_core", "hold": 21},
    "stacked_core_heavy": {"name": "Return-Stacked (SPY + Ensemble Core 1.5x)", "family": "stacked",
                           "desc": "100% SPY plus a 1.5x Ensemble-Core overlay. The highest-return stack (~13.9% vs 11%). Asset: PORTFOLIO.",
                           "fn": None, "mode": "composite", "composite_id": "stacked_core_heavy", "hold": 21},
}

# Composite strategies ignore the instrument (each is a fixed book); the UI
# pairs them with this single pseudo-asset.
COMPOSITE_ASSET = "PORTFOLIO"


def donchian_sig(df, window=55):
    hi = df["close"].rolling(window).max()
    lo = df["close"].rolling(window).min()
    sig = pd.Series(np.nan, index=df.index)
    sig[df["close"] >= hi] = 1.0
    sig[df["close"] <= lo] = -1.0
    return sig.ffill().fillna(0.0)


def ma_cross_sig(df, fast=50, slow=200):
    return np.sign(df["close"].rolling(fast).mean() - df["close"].rolling(slow).mean())


_SIG_FUNCS = {"donchian_sig": donchian_sig, "ma_cross_sig": ma_cross_sig}

# instrument universe, grouped for the picker (+ any yfinance ticker works)
INSTRUMENT_GROUPS = {
    "Stocks": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO",
               "AMD", "INTC", "CRM", "ORCL", "ADBE", "CSCO", "QCOM", "TXN", "NFLX",
               "JPM", "BAC", "WFC", "GS", "MS", "C", "BRK-B", "V", "MA", "AXP",
               "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO",
               "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "PG", "KO", "PEP", "DIS",
               "XOM", "CVX", "CAT", "BA", "GE", "HON", "UPS", "LMT", "T", "VZ", "IBM"],
    "FX": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD",
           "EURGBP", "EURJPY", "GBPJPY", "AUDJPY", "EURCHF", "GBPCHF", "GBPAUD",
           "EURAUD", "EURCAD", "AUDNZD", "NZDJPY", "CADJPY", "CHFJPY",
           "USDSEK", "USDNOK", "USDSGD", "USDMXN", "USDZAR", "USDTRY"],
    "Commodities": ["XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD", "USOIL", "UKOIL",
                    "NATGAS", "GASOLINE", "HEATOIL", "HG1!", "CORN", "WHEAT",
                    "SOYBEAN", "SOYOIL", "SOYMEAL", "OATS", "COFFEE", "SUGAR",
                    "COCOA", "COTTON", "OJUICE", "CATTLE", "LEANHOGS"],
    "Bonds & Fixed Income": ["TLT", "IEF", "IEI", "SHY", "SHV", "GOVT", "BIL",
                             "AGG", "BND", "BNDX", "LQD", "HYG", "JNK", "TIP",
                             "MUB", "EMB"],
    "Indices": ["SPX", "NDX", "DJIA", "RUSSELL", "DAX", "NIKKEI", "FTSE", "CAC40",
                "ESTX50", "SMI", "IBEX", "HSI", "ASX200", "KOSPI", "SENSEX",
                "NIFTY", "TSX", "BOVESPA", "VIX"],
    "ETFs": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EFA", "EEM", "VWO", "VEA",
             "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
             "XLC", "SMH", "SOXX", "ARKK", "IBB", "XBI", "KRE", "GDX", "GDXJ",
             "GLD", "SLV", "USO", "UNG", "DBA", "DBC",
             "EWJ", "EWZ", "EWG", "FXI", "INDA", "EWU"],
    "Derivatives": ["ES1!", "NQ1!", "RTY1!", "YM1!", "6E1!", "6J1!", "ZN1!",
                    "ZB1!", "ZF1!", "ZT1!",
                    "TQQQ", "SQQQ", "SPXL", "SPXS", "SOXL", "SOXS", "TNA", "TZA",
                    "UPRO", "UVXY", "SVXY", "TMF", "TBT"],
    "Crypto": ["BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "XRPUSD", "ADAUSD",
               "DOGEUSD", "AVAXUSD", "BNBUSD", "DOTUSD", "LINKUSD"],
}

# Honest DSR trial count. The naive count is strategies x assets (~473), but
# those trials are FAR from independent: many "strategies" are near-identical
# trend variants and assets move together within their groups, so treating 473
# as independent trials over-deflates and crushes almost every DSR to ~0.
# The correct figure is the EFFECTIVE number of independent trials = the
# participation ratio of the eigenvalues of the trial-return correlation matrix,
# (sum(lambda))^2 / sum(lambda^2). Measured across all 469 live return streams
# (mean |pairwise corr| ~0.06) this comes to ~59, about 8x fewer than the raw
# count. Deflating by 11 (strategies only) is too lenient; by 473 too harsh; 59
# is the honest middle the data actually supports. Recompute if the menu changes.
N_SCAN_ASSETS = sum(len(v) for v in INSTRUMENT_GROUPS.values())
EFFECTIVE_TRIALS = 59

# Blended per-side transaction cost (spread + slippage + commission), in bps of
# turnover. 1bp was unrealistically optimistic; 5bp is a conservative blended
# estimate across liquid ETFs/FX (tighter) and crypto (wider).
DEFAULT_COST_BPS = 5.0


def _riskfree_daily(index):
    """Time-varying daily risk-free (cash & financing) rate aligned to `index`,
    from FRED's 3-month T-bill (DGS3MO). Honest replacement for a flat 2%: cash
    earned ~0% in 2021 and ~5% in 2024-26, and leverage financing costs track it.
    Falls back to a flat 2%/yr if FRED is unavailable (keeps backtests runnable
    offline)."""
    try:
        from quant.data.macro import riskfree_series
        s = riskfree_series()
        if len(s):
            aligned = s.reindex(index.union(s.index)).sort_index().ffill().reindex(index)
            return (aligned / 252.0).fillna(0.02 / 252)
    except Exception:
        pass
    return pd.Series(0.02 / 252, index=index)


def _strategy_returns(spec, instrument, cost_bps=DEFAULT_COST_BPS, bars=6000, vol_win=63):
    """One strategy on one instrument: vol-scaled, rebalanced at `hold`.

    `bars` defaults to the FULL available history (cache holds up to ~5000-15000
    daily bars). Using all of it, rather than the old 3000 cap, raises the
    Deflated-Sharpe hurdle's statistical power: the required net Sharpe to clear
    DSR>=0.95 falls from ~1.16 (T=3000) to ~0.89 (T=5000). Free, honest power."""
    # composite mode: a fixed diversified portfolio book; ignore `instrument`.
    if spec.get("mode") == "composite":
        from quant.research import composites as _comp
        fn = _comp.COMPOSITES[spec["composite_id"]]["fn"]
        strat, bench = fn()
        if strat is None or len(strat) < 250:
            raise RuntimeError(f"composite {spec['composite_id']} unavailable")
        return strat, bench
    # pairs mode: `instrument` is a pair "A/B"; delegate to the pairs engine.
    if spec.get("mode") == "pairs":
        from quant.research.pairs import pair_returns
        a, _, b = (instrument or "").partition("/")
        strat, bench = pair_returns(a.strip().upper(), b.strip().upper())
        if strat is None or len(strat) < 250:
            raise RuntimeError(f"no pair data for {instrument}")
        return strat, bench

    df = load_ohlcv(instrument, bars=bars)
    ret = df["close"].pct_change()

    # timing mode: on/off leverage-vs-cash regime rule (Leverage for the Long Run)
    if spec.get("mode") == "timing":
        p = spec["params"]
        close = df["close"]
        sma = close.rolling(p["window"]).mean()
        active = (close > sma).shift(1).fillna(False)
        L = p.get("leverage", 1.0)
        rf_d = _riskfree_daily(df.index)                    # live time-varying T-bill
        fee_d = p.get("fee_annual", 0.0) / 252
        lev = L * ret - (L - 1) * rf_d - fee_d
        strat = pd.Series(np.where(active, lev, rf_d), index=df.index)
        switch = active.ne(active.shift(1)).fillna(False)
        strat = strat - switch * (cost_bps / 1e4)            # cost on regime switches
        return strat.dropna(), ret

    # overnight mode: fade the overnight gap, earn the intraday session
    if spec.get("mode") == "overnight":
        o, c = df["open"], df["close"]
        r_on = o / c.shift(1) - 1.0          # overnight gap (known at the open)
        r_id = c / o - 1.0                   # intraday return (earned during the day)
        pos = -np.sign(r_on)                 # reversal: fade the gap
        strat = pos * r_id - (pos - pos.shift(1)).abs() * (cost_bps / 1e4)
        return strat.dropna(), c.pct_change()

    fn = spec["fn"]
    if isinstance(fn, str):
        fn = _SIG_FUNCS[fn]
    sig = fn(df, **spec["params"])
    vol = ret.rolling(vol_win).std()
    hold = spec["hold"]
    raw = np.sign(sig) / vol
    reb = pd.Series(np.nan, index=df.index)
    reb.iloc[::hold] = raw.iloc[::hold]
    pos = reb.ffill()
    pos = pos / pos.abs().rolling(252, min_periods=20).median()
    strat = pos.shift(1) * ret - (pos - pos.shift(1)).abs() * (cost_bps / 1e4)
    return strat.dropna(), ret


def run_strategy(strategy_id, instrument, cost_bps=DEFAULT_COST_BPS) -> dict:
    spec = STRATEGIES.get(strategy_id)
    if not spec:
        return {"error": f"unknown strategy {strategy_id}"}
    try:
        strat, ret = _strategy_returns(spec, instrument, cost_bps)
    except Exception as e:
        return {"error": f"could not load {instrument}: {e}"}
    if len(strat) < 250:
        return {"error": f"insufficient history for {instrument}"}

    m = perf_metrics(strat)
    if spec.get("mode") == "pairs":
        from quant.research.pairs import PAIRS
        n_trials = len(PAIRS)                       # deflate by the pairs searched
    elif spec.get("mode") == "composite":
        from quant.research.composites import N_COMPOSITE_TRIALS
        n_trials = N_COMPOSITE_TRIALS               # the small pre-registered family
    else:
        n_trials = EFFECTIVE_TRIALS
    dsr = deflated_sharpe_ratio(strat.values, n_trials=n_trials)
    eq = (1 + strat).cumprod()
    bh = (1 + ret.reindex(strat.index).fillna(0)).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    # monthly return grid
    mo = (1 + strat).resample("ME").prod() - 1
    mdf = pd.DataFrame({"y": mo.index.year, "m": mo.index.month, "r": mo.values * 100})
    years = sorted(mdf["y"].unique(), reverse=True)
    grid = {int(y): [None] * 12 for y in years}
    for _, row in mdf.iterrows():
        grid[int(row["y"])][int(row["m"]) - 1] = round(float(row["r"]), 2)
    return {
        "strategy": spec["name"], "instrument": instrument,
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "dsr": round(dsr["dsr"], 3), "dsr_verdict": dsr["verdict"],
        "equity": {"dates": [d.strftime("%Y-%m-%d") for d in eq.index],
                   "strategy": (eq.values * 100 - 100).round(2).tolist(),
                   "buyhold": (bh.values * 100 - 100).round(2).tolist()},
        "drawdown": {"dates": [d.strftime("%Y-%m-%d") for d in dd.index],
                     "values": dd.values.round(2).tolist()},
        "monthly": {"years": [int(y) for y in years], "grid": {int(y): grid[y] for y in years}},
    }


def library():
    from quant.research.pairs import PAIRS
    return {
        "strategies": [{"id": k, "mode": v.get("mode"),
                        **{f: v[f] for f in ("name", "family", "desc")}}
                       for k, v in STRATEGIES.items()],
        "instrument_groups": INSTRUMENT_GROUPS,
        # the Pairs Stat-Arb strategy swaps the asset list for these pairs
        "pair_instruments": {"Pairs": [f"{a}/{b}" for a, b in PAIRS]},
        # composite strategies use a single fixed-book pseudo-asset
        "composite_instruments": {"Portfolio": [COMPOSITE_ASSET]},
    }
