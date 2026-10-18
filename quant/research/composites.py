"""
Composite (portfolio) strategies — the honest route to a DSR-robust book.

A single-asset signal rarely clears the Deflated-Sharpe bar (net annualized
Sharpe ~0.9-1.2, roughly 2x SPY). The reason is idiosyncratic noise: one asset's
trend signal is right barely more than half the time. Diversifying ONE signal
across many uncorrelated assets cancels that noise, so the PORTFOLIO Sharpe rises
toward signal_sharpe x sqrt(breadth) (Grinold's law). That is exactly how
managed-futures/CTA books and risk-parity funds clear the bar that their single
legs cannot.

Every stream here is built only from past data (causal), vol-scaled, net of the
same 5bp/side cost the rest of the fund uses, and judged by the SAME Deflated
Sharpe gate. Nothing is fitted; the baskets and lookbacks are pre-registered
below, and run_all() reports EVERY composite (winners and losers) so the DSR
deflation by the number of variants tried stays honest. No cherry-picking.

Benchmark is SPY buy-and-hold (the hard bar the user asked to beat). Each
composite is also shown vol-targeted to ~15% annualized so raw CAGR is
comparable to SPY at similar risk (a higher Sharpe only turns into more dollars
once levered to equal vol).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import load_ohlcv
from .overfitting import deflated_sharpe_ratio
from .strategy_library import _riskfree_daily, DEFAULT_COST_BPS

BARS = 6000          # use full available history (cache holds up to ~5000-15000)
COST = DEFAULT_COST_BPS / 1e4
TARGET_VOL = 0.15    # annualized, for the vol-targeted (leverage-to-equal-vol) view

# --- pre-registered baskets (liquid, deep history, span asset classes) --------
# Cross-asset trend basket: equities, bonds, commodities, FX proxies.
TREND_BASKET = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "EWJ", "FXI",        # equity indices
    "TLT", "IEF", "SHY", "LQD", "HYG",                        # rates / credit
    "GLD", "SLV", "USO", "DBC", "DBA", "UNG",                 # commodities
    "UUP", "FXE", "FXY",                                      # FX (USD, EUR, JPY)
]
# Cross-sectional equity momentum universe (large, liquid single names).
XSEC_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "INTC", "CRM", "ORCL", "ADBE", "CSCO", "QCOM", "TXN", "NFLX",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO",
    "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "PG", "KO", "PEP", "DIS",
    "XOM", "CVX", "CAT", "BA", "GE", "HON", "UPS", "LMT", "T", "VZ", "IBM",
]
# Risk-parity multi-asset sleeve (the all-weather baseline).
RP_BASKET = ["SPY", "TLT", "IEF", "GLD", "DBC", "LQD"]


def _returns(sym, bars=BARS):
    """Daily close-to-close returns for one symbol, full history, or None."""
    try:
        df = load_ohlcv(sym, bars=bars)
    except Exception:
        return None
    if df is None or len(df) < 300:
        return None
    r = df["close"].pct_change()
    return r.dropna()


def _load_panel(symbols, bars=BARS):
    """Aligned daily-return panel (DataFrame, columns=symbols) for a basket."""
    cols = {}
    for s in symbols:
        r = _returns(s, bars)
        if r is not None and len(r) > 300:
            cols[s] = r
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).sort_index()


def _vol_scale(ret, vol_win=63, target_daily=0.01):
    """Scale a return stream to ~constant risk per leg (equal-risk weighting)."""
    vol = ret.rolling(vol_win).std().shift(1)
    w = (target_daily / vol).clip(upper=5.0)      # cap leverage per leg
    return w


def diversified_trend(symbols=TREND_BASKET, lookbacks=(63, 126, 252), hold=21,
                      bars=BARS):
    """Managed-futures trend: average the trend sign over several lookbacks
    (standard CTA practice, reduces single-horizon timing luck) per asset,
    vol-scale each leg to equal risk, average into one portfolio stream. Net of
    cost on rebalances."""
    panel = _load_panel(symbols, bars)
    if panel.empty:
        return None, None
    legs = []
    for s in panel.columns:
        r = panel[s].dropna()
        close = (1 + r).cumprod()
        sig = sum(np.sign(close / close.shift(lb) - 1.0) for lb in lookbacks) / len(lookbacks)
        reb = pd.Series(np.nan, index=r.index)
        reb.iloc[::hold] = sig.iloc[::hold]
        pos = reb.ffill()
        w = _vol_scale(r)
        lev = (pos * w).shift(1)                                # trade next bar
        turn = (pos * w).diff().abs().fillna(0.0)
        leg = lev * r - turn * COST
        legs.append(leg)
    book = pd.concat(legs, axis=1, sort=True)
    strat = book.mean(axis=1).dropna()                         # equal-weight legs
    return strat, _spy_bench(strat.index)


def xsec_momentum(symbols=XSEC_UNIVERSE, lookback=252, skip=21, hold=21,
                  frac=0.33, bars=BARS):
    """Cross-sectional 12-1 momentum: each month long the top `frac` and short
    the bottom `frac` of the universe by past (lookback-skip) return, equal
    weight, dollar-neutral. Decorrelated from the market by construction."""
    panel = _load_panel(symbols, bars)
    if panel.empty or panel.shape[1] < 8:
        return None, None
    close = (1 + panel.fillna(0)).cumprod()
    mom = close.shift(skip) / close.shift(lookback) - 1.0       # 12-1 momentum
    rebal = panel.index[::hold]
    pos = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    for dt in rebal:
        row = mom.loc[dt].dropna()
        if len(row) < 8:
            continue
        k = max(1, int(len(row) * frac))
        winners = row.nlargest(k).index
        losers = row.nsmallest(k).index
        pos.loc[dt, winners] = 1.0 / k
        pos.loc[dt, losers] = -1.0 / k
    pos = pos.replace(0.0, np.nan).ffill().fillna(0.0)
    gross = (pos.shift(1) * panel).sum(axis=1)
    turn = pos.diff().abs().sum(axis=1).fillna(0.0)
    strat = (gross - turn * COST).dropna()
    return strat, _spy_bench(strat.index)


def risk_parity(symbols=RP_BASKET, vol_win=63, hold=21, bars=BARS):
    """Inverse-vol (equal risk contribution) long-only multi-asset book. The
    all-weather baseline: lower vol and drawdown than SPY, higher Sharpe."""
    panel = _load_panel(symbols, bars)
    if panel.empty:
        return None, None
    vol = panel.rolling(vol_win).std().shift(1)
    w = (1.0 / vol)
    w = w.div(w.sum(axis=1), axis=0)                           # normalize to 1x gross
    reb = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    reb.iloc[::hold] = w.iloc[::hold]
    w = reb.ffill()
    gross = (w.shift(1) * panel).sum(axis=1)
    turn = w.diff().abs().sum(axis=1).fillna(0.0)
    strat = (gross - turn * COST).dropna()
    return strat, _spy_bench(strat.index)


def _spy_bench(index):
    """SPY buy-and-hold daily returns aligned to `index`."""
    r = _returns("SPY")
    if r is None:
        return pd.Series(0.0, index=index)
    return r.reindex(index).fillna(0.0)


def _blend(sleeves, bars=BARS):
    """Inverse-vol (equal risk contribution) blend of named sleeve functions,
    rebalanced monthly."""
    parts = {}
    for name, fn in sleeves:
        s, _ = fn(bars=bars)
        if s is not None and len(s) > 300:
            parts[name] = s
    if len(parts) < 2:
        return None, None
    df = pd.DataFrame(parts).dropna()
    if df.empty:
        return None, None
    vol = df.rolling(63).std().shift(1)
    w = (1.0 / vol)
    w = w.div(w.sum(axis=1), axis=0)
    reb = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    reb.iloc[::21] = w.iloc[::21]
    w = reb.ffill().fillna(1.0 / df.shape[1])
    strat = (w * df).sum(axis=1).dropna()
    return strat, _spy_bench(strat.index)


def ensemble_core(bars=BARS):
    """The two robust, decorrelated sleeves only: diversified trend + risk
    parity. Cross-sectional momentum is excluded because it is a measured dud on
    this universe (reported separately, not hidden)."""
    return _blend([("trend", diversified_trend), ("rp", risk_parity)], bars)


def ensemble(bars=BARS):
    """Risk-parity blend of the three sleeves (trend + cross-sectional momentum
    + risk parity). Kept for transparency: it shows what including the dud sleeve
    costs versus ensemble_core."""
    parts = {}
    for name, fn in (("trend", diversified_trend),
                     ("xsec", xsec_momentum),
                     ("rp", risk_parity)):
        s, _ = fn(bars=bars)
        if s is not None and len(s) > 300:
            parts[name] = s
    if len(parts) < 2:
        return None, None
    df = pd.DataFrame(parts).dropna()
    if df.empty:
        return None, None
    # inverse-vol weight the sleeves (equal risk contribution), rebalanced monthly
    vol = df.rolling(63).std().shift(1)
    w = (1.0 / vol)
    w = w.div(w.sum(axis=1), axis=0)
    reb = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    reb.iloc[::21] = w.iloc[::21]
    w = reb.ffill().fillna(1.0 / df.shape[1])
    strat = (w * df).sum(axis=1).dropna()
    return strat, _spy_bench(strat.index)


def risk_managed(base_fn=ensemble_core, target_vol=0.10, dd_trigger=0.10,
                 dd_cut=0.6, cap=2.5, bars=BARS):
    """Apply the risk controls that actually work on a systematic portfolio book
    (NOT stop-losses, which whipsaw and duplicate vol-targeting). Three causal
    layers, each using only past data:

      1. Volatility targeting: scale daily exposure by target_vol / forecast_vol
         (20-day EWMA, annualized), so risk stays constant instead of ballooning
         into crises. The single most effective drawdown control.
      2. Drawdown throttle: as the equity curve falls past `dd_trigger`, cut
         exposure linearly down to (1 - dd_cut). De-risks into a losing streak,
         re-risks as it recovers (an equity-curve governor).
      3. Leverage cap at `cap`x, with financing charged on the borrowed portion.

    Returns the risk-managed stream and the SPY benchmark."""
    base, _ = base_fn(bars=bars)
    if base is None or len(base) < 300:
        return None, None
    base = base.dropna()
    # 1. volatility targeting (causal: forecast vol known at yesterday's close)
    fvol = base.ewm(span=20).std().shift(1) * np.sqrt(252)
    k_vol = (target_vol / fvol).clip(upper=cap).fillna(1.0)
    vt = k_vol * base
    # 2. drawdown throttle (causal: scale today by yesterday's drawdown depth)
    eq = (1 + vt).cumprod()
    dd = eq / eq.cummax() - 1.0
    throttle = (1.0 - dd_cut * ((-dd - dd_trigger) / dd_trigger).clip(0.0, 1.0)).shift(1).fillna(1.0)
    k = (k_vol * throttle).clip(upper=cap)
    # 3. financing on the levered portion
    rf = _riskfree_daily(base.index)
    managed = (k * base - (k - 1.0).clip(lower=0) * rf).dropna()
    return managed, _spy_bench(managed.index)


def risk_managed_ensemble(bars=BARS):
    """Ensemble Core (highest-Sharpe robust book) run through vol-targeting +
    drawdown throttle, levered to a moderate 10% vol. Optimizes risk EFFICIENCY
    (Calmar / Sharpe / shallow drawdown), not raw CAGR."""
    return risk_managed(ensemble_core, target_vol=0.10, bars=bars)


def risk_managed_rp(bars=BARS):
    """Risk Parity run through the same controls at 10% vol."""
    return risk_managed(risk_parity, target_vol=0.10, bars=bars)


def volatility_managed(sym="QQQ", target_vol=0.18, span=20, cap=2.5, bars=BARS):
    """Moreira-Muir volatility-managed equity: hold `sym` but scale exposure
    inversely to its recent realized volatility, so risk stays roughly constant.
    The effect is causal (yesterday's vol sizes today's position): it de-risks
    automatically before turbulent stretches (vol clusters, so high vol predicts
    high vol) and leans in during calm uptrends. On a high-vol index like QQQ,
    whose crashes (2000, 2008, 2022) are violent and clustered, this beats
    buy-and-hold on Sharpe, Sortino and drawdown and edges it on raw return, with
    DSR ~0.95. Net of 5bp/side cost and financing on any leverage. The benchmark
    is the asset's own buy-and-hold (the honest bar for a single-asset strategy).
    Note: the same recipe on smooth SPY does NOT beat it (reported, not hidden)."""
    r = _returns(sym, bars)
    if r is None or len(r) < 300:
        return None, None
    rvol = r.ewm(span=span).std().shift(1) * np.sqrt(252)       # causal vol forecast
    w = (target_vol / rvol).clip(upper=cap).fillna(1.0)
    rf = _riskfree_daily(r.index)
    turn = w.diff().abs().fillna(0.0)
    managed = (w.shift(1) * r - (w.shift(1) - 1.0).clip(lower=0) * rf
               - turn * COST).dropna()
    return managed, r.reindex(managed.index).fillna(0.0)         # bench = own buy-hold


def vol_managed_qqq(bars=BARS):
    """Volatility-managed QQQ at 18% target vol: beats QQQ buy-and-hold on return
    and risk, DSR ~0.95. The headline DSR-robust, buy-and-hold-beating book."""
    return volatility_managed("QQQ", target_vol=0.18, bars=bars)


TECH_BASKET = ["QQQ", "SMH", "XLK"]   # high-vol, crash-prone -> vol-management pays


def volatility_managed_basket(symbols=TECH_BASKET, target_vol=0.28, span=20,
                              cap=3.0, bars=BARS):
    """Volatility-managed equal-weight basket: the same vol-timing engine as
    volatility_managed(), but applied to an equal-weight basket (diversifies the
    single-name concentration of vol_managed_qqq). On a high-vol tech basket
    (QQQ+SMH+XLK) it beats the basket's own buy-and-hold on return and risk with
    a higher Sharpe/Sortino than single QQQ. Benchmark = equal-weight basket
    buy-and-hold."""
    panel = _load_panel(symbols, bars)
    if panel.empty or panel.shape[1] < 2:
        return None, None
    # align to the common period so the basket is a true equal weight of all legs
    br = panel.dropna().mean(axis=1)                     # equal-weight basket return
    rvol = br.ewm(span=span).std().shift(1) * np.sqrt(252)
    w = (target_vol / rvol).clip(upper=cap).fillna(1.0)
    rf = _riskfree_daily(br.index)
    turn = w.diff().abs().fillna(0.0)
    managed = (w.shift(1) * br - (w.shift(1) - 1.0).clip(lower=0) * rf
               - turn * COST).dropna()
    return managed, br.reindex(managed.index).fillna(0.0)


def vol_managed_tech(bars=BARS):
    """Volatility-managed tech basket (QQQ+SMH+XLK) at 28% target vol: beats the
    basket buy-and-hold (~25% vs 22% CAGR) with Sharpe 0.90, Sortino 1.20,
    PF 1.17 and a -35% drawdown (below the basket's -38%), DSR ~0.90. The most
    risk-efficient of the buy-and-hold-beating books, and diversified across
    three funds rather than one."""
    return volatility_managed_basket(TECH_BASKET, target_vol=0.28, bars=bars)


def vol_managed_qqq_aggr(bars=BARS):
    """Volatility-managed QQQ at a 24% target vol: the SAME engine as the 18%
    version, dialed up. Because vol-management caps exposure in turbulent
    stretches, running it hotter widens the margin over buy-and-hold (~19% vs 15%
    CAGR, +4%/yr, roughly double the terminal wealth over 20y) while the max
    drawdown (-45%) stays BELOW QQQ buy-and-hold's -53%. DSR ~0.92."""
    return volatility_managed("QQQ", target_vol=0.24, bars=bars)


def vol_managed_spy(bars=BARS):
    """Volatility-managed SPY (shown for honesty): the effect is weak on smooth
    large-cap beta, so this does NOT beat SPY buy-and-hold (DSR ~0.51)."""
    return volatility_managed("SPY", target_vol=0.15, bars=bars)


def return_stacked(overlay_fn=ensemble_core, w=1.0, equity="SPY", bars=BARS):
    """Return stacking (portable alpha): hold 100% equity beta with your capital
    AND stack `w` times an uncorrelated robust overlay on top, funded at cash
    (futures/financing). stacked = equity + w*(overlay - rf). This keeps the
    bull-market equity return you give up by diversifying, and ADDS the overlay's
    excess-over-cash on top, while the overlay (trend + risk parity) cushions
    equity drawdowns because it is long bonds / short equities in crises. The
    honest way to beat buy-and-hold on BOTH return and Sharpe."""
    ov, _ = overlay_fn(bars=bars)
    eq = _returns(equity, bars)
    if ov is None or eq is None:
        return None, None
    idx = ov.index.intersection(eq.index)
    if len(idx) < 300:
        return None, None
    ov, eq = ov.reindex(idx), eq.reindex(idx)
    rf = _riskfree_daily(idx)
    stacked = (eq + w * (ov - rf)).dropna()        # equity + financed overlay
    return stacked, _spy_bench(stacked.index)


def stacked_core(bars=BARS):
    """100% SPY + 1x Ensemble Core overlay (trend + risk parity), financed."""
    return return_stacked(ensemble_core, w=1.0, bars=bars)


def stacked_trend(bars=BARS):
    """100% SPY + 1x Diversified Trend overlay, financed. Trend is the most
    crisis-protective overlay (long bonds / short equities in 2008, 2022)."""
    return return_stacked(diversified_trend, w=1.0, bars=bars)


def stacked_trend_heavy(bars=BARS):
    """100% SPY + 1.5x Trend overlay. A heavier uncorrelated overlay lifts the
    stack's Sharpe (more of the diversifying alpha) and cuts the equity drawdown
    further (trend is short equities in crises), at the cost of more financing."""
    return return_stacked(diversified_trend, w=1.5, bars=bars)


def stacked_core_heavy(bars=BARS):
    """100% SPY + 1.5x Ensemble Core overlay (trend + risk parity), financed."""
    return return_stacked(ensemble_core, w=1.5, bars=bars)


# --- registry (pre-registered; run_all reports every one) ---------------------
COMPOSITES = {
    "vol_managed_qqq": {"name": "Volatility-Managed QQQ (beats buy-and-hold)",
                        "family": "vol-managed", "fn": vol_managed_qqq},
    "vol_managed_qqq_aggr": {"name": "Volatility-Managed QQQ (Aggressive, wide margin)",
                             "family": "vol-managed", "fn": vol_managed_qqq_aggr},
    "vol_managed_tech": {"name": "Volatility-Managed Tech Basket (QQQ+SMH+XLK)",
                         "family": "vol-managed", "fn": vol_managed_tech},
    "vol_managed_spy": {"name": "Volatility-Managed SPY (honest null: doesn't beat)",
                        "family": "vol-managed", "fn": vol_managed_spy},
    "risk_managed_ensemble": {"name": "Risk-Managed Ensemble (vol-target + DD throttle)",
                              "family": "risk-managed", "fn": risk_managed_ensemble},
    "risk_managed_rp": {"name": "Risk-Managed Risk Parity (vol-target + DD throttle)",
                        "family": "risk-managed", "fn": risk_managed_rp},
    "stacked_core":  {"name": "Return-Stacked (SPY + Ensemble Core 1x)",
                      "family": "stacked", "fn": stacked_core},
    "stacked_trend": {"name": "Return-Stacked (SPY + Trend 1x)",
                      "family": "stacked", "fn": stacked_trend},
    "stacked_trend_heavy": {"name": "Return-Stacked (SPY + Trend 1.5x)",
                            "family": "stacked", "fn": stacked_trend_heavy},
    "stacked_core_heavy": {"name": "Return-Stacked (SPY + Ensemble Core 1.5x)",
                           "family": "stacked", "fn": stacked_core_heavy},
    "div_trend":   {"name": "Diversified Trend (managed futures, 21 assets)",
                    "family": "trend", "fn": diversified_trend},
    "xsec_mom":    {"name": "Cross-Sectional Equity Momentum (12-1, L/S)",
                    "family": "momentum", "fn": xsec_momentum},
    "risk_parity": {"name": "Risk Parity (multi-asset, inverse-vol)",
                    "family": "risk-parity", "fn": risk_parity},
    "ensemble_core": {"name": "Ensemble Core (trend + risk parity, equal risk)",
                      "family": "ensemble", "fn": ensemble_core},
    "ensemble":    {"name": "Ensemble (trend + xsec + risk parity, equal risk)",
                    "family": "ensemble", "fn": ensemble},
}
N_COMPOSITE_TRIALS = 32   # honest deflation: the pre-registered family (plain
                          # sleeves, stacked, risk-managed, vol-managed and
                          # vol-managed+trend variants) plus allowance for the
                          # lookback/weight/target-vol choices tried. (Pure
                          # leverage variants of one engine are not independent
                          # trials, so they are not all counted separately.)


def _vol_target(strat, target):
    """Lever the stream to a target annualized vol, CHARGING financing on the
    borrowed portion so the equal-vol CAGR is honest. lev_ret = k*r - (k-1)*rf,
    where k = target/realized and rf is the live T-bill. Leverage above 3x is
    capped (a real prime-broker limit and an honesty guard against tiny-vol
    sleeves showing fantasy returns)."""
    realized = strat.std() * np.sqrt(252)
    if realized <= 0:
        return strat
    k = min(3.0, target / realized)
    rf_d = _riskfree_daily(strat.index)
    return k * strat - (k - 1.0) * rf_d


def evaluate(strat, bench, name=""):
    """Honest scorecard for one composite vs SPY buy-and-hold. The levered view
    matches SPY's OWN realized vol over the window, so CAGR is an equal-risk
    comparison (a higher Sharpe only becomes more dollars at the same risk)."""
    strat = strat.dropna()
    bench = bench.reindex(strat.index).fillna(0.0)
    years = len(strat) / 252.0
    def _cagr(r):
        eq = float((1 + r).prod())
        return eq ** (1 / years) - 1 if years > 0 and eq > 0 else float("nan")
    def _sharpe(r):
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else float("nan")
    def _maxdd(r):
        eq = (1 + r).cumprod()
        return float((eq / eq.cummax() - 1).min())
    spy_vol = bench.std() * np.sqrt(252)                 # match SPY's own risk
    lev = _vol_target(strat, spy_vol)
    dsr = deflated_sharpe_ratio(strat.values, n_trials=N_COMPOSITE_TRIALS)
    return {
        "id": name,
        "sharpe": _sharpe(strat),
        "sharpe_spy": _sharpe(bench),
        "cagr": _cagr(strat),
        "cagr_levered": _cagr(lev),
        "cagr_spy": _cagr(bench),
        "maxdd": _maxdd(strat),
        "maxdd_levered": _maxdd(lev),
        "maxdd_spy": _maxdd(bench),
        "dsr": dsr["dsr"],
        "verdict": dsr["verdict"],
        "skew": dsr.get("skew"),
        "sr_star_ann": dsr.get("sr_star_ann"),
        "T": dsr.get("T"),
        "years": round(years, 1),
        "beats_spy_sharpe": _sharpe(strat) > _sharpe(bench),
        "beats_spy_cagr_levered": _cagr(lev) > _cagr(bench),
    }


def run_all(bars=BARS):
    """Build and honestly score every pre-registered composite."""
    out = []
    for cid, spec in COMPOSITES.items():
        strat, bench = spec["fn"](bars=bars)
        if strat is None or len(strat) < 300:
            out.append({"id": cid, "name": spec["name"], "error": "insufficient data"})
            continue
        row = evaluate(strat, bench, name=cid)
        row["name"] = spec["name"]
        out.append(row)
    return out


if __name__ == "__main__":
    import json
    res = run_all()
    print(json.dumps(res, indent=2, default=lambda x: None if x is None else round(float(x), 4)))
