"""
Dashboard compute layer — turns the quant engine into JSON payloads for the UI.
Everything is derived from real data via the existing modules. Heavy items are
cached in-process so the UI stays snappy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.research.reversal_book import _instrument_series, BOOK
from quant.research.metrics import perf_metrics, derive_trades
from quant.research.overfitting import deflated_sharpe_ratio

# Estimated number of strategy variants tried (signal families × lookbacks) —
# the multiple-testing count the DSR deflates the Sharpe against.
N_TRIALS = 20
from quant.data.loader import load_ohlcv
from quant.committee import Committee
from quant.agents.decision_tree import DecisionTreeAgent
from quant.agents.random_forest import RandomForestAgent
from quant.agents.gradient_boost import GradientBoostAgent
from quant.agents.lasso_factors import LassoFactorAgent
from quant.agents.rates_macro import RatesMacroAgent
from quant.agents.reversal import ReversalAgent
from quant.agents.info_diffusion import InfoDiffusionAgent
from quant.agents.hill_tail import HillTailAgent
from quant.agents.vol_regime import VolRegimeAgent

_CACHE: dict = {}


# ---------- series helpers ----------
def _equity(daily: pd.Series):
    eq = (1 + daily).cumprod()
    return {"dates": [d.strftime("%Y-%m-%d") for d in eq.index],
            "values": (eq.values * 100 - 100).round(3).tolist()}


def _buyhold(ret: pd.Series, index):
    """Cumulative return of simply holding the asset(s) long, over `index`."""
    r = ret.reindex(index).fillna(0)
    eq = (1 + r).cumprod()
    return {"dates": [d.strftime("%Y-%m-%d") for d in eq.index],
            "values": (eq.values * 100 - 100).round(3).tolist()}


def _drawdown(daily: pd.Series):
    eq = (1 + daily).cumprod()
    dd = (eq / eq.cummax() - 1) * 100
    return {"dates": [d.strftime("%Y-%m-%d") for d in dd.index],
            "values": dd.values.round(3).tolist()}


def _rolling_sharpe(daily: pd.Series, win=126):
    rs = daily.rolling(win).mean() / daily.rolling(win).std() * np.sqrt(252)
    rs = rs.dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in rs.index],
            "values": rs.values.round(3).tolist()}


def _monthly_grid(daily: pd.Series):
    m = (1 + daily).resample("ME").prod() - 1
    df = pd.DataFrame({"y": m.index.year, "m": m.index.month, "r": m.values * 100})
    years = sorted(df["y"].unique(), reverse=True)
    grid = {int(y): [None] * 12 for y in years}
    for _, row in df.iterrows():
        grid[int(row["y"])][int(row["m"]) - 1] = round(float(row["r"]), 2)
    return {"years": [int(y) for y in years], "grid": {int(y): grid[y] for y in years}}


def _trade_dist(position: pd.Series, daily: pd.Series):
    pos = np.sign(position.reindex(daily.index).fillna(0).to_numpy())
    longs = shorts = 0
    i, n = 0, len(pos)
    while i < n:
        if pos[i] == 0:
            i += 1; continue
        s = pos[i]
        while i < n and pos[i] == s:
            i += 1
        if s > 0:
            longs += 1
        else:
            shorts += 1
    return {"long": longs, "short": shorts, "total": longs + shorts}


def _score(m):
    """Horizon-style 0-100 'viability' score from Sharpe / DD / profit factor."""
    sh = m.get("sharpe", 0) or 0
    dd = abs(m.get("max_drawdown", 0) or 0)
    pf = m.get("profit_factor", 1) or 1
    s = 50 + 22 * np.tanh(sh) - 60 * dd + 12 * np.tanh(pf - 1)
    return int(max(1, min(99, round(s))))


# ---------- strategies ----------
def build_strategies(cost_bps=1.0):
    if "strategies" in _CACHE:
        return _CACHE["strategies"]
    daily, positions, rets, out = {}, {}, {}, []
    label = {}
    for cls, syms in BOOK.items():
        for sym in syms:
            try:
                pos, strat, ret = _instrument_series(sym, cost_bps=cost_bps)
            except Exception:
                continue
            daily[sym] = strat; positions[sym] = pos; rets[sym] = ret; label[sym] = cls

    def make(name, dseries, pseries, cls, desc, bh=None):
        m = perf_metrics(dseries, pseries)
        if not m:
            return
        out.append({
            "id": name, "name": name, "category": cls, "description": desc,
            "metrics": m,
            "overfit": deflated_sharpe_ratio(dseries.values, N_TRIALS),
            "equity": _equity(dseries), "buyhold": bh, "drawdown": _drawdown(dseries),
            "rolling_sharpe": _rolling_sharpe(dseries), "monthly": _monthly_grid(dseries),
            "trades": _trade_dist(pseries, dseries) if pseries is not None else None,
        })

    # per-instrument
    for sym in daily:
        make(sym, daily[sym], positions[sym], label[sym],
             f"7-day reversal on {sym}, vol-targeted, net {cost_bps}bps/side",
             bh=_buyhold(rets[sym], daily[sym].index))

    # aggregates (equal-risk average of instrument returns)
    def combine(symlist, name, cls):
        cols = [daily[s] for s in symlist if s in daily]
        if not cols:
            return
        D = pd.concat(cols, axis=1).dropna(how="all")
        port = D.mean(axis=1)
        m = perf_metrics(port)
        # equal-weight buy & hold of the same underlyings
        bh_ret = pd.concat([rets[s] for s in symlist if s in rets], axis=1).mean(axis=1)
        out.append({
            "id": name, "name": name, "category": cls, "description":
                f"Equal-risk reversal book across {len([s for s in symlist if s in daily])} instruments",
            "metrics": m,
            "overfit": deflated_sharpe_ratio(port.values, N_TRIALS),
            "equity": _equity(port), "buyhold": _buyhold(bh_ret, port.index),
            "drawdown": _drawdown(port),
            "rolling_sharpe": _rolling_sharpe(port), "monthly": _monthly_grid(port),
            "trades": None,
            "components": {s: _equity(daily[s]) for s in symlist if s in daily},
        })

    for cls, syms in BOOK.items():
        combine(syms, f"{cls} Reversal Book", cls)
    combine([s for syms in BOOK.values() for s in syms], "Multi-Asset Portfolio", "Portfolio")

    out.sort(key=lambda s: -s["metrics"].get("sharpe", 0))
    _CACHE["strategies"] = out
    return out


# ---------- live signals (agents) ----------
def _committee():
    return Committee(
        directional=[ReversalAgent(), InfoDiffusionAgent(), DecisionTreeAgent(),
                     RandomForestAgent(n_estimators=120), GradientBoostAgent(max_iter=150),
                     LassoFactorAgent(), RatesMacroAgent()],
        risk=[HillTailAgent(), VolRegimeAgent()],
    )


def live_signals(instrument="XAUUSD"):
    df = load_ohlcv(instrument, bars=2000)
    c = _committee()
    dec = c.decide(instrument, df)
    return {"instrument": instrument, "last_price": round(float(df["close"].iloc[-1]), 4),
            "as_of": df.index[-1].strftime("%Y-%m-%d"), "decision": dec.to_dict()}


# ---------- surfaces ----------
def surfaces(instrument="XAUUSD"):
    from quant.analytics import surfaces as S
    from quant.analytics.pca_curve import curve_pca
    df = load_ohlcv(instrument, bars=3000)
    rets = df["close"].pct_change().dropna().to_numpy()
    payload = {
        "tail": S.tail_exponent_surface(rets),
        "gamma": S.gamma_surface(),
        "charm": S.charm_surface(),
        "return_dist": S.return_distribution(rets),
        "pca": curve_pca(),
    }
    try:
        vs = S.vol_surface("SPY")
        payload["vol"] = vs
    except Exception:
        payload["vol"] = None
    return payload


# ---------- strategy lab (DSR-gated search) ----------
def strategy_lab():
    if "lab" in _CACHE:
        return _CACHE["lab"]
    from quant.research.strategy_search import StrategySearchAgent, CombinationSearchAgent
    combo = CombinationSearchAgent(dsr_gate=0.90).search()
    single = StrategySearchAgent(dsr_gate=0.90).search()
    out = {
        "combo": combo,
        "single": {"leaderboard": single["leaderboard"][:12],
                   "winner": single["winner"], "n_trials": single["n_trials"],
                   "note": single["note"]},
    }
    _CACHE["lab"] = out
    return out


def live_paper():
    """Live Alpaca paper account for the FX-ETF reversal book: positions + P&L,
    equity curve (Alpaca portfolio history), forward metrics and portfolio risk.
    Not cached — reflects the live account on every call."""
    import datetime as _dt
    try:
        from alpaca.client import Alpaca
    except Exception as e:
        return {"error": f"alpaca module unavailable: {e}"}
    try:
        a = Alpaca()
    except Exception as e:
        return {"error": str(e)}

    START_EQUITY = 50000.0
    try:
        acct = a.account()
        equity = float(acct.get("equity", 0))
        positions = []
        for p in a.positions():
            qty = float(p["qty"])
            positions.append({
                "symbol": p["symbol"],
                "side": "long" if qty >= 0 else "short",
                "qty": abs(qty),
                "avgPrice": float(p["avg_entry_price"]),
                "lastPrice": float(p["current_price"]),
                "pl": float(p["unrealized_pl"]),
                "plPercent": float(p["unrealized_plpc"]) * 100,
            })
        trades = []
        for o in a.orders(status="all", limit=15):
            if o.get("filled_at") and o.get("filled_avg_price"):
                trades.append({"symbol": o.get("symbol"), "side": o.get("side"),
                               "qty": float(o.get("filled_qty") or o.get("qty") or 0),
                               "price": float(o["filled_avg_price"])})
    except Exception as e:
        return {"error": f"alpaca account read failed: {e}"}

    # equity curve from Alpaca portfolio history
    hist = []
    try:
        ph = a.portfolio_history(period="1M", timeframe="1D")
        for t, eqv in zip(ph.get("timestamp") or [], ph.get("equity") or []):
            if eqv:
                hist.append({"date": _dt.datetime.utcfromtimestamp(int(t)).strftime("%Y-%m-%d"),
                             "equity": float(eqv)})
    except Exception:
        pass

    unreal = sum(p["pl"] for p in positions)
    snap = {
        "account": {"equity": equity, "cash": float(acct.get("cash", 0)),
                    "buying_power": float(acct.get("buying_power", 0)),
                    "unrealizedPnl": unreal, "realizedPnl": 0.0},
        "positions": positions, "trades": trades, "history": hist,
        "start_equity": START_EQUITY,
    }

    # forward out-of-sample tracking
    eq = np.array([h["equity"] for h in hist], dtype=float)
    fwd = {"days_live": len(eq), "forward_sharpe": None, "forward_max_dd": None,
           "inception_return_pct": 0.0}
    if len(eq) >= 2:
        rets = np.diff(eq) / eq[:-1]
        fwd["inception_return_pct"] = round((eq[-1] / eq[0] - 1) * 100, 3)
        peak = np.maximum.accumulate(eq)
        fwd["forward_max_dd"] = round(float((eq / peak - 1).min()) * 100, 2)
        if len(rets) >= 20 and rets.std() > 0:
            fwd["forward_sharpe"] = round(float(rets.mean() / rets.std() * np.sqrt(252)), 2)
    snap["forward"] = fwd

    # portfolio-level risk of the current book (signed notional weights)
    try:
        from quant.risk.portfolio_risk import portfolio_risk
        raw = {p["symbol"]: (1 if p["side"] == "long" else -1) * p["qty"] * p["lastPrice"]
               for p in positions}
        tot = sum(abs(v) for v in raw.values()) or 1.0
        w = {k: v / tot for k, v in raw.items()}          # normalize to fractional weights
        snap["portfolio_risk"] = portfolio_risk(w) if w else None
    except Exception as e:
        snap["portfolio_risk"] = {"error": str(e)[:80]}
    return snap


def strategy_library():
    from quant.research.strategy_library import library
    return library()


def strategy_run(strategy_id, instrument):
    from quant.research.strategy_library import run_strategy
    return run_strategy(strategy_id, (instrument or "").strip().upper())


def kalshi_scan():
    if "kalshi" in _CACHE:
        return _CACHE["kalshi"]
    from quant.agents.kalshi_arb import KalshiArbitrageAgent
    r = KalshiArbitrageAgent(min_edge_cents=0.3).scan(max_markets=1500)
    _CACHE["kalshi"] = r
    return r
