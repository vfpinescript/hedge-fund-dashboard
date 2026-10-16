"""
Dashboard compute layer — turns the quant engine into JSON payloads for the UI.
Everything is derived from real data via the existing modules. Heavy items are
cached in-process so the UI stays snappy.
"""
from __future__ import annotations

import json
import os
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

# --- background warming for the heavy full-universe grids ---------------------
import threading as _threading

_WARM_LOCK = _threading.Lock()
_WARMING: set = set()


def _ensure_warm(key, fn):
    """Compute `fn` in a background thread if it is not cached and not already
    running, so the request thread never blocks on the multi-minute grids."""
    if key in _CACHE:
        return
    with _WARM_LOCK:
        if key in _WARMING:
            return
        _WARMING.add(key)

    def _run():
        try:
            fn()
        except Exception:
            pass
        finally:
            _WARMING.discard(key)

    _threading.Thread(target=_run, daemon=True).start()


def matrix_status():
    """Cached matrix, or {'computing': True} while the background warm runs."""
    if "matrix" in _CACHE:
        return _CACHE["matrix"]
    _ensure_warm("matrix", strategy_matrix)
    return {"computing": True, "assets_total": len(MATRIX_ASSETS)}


def forward_status():
    if "forward" in _CACHE:
        return _CACHE["forward"]
    _ensure_warm("forward", forward_test)
    return {"computing": True}


def _safe_sharpe(r, ann=np.sqrt(252), floor=1e-7, cap=50.0):
    """Sharpe with degenerate-series protection: a near-constant return series
    has ~0 variance, which otherwise yields an absurd ratio (e.g. 9e16, a pure
    divide-by-near-zero artifact). Returns None when not meaningful."""
    sd = float(r.std())
    if not np.isfinite(sd) or sd < floor:
        return None
    v = float(r.mean() / sd * ann)
    return round(v, 2) if np.isfinite(v) and abs(v) <= cap else None


def _safe_sortino(r, ann=np.sqrt(252), floor=1e-7, cap=50.0):
    dn = r[r < 0]
    sd = float(dn.std()) if len(dn) else 0.0
    if not np.isfinite(sd) or sd < floor:
        return None
    v = float(r.mean() / sd * ann)
    return round(v, 2) if np.isfinite(v) and abs(v) <= cap else None


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

    # live book construction: regime, ERC sleeve weights, book vol targeting
    try:
        from alpaca.multi_strategy import compute_combined
        cc = compute_combined(a)
        snap["book"] = {
            "weights": cc["weights"], "correlations": cc["correlations"],
            "regime": cc["regime"], "book_vol": cc["book_vol"],
            "target_vol": cc["target_vol"], "vol_scale": cc["vol_scale"],
            "invested": cc["invested"], "lev_regime": cc["lev_regime"],
            "crypto_regime": cc["crypto_regime"],
            "trend_regime": cc.get("trend_regime"), "reversal_cap": cc.get("reversal_cap"),
            "trend_basket": cc.get("trend_basket"),
            "crypto_risk_cap": cc.get("crypto_risk_cap"),
            "crypto_risk_before": cc.get("crypto_risk_before"),
            "crypto_risk_after": cc.get("crypto_risk_after"),
            "crypto_scale": cc.get("crypto_scale"),
            "anomaly_score": cc.get("anomaly_score"),
            "anomaly_scale": cc.get("anomaly_scale"),
            "anomaly_threshold": cc.get("anomaly_threshold"),
            "anomaly_flag": cc.get("anomaly_flag"),
            "nfci": cc.get("nfci"), "nfci_scale": cc.get("nfci_scale"),
        }
    except Exception as e:
        snap["book"] = {"error": str(e)[:100]}
    return snap


def anomaly_scan():
    """Data-quality & anomaly scan over the core book assets using the three
    robust outlier detectors (GARCH vol-filter, MCD Mahalanobis, LOF). Report:
      - market anomaly score (cross-sectional robust Mahalanobis of today's move)
      - per-asset conditional-sigma outliers and bad ticks cleaned on load
    Cached per process."""
    if "anomaly" in _CACHE:
        return _CACHE["anomaly"]
    from quant.data.loader import load_ohlcv
    from quant.risk.anomaly import (conditional_vol, garch_outliers,
                                    clean_series, market_anomaly, lof_scores)
    core = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "USDCAD",
            "XAUUSD", "USOIL", "SPX", "NDX", "BTCUSD", "ETHUSD"]
    rets = {}
    assets = []
    for sym in core:
        try:
            raw = load_ohlcv(sym, bars=400, clean=False)          # pre-clean prices
            close = raw["close"]
            cleaned, n_bad = clean_series(close, kappa=8.0, method="ewma")
            r = cleaned.pct_change().dropna()
            if len(r) < 120:
                continue
            mask, sig = garch_outliers(r, kappa=4.0)
            last_sigma = float(sig.dropna().iloc[-1]) if sig.notna().any() else None
            recent = [str(pd.Timestamp(t).date()) for t in r.index[mask.values]][-5:]
            assets.append({
                "symbol": sym, "bars": len(r),
                "bad_ticks_cleaned": int(n_bad),
                "sigma_outliers": int(mask.sum()),
                "cond_vol_annual": round(last_sigma * np.sqrt(252) * 100, 1) if last_sigma else None,
                "recent_flags": recent,
            })
            rets[sym] = r
        except Exception:
            continue
    panel = pd.DataFrame(rets).dropna()
    mkt = market_anomaly(panel) if not panel.empty else {"ok": False, "reason": "no data"}
    lof_days = None
    if not panel.empty and len(panel) > 40:
        try:
            _, lof_mask = lof_scores(panel.values, k=20)
            lof_days = int(lof_mask.sum())
        except Exception:
            lof_days = None
    out = {"as_of": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
           "kappa": 4.0, "clean_kappa": 8.0,
           "assets": assets, "market": mkt, "lof_outlier_days": lof_days,
           "total_bad_ticks": sum(a["bad_ticks_cleaned"] for a in assets)}
    _CACHE["anomaly"] = out
    return out


def pairs_trading():
    """Statistical-arbitrage backtest across ~50 economically-linked pairs, net of
    transaction + short-borrow costs and DSR-deflated by the number of pairs.
    Cached (first run loads prices for ~100 tickers)."""
    if "pairs" in _CACHE:
        return _CACHE["pairs"]
    from quant.research.pairs import run_all
    out = run_all()
    _CACHE["pairs"] = out
    return out


def congress_trades():
    """Recent US congressional stock/bond disclosures (Senate + House) via FMP,
    plus a most-active-tickers summary. Informational, not a signal. Cached."""
    if "congress" in _CACHE:
        return _CACHE["congress"]
    from quant.data.congress import latest_trades, latest_insider, amount_mid
    trades = latest_trades()
    from collections import defaultdict
    agg = defaultdict(lambda: {"ticker": "", "buys": 0, "sells": 0, "vol": 0.0})
    tot_buys = tot_sells = 0
    for t in trades:
        ty = (t.get("type") or "").lower()
        is_buy = ty.startswith("p") or ty.startswith("buy")
        is_sell = ty.startswith("s")
        if is_buy:
            tot_buys += 1
        elif is_sell:
            tot_sells += 1
        tk = t.get("ticker")
        if not tk:
            continue
        a = agg[tk]
        a["ticker"] = tk
        if is_buy:
            a["buys"] += 1
        elif is_sell:
            a["sells"] += 1
        a["vol"] += amount_mid(t.get("amount"))
    ranked = sorted(agg.values(), key=lambda x: -(x["buys"] + x["sells"]))
    for a in ranked:
        a["vol"] = round(a["vol"])
        a["net"] = a["buys"] - a["sells"]
    top = ranked[:15]
    net_bought = sorted([a for a in ranked if a["net"] > 0], key=lambda x: -x["net"])[:8]
    net_sold = sorted([a for a in ranked if a["net"] < 0], key=lambda x: x["net"])[:8]
    # corporate insider feed (separate dataset)
    insider = latest_insider()
    out = {"as_of": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
           "n": len(trades), "configured": bool(os.environ.get("FMP_API_KEY")),
           "trades": trades[:250], "top_tickers": top,
           "net": {"buys": tot_buys, "sells": tot_sells,
                   "net_bought": net_bought, "net_sold": net_sold},
           "insider": insider[:40]}
    _CACHE["congress"] = out
    return out


def news_view(ticker):
    """Experimental LLM news/sentiment analyst for one ticker. Report-only, not a
    signal; delegates to quant.agents.news_analyst (handles caching + graceful
    degradation when no ANTHROPIC_API_KEY is set)."""
    from quant.agents.news_analyst import analyze
    return analyze(ticker)


def macro_monitor():
    """Live macro (FRED) + energy (EIA) dashboard payload: headline indicators,
    chart series, and a few derived regime flags. Cached per process (the macro
    module also caches the raw series with a 6h TTL)."""
    if "macro" in _CACHE:
        return _CACHE["macro"]
    from quant.data import macro as M

    def fcard(sid, label, unit="%"):
        d, v = M.fred_latest(sid)
        return {"label": label, "value": v, "date": d, "unit": unit}

    def ecard(sid, label, unit):
        d, v = M.eia_latest(sid)
        return {"label": label, "value": v, "date": d, "unit": unit}

    def fseries(sid, start="2022-01-01"):
        try:
            s = M.fred_series(sid)
            s = s[s.index >= start].resample("W").last().dropna()
            return {"dates": [str(d.date()) for d in s.index],
                    "values": [round(float(x), 3) for x in s.values]}
        except Exception:
            return {"dates": [], "values": []}

    def eseries(sid, start="2022-01-01"):
        try:
            s = M.eia_series(sid)
            s = s[s.index >= start].resample("W").last().dropna()
            return {"dates": [str(d.date()) for d in s.index],
                    "values": [round(float(x), 2) for x in s.values]}
        except Exception:
            return {"dates": [], "values": []}

    rates = [fcard("DGS3MO", "3M T-Bill"), fcard("DGS2", "2Y"), fcard("DGS10", "10Y"),
             fcard("DGS30", "30Y"), fcard("DFF", "Fed Funds")]
    # CPI YoY from the index level
    cpi_yoy = None
    try:
        cpi = M.fred_series("CPIAUCSL")
        if len(cpi) > 13:
            cpi_yoy = round(float(cpi.iloc[-1] / cpi.iloc[-13] - 1) * 100, 2)
    except Exception:
        pass

    curve = fcard("T10Y2Y", "10Y-2Y Slope")
    risk = {"hy_oas": fcard("BAMLH0A0HYM2", "HY Credit Spread (OAS)"),
            "vix": fcard("VIXCLS", "VIX", unit="pts"),
            "nfci": fcard("NFCI", "Financial Conditions (NFCI)", unit="idx"),
            "dollar": fcard("DTWEXBGS", "Broad Dollar Index", unit="idx"),
            "breakeven": fcard("T10YIE", "10Y Breakeven Inflation"),
            "unemployment": fcard("UNRATE", "Unemployment"),
            "cpi_yoy": {"label": "CPI (YoY)", "value": cpi_yoy, "date": None, "unit": "%"}}

    energy = {"wti": ecard("PET.RWTC.D", "WTI Crude", "$/bbl"),
              "brent": ecard("PET.RBRTE.D", "Brent Crude", "$/bbl"),
              "natgas": ecard("NG.RNGWHHD.D", "Henry Hub Nat Gas", "$/MMBtu"),
              "crude_stocks": ecard("PET.WCESTUS1.W", "US Crude Stocks", "Mbbl")}

    charts = {"curve": fseries("T10Y2Y"), "hy_oas": fseries("BAMLH0A0HYM2"),
              "vix": fseries("VIXCLS"), "ust10y": fseries("DGS10"),
              "wti": eseries("PET.RWTC.D"), "natgas": eseries("NG.RNGWHHD.D")}

    # derived regime flags (honest, rule-based read of the live data)
    flags = []
    cv = curve["value"]
    if cv is not None:
        flags.append({"k": "Yield curve", "v": ("inverted" if cv < 0 else "positive") + f" ({cv:+.2f}%)",
                      "tone": "neg" if cv < 0 else "pos"})
    hy = risk["hy_oas"]["value"]
    if hy is not None:
        flags.append({"k": "Credit stress", "v": ("elevated" if hy > 5 else "moderate" if hy > 4 else "calm") + f" ({hy:.2f}%)",
                      "tone": "neg" if hy > 5 else "warn" if hy > 4 else "pos"})
    nf = risk["nfci"]["value"]
    if nf is not None:
        flags.append({"k": "Financial conditions", "v": ("tight" if nf > 0 else "loose") + f" ({nf:+.2f})",
                      "tone": "neg" if nf > 0 else "pos"})
    vx = risk["vix"]["value"]
    if vx is not None:
        flags.append({"k": "Equity vol", "v": ("elevated" if vx > 25 else "low") + f" ({vx:.1f})",
                      "tone": "neg" if vx > 25 else "pos"})

    out = {"as_of": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
           "rates": rates, "curve": curve, "risk": risk, "energy": energy,
           "charts": charts, "flags": flags}
    _CACHE["macro"] = out
    return out


def macro_signals():
    """Economically-grounded macro/energy overlays on broad US equity (SPY): each
    holds equity only while a 'risk-on' condition holds, else sits in cash at the
    live T-bill rate. Honestly scored vs buy & hold, deflated for the handful of
    overlays tried. These are slow drawdown-control tools, not return engines, and
    the DSR tells you whether each actually earns its place. Cached."""
    if "macro_sig" in _CACHE:
        return _CACHE["macro_sig"]
    from quant.data import macro as M
    from quant.data.loader import load_ohlcv
    from quant.research.overfitting import deflated_sharpe_ratio

    try:
        spy = load_ohlcv("SPY", bars=3000)["close"]
    except Exception as e:
        return {"error": f"SPY load failed: {e}"}
    ret = spy.pct_change().dropna()
    rf_s = M.riskfree_series()
    if len(rf_s):
        rf = (rf_s.reindex(ret.index.union(rf_s.index)).sort_index().ffill()
              .reindex(ret.index) / 252.0).fillna(0.02 / 252)
    else:
        rf = pd.Series(0.02 / 252, index=ret.index)

    def metrics(r):
        r = r.dropna()
        eq = (1 + r).cumprod()
        cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1) if len(r) else 0.0
        sh = _safe_sharpe(r)
        mdd = float((eq / eq.cummax() - 1).min())
        return cagr, sh, mdd

    N_OVERLAYS = 3  # honest trial count for this small macro search

    def overlay(cond, name, desc):
        try:
            # align the signal to SPY dates, then backtest ONLY over the window
            # where the signal actually exists (do not force pre-signal years to
            # cash, which would understate the strategy).
            c = cond.astype(float).reindex(ret.index.union(cond.index)).sort_index() \
                    .ffill().reindex(ret.index)
            valid = c.dropna().index
            if len(valid) < 250:
                return {"name": name, "desc": desc, "error": "insufficient overlapping history"}
            r = ret.reindex(valid)
            rfx = rf.reindex(valid)
            active = (c.reindex(valid) > 0.5).shift(1).fillna(False)
            strat = pd.Series(np.where(active, r.values, rfx.values), index=valid)
            cagr, sh, mdd = metrics(strat)
            bh_c, _, bh_dd = metrics(r)                      # matched buy & hold (same window)
            dsr = deflated_sharpe_ratio(strat.dropna().values, n_trials=N_OVERLAYS)
            return {"name": name, "desc": desc, "cagr": round(cagr * 100, 1),
                    "sharpe": sh, "max_dd": round(mdd * 100, 1),
                    "dsr": round(float(dsr["dsr"]), 3) if np.isfinite(dsr["dsr"]) else None,
                    "verdict": dsr["verdict"], "pct_invested": round(float(active.mean()) * 100),
                    "years": round(len(valid) / 252, 1),
                    "bh_cagr": round(bh_c * 100, 1), "bh_max_dd": round(bh_dd * 100, 1)}
        except Exception as e:
            return {"name": name, "desc": desc, "error": str(e)[:60]}

    bcagr, bsh, bmdd = metrics(ret)
    rows = [{"name": "Buy & Hold SPY", "desc": "always invested (baseline, full window)",
             "cagr": round(bcagr * 100, 1), "sharpe": bsh, "max_dd": round(bmdd * 100, 1),
             "dsr": None, "verdict": "baseline", "pct_invested": 100,
             "years": round(len(ret) / 252, 1), "bh_cagr": None, "bh_max_dd": None}]
    try:
        hy = M.fred_series("BAMLH0A0HYM2")
        rows.append(overlay(hy < hy.rolling(126).mean(),
                            "Credit filter", "in SPY only when HY credit spread is below its 6-month average (risk-off when spreads widen)"))
    except Exception:
        pass
    try:
        cu = M.fred_series("T10Y2Y")
        rows.append(overlay(cu > 0, "Curve filter", "in SPY only when the 10Y-2Y curve is positive (step aside when inverted)"))
    except Exception:
        pass
    try:
        nf = M.fred_series("NFCI")
        rows.append(overlay(nf < 0, "Conditions filter", "in SPY only when financial conditions (NFCI) are loose (< 0)"))
    except Exception:
        pass

    out = {"signals": rows,
           "note": "Overlays on SPY, 1987-present where data allows, deflated for 3 trials. "
                   "They mostly trade return for smaller drawdowns; judge them on whether the DSR "
                   "clears and whether the drawdown improvement is worth the lost upside, not on CAGR alone."}
    _CACHE["macro_sig"] = out
    return out


def desk_accounts():
    """The two additional Alpaca paper desks, each a separate account running one
    strategy: Trend (MA timing on SPY/QQQ/BTC/ETH) and Reversal (7d L/S on equity
    ETFs). One snapshot per desk for the Live Paper page. Not cached."""
    import datetime as _dt
    from alpaca.client import Alpaca
    from alpaca.desks import DESKS, ma_timing_targets, reversal_targets
    START = {"trend": 50000.0, "reversal": 100000.0}
    LABEL = {"trend": "Trend Desk · MA Timing", "reversal": "Reversal Desk · 7d Long/Short"}
    out = []
    for name, cfg in DESKS.items():
        d = {"name": name, "label": LABEL.get(name, name), "strategy": cfg["strategy"],
             "assets": cfg["assets"], "start_equity": START.get(name)}
        try:
            a = Alpaca(key_env=cfg["key_env"], secret_env=cfg["secret_env"])
            acct = a.account()
            equity = float(acct.get("equity", 0))
            positions = []
            for p in a.positions():
                qty = float(p["qty"])
                positions.append({
                    "symbol": p["symbol"], "side": "long" if qty >= 0 else "short",
                    "qty": abs(qty), "avgPrice": float(p["avg_entry_price"]),
                    "lastPrice": float(p["current_price"]), "pl": float(p["unrealized_pl"]),
                    "plPercent": float(p["unrealized_plpc"]) * 100})
            hist = []
            try:
                ph = a.portfolio_history(period="1M", timeframe="1D")
                for t, eqv in zip(ph.get("timestamp") or [], ph.get("equity") or []):
                    if eqv:
                        hist.append({"date": _dt.datetime.utcfromtimestamp(int(t)).strftime("%Y-%m-%d"),
                                     "equity": float(eqv)})
            except Exception:
                pass
            start_eq = d["start_equity"] or (hist[0]["equity"] if hist else equity)
            try:
                if cfg["strategy"] == "ma_timing":
                    _, note = ma_timing_targets(a, cfg["assets"], cfg["gross"], equity)
                else:
                    _, note = reversal_targets(a, cfg["assets"], cfg["gross"], equity)
            except Exception:
                note = None
            d.update({"ok": True, "equity": equity, "cash": float(acct.get("cash", 0)),
                      "unrealizedPnl": sum(p["pl"] for p in positions),
                      "return_pct": round((equity / start_eq - 1) * 100, 2) if start_eq else None,
                      "positions": positions, "history": hist, "note": note})
        except Exception as e:
            d.update({"ok": False, "error": str(e)[:140]})
        out.append(d)
    return {"desks": out}


def strategy_library():
    from quant.research.strategy_library import library
    return library()


def strategy_run(strategy_id, instrument):
    from quant.research.strategy_library import run_strategy
    return run_strategy(strategy_id, (instrument or "").strip().upper())


def analytics(strategy_id, instrument, horizon=252, n_paths=2000):
    """Monte Carlo fan + return distribution + rolling Sharpe for one strategy
    on one asset. The MC resamples the strategy's own daily returns (block
    bootstrap, preserving vol clustering) to show the RANGE of 1-year outcomes,
    not a single backtest line. It inherits the backtest's cost assumptions, so
    treat it as the spread of possibilities, not a forecast."""
    from quant.research.strategy_library import _strategy_returns, STRATEGIES
    inst = (instrument or "").strip().upper()
    spec = STRATEGIES.get(strategy_id)
    if not spec:
        return {"error": f"unknown strategy {strategy_id}"}
    try:
        strat, _ = _strategy_returns(spec, inst)
    except Exception as e:
        return {"error": f"could not load {inst}: {e}"}
    r = strat.dropna().values
    if len(r) < 250:
        return {"error": f"insufficient history for {inst}"}

    # block bootstrap, vectorised
    rng = np.random.default_rng(7)
    block = 20
    n = len(r)
    nblocks = int(np.ceil(horizon / block))
    starts = rng.integers(0, n - block, size=(n_paths, nblocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, nblocks * block)[:, :horizon]
    sims = r[idx]                                   # (n_paths, horizon) daily returns
    cum = np.cumprod(1 + sims, axis=1)              # wealth paths
    bands = {str(p): (np.percentile(cum, p, axis=0) * 100 - 100).round(2).tolist()
             for p in (5, 25, 50, 75, 95)}
    final = cum[:, -1]
    # drawdown distribution across paths
    peak = np.maximum.accumulate(cum, axis=1)
    mdd = (cum / peak - 1).min(axis=1)
    samp = rng.integers(0, n_paths, size=30)
    mc = {
        "days": horizon, "bands": bands,
        "sample_paths": [(cum[i] * 100 - 100).round(2).tolist() for i in samp],
        "median_return_pct": round(float(np.median(final) - 1) * 100, 1),
        "p5_return_pct": round(float(np.percentile(final, 5) - 1) * 100, 1),
        "p95_return_pct": round(float(np.percentile(final, 95) - 1) * 100, 1),
        "prob_loss_pct": round(float((final < 1).mean()) * 100, 1),
        "median_max_dd_pct": round(float(np.median(mdd)) * 100, 1),
        "worst_max_dd_pct": round(float(np.percentile(mdd, 5)) * 100, 1),
    }
    # return distribution histogram (daily, %)
    counts, edges = np.histogram(r * 100, bins=60)
    dist = {"counts": counts.tolist(), "edges": edges.round(3).tolist(),
            "mean": round(float(r.mean()) * 100, 3), "std": round(float(r.std()) * 100, 3),
            "skew": round(float(((r - r.mean()) ** 3).mean() / (r.std() ** 3 + 1e-12)), 2),
            "kurt": round(float(((r - r.mean()) ** 4).mean() / (r.std() ** 4 + 1e-12)), 2)}
    # rolling Sharpe
    s = strat.dropna()
    rs = (s.rolling(126).mean() / s.rolling(126).std() * np.sqrt(252)).dropna()
    rolling_sharpe = {"dates": [d.strftime("%Y-%m-%d") for d in rs.index],
                      "values": rs.values.round(3).tolist()}
    return {"strategy": spec["name"], "instrument": inst,
            "mc": mc, "dist": dist, "rolling_sharpe": rolling_sharpe}


# Assets shown down the side of the Strategy Matrix / Forward Test. These now
# mirror the FULL browsable universe on the Backtest page, so every strategy x
# asset you can open there also appears ranked in Forward Test and colour-coded
# in the Matrix. The grid is large (~214 assets x 11 strategies), so the first
# computation is slow and then cached; it is warmed in the background on startup.
def _full_matrix_assets():
    from quant.research.strategy_library import INSTRUMENT_GROUPS
    seen, out = set(), []
    for syms in INSTRUMENT_GROUPS.values():
        for s in syms:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


MATRIX_ASSETS = _full_matrix_assets()


# out-of-sample tracking start: performance from here forward is the honest test
FORWARD_START = "2026-07-01"


def forward_test(start=FORWARD_START):
    """Forward/out-of-sample test of EVERY strategy on EVERY asset since `start`.
    Each combo's daily strategy return is recomputed from live data and measured
    only over the out-of-sample window, so the leaderboard reflects what has held
    up going forward, not what fit the backtest. Cached per process."""
    if "forward" in _CACHE:
        return _CACHE["forward"]
    from quant.research.strategy_library import _strategy_returns, STRATEGIES
    # every (strategy, asset) browsable on the Backtest page, pairs included
    combos = []
    for sid, spec in STRATEGIES.items():
        if spec.get("mode") == "pairs":
            from quant.research.pairs import PAIRS
            combos += [(sid, spec, f"{a}/{b}") for a, b in PAIRS]
        else:
            combos += [(sid, spec, asset) for asset in MATRIX_ASSETS]
    rows = []
    for sid, spec, asset in combos:
        try:
            strat, _ = _strategy_returns(spec, asset)
            fwd = strat[strat.index >= start].dropna()
            if len(fwd) < 10:
                continue
            eq = (1 + fwd).cumprod()
            tot = (eq.iloc[-1] - 1) * 100
            sh = _safe_sharpe(fwd)
            so = _safe_sortino(fwd)
            mdd = float((eq / eq.cummax() - 1).min()) * 100
            rows.append({"strategy": spec["name"], "strategy_id": sid, "asset": asset,
                         "days": len(fwd), "return": round(tot, 1),
                         "sharpe": sh, "sortino": so,
                         "max_dd": round(mdd, 1),
                         "rr": round(tot / abs(mdd), 2) if mdd else None})
        except Exception:
            continue
    rows.sort(key=lambda x: -(x["sharpe"] if x["sharpe"] is not None else -1e9))
    out = {"start": start, "n": len(rows),
           "as_of": pd.Timestamp.utcnow().strftime("%Y-%m-%d"), "rows": rows}
    _CACHE["forward"] = out
    return out


def forward_series(strategy_id, instrument, start=FORWARD_START):
    """Out-of-sample equity curve of ONE strategy on ONE asset since `start`.
    This is the forward test drawn as a line (strategy vs buy & hold), measured
    only on data after the tracking start — not the in-sample backtest."""
    from quant.research.strategy_library import _strategy_returns, STRATEGIES
    inst = (instrument or "").strip().upper()
    spec = STRATEGIES.get(strategy_id)
    if not spec:
        return {"error": f"unknown strategy {strategy_id}"}
    try:
        strat, bench = _strategy_returns(spec, inst)
    except Exception as e:
        return {"error": f"could not compute {strategy_id} on {inst}: {str(e)[:80]}"}
    fwd = strat[strat.index >= start].dropna()
    if len(fwd) < 2:
        return {"error": "not enough out-of-sample data yet for this combination"}
    bh = bench.reindex(fwd.index).fillna(0.0)
    eq = (1 + fwd).cumprod()
    bheq = (1 + bh).cumprod()
    tot = float((eq.iloc[-1] - 1) * 100)
    mdd = float((eq / eq.cummax() - 1).min()) * 100
    return {
        "strategy": spec["name"], "strategy_id": strategy_id, "asset": inst,
        "start": start, "days": len(fwd),
        "dates": [d.strftime("%Y-%m-%d") for d in fwd.index],
        "equity_pct": [round(float(v - 1) * 100, 2) for v in eq],
        "benchmark_pct": [round(float(v - 1) * 100, 2) for v in bheq],
        "return": round(tot, 1), "sharpe": _safe_sharpe(fwd),
        "sortino": _safe_sortino(fwd), "max_dd": round(mdd, 1),
    }


def strategy_matrix():
    """Every strategy applied to every tradeable asset at once — a grid of
    total return / Sharpe / Sortino / DSR per (asset, strategy). Cached."""
    if "matrix" in _CACHE:
        return _CACHE["matrix"]
    from quant.research.strategy_library import run_strategy, STRATEGIES
    from quant.research.pairs import PAIRS
    single = [{"id": k, "name": v["name"], "family": v["family"]}
              for k, v in STRATEGIES.items() if v.get("mode") != "pairs"]
    pairs_meta = [{"id": k, "name": v["name"], "family": v["family"]}
                  for k, v in STRATEGIES.items() if v.get("mode") == "pairs"]
    strat_meta = single + pairs_meta           # all columns

    def _cell(sid, asset):
        try:
            r = run_strategy(sid, asset)
            if "error" in r:
                return None
            m = r["metrics"]
            return {"cagr": round(m.get("cagr", 0) * 100, 1),
                    "total_return": round(m.get("total_return", 0) * 100, 1),
                    "sharpe": round(m.get("sharpe", 0), 2),
                    "sortino": round(m.get("sortino", 0), 2),
                    "dsr": r.get("dsr"), "verdict": r.get("dsr_verdict")}
        except Exception:
            return None

    rows = []
    for asset in MATRIX_ASSETS:                # single-ticker rows
        cells = {s["id"]: _cell(s["id"], asset) for s in single}
        for s in pairs_meta:
            cells[s["id"]] = None              # pairs don't run on a single ticker
        rows.append({"asset": asset, "cells": cells})
    for a, b in PAIRS:                         # pair rows
        pair = f"{a}/{b}"
        cells = {s["id"]: None for s in single}
        for s in pairs_meta:
            cells[s["id"]] = _cell(s["id"], pair)
        rows.append({"asset": pair, "cells": cells, "is_pair": True})

    assets = list(MATRIX_ASSETS) + [f"{a}/{b}" for a, b in PAIRS]
    out = {"strategies": strat_meta, "assets": assets, "rows": rows}
    _CACHE["matrix"] = out
    return out


def kalshi_scan():
    if "kalshi" in _CACHE:
        return _CACHE["kalshi"]
    from quant.agents.kalshi_arb import KalshiArbitrageAgent
    r = KalshiArbitrageAgent(min_edge_cents=0.3).scan(max_markets=1500)
    _CACHE["kalshi"] = r
    return r
