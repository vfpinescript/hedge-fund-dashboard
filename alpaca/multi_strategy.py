"""
Combined target book for the Alpaca paper account: three uncorrelated sleeves,
weighted by Equal Risk Contribution (correlation-aware risk parity) rather than
by hand.

  1. FX reversal      — the DSR-robust FX-ETF short-term reversal book.
  2. Leverage rotation — SPY above its 200d MA -> SSO (2x), else BIL.
  3. Crypto trend      — BTC/ETH above their 200d MA -> hold, else cash.

Each sleeve's own daily return series is reconstructed, their covariance is
formed, and ERC weights are solved so every sleeve contributes the same risk.
This is the honest, return-estimation-free way to combine uncorrelated edges.
DRY_RUN prints the plan only.
"""
from __future__ import annotations
import os
import json

import numpy as np
import pandas as pd

from alpaca.client import Alpaca
from alpaca.strategy import _closes, REVERSAL_WINDOW, VOL_WINDOW

INVESTED = float(os.environ.get("INVESTED", "0.90"))   # fraction of equity deployed
MAX_POS_PCT = float(os.environ.get("MAX_POS_PCT", "0.20"))   # cap any single name at 20% of equity
MA_WINDOW = 200
MIN_TRADE_DOLLARS = 50.0
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"
CRYPTO_ORDER = {"BTCUSD": "BTC/USD", "ETHUSD": "ETH/USD"}
FALLBACK_W = {"fx": 0.50, "leverage": 0.25, "crypto": 0.25}   # if ERC can't solve


# ---------- sleeve return series (for the covariance / ERC weighting) ----------
def _sleeve_return_series(a, lookback=504) -> pd.DataFrame:
    px = _closes(a)
    ret = px.pct_change()
    mom = px / px.shift(REVERSAL_WINDOW) - 1.0
    vol = ret.rolling(VOL_WINDOW).std()
    raw = -np.sign(mom) / vol
    w = raw.div(raw.abs().sum(axis=1), axis=0)
    fx = (w.shift(1) * ret).sum(axis=1).rename("fx")

    spy = a.daily_bars("SPY", days=700)
    sc = pd.Series([b["c"] for b in spy], index=pd.to_datetime([b["t"][:10] for b in spy]))
    sret = sc.pct_change()
    above = (sc > sc.rolling(MA_WINDOW).mean()).shift(1).fillna(False)
    lev = pd.Series(np.where(above, 2 * sret, 0.0), index=sc.index).rename("leverage")

    cb = a.crypto_bars(days=700)
    legs = []
    for sym in ("BTC/USD", "ETH/USD"):
        b = cb.get(sym) or []
        if not b:
            continue
        s = pd.Series([x["c"] for x in b], index=pd.to_datetime([x["t"][:10] for x in b]))
        r = s.pct_change()
        ab = (s > s.rolling(MA_WINDOW).mean()).shift(1).fillna(False)
        legs.append(pd.Series(np.where(ab, r, 0.0), index=s.index))
    cry = (sum(legs) / len(legs)).rename("crypto") if legs else pd.Series(dtype=float, name="crypto")

    df = pd.concat([fx, lev, cry], axis=1).dropna()
    return df.iloc[-lookback:]


TARGET_VOL = float(os.environ.get("TARGET_VOL", "0.10"))   # 10% annualised book vol


def _erc_weights(cov: np.ndarray, cols) -> dict:
    """Equal Risk Contribution weights from a (shrunk) covariance."""
    from scipy.optimize import minimize
    n = len(cov)

    def obj(w):
        w = np.asarray(w)
        rc = w * (cov @ w)
        rc = rc / rc.sum()
        return ((rc - 1.0 / n) ** 2).sum()

    res = minimize(obj, [1.0 / n] * n, bounds=[(0.02, 1.0)] * n,
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1.0},),
                   method="SLSQP")
    if not res.success:
        raise RuntimeError("erc did not converge")
    w = res.x / res.x.sum()
    return {list(cols)[i]: float(w[i]) for i in range(n)}


def sleeve_weights(a):
    """Returns (ERC weights, correlation dict, annualised book vol at those weights)."""
    try:
        df = _sleeve_return_series(a)
        if len(df) < 60:
            return dict(FALLBACK_W), None, None
        from quant.risk.cov import shrunk_cov
        cov = shrunk_cov(df)
        w = _erc_weights(cov, df.columns)
        wv = np.array([w[c] for c in df.columns])
        book_vol = float(np.sqrt(max(wv @ cov @ wv, 0)) * np.sqrt(252))
        return w, df.corr().round(2).to_dict(), round(book_vol, 4)
    except Exception:
        return dict(FALLBACK_W), None, None


# ---------- per-sleeve targets (given a dollar budget) ----------
def _fx_targets(a, gross, prices):
    px = _closes(a)
    if px.empty:
        return {}
    ret = px.pct_change()
    mom = px / px.shift(REVERSAL_WINDOW) - 1.0
    vol = ret.rolling(VOL_WINDOW).std()
    raw = -np.sign(mom) / vol
    last = raw.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()
    if last.empty or last.abs().sum() == 0:
        return {}
    w = last / last.abs().sum()
    out = {}
    for s in w.index:
        prices[s] = float(px.iloc[-1][s])
        out[s] = (float(w[s]) * gross) / prices[s]
    return out


def _leverage_sleeve(a, gross, prices):
    bars = a.daily_bars("SPY", days=260)
    closes = np.array([b["c"] for b in bars], dtype=float)
    above = closes[-1] > closes[-MA_WINDOW:].mean()
    sym = "SSO" if above else "BIL"
    price = a.latest_trade(sym) or a.daily_bars(sym, days=3)[-1]["c"]
    prices[sym] = float(price)
    return {sym: gross / float(price)}, ("risk-on -> SSO" if above else "risk-off -> BIL")


def _crypto_sleeve(a, gross, prices):
    bars = a.crypto_bars()
    tgt, held = {}, []
    per = gross / 2.0
    for order_sym, pos_sym in [("BTC/USD", "BTCUSD"), ("ETH/USD", "ETHUSD")]:
        b = bars.get(order_sym) or []
        if len(b) < MA_WINDOW + 1:
            continue
        c = np.array([x["c"] for x in b], dtype=float)
        prices[pos_sym] = float(c[-1])
        if c[-1] > c[-MA_WINDOW:].mean():
            tgt[pos_sym] = per / float(c[-1])
            held.append(pos_sym.replace("USD", ""))
    regime = ("hold " + "+".join(held)) if held else "all below MA -> cash"
    return tgt, regime


def _regime(a):
    """Equity regime (jump model on SPY): turbulent -> scale the whole book down."""
    try:
        from quant.research.regime import current_regime
        bars = a.daily_bars("SPY", days=800)
        close = pd.Series([b["c"] for b in bars],
                          index=pd.to_datetime([b["t"][:10] for b in bars]))
        return current_regime(close)
    except Exception as e:
        return {"state": "unknown", "exposure_scale": 1.0, "error": str(e)[:60]}


def compute_combined(a=None):
    a = a or Alpaca()
    equity = float(a.account().get("equity", 0))
    weights, corr, book_vol = sleeve_weights(a)
    reg = _regime(a)
    # book-level vol targeting as a CAP: de-lever when book vol exceeds the
    # target, never lever up (risk-reduction mandate, not a vol booster).
    vol_scale = 1.0
    if book_vol and book_vol > 0:
        vol_scale = float(np.clip(TARGET_VOL / book_vol, 0.3, 1.0))
    budget = equity * INVESTED * reg["exposure_scale"] * vol_scale
    prices = {}
    fx = _fx_targets(a, budget * weights["fx"], prices)
    lev, lev_regime = _leverage_sleeve(a, budget * weights["leverage"], prices)
    cry, cry_regime = _crypto_sleeve(a, budget * weights["crypto"], prices)
    targets = {}
    for d in (fx, lev, cry):
        for k, v in d.items():
            targets[k] = targets.get(k, 0.0) + v
    # per-position concentration cap: no single name above MAX_POS_PCT of equity
    cap_dollars = equity * MAX_POS_PCT
    capped = []
    for sym in list(targets):
        px = prices.get(sym, 0)
        if px and abs(targets[sym] * px) > cap_dollars:
            targets[sym] = np.sign(targets[sym]) * cap_dollars / px
            capped.append(sym)
    return {"equity": equity, "weights": {k: round(v, 3) for k, v in weights.items()},
            "position_cap_pct": MAX_POS_PCT, "capped": capped,
            "correlations": corr, "book_vol": book_vol, "target_vol": TARGET_VOL,
            "vol_scale": round(vol_scale, 3), "invested": round(INVESTED * reg["exposure_scale"] * vol_scale, 3),
            "targets": targets, "prices": prices,
            "regime": reg, "lev_regime": lev_regime, "crypto_regime": cry_regime}


def plan(a=None):
    a = a or Alpaca()
    t = compute_combined(a)
    prices = t["prices"]
    positions = {p["symbol"]: float(p["qty"]) for p in a.positions()}
    orders = []
    for sym, want in t["targets"].items():
        cur = positions.get(sym, 0.0)
        delta = round(want - cur, 6)
        px = prices.get(sym, 0)
        if px and abs(delta * px) < MIN_TRADE_DOLLARS:
            continue
        orders.append({"symbol": sym, "side": "buy" if delta > 0 else "sell",
                       "qty": abs(delta), "target": round(want, 4), "cur": cur})
    for sym, cur in positions.items():
        if sym not in t["targets"] and abs(cur) > 0:
            orders.append({"symbol": sym, "side": "sell" if cur > 0 else "buy",
                           "qty": abs(cur), "target": 0.0, "cur": cur})
    return {"equity": t["equity"], "weights": t["weights"], "correlations": t["correlations"],
            "book_vol": t["book_vol"], "target_vol": t["target_vol"], "vol_scale": t["vol_scale"],
            "invested": t["invested"], "regime": t["regime"],
            "position_cap_pct": t["position_cap_pct"], "capped": t["capped"],
            "lev_regime": t["lev_regime"], "crypto_regime": t["crypto_regime"],
            "n_orders": len(orders), "orders": orders}


def execute(a=None):
    a = a or Alpaca()
    p = plan(a)
    if DRY_RUN:
        p["dry_run"] = True
        return p
    eq_open = a.clock().get("is_open")
    submitted = []
    for o in p["orders"]:
        is_crypto = o["symbol"] in CRYPTO_ORDER
        if not is_crypto and not eq_open:
            submitted.append({"symbol": o["symbol"], "skipped": "market_closed"})
            continue
        order_sym = CRYPTO_ORDER.get(o["symbol"], o["symbol"])
        qty = round(o["qty"], 6 if is_crypto else 3)
        r = a.submit_order(order_sym, qty, o["side"])
        submitted.append({"symbol": o["symbol"], "side": o["side"], "qty": qty,
                          "status": r.get("status"), "error": r.get("message")})
    p["submitted"] = submitted
    p["dry_run"] = False
    return p


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
