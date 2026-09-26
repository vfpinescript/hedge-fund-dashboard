"""
"Leverage for the Long Run" (Gayed, 2016 Dow Award) generalised across instruments.

Rule: when the asset closes ABOVE its N-day moving average, hold it (optionally
levered L x, paying financing on the borrowed (L-1) plus a 1%/yr leveraged-ETF
fee); when at or BELOW the MA, hold cash (the risk-free / T-bill proxy). The
signal is known from the prior close, so exposure earns only the next day's return.

We run it on many instruments and rank by the Deflated-Sharpe-gated, risk-adjusted
result to answer: which instrument does this timing-plus-leverage rule fit best?
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from quant.data.loader import load_ohlcv                     # noqa: E402
from quant.research.metrics import perf_metrics              # noqa: E402
from quant.research.overfitting import deflated_sharpe_ratio  # noqa: E402

WINDOW = 200        # canonical 200-day SMA
LEVERAGE = 2.0      # the paper's declared main portfolio (2x rotation)
FEE_ANNUAL = 0.01   # 1%/yr leveraged-ETF fee
BARS = 9000         # ~35y where available

# ETF / ticker, asset class, human label
UNIVERSE = [
    ("SPY", "US Equity", "S&P 500"),
    ("QQQ", "US Equity", "Nasdaq 100"),
    ("IWM", "US Equity", "Russell 2000"),
    ("DIA", "US Equity", "Dow 30"),
    ("EEM", "EM Equity", "Emerging Mkts"),
    ("EWJ", "Intl Equity", "Japan"),
    ("GLD", "Commodity", "Gold"),
    ("SLV", "Commodity", "Silver"),
    ("USO", "Commodity", "Crude Oil"),
    ("TLT", "Rates", "20y Treasuries"),
    ("HYG", "Credit", "High Yield"),
    ("FXE", "FX", "EUR/USD"),
    ("FXY", "FX", "JPY/USD"),
    ("BTC-USD", "Crypto", "Bitcoin"),
    ("ETH-USD", "Crypto", "Ethereum"),
]


def _rf_daily(index) -> pd.Series:
    """13-week T-bill yield (^IRX) -> daily risk-free, aligned to `index`."""
    try:
        import yfinance as yf
        irx = yf.download("^IRX", period="max", progress=False, auto_adjust=False)["Close"]
        if isinstance(irx, pd.DataFrame):
            irx = irx.iloc[:, 0]
        irx.index = pd.to_datetime(irx.index).tz_localize(None)
        rf = (irx / 100.0 / 252.0).reindex(index).ffill().fillna(0.0)
        return rf
    except Exception:
        return pd.Series(0.02 / 252, index=index)   # 2%/yr fallback


def rotation_returns(close: pd.Series, rf: pd.Series, window=WINDOW,
                     leverage=1.0, fee_annual=0.0) -> pd.Series:
    r = close.pct_change()
    sma = close.rolling(window).mean()
    active = (close > sma).shift(1).fillna(False)          # known before the day
    fee_d = fee_annual / 252.0
    lev_ret = leverage * r - (leverage - 1.0) * rf - fee_d  # active-state return
    strat = pd.Series(np.where(active, lev_ret, rf), index=close.index)
    return strat.dropna()


def _row(name, cls, strat, n_trials):
    m = perf_metrics(strat)
    if not m:
        return None
    dsr = deflated_sharpe_ratio(strat.values, n_trials)
    eq = (1 + strat).cumprod()
    cagr = eq.iloc[-1] ** (252 / len(strat)) - 1
    return {
        "instrument": name, "class": cls, "n_days": len(strat),
        "cagr_pct": round(cagr * 100, 1), "sharpe": round(m["sharpe"], 3),
        "max_dd_pct": round(m["max_drawdown"] * 100, 1),
        "sortino": round(m.get("sortino", 0), 2),
        "dsr": round(dsr["dsr"], 3), "verdict": dsr["verdict"],
    }


def run():
    loaded = {}
    for sym, cls, label in UNIVERSE:
        try:
            df = load_ohlcv(sym, bars=BARS)
            if df is None or len(df) < WINDOW + 300:
                continue
            loaded[sym] = (cls, label, df["close"].astype(float))
        except Exception:
            continue
    n_trials = max(20, len(loaded) * 2)   # timing + leverage variants
    rows_2x, rows_1x, rows_bh = [], [], []
    for sym, (cls, label, close) in loaded.items():
        rf = _rf_daily(close.index)
        s2 = rotation_returns(close, rf, leverage=LEVERAGE, fee_annual=FEE_ANNUAL)
        s1 = rotation_returns(close, rf, leverage=1.0, fee_annual=0.0)
        bh = close.pct_change().dropna()
        for bucket, series in ((rows_2x, s2), (rows_1x, s1), (rows_bh, bh)):
            r = _row(label, cls, series, n_trials)
            if r:
                bucket.append(r)
    return {"n_trials": n_trials,
            "rotation_2x": sorted(rows_2x, key=lambda x: -x["sharpe"]),
            "rotation_1x": sorted(rows_1x, key=lambda x: -x["sharpe"]),
            "buy_hold": {r["instrument"]: r for r in rows_bh}}


if __name__ == "__main__":
    import json
    out = run()
    print(f"DSR n_trials = {out['n_trials']}\n")
    print("=== 2x LEVERAGE ROTATION (above 200d MA -> 2x asset, else T-bills) ===")
    print(f"{'instrument':16}{'class':13}{'CAGR%':>7}{'Sharpe':>8}{'MaxDD%':>8}{'DSR':>7}  verdict   vs B&H Sharpe")
    bh = out["buy_hold"]
    for r in out["rotation_2x"]:
        b = bh.get(r["instrument"], {})
        print(f"{r['instrument']:16}{r['class']:13}{r['cagr_pct']:>7}{r['sharpe']:>8}"
              f"{r['max_dd_pct']:>8}{r['dsr']:>7}  {r['verdict']:<11}{b.get('sharpe','-')}")
    print("\n=== 1x TIMING ONLY (above MA -> asset, else T-bills; no leverage) ===")
    for r in out["rotation_1x"][:8]:
        print(f"{r['instrument']:16}{r['class']:13}{r['cagr_pct']:>7}{r['sharpe']:>8}{r['max_dd_pct']:>8}{r['dsr']:>7}  {r['verdict']}")
