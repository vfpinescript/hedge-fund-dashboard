"""
Data access + feature engineering.

`load_ohlcv` is the single seam where real market data enters the system.
Today it raises NotImplementedError for the live path — we wire it to the
TradingView CDP bridge (or a real vendor) in build step 5. Until then, agents
are exercised on `synthetic_ohlcv`, which is ALWAYS labelled synthetic.

`make_features` builds the feature panel the ML agents share (returns, realized
vol, RSI, distance-from-MA). Labels are forward-return sign, with a temporal
(never shuffled) train/test split — the discipline the screenshots stress.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Map TradingView-style tickers (from rules.json) -> Yahoo Finance symbols.
# Passthrough for anything already in Yahoo format.
SYMBOL_MAP = {
    # FX majors
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X", "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X",
    # Commodities
    "XAUUSD": "GC=F", "XAGUSD": "SI=F", "USOIL": "CL=F", "UKOIL": "BZ=F",
    "NATGAS": "NG=F", "HG1!": "HG=F",
    # Rates / bonds  (^TNX etc. are yields in %; ZN/ZB are note/bond futures)
    "US10Y": "^TNX", "US02Y": "^IRX", "US30Y": "^TYX",
    "ZN1!": "ZN=F", "ZB1!": "ZB=F",
    # Equity index futures / indices
    "ES1!": "ES=F", "NQ1!": "NQ=F", "SPX": "^GSPC", "NDX": "^NDX",
    # --- extended universe ---
    # FX crosses
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X", "AUDJPY": "AUDJPY=X", "EURCHF": "EURCHF=X",
    # Metals & softs
    "XPTUSD": "PL=F", "XPDUSD": "PA=F", "CORN": "ZC=F", "WHEAT": "ZW=F",
    "SOYBEAN": "ZS=F", "COFFEE": "KC=F", "SUGAR": "SB=F",
    # Crypto
    "BTCUSD": "BTC-USD", "ETHUSD": "ETH-USD", "SOLUSD": "SOL-USD",
    "LTCUSD": "LTC-USD",
    # Global equity indices
    "RTY1!": "RTY=F", "DAX": "^GDAXI", "NIKKEI": "^N225", "FTSE": "^FTSE",
    "ESTX50": "^STOXX50E", "HSI": "^HSI",
}

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data_cache"
_CACHE_TTL_SECONDS = 6 * 3600   # refresh daily bars at most every 6h


def to_yahoo(instrument: str) -> str:
    return SYMBOL_MAP.get(instrument.upper(), instrument)


def load_ohlcv(instrument: str, timeframe: str = "1d", bars: int = 2000,
               use_cache: bool = True) -> pd.DataFrame:
    """Daily OHLCV from Yahoo Finance (free). Columns: open/high/low/close/volume.

    Cached to data_cache/ as parquet with a 6h TTL. `timeframe` currently
    supports Yahoo intervals ('1d','1h', ...); default daily.
    Raises RuntimeError if the symbol returns no data (agent should abstain).
    """
    import yfinance as yf

    yh = to_yahoo(instrument)
    _CACHE_DIR.mkdir(exist_ok=True)
    cache = _CACHE_DIR / f"{yh.replace('=','_').replace('^','idx_').replace('!','')}_{timeframe}.parquet"

    if use_cache and cache.exists() and (time.time() - cache.stat().st_mtime) < _CACHE_TTL_SECONDS:
        df = pd.read_parquet(cache)
    else:
        # ~2000 daily bars ≈ 8y; cap request span accordingly.
        period = "max" if bars > 1250 else "5y"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = yf.download(yh, period=period, interval=timeframe,
                              progress=False, auto_adjust=True)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"no data for {instrument} ({yh})")
        # Flatten possible multi-index columns from yfinance.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df.index.name = "date"
        try:
            df.to_parquet(cache)
        except Exception:
            pass  # cache is best-effort

    if len(df) > bars:
        df = df.iloc[-bars:]
    return df


def synthetic_ohlcv(
    bars: int = 1500,
    seed: int = 0,
    mu: float = 0.0003,
    sigma: float = 0.012,
    regime_shift: bool = True,
) -> pd.DataFrame:
    """SYNTHETIC price series (GBM + optional vol regimes). Tests only."""
    rng = np.random.default_rng(seed)
    vol = np.full(bars, sigma)
    if regime_shift:
        # inject a couple of higher-vol regimes so risk agents have signal
        vol[bars // 3: bars // 3 + 80] *= 3.0
        vol[2 * bars // 3: 2 * bars // 3 + 60] *= 4.0
    rets = rng.normal(mu, vol)
    close = 100.0 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.normal(0, 0.003, bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, bars)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1_000, 10_000, bars).astype(float)
    idx = pd.date_range("2018-01-01", periods=bars, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def make_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Feature panel shared by the ML agents. Matches the screenshots'
    columns: returns, realized vol, RSI, distance-from-MA, volume z-score."""
    c = ohlcv["close"]
    f = pd.DataFrame(index=ohlcv.index)
    f["ret_1d"] = c.pct_change()
    f["ret_5d"] = c.pct_change(5)
    f["vol_20d"] = f["ret_1d"].rolling(20).std()
    f["rsi_14"] = rsi(c, 14)
    sma50 = c.rolling(50).mean()
    f["dist_sma50"] = (c - sma50) / sma50
    v = ohlcv["volume"]
    f["volume_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    return f


def make_features_wide(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Expanded candidate-signal panel for Lasso (many features, most useless).
    The screenshots' point: throw ~30 signals at Lasso, let L1 kill the noise."""
    c = ohlcv["close"]
    v = ohlcv["volume"]
    f = pd.DataFrame(index=ohlcv.index)
    for w in (1, 2, 3, 5, 10, 20):
        f[f"ret_{w}d"] = c.pct_change(w)
    for w in (5, 10, 20, 50):
        f[f"vol_{w}d"] = c.pct_change().rolling(w).std()
    for w in (7, 14, 21):
        f[f"rsi_{w}"] = rsi(c, w)
    for w in (10, 20, 50, 100):
        sma = c.rolling(w).mean()
        f[f"dist_sma{w}"] = (c - sma) / sma
    f["volume_z"] = (v - v.rolling(20).mean()) / v.rolling(20).std()
    f["hl_range"] = (ohlcv["high"] - ohlcv["low"]) / c
    f["mom_accel"] = c.pct_change(5) - c.pct_change(5).shift(5)
    return f


# Yahoo constant-maturity Treasury yield tickers (percent).
YIELD_TENORS = {"^IRX": 0.25, "^FVX": 5.0, "^TNX": 10.0, "^TYX": 30.0}


def load_yield_curve(bars: int = 1000, use_cache: bool = True) -> pd.DataFrame:
    """Daily Treasury yields (%) across available tenors. Columns are years to
    maturity (0.25, 5, 10, 30). Free via yfinance — no FRED key needed."""
    cols = {}
    for tk, yrs in YIELD_TENORS.items():
        try:
            df = load_ohlcv(tk, bars=bars, use_cache=use_cache)
            cols[yrs] = df["close"]
        except Exception:
            continue
    if not cols:
        raise RuntimeError("no yield tenors available")
    curve = pd.DataFrame(cols).sort_index(axis=1).dropna()
    return curve


def load_option_chain(instrument: str, expiry_index: int = 1) -> dict:
    """Option chain snapshot via yfinance (free, no key). Returns calls/puts
    DataFrames with impliedVolatility, openInterest, bid/ask, plus spot & expiry.
    Works for equity/ETF/index underlyings (e.g. SPY, QQQ). Raises for symbols
    with no listed options."""
    import yfinance as yf
    yh = to_yahoo(instrument)
    tk = yf.Ticker(yh)
    exps = tk.options
    if not exps:
        raise RuntimeError(f"no listed options for {instrument} ({yh})")
    exp = exps[min(expiry_index, len(exps) - 1)]
    ch = tk.option_chain(exp)
    spot = tk.history(period="1d")["Close"].iloc[-1]
    return {"expiry": exp, "spot": float(spot), "calls": ch.calls, "puts": ch.puts}


def label_forward(ohlcv: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Binary label: 1 if price is higher `horizon` bars ahead, else 0."""
    c = ohlcv["close"]
    return (c.shift(-horizon) > c).astype(int)


def temporal_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.7):
    """Chronological split — NEVER shuffle time series (screenshots' rule)."""
    xy = X.join(y.rename("y")).dropna()
    cut = int(len(xy) * train_frac)
    Xtr, ytr = xy.iloc[:cut].drop(columns="y"), xy.iloc[:cut]["y"]
    Xte, yte = xy.iloc[cut:].drop(columns="y"), xy.iloc[cut:]["y"]
    return Xtr, ytr, Xte, yte
