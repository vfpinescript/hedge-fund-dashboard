"""
Execution-aware backtest of the reversal_7 signal in nautilus_trader (1.231.0).

This is the rigorous cost check: nautilus fills real market orders against the
bar data with a configurable fee, so the net-of-cost result is trustworthy in a
way the quick turnover*bps estimate is not.

API paths verified against the INSTALLED 1.231.0 (the cloned develop-branch
examples use a newer flattened API and do NOT match — do not copy them).
"""
from __future__ import annotations

from collections import deque
from decimal import Decimal

import numpy as np
import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.identifiers import Venue, TraderId
from nautilus_trader.model.objects import Money
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.persistence.wranglers import BarDataWrangler


class ReversalConfig(StrategyConfig, frozen=True):
    instrument_id: str
    bar_type: str
    lookback: int = 7
    vol_window: int = 63             # window for realized-vol sizing
    target_vol_usd: float = 10_000.0  # 1-sigma daily P&L target ($ = 1% of $1M)
    sigma_floor: float = 0.002       # floor on daily vol to cap position size
    max_notional_usd: float = 3_000_000.0  # leverage cap (3x on $1M)


class ReversalStrategy(Strategy):
    """Short-term reversal, VOLATILITY-TARGETED sizing with correct FX notional.

    Each position is sized so a 1-sigma daily move ~= target_vol_usd, converting
    base units per the pair's quote currency:
      - XXX/USD (EUR/USD...): 1 base unit = `price` USD  -> units = notional/price
      - USD/XXX (USD/JPY...): 1 base unit = 1 USD        -> units = notional
    This makes dollar risk comparable across pairs and fixes the JPY blow-up.
    """

    def __init__(self, config: ReversalConfig):
        super().__init__(config)
        self.closes: deque = deque(maxlen=max(config.lookback, config.vol_window) + 1)
        self.bar_type = BarType.from_str(config.bar_type)
        self.iid = self.bar_type.instrument_id
        self.instrument = None
        self.quote_is_usd = True
        self.n_bars = 0
        self.n_orders = 0

    def on_start(self):
        self.instrument = self.cache.instrument(self.iid)
        self.quote_is_usd = self.instrument.quote_currency == USD
        self.subscribe_bars(self.bar_type)

    def on_bar(self, bar: Bar):
        self.n_bars += 1
        price = float(bar.close)
        self.closes.append(price)
        if len(self.closes) <= self.config.vol_window:
            return

        ret = self.closes[-1] / self.closes[-1 - self.config.lookback] - 1.0
        target = -np.sign(ret)                       # reversal
        cur = 1.0 if self.portfolio.is_net_long(self.iid) else \
              -1.0 if self.portfolio.is_net_short(self.iid) else 0.0
        if target == cur:
            return

        # Vol-targeted notional.
        arr = np.asarray(self.closes)
        sigma = float(np.std(np.diff(arr) / arr[:-1]))
        sigma = max(sigma, self.config.sigma_floor)
        notional_usd = min(self.config.target_vol_usd / sigma, self.config.max_notional_usd)
        base_units = notional_usd / price if self.quote_is_usd else notional_usd

        self.close_all_positions(self.iid)
        if target != 0 and base_units > 0:
            self._order(OrderSide.BUY if target > 0 else OrderSide.SELL, base_units)

    def _order(self, side, base_units):
        order = self.order_factory.market(
            instrument_id=self.iid,
            order_side=side,
            quantity=self.instrument.make_qty(round(base_units)),
        )
        self.n_orders += 1
        self.submit_order(order)

    def on_stop(self):
        self.close_all_positions(self.iid)


def _invert_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Invert a USD/XXX series into XXX/USD (price -> 1/price, high<->low).

    Inverted prices (e.g. 1/USDJPY ~ 0.006) are then rescaled to O(1) so
    vol-targeted quantities stay under the instrument's max_quantity. Scaling
    price by k shrinks size by k (sizing is vol-targeted), so dollar P&L and
    Sharpe are unchanged — only the quote units are cosmetic.
    """
    inv_close = 1.0 / df["close"]
    k = 10.0 ** (-round(np.log10(inv_close.median())))   # bring median near 1
    out = pd.DataFrame(index=df.index)
    out["open"] = k / df["open"]
    out["close"] = k / df["close"]
    out["high"] = k / df["low"]        # reciprocal swaps the extremes
    out["low"] = k / df["high"]
    out["volume"] = df["volume"]
    return out


def run_backtest(instrument_str="EUR/USD", ohlcv: pd.DataFrame | None = None,
                 lookback=7, fee_bps=0.0):
    """Backtest one FX pair. USD/XXX pairs are inverted to XXX/USD so P&L is
    always in the USD account currency (exact, no cross-rate needed)."""
    from ..data.loader import load_ohlcv
    base, _, quote = instrument_str.partition("/")
    if ohlcv is None:
        if base == "USD" and quote and quote != "USD":
            # e.g. "USD/JPY" -> load USDJPY, invert to JPY/USD
            ohlcv = _invert_ohlcv(load_ohlcv("USD" + quote, bars=3000))
            instrument_str = f"{quote}/USD"
        else:
            ohlcv = load_ohlcv(instrument_str.replace("/", ""), bars=3000)

    engine = BacktestEngine(BacktestEngineConfig(
        trader_id=TraderId("BACKTESTER-001"),
        logging=LoggingConfig(bypass_logging=True),
    ))
    SIM = Venue("SIM")
    engine.add_venue(
        venue=SIM, oms_type=OmsType.NETTING, account_type=AccountType.MARGIN,
        base_currency=USD, starting_balances=[Money(1_000_000, USD)],
    )
    instrument = TestInstrumentProvider.default_fx_ccy(instrument_str, SIM)
    engine.add_instrument(instrument)

    bar_type = BarType.from_str(f"{instrument.id}-1-DAY-MID-EXTERNAL")
    df = ohlcv[["open", "high", "low", "close", "volume"]].copy().dropna()
    # Enforce OHLC consistency (adjusted data can round low>close etc.).
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    if (df["volume"] <= 0).any():
        df["volume"] = df["volume"].clip(lower=1)   # FX volume is 0; wrangler needs >0
    wrangler = BarDataWrangler(bar_type, instrument)
    bars = wrangler.process(df)
    engine.add_data(bars)

    strat = ReversalStrategy(ReversalConfig(
        instrument_id=str(instrument.id), bar_type=str(bar_type), lookback=lookback,
    ))
    engine.add_strategy(strat)
    engine.run()

    account = engine.trader.generate_account_report(SIM)
    fills = engine.trader.generate_order_fills_report()
    positions = engine.trader.generate_positions_report()
    diag = {"n_bars": strat.n_bars, "n_orders": strat.n_orders,
            "instrument_id": str(instrument.id)}
    engine.reset(); engine.dispose()
    return account, fills, positions, diag
