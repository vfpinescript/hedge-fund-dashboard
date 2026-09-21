# Hedge Fund v2 — Quant Architecture

**Status:** Phase 1 (risk engine) done in `~/claudeverstradingview`. Phase 2 (this
document) = the multi-agent signal committee.

**Prime directive:** every agent runs a *specific, named calculation* from
`Quant_Analysis/`. No agent invents a number. If the data an agent needs does
not exist, the agent returns `abstain`, never a guess.

---

## 1. Fund parameters

| Parameter | Value |
|---|---|
| Account equity | $1,000,000 |
| Risk per trade | 1.0% ($10,000) |
| Asset classes | Forex, Commodities, Fixed Income (bonds), Derivatives (futures/options) |
| Sizing / risk engine | `~/claudeverstradingview/src/core/risk.js` (built, 16 tests passing) |

Risk limits live in `~/claudeverstradingview/rules.json → risk_model`. The
committee produces a **direction + conviction**; the JS risk engine turns an
approved direction into a **position size** and enforces portfolio limits.
The two halves are deliberately separate: signals decide *what*, risk decides
*how much and whether allowed*.

---

## 2. The committee model

Each analysis method from `Quant_Analysis/` becomes one **agent**. Agents do not
all answer the same question — they have three distinct roles:

### 2a. Directional agents → vote `bull / bear / neutral` (+ conviction 0..1)
These produce a tradable directional opinion on one instrument.

| Agent | Method (screenshot) | Core inputs | Data status |
|---|---|---|---|
| `decision_tree` | Decision Trees | OHLCV features (returns, vol, RSI, dist-from-MA) → up/down label | ✅ obtainable (OHLCV) |
| `random_forest` | Random Forests | same feature panel, ensemble vote | ✅ obtainable |
| `gradient_boost` | XGBoost* | same feature panel, boosted | ✅ obtainable (*sklearn HistGB, see note) |
| `lasso_factors` | Lasso | many candidate signals → sparse survivors → sign | ✅ obtainable |
| `rates_macro` | Stocks vs interest rates | index level vs 2y/10y yield regime | ⚠️ needs yield series |
| `hawkes_regime` | Hawkes process | trade timestamps → branching ratio (stability/aftershock) | ❌ needs tick data |
| `charm_flow` | Charm surface | option chain Greeks, dealer positioning | ❌ needs option chain |
| `net_premium_flow` | Net premium flow | options order flow (calls vs puts $ flow) | ❌ needs flow feed |

### 2b. Risk agents → output a `risk_scale` (0..1) and/or hard `veto`
They never pick a direction. They shrink size or block trades.

| Agent | Method | Output | Data status |
|---|---|---|---|
| `hill_tail` | Hill estimator | tail index α; fat-tail VaR vs Gaussian; veto if α<3 (infinite variance regime) | ✅ obtainable (returns) |
| `vol_regime` | Volatility surface / smile | realized/implied vol level & skew → size scaler | ⚠️ implied needs chain; realized ✅ |

### 2c. Pricing agents → fair-value a specific derivative (no direction)
Used once the committee wants to *express* a view via options/futures.

| Agent | Method | Output | Data status |
|---|---|---|---|
| `black_scholes` | Black-Scholes | European option price + analytic Greeks | ✅ (needs S,K,T,r,σ) |
| `american_psor` | Crank-Nicolson + PSOR | American option price + early-exercise boundary | ✅ (numerical) |
| `mc_greeks` | Monte-Carlo autodiff | MC price + Greeks (finite-diff fallback, no JAX) | ✅ |
| `greeks_book` | Greeks reference | portfolio Greek aggregation (Δ,Γ,Θ,V,ρ,vanna,charm) | ✅ (given positions) |
| `pca_curve` | PCA | yield-curve level/slope/curvature factors, curve hedges | ⚠️ needs tenor set |

\* **XGBoost note:** true `xgboost` needs the OpenMP runtime (`libomp`), absent on
this machine (no Homebrew). `gradient_boost` uses sklearn
`HistGradientBoostingClassifier` — same algorithm family
(`F_m = F_{m-1} + η·h_m`, histogram-binned, early stopping). Swap to real
xgboost later with `brew install libomp && pip install xgboost`.

### 2d. Aggregation → the CIO
`committee.py` combines the outputs:
1. Directional agents vote; each vote weighted by (conviction × agent_weight).
2. Net score → `bull / bear / neutral` with an aggregate conviction.
3. Risk agents apply `risk_scale` (multiply) and can `veto` (force neutral/flat).
4. Output a `Decision` object: `{instrument, direction, conviction, risk_scale,
   vetoes, rationale[]}`.
5. Hand the `Decision` to the JS risk engine (`evaluateTrade`) for sizing +
   portfolio-limit checks. **No order is ever placed by this system.**

---

## 3. Data policy (the anti-hallucination rule)

| Tier | Source | Feeds |
|---|---|---|
| Have now | TradingView Desktop via CDP (`claudeverstradingview`) | daily/intraday OHLCV, standard indicators, quotes |
| Have now | Computable from OHLCV | returns, realized vol, RSI, MA distance, tail index, PCA-on-prices |
| Missing | Option chain + Greeks | vol surface, charm, BS/American *market* calibration |
| Missing | Tick / trade prints | Hawkes branching ratio, true order flow |
| Missing | Yield-curve tenors, options $ flow | rates_macro (full), net_premium_flow |

**Rule:** an agent whose feed is Missing must `abstain` (contribute nothing to
the vote) until a real feed is wired. Agents never fabricate their input.
Synthetic data is allowed ONLY in tests and is always labelled synthetic.

---

## 4. Build order

1. ✅ Risk engine (`risk.js`) + config → done.
2. Framework core: `base.py` (Signal/Agent contracts), `committee.py`, `data/loader.py`.
3. Reference agents end-to-end w/ tests: `hill_tail` (risk), `black_scholes` (pricing), `decision_tree` (directional).
4. Remaining OHLCV-only agents: `random_forest`, `gradient_boost`, `lasso_factors`, `vol_regime`, `american_psor`, `mc_greeks`, `pca_curve`.
5. Wire a real data source → unblock `rates_macro`, then the feed-gated agents.
6. Backtest harness: run the committee historically, measure hit-rate / PnL / drawdown before any capital.

---

## 5. What this system does NOT do
- It does not place, modify, or close orders. Execution is manual, by the human.
- It does not give investment advice. It computes signals and prices.
- It does not claim a signal is profitable until the backtest harness (step 6) proves edge out-of-sample.
