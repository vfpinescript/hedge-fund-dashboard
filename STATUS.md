# Hedge Fund v2 — Full Status & Roadmap

_Last updated: 2026-08-30_

A hedge fund is far more than math. This tracks **every** function, not just the
quant. Honest status labels: ✅ done · 🟡 in progress · ⛔ not started · 🚫 needs
a licensed human (I can draft/organize but cannot execute).

---

## 0. Executive reality check

We are at the **pre-strategy research** stage. To actually run money you need
four things, and today we have **one partially**:
1. A **strategy with proven edge** — 🟡 infrastructure built, **edge NOT yet found** (attribution shows ~0 IC).
2. **Legal + regulatory structure** — ⛔ not started.
3. **Capital** (LPs / seed) — ⛔ not started.
4. **Service providers** (admin, prime broker, auditor, counsel) — ⛔ not started.

Nothing here trades real money. Everything built so far is research + tooling.

---

## 1. Investment / Front Office

### 1a. Quant research & strategy — 🟡
**Done**
- Risk & position-sizing engine (`claudeverstradingview/src/core/risk.js`), 16 tests.
- Signal-committee framework (`quant/`): agent contracts, weighted vote + risk veto.
- 12 of 15 agents built & unit-tested: decision_tree, random_forest, gradient_boost,
  lasso_factors, rates_macro, hill_tail, vol_regime, black_scholes, american_psor,
  mc_greeks, greeks_book, pca_curve. (43 tests total.)
- Live free data (yfinance): FX, commodities, index futures, Treasury yields, option chains.
- Walk-forward backtest harness + **per-agent attribution (IC)**.

**Key finding:** current agents have **no measurable edge** (mean IC ≈ 0, t-stats within ±0.5).
This is a *result*, not a failure — the harness did its job.

**Left to do**
- Find actual edge: richer features (cross-asset, seasonality, term structure, positioning),
  alternative signals, or the blocked data feeds.
- 3 agents blocked on paid data: hawkes_regime (ticks), net_premium_flow (signed option flow),
  full implied-vol surface (real-time chain).
- Ensemble weighting by evidence once any agent earns positive IC.
- Adopt `nautilus_trader` as the production backtest/execution engine once a strategy proves out.

### 1b. Portfolio construction — 🟡
- Sizing + correlation/heat limits exist in the risk engine.
- Left: multi-strategy capital allocation, factor/exposure netting, drawdown control policy.

### 1c. Execution / trading — ⛔ / 🚫
- No broker connectivity. No order routing. **I will not place live orders.**
- Left (human-owned): choose broker/prime, build (or use Nautilus) execution, TCA, borrow/financing.

---

## 2. Risk Management (independent) — 🟡
- Tactical risk built (tail index, vol targeting, heat, circuit breakers).
- Left: an **independent** risk function separate from research, firm-wide VaR/stress,
  limit monitoring, liquidity risk, counterparty risk, a written risk policy.

---

## 3. Data & Technology — 🟡
**Done:** free daily data pipeline w/ cache, feature engineering, Python research stack, git repos.
**Left:** production data feeds (paid), a research database (point-in-time, survivorship-bias-free),
scheduling/monitoring, secrets management (keys currently in a local `.env`), disaster recovery,
security review before anything touches real accounts.

---

## 4. Middle Office — ⛔
- **Valuation / pricing policy** (independent price verification): ⛔
- **Performance measurement & attribution** (GIPS-style, official track record): ⛔ (research attribution ≠ audited track record)
- **Reconciliation** (positions/cash vs broker & administrator): ⛔

---

## 5. Back Office / Operations — ⛔ / 🚫
- **Fund administrator** (NAV, books, investor register): ⛔ needs to be hired.
- **Prime broker / custody**: 🚫 human — no assets in custody.
- **Trade settlement, cash/treasury management, corporate actions**: ⛔.
- **Fund accounting & audited financials**: 🚫 needs an audit firm.

---

## 6. Legal & Structure — 🚫 (I can draft/organize; a lawyer must execute)
- **Entities**: management company (LLC), General Partner entity, the Fund (LP/LLC), possibly
  offshore feeder — ⛔.
- **Fund documents**: PPM / offering memorandum, LPA, subscription docs, side letters — ⛔.
- **Service agreements**: admin, PB, auditor, counsel — ⛔.
- **IP / operating agreements / employment** — ⛔.

---

## 7. Regulatory & Compliance — 🚫
- **Registration**: e.g. SEC RIA (Form ADV) and/or CFTC/NFA (CPO/CTA) if trading futures/options;
  or exemptions (e.g. de minimis). Jurisdiction TBD — ⛔.
- **Compliance program**: written policies, code of ethics, personal-trading rules,
  AML/KYC on investors, marketing rules, recordkeeping — ⛔.
- **Chief Compliance Officer** (can be outsourced) — ⛔.
- **Blue-sky / Reg D (US) or local private-placement filings** for fundraising — ⛔.

---

## 8. Capital / Investor Relations — 🚫
- **Seed/anchor capital or founder capital** — ⛔ (nothing to raise on without a track record).
- **Investor materials** (pitch deck, DDQ, tear sheet) — ⛔ (premature until edge + track record exist).
- **Track record** — ⛔ (need ≥12–24 months of real or verified paper performance first).

---

## 9. Finance & Corporate — ⛔
- Management-company budgeting, expense tracking, fee model (e.g. 2/20), banking — ⛔.
- Insurance: D&O, E&O/professional indemnity — ⛔.

---

## 10. Governance & People — ⛔
- Team/roles (PM, trader, ops, compliance), advisory board, fund directors (offshore) — ⛔.

---

## What I (Claude) can vs cannot do
**Can:** quant research, code, backtesting, risk models, data engineering, drafting documents,
organizing structure, building the tech, preparing checklists for the human-owned items.
**Cannot / will not:** place trades or move money, form legal entities, register with regulators,
custody assets, raise capital, or give investment/legal/tax advice. Those need licensed
professionals — lawyer, compliance consultant, auditor, prime broker, and you.

---

## Recommended next 3 steps (quant track)
1. **Hunt for edge with better inputs** — engineer richer/cross-asset features and re-run
   attribution; keep only signals with positive, consistent IC.
2. **Decide on data budget** — the highest-value blocked signals (options flow, positioning)
   need a paid feed; without it, alpha is much harder.
3. **Once one signal earns its IC**, port it into `nautilus_trader` for a rigorous,
   execution-aware backtest before anything else.

## Recommended parallel step (business track — human-owned)
Talk to a **hedge-fund formation lawyer** and a **compliance consultant** early to scope
structure, jurisdiction, and cost — even while research continues. Funds routinely spend
6–12 months and significant fees on setup; starting the conversation early is free.
