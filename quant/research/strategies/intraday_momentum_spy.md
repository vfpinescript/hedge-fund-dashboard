# Market Intraday Momentum — SPY (long-only)

Source: Gao, Han, Li & Zhou, "Market Intraday Momentum", JFE 2018.
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2440866

## Rule
- First-half-hour return r1 = (P[10:00 ET] / P[prev close]) - 1
- If r1 > 0: go long SPY at 15:30 ET, exit at 16:00 ET (long-only version;
  the paper is long/short — short leg omitted because Astral is long-only).

## Astral compiled spec (30m bars)
- prev_close = VALUE_WHEN(HOUR()*60+MINUTE() == 960, SPY.close)
- p1000      = VALUE_WHEN(HOUR()*60+MINUTE() == 600, SPY.close)
- fh_ret     = p1000 / prev_close - 1
- signal first_half_up: fh_ret > 0
- rule enter (daily 15:30 NY, if first_half_up): buy SPY 95% equity
- rule exit  (daily 16:00 NY): sell SPY 100%

## Status
- Authored & validated in Astral (free tier). NOT backtested — free tier
  allows 0 backtests. Must be validated through our own DSR/PBO gate before
  it counts as an edge.
- Requires 30-min intraday data (daily engine cannot see the 15:30-16:00 window).
