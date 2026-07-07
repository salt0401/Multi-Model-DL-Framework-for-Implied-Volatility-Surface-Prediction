# Mag 7 US Options Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Checkbox steps.

**Goal:** Port the pipeline to US single-stock options (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA): acquire 2019–2026 EOD chains, adapt the data layer, train pooled HyperIV + per-ticker flow forecasters, compare against the TXO results ("does a deeper market model better?"), and prototype an honestly-costed vol strategy.

**Architecture:** New parallel data layer (`src/us_dataset.py`, `dataset/us_options/`), reusing the existing model code unchanged (`hyperiv.py`, `flow_surface.py`). One pooled cross-ticker HyperIV (the reference set identifies the surface — cross-asset pooling is the paper's own transfer story). Per-ticker flow matching over PCA factors of the short-dated surface, with snapshot-gap-days as a condition. Strategy: delta-neutral ATM straddle timing from the flow forecast, costed with real bid/ask spreads.

**Data source (verified empirically this session):** DoltHub `post-no-preference/options` free SQL API — no auth; columns (date, act_symbol, expiration, strike, call_put, bid, ask, vol, delta, gamma, theta, vega, rho); provider-computed IVs (sidesteps American-exercise IV extraction); coverage ≈ 2019-02 →ongoing; snapshots ~Mon/Wed/Fri (weekly Saturdays in the 2019 era — cadence changed over time); ~140 rows/ticker/snapshot; 3 expirations (τ ≈ 11–46 d), strikes ≈ ±30% moneyness; a (date,symbol) point query returns a full chain in one page (paginate defensively if exactly 200 rows return).

## Global Constraints

- Free API: ≤3 concurrent requests, retry with backoff on failures, resume-safe manifest; total ≈ 9–10K requests.
- **US branch models the short-dated surface only** (3 expirations, τ ≲ 46 d) — state this in every comparison with TXO (τ up to 2 y).
- Spot data MUST be split-unadjusted (yfinance `auto_adjust=False`) to match as-traded strikes (AAPL 4:1 2020, TSLA 3:1 2022, AMZN/GOOGL 20:1 2022, NVDA 10:1 2024 — adjusted closes would corrupt logm by the split factor on pre-split dates).
- META traded as `FB` before 2022-06: fetch both symbols, map FB→META.
- Saturday-dated snapshots map to Friday's close for spot join.
- w = vol² · τ, τ = calendar days / 365.25. Consume provider IV directly — no Newton–Raphson here.
- Chronological splits: train ≤ 2024-12-31, val = last 15% of train snapshots, test 2025-01-01 → end (~1.5 y incl. the Apr-2025 tariff shock).
- Irregular snapshot gaps (2–3 calendar days): gap-days is a REQUIRED condition feature for the forecaster; forecast target = next snapshot, not next day.
- Reuse `hyperiv.py` / `flow_surface.py` unchanged wherever possible; new code lives in us-suffixed files.
- Commit+push at the end (authorized in this session's directive).

**Completion check:** (1) `dataset/us_options/` populated for 7 tickers + spots; (2) full pytest suite green incl. new `tests/test_us_dataset.py`; (3) `models/HyperIVModel_us.pt` + `logs/hyperiv_us_results.json` (per-ticker test metrics + violation rates); (4) per-ticker `logs/flow_us_eval.json` with FM/RW/VAR + DM; (5) `logs/strategy_us_results.json` + PnL figure with and without spread costs; (6) `docs/mag7_us_branch_report.md` with the TXO comparison; (7) pushed.

**Not doing:** Models 2/3 ports (local-vol PINN & crisis adjuster are TXO-specific research artifacts); full-chain (all-strike/all-expiry) US data (not in the free source); intraday anything; live trading integration.

## Tasks

### M1: Chain fetcher — `scripts/download_us_options.py`
Probe AAPL on every calendar date 2019-01-01→today (COUNT point query) to build the snapshot calendar; fetch full chains per (ticker, active date) for AAPL MSFT GOOGL AMZN NVDA META TSLA FB; paginate if a page returns exactly 200 rows; write `dataset/us_options/chains_{TICKER}.csv` + `manifest.json` (fetched keys, empties) for resume. Verify: row counts per ticker logged; spot-check 3 random chains against direct API queries.

### M2: Spot/aux — extend the fetcher
yfinance `auto_adjust=False` daily OHLC + dividends for the 7 (+FB history via META raw = fine, yfinance maps), ^VIX, ^IRX → `dataset/us_options/spots.csv`, `vix_us.csv`. Verify: AAPL raw close on 2020-08-28 ≈ 499 (pre-split as-traded), on 2020-08-31 ≈ 129.

### M3: `src/us_dataset.py` + `tests/test_us_dataset.py`
`UsOptionsProcessor`: load chains (FB→META), join spot (Sat→Fri), filters (bid>0, 0.01<vol<5, |logm|≤0.6, τ≥3/365), w=vol²τ, per-(ticker,date) y_atm (interp at logm=0 across the day's options, per expiration then across τ), `prepare_hyperiv_surfaces()` (pooled list of (ticker, date, tensors)), `prepare_surface_panel(ticker, train_end)` (fixed τ grid {15,30,45}/365.25 × 15-point logm grid from train quantiles; griddata linear+nearest; conditions: own return, RV5, ΔIV skew/term proxies from surface, VIX level/change, gap_days). Tests: synthetic frames — split-unadjusted join correctness, Sat mapping, FB continuity, filter bounds, panel shapes, gap-days values ∈ {2,3,…}.

### M4: Pooled HyperIV — `src/train_hyperiv_us.py` + `[hyperiv_us]` config
Reuse HyperIVModel/HyperIVLoss verbatim. Pool all tickers' surfaces; chronological split by snapshot date; train-stat normalization; spike guard; per-ticker AND pooled test metrics + violation rates → `logs/hyperiv_us_results.json`, checkpoint `models/HyperIVModel_us.pt`.

### M5: Per-ticker flow — `src/train_flow_us.py` + `[flow_us]` config
Reuse flow_surface module; increments formulation; conditions include z_today + market features + gap_days; 7 checkpoints `models/FlowSurface_{TICKER}.pt`. Cheap (CPU, ~4 min each).

### M6: Evaluation — `src/evaluate_us.py`
Per ticker: FM (100 samples) vs RW (next-snapshot persistence) vs VAR(1): tv-RMSE/IV-RMSE/IV-MAPE, DM (squared-error differentials, NW lag 5), CRPS, coverage, violation rates (actual/RW/FM) → `logs/flow_us_eval.json` + cross-ticker summary table + explicit TXO-vs-US comparison block (with the short-dated caveat and the 1-day-vs-1-snapshot horizon caveat).

### M7: Strategy — `src/strategy_us.py`
Signal: FM mean forecast of 30d-interpolated ATM IV change vs today; per snapshot per ticker: if |signal| > θ (θ from train-period signal std), enter nearest-30d ATM straddle (long if up, short if down) at MID, exit next snapshot at MID; delta-hedged approximation: subtract |Δ_straddle| × spot move (Δ from provider greeks); costs: half bid-ask spread per leg per side (real spreads from data). Report per-ticker and pooled: mean PnL/trade, hit rate, annualized Sharpe (by snapshot frequency), max drawdown, PnL curves with/without costs → `logs/strategy_us_results.json` + `logs/strategy_us_pnl.png`. State assumptions bluntly (mid fills, no early exercise, no borrow costs).

### M8: Report + commit
`docs/mag7_us_branch_report.md`: data caveats, per-ticker metrics, TXO comparison verdict, strategy results with honesty section; README/ARCHITECTURE cross-links; full suite; 2–3 commits; push.
