# Full US Pivot — Implementation Plan

**Goal:** Replace TAIEX/TXO entirely with US equity options (Mag 7 + SPY) across the whole pipeline, changing model slots where the data structure demands it.

**Completion check (named up front):**
```bash
cd src && python -m pytest ../tests -q            # all green incl. new US tests
python fit_surfaces_us.py --report                # M1: CV RMSE < 1.0 vol pts, 0 arb violations by construction
python evaluate_us.py --n_samples 100             # M5: per-ticker DM vs RW *and* vs per-ticker baseline
python strategy_us.py --universe SPY --cost full  # economics: cost-neutral bar reported, not just Sharpe
```
plus `docs/us_pipeline_report.md` written and `git push` clean.

## Constraints (from the user, verbatim)
- "Let's replace completely with US market" — US is now the primary and only live pipeline.
- "you could change the model if you believe the change of data requires different model" — model swaps authorized where justified.

## Measured facts driving every decision
- 3–4 expiries/snapshot, τ ∈ [10, 67] d (median 28), ~29 strikes, Mon/Wed/Fri, 2019-02→2026-08, 8 symbols (Mag 7 + SPY).
- Earnings dominate: an expiry spanning an announcement carries **+2.5 (AAPL) to +14.2 (TSLA) vol points**; post-event front-expiry crush **−8.6 to −28.2 vol points**, ~99.5% consistent over 151 events. Earnings variance is **23–58% (median 37%)** of front-expiry total variance.
- My own decomposition (NNLS on ATM term structure) gives R²=0.99 and implied moves AAPL 5.2% / NVDA 8.7% / TSLA 9.2%, agreeing with the independent Dubinsky–Johannes term-structure estimator (pooled median 6.67%).
- **CORRECTION to the committed report:** the penalty toxicity is NOT "the market violates no-arbitrage around earnings." Measured: negative forward variance in 1.07% of cells, **0.00% at ATM**, and *less* frequent across an earnings gap (0.51%) than without (1.20%) — earnings *raises* w, easing the calendar bound. True causes: (a) **units incommensurability** — `relu(-dw/dτ)` is O(0.10–0.42) yr⁻¹ vs MSE O(5e-6) in w²; no weight makes them commensurable and w→0 zeroes both, so the degenerate optimum is structural; (b) model-side ringing against a forward-variance step of 1.37×–2.21×.
- Frozen `rho_0=-0.95` (a TAIEX index artifact) is falsified for single names: measured median ρ −0.52 (SPY) to −0.12 (TSLA); <2.1% of slices compatible with ρ<−0.90; freezing costs **3.2×–4.7× CV RMSE** (6.198 vs 1.818 vol pts) against a 0.388 vol-pt bid-ask noise floor.
- Cost-neutral bar BE_acc = 0.5 + (straddle spread/vega)/(2·E|ΔIV|) = **68.7–99.1% on Mag 7 (pooled 72.4%) vs 52.5% on SPY**; achieved 54%. A 60% model — better than anything published — still loses on Mag 7 straddles.
- Cross-section: PC1 explains ~77% of single-name IV level/skew variation. So "7/7 significant" is **not** 7 independent replications; effective independent tests ≈ 1–2.

## Slot decisions
| Slot | Decision | Why |
|---|---|---|
| M1 eSSVI+5NN | **REPLACE** → Global eSSVI (Mingone 2022) on de-evented variance + per-slice SVI-JW | frozen ρ falsified; ρ(τ) decay unidentifiable on 3–4 expiries; arb-free by construction kills the penalty problem; scipy, CPU, ms/snapshot |
| M2 ICNN Dupire | **REPLACE** → closed-form Dupire on de-evented SVI + explicit event jump | motivation (repair M1's butterfly violations) vanishes under an arb-free backbone; local vol undefined across a predictable jump; ICNN smooths away an 88–174% one-day event local vol |
| M3 TFT adjust | **DROP** | its target (correct a *globally* fitted M1) is empty under per-snapshot calibration; would smear a deterministic quarterly sawtooth into a learned "regime" |
| M4 HyperIV | **KEEP + event head** | only US-trained working slot (pooled MAPE 4.97%); add additive event term + cumulative-softplus τ so dw/dτ≥0 structurally |
| M5 flow matching | **KEEP + pool + de-event + seed ensemble** | increments formulation works; pooling gated on beating per-ticker on ≥5/7 |

## Task order (value / effort)
1. **T1 Earnings calendar repair (BLOCKING).** yfinance covers only 2/4 announcements in 2025 and 1 in 2026 — the gap is inside the test window. Rebuild by IV-crush detection + quarterly-cadence prior, *validated against the 2019–2024 ground truth for precision/recall*. Document that reconstructing a publicly-pre-announced schedule is legitimate backtest information.
2. **T2 Correct the false claim** in `docs/mag7_us_branch_report.md` §3.2 + this session's summary; make `w_calendar` structurally unreachable rather than merely 0.0.
3. **T3 M1 = Global eSSVI** on de-evented variance (`src/svi_us.py`, `src/fit_surfaces_us.py`). Gates asserted not penalized: ρ∈[−0.7,0.2], strike-axis CV RMSE < 1.0 vol pts, wing ratio ∈[1.2,4.0], zero arb violations by construction.
4. **T4 M5 seed ensemble** (~20 lines, minutes of compute, highest-certainty lever).
5. **T5 Economics restructure**: cost-neutral bar as the headline metric; move the tradeable test to SPY; full-quoted-spread default; earnings short-vol as a *power-limited* event study (n=38, t=0.74).
6. **T6 M5 pooling + event clock**, gated on DM vs per-ticker baselines.
7. **T7 M4 event head** + structural monotonicity.
8. **T8 M2 analytic Greeks** from SVI (last; nothing consumes it until M3's role is settled).

---

# Phase 2 — Deferred items + SPY strategy (2026-08-09)

**Completion check:**
```bash
cd src
python train_flow_us_pooled.py --seeds 10        # pooled + ensemble, incl. SPY
python evaluate_us_pooled.py                     # GATE: pooled must beat per-ticker on >=5/7 by DM
python strategy_spy.py --cost-fraction 1.0       # SPY economics + cost-neutral bar + sensitivity
python train_hyperiv_us.py --on_gpu              # M4 with event head
python greeks_us.py --report                     # M2 replacement: analytic SVI Greeks + event jump
python -m pytest ../tests -q                     # all green
```

**P2 tasks**
1. **M5 pooled + seed ensemble + event clock.** Per-ticker log-mean before SVD (else PC1 encodes the TSLA-vs-MSFT level gap, not dynamics); shared basis; per-ticker z-scored scores. Conditioning blocks: own state, market (SPY scores + VIX + dispersion), zero-parameter cross-section (mean z, deviation), earnings (n_events, days-to-earnings), gap-days. Tiny ticker embedding (8-dim + dropout) so training starts at full pooling. 10-seed velocity ensemble. **Gate:** pooled must beat the existing per-ticker models on ≥5/7 by DM and on pooled CRPS, else keep per-ticker and report that.
2. **Block bootstrap over dates** replacing the invalid "7/7 independent tests" framing.
3. **SPY strategy + cost-neutral bar as headline.** `BE_acc = 0.5 + (straddle_spread/vega)/(2·E|ΔIV|)`. Default cost = **full quoted spread** (was half). Report a sensitivity curve over cost fraction 0.5→1.0 and over achieved accuracy, not a point Sharpe.
4. **M4 event head.** `w = w_diff + n_events·σ_j²·s(k)`; σ_j per-ticker learned scalar initialized at measured medians; `n_events` as a multiplicative gate, never a raw scalar input. Make `w_calendar` structurally unreachable, not merely weight 0.
5. **M2 replacement.** Closed-form Dupire local variance on the de-evented SVI surface + explicit discrete event jump; analytic vanna/volga/∂σ_LV/∂K from SVI parameters.

## Not doing
- No new option-data vendor (3–4 expiries is the ceiling of the free source; documented as the binding limitation).
- No calendar spreads / dispersion / pre-earnings long-vol strategies (research: negative net at this universe and resolution; sub-10-day front leg doesn't exist in the data).
- No delta-space fitting (outer 30% of strikes occupy 8.8% of the delta range here).
- No deletion of TXO code: `model1_research/`, `model2_research/`, `model3_research/` retained whole as the published index-market comparison. TXO becomes archive, not live path.
- No intraday execution modeling.
