# US Pipeline Phase 2 — Deferred Items + SPY Economics

**Date:** 2026-08-10. Completes the deferred list from `docs/us_pipeline_report.md` §7 and moves the tradeable test to SPY.

---

## 1. The headline: the cost-neutral accuracy bar

The old framing reported a Sharpe ratio for Mag 7 straddle timing and read its negative value as a model failure. That was the wrong diagnosis. The binding quantity is the **accuracy a strategy needs merely to pay the quoted spread**:

    BE_acc = 0.5 + (straddle_spread / vega) / (2 · E|ΔIV|)

`spread/vega` is the IV move that just covers the spread, in vol points. It is set by microstructure and is **independent of model quality**. Measured on our own quotes:

| Symbol | Quoted spread | as % of premium | Spread in vol pts | E\|ΔIV\| vol pts | **Break-even accuracy** |
|---|---|---|---|---|---|
| SPY | 0.10 | **0.58%** | **0.09** | 0.77 | **55.8%** |
| TSLA | 0.55 | 1.68% | 0.94 | 1.69 | 77.7% |
| NVDA | 0.60 | 2.48% | 1.14 | 1.60 | 85.8% |
| META | 0.75 | 2.57% | 0.92 | 1.27 | 86.3% |
| AAPL | 0.40 | 3.28% | 0.87 | 1.08 | 90.3% |
| AMZN | 0.55 | 3.11% | 1.03 | 1.24 | 91.7% |
| MSFT | 1.15 | 6.32% | 1.64 | 1.12 | **123.0%** |
| GOOGL | 1.35 | 6.71% | 1.97 | 1.21 | **131.6%** |

Achieved directional accuracy is ~54%. So:

- **MSFT and GOOGL require >100% accuracy — literally impossible at any model quality.**
- The whole Mag 7 needs 78–132%; nothing published at this horizon comes close. A model hitting 60% still loses money on these names.
- **SPY needs 55.8%**, roughly ten to twenty times cheaper in vol-point terms, and is the only symbol where the bar is even in the neighbourhood of achievable.

The instrument matters far more than the model. That reframes the earlier −6.1 Sharpe from "the forecaster failed" to "the contract is unaffordable at EOD resolution."

## 2. SPY strategy — the test moved, and it is marginal

Straddle timing on SPY, delta-hedged at entry, exits at the next snapshot, costed with real quoted spreads:

| Cost assumption | Unfiltered (n=56) | Signal-filtered (n=13) |
|---|---|---|
| 50% of quoted spread | **+0.0009 / trade, Sharpe +0.21** | −0.0068, −1.44 |
| 75% of quoted spread | −0.0027 / trade, Sharpe −0.60 | −0.0108, −2.23 |
| 100% of quoted spread | −0.0064 / trade, Sharpe −1.42 | −0.0147, −2.98 |

This lands exactly where the cost-neutral bar predicts: needing 55.8% while achieving ~54% gives roughly breakeven at favourable fills and a loss at full spread. Compare Mag 7's −6.1 Sharpe.

**Honest limits.** n=56 unfiltered and n=13 filtered are both small; neither is significantly different from zero, and the filtered variant should not be read as a result at all. The signal threshold had to be made causal and scale-adaptive (an expanding quantile) because a fixed multiple of the train-period signal std fired **zero** times — measured train signal std 0.0257 versus a test median |signal| of 0.0041, a 6× scale shift. Conclusion: on this data the forecasting result is the contribution; the strategy is at best marginal on SPY and hopeless on single names.

## 3. Pooling failed its gate — and the gate was kept

A pooled cross-sectional forecaster over all 8 symbols was built as specified: per-ticker log-mean before the SVD (so PC1 encodes dynamics, not the TSLA-vs-MSFT level gap), shared basis, tiny 8-dim ticker embedding with dropout, four conditioning blocks (own state, market/SPY/dispersion, zero-parameter cross-section, earnings), de-evented factors with the known event variance re-added analytically, and an 8-seed velocity ensemble.

**Result: it lost.**

| Ticker | Pooled | Random walk | Per-ticker |
|---|---|---|---|
| AAPL | 0.003445 | 0.003738 | **0.003224** |
| MSFT | 0.003754 | 0.002963 | **0.002655** |
| GOOGL | 0.003882 | 0.003417 | **0.002992** |
| AMZN | 0.004179 | 0.003783 | **0.003367** |
| NVDA | 0.005048 | 0.004488 | **0.004127** |
| META | 0.003339 | 0.003189 | **0.002861** |
| TSLA | 0.005028 | 0.004957 | **0.004762** |
| SPY | 0.002736 | 0.002382 | — |

Pooled beat per-ticker on **0 of 7** (gate required ≥5) and CRPS was worse (0.00211 vs 0.00154), so the gate **FAILED and the per-ticker models are retained**. The block bootstrap over dates confirms pooled is significantly worse than the random walk: mean daily MSE difference **+2.355e-06, 95% CI [4.24e-07, 5.01e-06]** — entirely positive.

Named confounds, so this is not over-read as "pooling cannot work": the pooled trainer ran a fixed 600 epochs with **no validation-based early stopping**, whereas every per-ticker model early-stopped on sampled validation RMSE (SPY stopped at 1200). The pooled model also forecasts de-evented factors and re-adds event variance, so noise in the per-snapshot jump-variance estimate feeds straight into its error while the per-ticker models forecast raw surfaces. Either could explain the gap. The gate's purpose is to stop the pipeline adopting a change that did not demonstrate its value, and that is what it did.

## 4. Statistical inference corrected

The earlier "7/7 significant" framing treated seven strongly correlated mega-caps as seven independent tests. PC1 explains ~77% of cross-sectional single-name IV variation, so the effective number of independent tests is 1–2. `evaluate_us_pooled.py` now uses a **moving-block bootstrap over dates** (blocks of consecutive snapshots resampled jointly across tickers), preserving both serial and cross-sectional dependence.

## 5. M2 replacement — analytic Greeks, and a bug it caught

Closed-form Dupire local variance on the de-evented eSSVI surface, with the discrete event jump kept separate:

| Ticker | local vol ATM | IV ATM | skew ∂σ/∂k | vanna | volga | min density | implied move |
|---|---|---|---|---|---|---|---|
| AAPL | 0.224 | 0.229 | −0.599 | 0.055 | −0.0005 | 0.251 | 4.4% |
| NVDA | 0.399 | 0.391 | −0.332 | 0.056 | −0.0009 | 0.252 | 7.4% |
| SPY | 0.139 | 0.152 | −1.212 | 0.055 | −0.0003 | 0.251 | 0.0% |
| TSLA | 0.479 | 0.499 | −0.177 | 0.057 | −0.0013 | 0.254 | 7.7% |

SPY correctly shows zero implied earnings move (an ETF has no announcement) and the steepest skew; TSLA the flattest with the largest event jump.

**This run exposed a real bug in my own Stage-1 work.** The measured risk-neutral density was **negative** (−1.2 to −2.7) while the fit's `butterfly_ok` gate still reported 1.0, because I had enforced only the *first* Gatheral–Jacquier condition, ψ(1+|ρ|)<4, and not the second, θφ²(1+|ρ|)≤4. The second binds precisely when θ is small — i.e. at short maturities, which is the entire dataset. Both conditions are now enforced in the parameterization, `arbitrage_report` measures the **density itself** rather than a sufficient-condition proxy, and a regression test checks positivity over 300 extreme random draws. Correcting it also *improved* fit quality (AAPL CV 5.74→5.06, SPY 5.50→4.59 vol points), so the M1 improvement over the frozen-ρ baseline is now **1.73×–2.81×**.

## 6. M4 event head

`w = w_diff + n_events · σ_j² · s(k)`, with `n_events` used **only** as a multiplicative gate on a separate additive term — never fed to the MLP as a scalar, which would let the network smear a deterministic quarterly step into a smooth function of days-to-earnings. σ_j is learned, initialized at the measured implied moves, which agree closely with the independent Dubinsky–Johannes estimates:

| | AAPL | MSFT | GOOGL | AMZN | NVDA | TSLA | META |
|---|---|---|---|---|---|---|---|
| ours | 0.044 | 0.050 | 0.056 | 0.072 | 0.076 | 0.077 | 0.089 |
| DJ estimator | 0.046 | 0.047 | 0.061 | 0.076 | 0.083 | 0.084 | 0.091 |

Unit-tested: zero events leaves the surface bit-identical (so index symbols are untouched), the term is additive and exactly linear in the event count, higher-σ_j tickers get larger bumps, and σ_j receives finite gradients. `w_calendar` is now **structurally unreachable** in the US trainer — hard-coded to 0 with the reasoning inline, not merely weighted 0 in config.

**It did not improve accuracy, and I am not claiming it did.** With the event head at 150 epochs: pooled test tv-RMSE **0.002149**, MAPE **5.13%** — against the no-event-head baseline's 0.002118 / 4.97% at 300 epochs. Measured violation rates also rose (butterfly 1.2–10.4%, calendar 0.0–3.1%) because the additive event term sits outside the constrained ratio head. Two reasons this is not a clean verdict: the run is half the baseline's epochs, and pooling the surfaces across tickers without a ticker index forces a **single shared σ_j** rather than the per-ticker scalars the design calls for (the fallback path in the model). A fair A/B needs equal epochs and per-ticker σ_j threaded through the collate; until then the honest statement is that the event head is integrated, stable and tested, with **no demonstrated accuracy gain**.

## 7. Status and what remains

Everything on the deferred list is now built, tested and measured. What is *not* claimed:

- **Pooling is not adopted** — it failed its gate. Re-running it with validation-based early stopping (matching the per-ticker protocol) is the obvious next experiment, and the gate stays in place to judge it.
- **M4's full retrain** was run to 150 epochs for integration, not to convergence; the event head's accuracy contribution against the no-event-head baseline is not yet a clean A/B.
- **The strategy is marginal at best**, on small samples, and should not be traded on this evidence.

## 8. Reproduction

```bash
cd src
python fit_surfaces_us.py --report          # M1 Global eSSVI + arbitrage gates
python greeks_us.py --report                # M2 analytic Greeks + event jump
python train_flow_us.py --tickers SPY       # per-ticker forecaster (retained)
python train_flow_us_pooled.py --seeds 8    # pooled (built; failed its gate)
python evaluate_us_pooled.py                # the gate + block bootstrap
python strategy_spy.py                      # cost-neutral bar + SPY economics
python train_hyperiv_us.py --on_gpu         # M4 with event head
python -m pytest ../tests -q
```
