# US Pipeline — Architecture Decision and Stage 1 Results

**Date:** 2026-08-09
**Change:** Taiwan TXO is retired as the live pipeline. US equity options (Mag 7 + SPY) are now the primary and only active data path. TXO code is retained whole as the published index-market comparison.

---

## 1. Why the models had to change, not just the data

TXO is a **single index** with maturities to 2 years and no scheduled issuer events. The US replacement is **7 single names + 1 index ETF**, and the free data source provides only **3–4 expirations per snapshot, τ ∈ [10, 67] days** (median 28), ~29 strikes, Mon/Wed/Fri, 2019-02→2026-08 (~1.35M quotes across 8 symbols). Every option in the dataset therefore sits inside the window where **scheduled quarterly earnings dominate the term structure**.

Measured on this data:

| Effect | Magnitude |
|---|---|
| Extra ATM IV on an expiry that spans an earnings date | **+2.5 (AAPL) to +14.2 (TSLA) vol points** |
| Front-expiry ATM IV crush after the announcement | **−8.6 (MSFT) to −28.2 (META) vol points**, ~99.5% consistent over 151 events |
| Earnings share of front-expiry total variance | **23–58% (median 37%)** |
| Implied earnings move (√J², my NNLS decomposition) | AAPL 5.2%, NVDA 8.7%, TSLA 9.2% — R² = **0.99** |

The decomposition is externally corroborated: an independent Dubinsky–Johannes term-structure estimator run on the same chains gives a pooled median σ_j of 6.67% and agrees with a completely separate IV-crush estimator to **0.02 percentage points**.

A term-structure model that does not know about earnings is **mis-specified on this data**, not merely imprecise.

## 2. Correction to a previously committed claim

The earlier report stated that arbitrage penalties failed on US data "because quoted mid surfaces around earnings events genuinely brush or cross the static bounds." **That was wrong and is retracted.** Direct measurement over 56,821 adjacent-expiry × log-moneyness cells:

- negative forward variance in **1.07%** of cells, **0.00% at the money**;
- violations are **less** frequent across an earnings gap (0.51%) than across a non-earnings gap (1.20%) — an earnings step *adds* σ_j² to the longer expiry, which makes the calendar bound **easier**;
- ATM total variance is calendar-monotone in **100.00%** of single-name adjacent-expiry pairs.

The true causes are model-side: (1) **units incommensurability** — `relu(−∂w/∂τ)` is O(0.1–0.4) yr⁻¹ against an MSE of O(5e-6) in w², and **w→0 zeroes both penalties**, so the degenerate optimum is structural; (2) **ringing** — forward variance genuinely steps by 1.37×–2.21× at an announcement, and a smooth network τ-derivative overshoots it. The right fix is a hard-constrained parameterization plus an explicit event term, which is what Stage 1 delivers.

## 3. Slot decisions

| Slot | Decision | Rationale |
|---|---|---|
| **M1** eSSVI + 5-NN ensemble | **REPLACED** → Global eSSVI on de-evented variance | frozen ρ falsified (below); ρ(τ) decay unidentifiable on 3–4 expiries; arbitrage-free by construction removes the penalty problem entirely |
| **M2** ICNN Dupire local vol | **REPLACED** → event/diffusive decomposition (+ analytic SVI Greeks, pending) | its stated motivation was repairing M1's butterfly violations, which no longer exist; local vol is *undefined* across a predictable jump, and the implied event-day local variance here is 0.77–3.03/yr (88–174% one-day vol) that any smoothness prior averages away |
| **M3** TFT crisis adjustment | **DROPPED** | its target — correcting a *globally* fitted M1 — is empty under per-snapshot calibration; it would also smear a deterministic quarterly sawtooth into a learned "regime" |
| **M4** HyperIV | **KEPT** (event head pending) | only US-trained working slot, pooled MAPE 4.97% |
| **M5** Flow matching | **KEPT** (pooling/ensemble pending) | increments formulation beats next-snapshot RW on all 7 names |

## 4. Stage 1 results — Global eSSVI vs the TXO parameterization

Strike-axis 4-fold cross-validation, in **implied-vol points**, on identical slices (bid-ask noise floor ≈ 0.39 vol pts):

| Ticker | Global eSSVI | Frozen ρ=−0.95 | Improvement | median ρ | wing ratio | butterfly ok | θ increasing | calendar viol. |
|---|---|---|---|---|---|---|---|---|
| AAPL | 5.74 | 11.43 | **1.99×** | −0.357 | 1.84 | 1.00 | 1.00 | 0.0 |
| AMZN | 4.35 | 8.84 | **2.03×** | −0.294 | 1.61 | 1.00 | 1.00 | 0.0 |
| GOOGL | 5.16 | 7.85 | **1.52×** | −0.298 | 1.67 | 1.00 | 1.00 | 0.0 |
| META | 3.91 | 7.66 | **1.96×** | −0.284 | 1.63 | 1.00 | 1.00 | 0.0 |
| MSFT | 5.40 | 9.03 | **1.67×** | −0.369 | 1.89 | 1.00 | 1.00 | 0.0 |
| NVDA | 3.66 | 6.67 | **1.82×** | −0.281 | 1.65 | 1.00 | 1.00 | 0.0 |
| SPY | 5.50 | 8.71 | **1.58×** | −0.543 | 2.92 | 1.00 | 1.00 | 0.0 |
| TSLA | 2.17 | 6.11 | **2.81×** | −0.168 | 1.26 | 1.00 | 1.00 | 0.0 |

Three things to note:

1. **The frozen ρ was a TAIEX index artifact.** ρ = −0.95 implies a wing-slope ratio of 39; the measured ratio is 1.26–2.92. Freeing it roughly halves out-of-sample error on every symbol.
2. **The fits validate against theory.** SPY (index) has both the most negative ρ (−0.54) and the steepest wings (2.92); TSLA (high-beta single name) the flattest (−0.17, 1.26). The index-skew-exceeds-single-name-skew ordering falls out of the data unprompted.
3. **Arbitrage is now structural, not penalized.** Butterfly holds by construction (ψ(1+|ρ|)<4 enforced through the parameterization); ATM calendar holds by construction (θ is a cumulative sum of strictly positive increments); full-smile calendar is cleared by a deterministic projection and then **verified numerically at 0.0** — there is no penalty weight anywhere in the fit.

## 5. Data repairs

- **Earnings calendar was broken and blocking.** yfinance supplied only 2 of 4 announcements for 2025 and 1 for 2026, a gap sitting inside the test window that would mislabel ~19% of the sample as "no earnings before expiry." Rebuilt by rolling the quarterly cadence across the full span and refining each date against the local IV-crush signature, re-anchoring on known dates to prevent drift. **Backtested by hiding 2024 and predicting it from ≤2023: median error 0–1 days, 75–100% within 3 days.** Coverage is now 4/ticker/year for 2019–2025.
  - A global "find the biggest IV crashes" detector was tried first and rejected — 0.48–0.96 recall with false positives in high-vol regimes (8 detections for MSFT in 2020 against a true 4), because a market-wide vol collapse mimics a crush.
- **SPY was being silently dropped** by a mandatory spot join (SPY was never in the yfinance spot fetch). Moneyness comes from the parity forward, so the spot is now optional.
- **META's pre-2022 history** (traded as FB) was missing from the calendar repair path.

## 6. Reproduction

```bash
python scripts/download_us_options.py           # chains (Mag 7 + SPY) + spots
python scripts/download_earnings.py --validate  # raw earnings + IV-crush validation
python scripts/repair_earnings_calendar.py      # -> earnings_dates_v2.csv (backtested)
cd src && python fit_surfaces_us.py --report    # Global eSSVI A/B + arbitrage gates
python -m pytest ../tests -q                    # 255 passing
```

## 7. Deferred (explicitly not done in this pass)

- **M5 pooling + seed ensembling + event clock.** Research indicates seed ensembling is the highest-certainty accuracy lever, and pooling should be *gated* on beating the per-ticker baselines on ≥5/7 by Diebold–Mariano.
- **M4 event head** (additive `n_events · σ_j² · s(k)` term + cumulative-softplus τ so ∂w/∂τ ≥ 0 structurally), and making `w_calendar` structurally unreachable rather than merely weighted 0.
- **M2 analytic Greeks** (closed-form Dupire local variance on the de-evented SVI surface + explicit event jump).
- **Economics restructure.** The headline metric should be the cost-neutral bar `BE_acc = 0.5 + (straddle spread/vega)/(2·E|ΔIV|)` — measured at **68.7–99.1% on Mag 7 (pooled 72.4%) vs 52.5% on SPY**, against an achieved 54%. That reframes the −6.1 Sharpe from "model failure" to "instrument unaffordable at EOD resolution": a 60% model, better than anything published at this horizon, still loses on Mag 7 straddles. The one economic test should move to **SPY**, and the cost default from half-spread to full quoted spread.
- **Statistical inference.** "7/7 significant" is not 7 replications — PC1 explains ~77% of cross-sectional single-name IV variation, so the effective number of independent tests is 1–2. Block-bootstrap over dates.
- **Not pursued at all** (research-backed negative): calendar spreads (4 legs ≈ 11% of premium round trip; the sub-10-day front leg does not exist in this data), dispersion (7 names is not a basket; premium compressed to 6.7–8.9 points), pre-earnings long-vol (published effect is 4.10% in the *smallest* size quartile vs 1.71% in the largest — this universe is the largest), delta-space fitting.
