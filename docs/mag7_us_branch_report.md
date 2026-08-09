# Mag 7 US Options Branch — Report

**Date:** 2026-07-06 (results appended on completion)
**Question:** Does a deeper market (US mega-cap single-stock options) model better than TXO — and can the model outputs drive a strategy?

---

## 1. Data

**Source:** DoltHub `post-no-preference/options` (free SQL API, verified this session). Provider-computed IVs and greeks — no Black-Scholes inversion on our side, which also sidesteps the American-exercise IV-extraction problem for single-stock options.

**Shape and coverage (measured):**
- 1,231 snapshot dates 2019-01 → 2026-07; cadence ~Mon/Wed/Fri in recent years (weekly Saturdays in the 2019 era). The forecast horizon is therefore **one snapshot (1–3 calendar days), not one trading day**.
- ~140 quotes per ticker-snapshot across **3 near-month expirations** (τ ≈ 11–46 days), strikes ≈ ±30% moneyness. The US branch models the **liquid short-dated surface**, versus TXO's τ ∈ [0.02, 2] years full curve. Every TXO comparison in this report carries that caveat.
- Tickers: AAPL, MSFT, GOOGL, AMZN, NVDA, META (incl. FB pre-2022 history), TSLA.

**Engineering safeguards:**
- Spots joined **split-unadjusted** (yfinance `auto_adjust=False`): five splits in-window (AAPL 4:1 2020, TSLA 3:1 2022, AMZN 20:1 2022, GOOGL 20:1 2022, NVDA 10:1 2024) would otherwise corrupt log-moneyness by the split factor against as-traded strikes.
- Saturday snapshots map to Friday closes (backward merge-asof).
- Filters: bid > 0, 0.01 < IV < 5, |logm| ≤ 0.6, τ ≥ 3 days; w = IV²·τ, τ in calendar days / 365.25.
- Splits: train ≤ 2024-12-31, val = last 15% of train snapshots, test 2025-01 → 2026-07 (~1.5 y, includes the Apr-2025 tariff shock).

## 2. Models

- **HyperIV (pooled, cross-ticker):** one model for all seven underlyings — the reference set identifies the surface, and pooling is exactly the cross-asset transfer setting of the ICML 2025 paper. Architecture identical to the TXO branch (incl. all fixes: corrected butterfly density, tanh/softplus ratio-to-ATM target net, residual hypernetwork, PIVOT price auxiliary, spike guard).
- **Flow forecaster (per ticker):** conditional OT flow matching over PCA factors of log total variance on a 3×15 (τ×logm) grid; **increments formulation** (established on TXO — the levels flow lost to the random walk); conditions = today's factor scores + [return, RV5, VIX level/change, ATM-30d level, term slope, skew] + **gap-days to the next snapshot** (irregular but known ahead).
- **Strategy:** at each test snapshot, forecast the next-snapshot 30d ATM IV; if the forecasted move exceeds θ (1 train-period signal std), trade the nearest-30d ATM straddle (long/short vol), delta-hedged at entry with provider deltas, exit next snapshot. Fills at mid; costs = half the real quoted spread per leg per side. Assumptions and limits are stated in §5.

## 3. Results

### 3.1 Flow forecaster vs baselines (test = 2025-01 → 2026-07, ~385 pairs/ticker)

**The flow forecaster beats the next-snapshot random walk on all 7 tickers, all statistically significant:**

| Ticker | Flow tv-RMSE | RW tv-RMSE | Improvement | DM stat | p | 90% coverage |
|---|---|---|---|---|---|---|
| AAPL | 0.003280 | 0.003745 | −12.4% | −8.21 | <1e-4 | 0.70 |
| MSFT | 0.002666 | 0.002978 | −10.5% | −11.80 | <1e-4 | 0.78 |
| GOOGL | 0.002956 | 0.003406 | −13.2% | −8.73 | <1e-4 | 0.81 |
| AMZN | 0.003356 | 0.003776 | −11.1% | −4.79 | <1e-4 | 0.48 |
| NVDA | 0.004210 | 0.004608 | −8.6% | −5.40 | <1e-4 | 0.84 |
| META | 0.002751 | 0.003111 | −11.6% | −7.46 | <1e-4 | 0.64 |
| TSLA | 0.004833 | 0.005067 | −4.6% | −2.42 | 0.016 | 0.87 |

**Statistical caveat added 2026-08-09 — "7/7 significant" is not 7 replications.** The first principal component explains ~77% of cross-sectional variation in single-name IV level and skew (Christoffersen–Fournier–Jacobs, RFS 2018) and ~87% of firm-level 30-day implied variance (Baruník et al. 2023). With that much common variance the *effective* number of independent tests here is 1–2, not 7, and the seven DM statistics are strongly dependent. The honest reading is "the result holds on a correlated basket of seven mega-caps," not "it replicated seven times." Block-bootstrapping over dates is the right inference procedure and is pending.

Directional accuracy of the 30d ATM IV forecast (sign of change): 48.4%–59.4%, pooled 53.6% — the RMSE gains come mostly from denoising/shrinkage of the surface dynamics rather than strong directional calls. Coverage remains under-dispersed on some names (AMZN 0.48) — same caveat as TXO: treat intervals as indicative. Full metrics incl. CRPS and violation rates: `logs/flow_us_eval.json`; figure `logs/flow_us_vs_rw.png`.

### 3.2 Pooled HyperIV (cross-ticker)

One model, seven underlyings, 4,803 pooled training surfaces (300 epochs, ~3 h on the RTX 4060). Test period 2025-01 → 2026-07 (2,701 surfaces):

| Ticker | tv-RMSE | MAPE | IV-RMSE | Calendar viol. | Butterfly viol. |
|---|---|---|---|---|---|
| AAPL | 0.00239 | 6.60% | 0.0372 | 3.37% | 9.34% |
| MSFT | 0.00201 | 6.14% | 0.0328 | 2.91% | 8.35% |
| GOOGL | 0.00233 | 5.71% | 0.0341 | 0.79% | 4.51% |
| AMZN | 0.00244 | 5.69% | 0.0345 | 0.98% | 4.89% |
| NVDA | 0.00238 | 4.40% | 0.0330 | 0.35% | 2.41% |
| META | 0.00176 | 4.21% | 0.0263 | 0.12% | 2.88% |
| TSLA | 0.00134 | 2.31% | 0.0166 | 0.18% | 0.93% |
| **Pooled** | **0.00212** | **4.97%** | **0.0311** | — | — |

The pooled cross-ticker model matches the TXO single-market HyperIV's accuracy (TXO: tv-RMSE 0.00215, MAPE 6.94%) while covering seven surfaces at once — the hypernetwork amortizes across assets exactly as the paper's transfer experiments suggested. Violation rates are nonzero by construction (fit-only training, see below); the highest rates (AAPL/MSFT) coincide with the tightest smiles, where quoted-mid structure sits closest to the static bounds.

**Arbitrage penalties are unusable here — but NOT for the reason first recorded (corrected 2026-08-09).** The original text in this section claimed the penalties were toxic "because quoted mid surfaces around earnings events genuinely brush or cross the static bounds." **That claim is false and has been retracted.** Direct measurement of negative forward variance (the exact calendar-arbitrage test) over 56,821 adjacent-expiry × fixed-log-moneyness cells finds violations in only 1.07% of cells, **0.00% at the money**, and they are *less* frequent across an earnings gap (0.51%) than across a non-earnings gap (1.20%) — because an earnings step *adds* σ_j² to the longer expiry, which makes calendar monotonicity **easier**, not harder. ATM total variance is calendar-monotone in 100.00% of single-name adjacent-expiry pairs, and the Gatheral–Jacquier butterfly bound has measured slack of 30–45×.

The real causes are two, and both are model-side:
1. **Units incommensurability.** `calendar_loss = mean(relu(-∂w/∂τ))` is in yr⁻¹ with a natural scale of 0.10–0.42, while the fit MSE is in w² at ~5e-6. No weight makes these commensurable, and **w→0 zeroes both penalties**, so the degenerate optimum is structural rather than earnings-specific.
2. **Model-side ringing.** Forward variance genuinely steps by 1.37× (TSLA) to 2.21× (META) across an earnings gap; a smooth network τ-derivative forced to reproduce that step overshoots and manufactures negative ∂w/∂τ *in the fit*, not in the data.

The correct fix is therefore a **hard-constrained parameterization** (arbitrage-free by construction) plus an explicit event term — not penalty tuning and not an earnings-aware penalty. This is what motivates the M1 replacement in the full US pivot (`docs/superpowers/plans/2026-08-09-us-full-pivot.md`).

## 4. TXO comparison — "does a deeper market model better?"

**Forecasting: yes, convincingly.** TXO's flow model beat its random walk by 11% on one surface (DM p<1e-4). The US branch replicates that result **7 out of 7 times** with improvements of 4.6–13.2% (median ≈ 11%) — on a 1.5-year out-of-sample window that includes the April-2025 tariff shock, and at a *harder* horizon (next snapshot = 1–3 calendar days vs TXO's 1 trading day). The deeper, more liquid market gives cleaner quotes (provider IVs from tighter markets), more consistent surface dynamics across correlated names, and the cross-sectional replication that a single-market study cannot provide.

**Fitting: comparable accuracy, different constraint economics.** The pooled cross-ticker HyperIV reaches TXO-like relative fit quality (see §3.2) with one model across seven underlyings — evidence for the hypernetwork's cross-asset amortization. But the no-arbitrage story *inverts*: TXO's index surface accepts penalty enforcement; US single-name short-dated surfaces violate the bounds in the raw data (earnings structure), so enforcement destroys fit. "Deeper market" does not mean "cleaner in every sense" — it means richer microstructure that the model must respect rather than regularize away.

**Caveats on comparability (stated in §1):** the US branch models only the liquid short-dated segment (3 expirations, τ ≤ 46 d) at Mon/Wed/Fri snapshots; TXO models the full curve (τ to 2 y) daily. tv-RMSE magnitudes across the two markets are not directly comparable (different IV levels and τ ranges); relative-to-RW comparisons and violation rates are the meaningful axis.

## 5. Strategy results and honesty section

**Verdict: statistically real forecastability does not survive trading costs in a naive expression.** 543 trades pooled across the 7 tickers (test period, θ = 1 train-σ signal threshold, delta-hedged ATM straddles held one snapshot):

| Measure | Gross (mid fills) | Net (half-spread costs) |
|---|---|---|
| Mean return / trade | ≈ breakeven (−2.4% to +1.5% by ticker) | **−0.5% to −8.1%** |
| Pooled hit rate | 37–55% | 23% |
| Pooled Sharpe (annualized) | ≈ 0 | **−6.1** |

Decomposition: the ~54% directional IV edge roughly pays for theta decay over the 2–3-day hold (hence gross ≈ 0), and the real quoted bid-ask — **~4–6% of straddle premium per round trip** on these names — decides the outcome. This is exactly the pattern the literature predicts (IV-forecast economic value disappearing under transaction costs), now measured with actual quoted spreads rather than assumed ones. TSLA, with the strongest directional accuracy (59.4%), comes closest to survival (net −0.45%/trade).

**What would be required to trade this signal** (out of scope, listed for future work): expressions with lower cost-per-unit-vega (multi-snapshot holds, spread structures instead of naked straddles, entry inside the quoted spread), position filtering on signal magnitude ≫ 1σ, and earnings-window handling. **Assumptions that flatter the backtest:** mid fills, single entry-time delta hedge, no early-exercise/assignment modeling, no financing. Assumptions that penalize it: no maker fills, full half-spread paid all four legs.

Artifacts: `logs/strategy_us_results.json`, `logs/strategy_us_pnl.png`, `logs/flow_us_directional.json`.

## 6. Reproduction

```bash
python scripts/download_us_options.py          # chains + spots (resume-safe)
cd src
python train_hyperiv_us.py --on_gpu            # pooled HyperIV -> logs/hyperiv_us_results.json
python train_flow_us.py                        # 7 flow models -> models/FlowSurface_*.pt
python evaluate_us.py                          # -> logs/flow_us_eval.json, flow_us_vs_rw.png
python strategy_us.py                          # -> logs/strategy_us_results.json, strategy_us_pnl.png
```
