# Model 3 US-Close Look-Ahead Leakage Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove look-ahead leakage from Model 3 (TFT adjustment model) inputs — `sp500_return` and `vix_change` dated day *t* are realized ~15h after Taiwan's close on day *t* — then retrain TFT+CPR and quantify how much of the recorded Test RMSE 0.1558 / MAPE 9.51% was leakage.

**Architecture:** Lag the two US-close features by one row of their own date grid **at merge time inside `DataProcessor.prepare_adjustment_data`** (src/dataset.py:424 and :433). Do NOT change `scripts/build_features.py` values or `Prepare_diffusion_data` — Model 5 conditions day-*t* features on a day-*t+1* target, where same-date alignment is legitimate. Keep exact-date merges (no merge_asof) so the dropna row-composition matches the recorded baseline as closely as possible.

**Tech Stack:** pandas, PyTorch 2.11+cu126, pytest; training on RTX 4060 Laptop 8GB.

## Global Constraints

- User-prescribed fix: "shift sp500_return by one Taiwan trading day for same-day-target uses" — implemented as `shift(1)` on the enhancement frame, whose rows are the Taiwan trading-day grid (built from TWII in build_features.py).
- Scope extension (justified by the requested audit): `vix_change` from `dataset/VIX.csv` is the **US CBOE VIX** (values match CBOE closes exactly, e.g. 14.23 on 2014-01-02) and is merged same-date at src/dataset.py:424 — identical leak class. It is lagged in the same change; the retrain therefore measures the **combined US-close leakage**, stated explicitly in the report.
- Do not touch: `Prepare_diffusion_data` (Model 5, legit), `load_vix_data` itself (shared with Model 5 path), `build_features.py` feature VALUES (docstring clarification only), the pre-existing uncommitted `getYATM` leakage-guard changes in src/dataset.py.
- Do NOT commit — user has not asked for commits; leave changes in the working tree (there are unrelated uncommitted hyperiv changes; do not mix or revert them).
- Retrain command (matches recorded winner exactly, README "TFT — float32", results JSONs confirm fp32 timings): from `model3_research/scripts/`: `python train_models.py --model tft --optimizer cpr --dtype float32`. Seed 42, epochs 1000 w/ early stopping (patience 100), lr 0.001, batch 256 — all from src/config.ini defaults.
- Completion check (named up front): `model3_research/logs/tft_cpr_fp32_results.json` exists with non-null `test_rmse`/`test_mape`, full `pytest tests/` passes, and the final report compares against 0.1558 / 9.51%.

## Audit verdicts (deliverable, established during planning)

| Feature | Source & timing | Verdict |
|---|---|---|
| `sp500_return` | US close day *t* (≈04:00–05:00 Taiwan day *t+1*) merged same-date | **Leakage** for same-day target → lag 1 Taiwan trading day |
| `vix_change` | US CBOE VIX close day *t*, same timing | **Leakage** (same class) → lag 1 row on VIX grid |
| `iv_skew`, `iv_term_slope` | Day-*t* TAIEX option cross-section (same quotes as target) | Contemporaneous at Taiwan close — **not a time-arrow violation**, but **target contamination**: aggregates include the quote whose tv_ratio is predicted. Keep (by design: model adjusts day-*t* surface given day-*t* market state; base model's y_atm/tv_pred are equally day-*t*). Document caveat: Model 3 RMSE is surface-reconstruction skill, not forecasting skill; if ever deployed pre-close, these must be lagged too. |
| `vrp_20d` | `atm_iv(t)² − rv_20d(t)²`, day-*t* quotes + TWII close | Same as above — keep, document |
| `rv_20d`, `futures_basis_pct`, `underlying_return` | TWII/TAIFEX day-*t* closes | Contemporaneous Taiwan close — legitimate for a close-time model |

Not doing: merge_asof retention of US-holiday rows (changes sample composition vs baseline); per-feature attribution runs (one retrain only, per request); lagging Taiwan-close features (redefines the model); committing.

---

### Task 1: Baseline test suite

**Files:** none modified.

- [ ] **Step 1: Run existing suite** — `python -m pytest tests/ -q` from repo root. Record pass/fail counts. If already failing, report before proceeding (working tree has uncommitted hyperiv changes).

### Task 2: Failing repro tests (TDD)

**Files:**
- Modify: `tests/test_dataset.py` (class `TestPrepareAdjustmentData`, after `test_dates_chronological_for_split`)

**Interfaces:** Consumes `_adj_setup` fixture (returns `dp, base_model`; vix injected on prs dates via `dp.load_vix_data`). Feature order in sequences: `[vix_change, underlying_return, logm, tau, tv_pred, itm_otm] + avail_cols`; with only `sp500_return` injected as enhancement, its index is 6. Last timestep of each sequence = target date (mock groups are single-row, so `sequences[i, -1, :]` is the target day's feature row).

- [ ] **Step 1: Write the failing tests**

```python
    def test_sp500_return_lagged_one_day(self, _adj_setup):
        """sp500_return dated day t is the US close realized ~15h AFTER
        Taiwan's close on day t; a day-t target may only see day t-1's value."""
        dp, base_model = _adj_setup
        dates = sorted(dp.prs_dataset['date'].unique())
        sp500 = np.arange(1, len(dates) + 1) * 0.01
        dp.load_enhancement_features = lambda: pd.DataFrame(
            {'date': dates, 'sp500_return': sp500})
        sequences, _, _, seq_dates = dp.prepare_adjustment_data(
            base_model, torch.device('cpu'), sequence_length=3)
        sp_idx = 6  # after the 6 base features
        date_pos = {pd.Timestamp(d): k for k, d in enumerate(dates)}
        assert len(seq_dates) > 0
        for i, d in enumerate(seq_dates):
            k = date_pos[pd.Timestamp(d)]
            expected = sp500[k - 1] if k >= 1 else 0.0
            assert sequences[i, -1, sp_idx].item() == pytest.approx(expected), \
                f'day {d}: saw same-day US close (leak)'

    def test_vix_change_lagged_one_day(self, _adj_setup):
        """vix_change comes from the US CBOE VIX close — same timing leak."""
        dp, base_model = _adj_setup
        vix_df = dp.load_vix_data().reset_index(drop=True)
        dates = sorted(dp.prs_dataset['date'].unique())
        sequences, _, _, seq_dates = dp.prepare_adjustment_data(
            base_model, torch.device('cpu'), sequence_length=3)
        date_pos = {pd.Timestamp(d): k for k, d in enumerate(dates)}
        assert len(seq_dates) > 0
        for i, d in enumerate(seq_dates):
            k = date_pos[pd.Timestamp(d)]
            expected = vix_df['vix_change'].iloc[k - 1] if k >= 1 else np.nan
            if np.isnan(expected):
                continue  # NaN rows are dropped by dropna, never surface
            assert sequences[i, -1, 0].item() == pytest.approx(expected), \
                f'day {d}: saw same-day US VIX close (leak)'
```

- [ ] **Step 2: Run to verify both fail** — `python -m pytest tests/test_dataset.py::TestPrepareAdjustmentData -q`. Expected: the two new tests FAIL (values equal same-day, not lagged); the four existing ones still pass.

### Task 3: Merge-time lag in prepare_adjustment_data

**Files:**
- Modify: `src/dataset.py:424` (vix merge) and `src/dataset.py:427-435` (enhancement merge)

- [ ] **Step 1: Implement.** Replace the vix merge line and enhancement block:

```python
        # vix_change and sp500_return are US closes: the value dated day t is
        # realized ~15h AFTER Taiwan's close on day t, so a same-day target
        # may only see the previous close. Lag them here (not in
        # build_features.py) because next-day-target consumers
        # (Prepare_diffusion_data) legitimately use same-date alignment.
        vix = vix.sort_values('date').copy()
        vix['vix_change'] = vix['vix_change'].shift(1)
        df = pd.merge(df, vix[['date', 'vix_change']], on='date', how='left')

        # Merge enhancement features if available
        enhance = self.load_enhancement_features()
        if enhance is not None:
            enhance_cols = ['sp500_return', 'iv_term_slope',
                            'iv_skew', 'vrp_20d', 'futures_basis_pct', 'rv_20d']
            avail_cols = [c for c in enhance_cols if c in enhance.columns]
            if avail_cols:
                enhance = enhance.sort_values('date').copy()
                if 'sp500_return' in avail_cols:
                    enhance['sp500_return'] = enhance['sp500_return'].shift(1)
                df = pd.merge(df, enhance[['date'] + avail_cols], on='date', how='left')
                for c in avail_cols:
                    df[c] = df[c].fillna(0.0)
```

- [ ] **Step 2: Run the two new tests** — expect PASS.
- [ ] **Step 3: Full suite** — `python -m pytest tests/ -q`, expect same-or-better than Task 1 baseline.

### Task 4: Docstring corrections

**Files:**
- Modify: `scripts/build_features.py:9` (docstring) and `:75` (comment)
- Modify: `src/dataset.py:403-408` (prepare_adjustment_data docstring note)

- [ ] **Step 1:** build_features.py line 9 → `- sp500_return: S&P 500 close-to-close return, dated at the US close (realized AFTER Taiwan's same-calendar-date close; same-day-target consumers must lag it — see dataset.py prepare_adjustment_data)`. At line 75 add comment: `# Dated at the US close: same-calendar-date Taiwan targets must use shift(1).`
- [ ] **Step 2:** Add to prepare_adjustment_data docstring: `US-close features (vix_change, sp500_return) are lagged one row so day-t sequences only see information available before Taiwan's day-t close.`
- [ ] **Step 3:** Re-run `python -m pytest tests/test_dataset.py -q` (guards against accidental code edits).

### Task 5: Retrain TFT+CPR (leak-free)

- [ ] **Step 1:** Confirm GPU free: `nvidia-smi`. Launch background from `model3_research/scripts/`: `python train_models.py --model tft --optimizer cpr --dtype float32`. Expected ~3.5h (recorded winner: 199 min + prep). Artifacts: `model3_research/logs/tft_cpr_fp32_results.json`, `.../tft_cpr_fp32_metrics.json`, `models/tft_cpr_fp32_AdjustmentModel.pt`, `tft_interpretability.json`.
- [ ] **Step 2:** Monitor startup log for: `input_dim=16`, train/val/test sequence counts comparable to archived runs, no errors.
- [ ] **Step 3:** On completion, read `tft_cpr_fp32_results.json` → `test_rmse`, `test_mape`.

### Task 6: Comparison & documentation

**Files:**
- Modify: `model3_research/README.md` (append a clearly-marked "Leakage-corrected results" section — do not rewrite existing tables)

- [ ] **Step 1:** Compute deltas vs recorded 0.1558 / 9.51% (and val 0.1494 / 8.92%). Caveat: single-seed comparison; both fixes bundled.
- [ ] **Step 2:** Append README section: the audit table above, the fix description, corrected metrics, delta, and the interpretability shift of sp500_return/vix_change importance (compare tft_interpretability.json feature importances if archived counterpart exists).
- [ ] **Step 3:** Verification-before-completion: rerun full pytest, `git status`/`git diff` review (no debug prints, no out-of-scope edits), quote results JSON in final report.
