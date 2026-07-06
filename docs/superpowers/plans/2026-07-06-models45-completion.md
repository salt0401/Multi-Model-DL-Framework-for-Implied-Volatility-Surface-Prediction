# Models 4 & 5 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the untrained "later part" of the TXO IV-surface pipeline: fix and train Model 4 (HyperIV, kept per SOTA review, upgraded with the PIVOT price auxiliary), and replace draft Model 5 (grid DDPM) with conditional OT flow matching over train-only PCA factors of log-total-variance, evaluated against random-walk and VAR baselines with arbitrage-violation reporting.

**Architecture:** Model 4 keeps the hypernetwork/set-transformer design from HyperIV (ICML 2025) with corrected Gatheral butterfly penalty, smooth (tanh) target MLP with softplus-positive output, input standardization, seeded eval reference sampling, fp32 training, and a Black-76 normalized price-space auxiliary loss (PIVOT, arXiv 2606.17065). Model 5 becomes: grid daily surfaces (train-only grid quantiles) → log(tv) → PCA (train-fit, ≈99% EV, ≤12 comps) → z-scored scores → conditional OT flow matching with a FiLM residual-MLP velocity field → Euler sampling (50 steps) → inverse transform (exp ⇒ positivity by construction). Evaluation: tv-RMSE / IV-RMSE / IV-MAPE vs random walk and VAR(1)-on-scores, Diebold–Mariano test, CRPS, 90% coverage, calendar+butterfly violation rates.

**Tech Stack:** PyTorch 2.11 cu126 (RTX 4060 8GB), scikit-learn PCA, scipy, numpy, pytest. No new dependencies.

## Global Constraints

- User: "no need to replace with model that only better a little bit" — Model 4 KEPT (HyperIV remains SOTA; GNO alternative needs 78GB), Model 5 REPLACED (three research tracks agree grid-DDPM at n=1458 is data-starved; FM+MLP+factors substantially better justified).
- Hardware budget: RTX 4060 Laptop 8GB VRAM / 32GB RAM / i5-13420H. FP64 on Ada is 1:64 throughput ⇒ new training runs use float32; tests keep the global float64 conftest convention (models must work in both dtypes — no hardcoded dtype in modules).
- Chronological splits only; train ≤ 2020-12-31, test ≥ 2021-01-01; all normalization/PCA/grid statistics from TRAIN data only.
- Data: dataset/prs_dataset_no_fat(clean).csv (2014–2021, 254K rows, 1,960 days). The 2022–2026 extension stays quarantined.
- No git commits unless the user asks (workspace stays dirty; suggest commit at end).
- Baseline (verified this session): 210 tests pass after 3 baseline repairs (conftest.py src path, test_train_integration import, `pip install ruptures`).

**Completion check (named up front):**
1. `python -m pytest tests/ model1_research/tests model2_research/tests -q` → all pass (incl. new tests).
2. `models/HyperIVModel.pt` exists; `logs/hyperiv_results.json` contains test RMSE/MAPE/IV-RMSE + calendar/butterfly violation rates.
3. `models/FlowSurfaceModel.pt` exists; `logs/flow_surface_eval.json` contains FM vs RW vs VAR(1) tv-RMSE/IV-RMSE, DM p-value, CRPS, 90% coverage, violation rates.
4. `docs/model45_completion_report.md` written; ARCHITECTURE.md/README.md statuses updated.

## Plan changes during execution

- **PLAN CHANGE 1 (Task 3/5):** HyperIVModel gained a residual hypernetwork parameterization (`flat_params = base_params + hyper_proj(embedding)`, base = normally-initialized flat MLP) because pure-delta generated weights start at ~0, hidden activations are tanh(0)=0, and gradient flow to generated weights dies — evidence: val RMSE frozen at 0.018774 across epochs 10–30.
- **PLAN CHANGE 2 (Task 3/5, root-caused via systematic debugging):** TargetNetwork output re-parameterized as `w = sqrt(yATM² + 0.002²) × softplus(f(·))` (ratio-to-ATM, output bias init 0.54 ⇒ ratio ≈ 1) because predicting RAW total variance (median 0.005) through softplus is fatally ill-conditioned: pre-activations sit at log(tv) ≈ −5..−9 where softplus' gradient ≈ tv ≈ 1e-6 kills upstream gradients — evidence: checkpoint predicted exactly 0 everywhere, train loss = E[tv²], upstream grad norms 1e-8, model worse than predict-yATM baseline (0.0187 vs 0.0111). Fix verified: val RMSE 0.0105 by epoch 10, test RMSE 0.0066 at 15 epochs.

**Not doing** (explicitly out of scope):
- 5-model consensus/stacking (no spec exists anywhere in docs; flagged as future work).
- Model 3 retraining (sp500_return same-day leakage flagged separately as a spawned task).
- 2022–2026 dataset integration; fixing the deprecated train_diffusion.py DDPM path (superseded; kept for reference with a deprecation note); vmap batching of HyperIV functional_call.

---

### Task 1: Pipeline leakage guard — getYATM train_end_date

**Files:**
- Modify: `src/dataset.py:48-51` (`__call__`)
- Test: `tests/test_dataset.py` (append)

`getYATM(train_end_date)` already supports the guard (dataset.py:282-285); `__call__` never passes it, so the synthetic-c6 ATM curve averages over test dates.

- [ ] **Step 1: failing test** — construct DataProcessor via `mock_config`, inject a tiny prs_dataset with pre-cutoff tv=0.01 and post-cutoff tv=1.0, call `getYATM(train_end_date=cutoff)`, assert `syn_dataset['y_atm'].max() < 0.1` (train-only mean); also assert `dp()`-equivalent path passes the config cutoff by checking `DataProcessor.__call__` source passes it (behavioral test below).

```python
class TestGetYATMLeakageGuard:
    def test_syn_curve_uses_train_dates_only(self, mock_config, mock_prs_dataset):
        from dataset import DataProcessor
        import pandas as pd
        dp = DataProcessor(mock_config)
        df = mock_prs_dataset.copy()
        cutoff = pd.Timestamp('2020-01-05')
        df.loc[df['date'] > cutoff, 'total_var'] = 5.0   # absurd post-cutoff tv
        dp.prs_dataset = df
        dp.syn_dataset = pd.DataFrame({'tau': [0.1, 0.5, 1.0], 'logm': [-2.0, 2.0, 2.5]})
        dp.getYATM(train_end_date=cutoff)
        assert dp.syn_dataset['y_atm'].max() < 1.0   # excludes the 5.0 rows

    def test_call_passes_config_cutoff(self):
        import inspect
        from dataset import DataProcessor
        src = inspect.getsource(DataProcessor.__call__)
        assert 'train_end_date' in src
```

- [ ] **Step 2:** run → FAIL (second test; first may pass only if guard given — it exercises the API).
- [ ] **Step 3: implement** — in `__call__`:

```python
def __call__(self):
    self.prs_dataset = self.preprocess()
    self.syn_dataset = self.synthesize()
    train_end = None
    try:
        from utils import parse_date
        train_end = parse_date(self.config['training']['train_end_date'])
    except Exception:
        pass
    self.getYATM(train_end_date=train_end)
```

- [ ] **Step 4:** `python -m pytest tests/test_dataset.py -q` → PASS; full suite still green.

### Task 2: Fix butterfly density formula (both copies) + HyperIV loss upgrade

**Files:**
- Modify: `src/hyperiv.py:189-218` (HyperIVLoss), `src/test.py:65`
- Test: `tests/test_hyperiv.py` (extend)

**Interfaces — Produces:** `HyperIVLoss(w_mse, w_calendar, w_butterfly, w_price).forward(tv_pred, tv_true, logm, grad_tau, grad_logm, grad_logm2, valid_mask=None) -> (total, mse, cal, but, price)`. `black76_call_price(logm, w)` module-level function. All masked reductions divide by valid count, not padded size.

- [ ] **Step 1: failing tests**

```python
def test_butterfly_matches_model1_reference():
    """g(k) must equal model1_research Loss_butterfly (w' SQUARED term)."""
    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from model1_research.model import Loss_butterfly
    torch.manual_seed(0)
    w = torch.rand(20, 1) * 0.05 + 0.01
    k = torch.randn(20, 1) * 0.3
    g1 = torch.rand(20, 1) * 0.1 - 0.05
    g2 = torch.rand(20, 1) * 0.1 - 0.05
    ref = Loss_butterfly()(w, k, g1, g2)
    _, _, _, but, _ = HyperIVLoss(w_price=0.0)(w.unsqueeze(0), w.unsqueeze(0), k.unsqueeze(0),
                                               torch.zeros_like(w).unsqueeze(0), g1.unsqueeze(0), g2.unsqueeze(0))
    assert torch.allclose(but, ref, atol=1e-10)

def test_price_aux_matches_scipy():
    """Black-76 normalized call: C = N(d1) - e^k N(d2)."""
    from scipy.stats import norm
    import numpy as np
    k, w = 0.05, 0.04
    d1 = (-k + w / 2) / np.sqrt(w); d2 = d1 - np.sqrt(w)
    expected = norm.cdf(d1) - np.exp(k) * norm.cdf(d2)
    from hyperiv import black76_call_price
    got = black76_call_price(torch.tensor([[k]]), torch.tensor([[w]]))
    assert abs(got.item() - expected) < 1e-8

def test_masked_loss_ignores_padding():
    torch.manual_seed(1)
    B, N = 2, 6
    args = [torch.rand(B, N, 1) * 0.05 + 0.01 for _ in range(2)]
    k = torch.randn(B, N, 1) * 0.2
    gt, g1, g2 = [torch.randn(B, N, 1) * 0.05 for _ in range(3)]
    mask = torch.zeros(B, N, dtype=torch.bool); mask[:, 4:] = True
    loss_fn = HyperIVLoss()
    full = loss_fn(args[0][:, :4], args[1][:, :4], k[:, :4], gt[:, :4], g1[:, :4], g2[:, :4])
    padded = loss_fn(args[0], args[1], k, gt, g1, g2, valid_mask=~mask)
    assert torch.allclose(full[0], padded[0], atol=1e-10)
```

- [ ] **Step 2:** run → FAIL (`black76_call_price` undefined; 5-tuple mismatch; mask kw missing).
- [ ] **Step 3: implement** in src/hyperiv.py:

```python
def black76_call_price(logm, w):
    """Normalized Black-76 call (forward=1, discount=1) from log-moneyness and total variance."""
    w = w.clamp(min=1e-10)
    sqrt_w = torch.sqrt(w)
    d1 = (-logm + w / 2) / sqrt_w
    d2 = d1 - sqrt_w
    normal = torch.distributions.Normal(0.0, 1.0)
    return normal.cdf(d1) - torch.exp(logm) * normal.cdf(d2)

# In HyperIVLoss: add w_price=0.1 param; masked means via
#   def _mmean(x, m): return (x * m).sum() / m.sum().clamp(min=1) if mask else x.mean()
# butterfly: g_k = (1 - (logm*grad_logm)/(2*w))**2 - grad_logm**2/4*(1/w + 0.25) + grad_logm2/2
# price:     price_loss = _mmean((black76_call_price(logm, tv_pred) - black76_call_price(logm, tv_true))**2, m)
# return total, mse, cal, but, price
```

Also fix `src/test.py:65`: `g = (1 - k[i]*dw/(2*w[i]))**2 - dw**2/4*(1/w[i] + 0.25) + d2w/2`.

- [ ] **Step 4:** update the two existing HyperIVLoss tests (4-tuple → 5-tuple) in tests/test_hyperiv.py; run `python -m pytest tests/test_hyperiv.py -q` → PASS.

### Task 3: TargetNetwork smooth activation + positive output + input standardization

**Files:**
- Modify: `src/hyperiv.py` (TargetNetwork, HyperIVModel)
- Test: `tests/test_hyperiv.py` (extend)

**Interfaces — Produces:** `TargetNetwork(hidden_dims)` = Linear/Tanh stack + final Linear + Softplus (structure change: `net` ends with `nn.Softplus()`; `_generate_target_params` unchanged since Softplus has no params but sequential indices shift — final Linear stays at even index because Tanh replaces ReLU 1:1 and Softplus appended AFTER last Linear at odd index). `HyperIVModel.set_normalization(mean, std)` registers 3-dim buffers (`feat_mean`, `feat_std`) applied to ref_set AND target (tau, logm, yATM) inside forward; buffers default 0/1 so behavior without stats is unchanged. `hyper_proj.bias[-1]` initialized to −4.6 so initial surfaces ≈ softplus(−4.6) ≈ 0.01 (data scale).

- [ ] **Step 1: failing tests**

```python
def test_target_network_positive_output_and_smooth():
    net = TargetNetwork(hidden_dims=(8, 4))
    x = torch.randn(50, 3, requires_grad=True)
    out = net(x)
    assert (out > 0).all()                                   # softplus head
    g = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    g2 = torch.autograd.grad(g[:, 1].sum(), x)[0]
    assert g2.abs().sum() > 0                                 # tanh ⇒ nonzero 2nd deriv

def test_model_normalization_buffers():
    m = HyperIVModel(embed_dim=16, n_heads=4, n_transformer_layers=1, target_hidden_dims=(8, 4))
    m.set_normalization(torch.tensor([0.3, 0.0, 0.02]), torch.tensor([0.2, 0.15, 0.01]))
    ref = torch.rand(2, 10, 3) * 0.05
    out = m(ref, torch.rand(2, 5, 1) * 0.5, torch.randn(2, 5, 1) * 0.1, torch.rand(2, 5, 1) * 0.05)
    assert all(torch.isfinite(o).all() for o in out)

def test_initial_predictions_at_data_scale():
    m = HyperIVModel(embed_dim=16, n_heads=4, n_transformer_layers=1, target_hidden_dims=(8, 4))
    tv, *_ = m(torch.rand(2, 10, 3) * 0.05, torch.rand(2, 5, 1), torch.randn(2, 5, 1) * 0.1, torch.rand(2, 5, 1) * 0.05)
    assert tv.median() < 0.2   # not 0.69-scale
```

- [ ] **Step 2:** run → FAIL.
- [ ] **Step 3: implement.** TargetNetwork: `layers.append(nn.Tanh())` instead of ReLU; after final Linear append `nn.Softplus()`. HyperIVModel `__init__`: `self.register_buffer('feat_mean', torch.zeros(3)); self.register_buffer('feat_std', torch.ones(3))`; `set_normalization(mean, std)` copies in. forward: `ref_norm = (ref_set - self.feat_mean) / self.feat_std`; targets: normalize each of tau/logm/yATM by respective mean/std AFTER `requires_grad_(True)` on originals (chain rule keeps grads w.r.t. originals correct). Init: `with torch.no_grad(): self.hyper_proj.bias[-1] = -4.6`.
- [ ] **Step 4:** `python -m pytest tests/test_hyperiv.py -q` → PASS. Note: `test_autograd_derivatives_nonzero` should now be strictly meaningful.

### Task 4: train_hyperiv.py — eval protocol, fp32, stats, metrics, violations

**Files:**
- Modify: `src/train_hyperiv.py`, `src/config.ini` ([hyperiv]: add `w_mse=1.0 w_calendar=10.0 w_butterfly=10.0 w_price=0.1 dtype=float32`)

Changes (no unit tests — integration verified by smoke run + full suite):
- [ ] dtype from config (`float32` default; `torch.set_default_dtype` accordingly; tensors from DataProcessor are float64 → cast in collate/eval).
- [ ] Compute feat stats from TRAIN surfaces only: concat (tau, logm, tv) over train surfaces → mean/std → `model.set_normalization(...)`.
- [ ] evaluate(): replace `list(range(n_reference))` with per-surface seeded sample `torch.randperm(n, generator=torch.Generator().manual_seed(9000 + idx))`; batch surfaces through `collate_surfaces` for speed; pass `valid_mask` to loss.
- [ ] train_one_epoch: pass `valid_mask=~t_mask` instead of premultiplied zeros.
- [ ] Checkpoint: `torch.save({'state_dict':..., 'feat_mean':..., 'feat_std':..., 'config': dict(hiv_cfg)}, path)`; loader handles legacy raw state_dict.
- [ ] After test eval: violation rates on test predictions — for each test surface, dense grid (5 taus × 41 logm in observed range) queried through the model; count `grad_tau < 0` fraction (calendar) and `g(k) < 0` fraction (butterfly, corrected formula). Write `logs/hyperiv_results.json` with all test metrics + violation rates + config; save `logs/hyperiv_test_fit.png` (pred vs true scatter + one day's smile).
- [ ] **Smoke run:** `cd src && python train_hyperiv.py --epochs 2` (CPU ok) → completes, writes json. Then full suite green.

### Task 5: Train Model 4

- [ ] `cd src && python train_hyperiv.py --on_gpu --epochs 500` (early stopping patience 50). Expected ~10–30 min fp32.
- [ ] Verify: `models/HyperIVModel.pt` exists; `logs/hyperiv_results.json` has test_rmse/test_mape/test_iv_rmse, calendar_violation_rate, butterfly_violation_rate. Record numbers in report. Sanity: test tv-RMSE same order as Model 1's 0.01977 (HyperIV solves a harder task — masked interpolation from 50 refs — so somewhat higher is acceptable; >10× worse means investigate before proceeding).

### Task 6: Surface panel data prep for Model 5

**Files:**
- Modify: `src/dataset.py` (new method `Prepare_surface_panel`)
- Test: `tests/test_flow_surface.py` (new file, first tests)

**Interfaces — Produces:** `dp.Prepare_surface_panel(train_end_date, n_tau_grid=10, n_logm_grid=20) -> dict` with keys: `dates` (list of N pd.Timestamp), `surfaces` (np.ndarray (N, n_tau*n_logm) float64, row-major = tau-major: index = i_tau * n_logm + i_logm), `conditions` (np.ndarray (N, 11)), `tau_grid` (n_tau,), `logm_grid` (n_logm,), `cond_names` (list). Grid quantiles from rows with date ≤ train_end_date ONLY. VIX joined with ffill (no 0.2 default). No pairing here — pairing/splitting happens in flow_surface.build_dataset.

- [ ] **Step 1: failing test** (synthetic df injected as in conftest patterns):

```python
def test_surface_panel_grid_train_only(mock_config, mock_prs_dataset):
    from dataset import DataProcessor
    import pandas as pd, numpy as np
    dp = DataProcessor(mock_config)
    df = mock_prs_dataset.copy()
    cutoff = pd.Timestamp('2020-01-05')
    # widen tau range only after cutoff — grid must ignore it
    df.loc[df['date'] > cutoff, 'tau'] = 5.0
    dp.prs_dataset = df
    panel = dp.Prepare_surface_panel(train_end_date=cutoff, n_tau_grid=3, n_logm_grid=4)
    assert panel['tau_grid'].max() < 3.0
    assert panel['surfaces'].shape[1] == 12
    assert len(panel['dates']) == len(panel['surfaces']) == len(panel['conditions'])
```

- [ ] **Step 2:** FAIL (method missing).
- [ ] **Step 3: implement** — same gridding core as Prepare_diffusion_data (griddata linear + nearest fill) but: quantiles on `df[df.date <= train_end_date]`; `np.meshgrid(tau_range, logm_range, indexing='ij')` for tau-major layout; conditions built for every surfaced day (not pairs): VIX via merge+ffill on the option-date index; enhancement features as before (NaN→0.0 retained, but the 7-col order fixed and returned as `cond_names`).
- [ ] **Step 4:** test passes; full suite green.

### Task 7: flow_surface.py — preprocessor, velocity net, FM, sampler, EMA

**Files:**
- Create: `src/flow_surface.py`
- Test: `tests/test_flow_surface.py` (extend)

**Interfaces — Produces:**
- `FactorPreprocessor(n_components, ev_target=0.99, max_components=12)`: `.fit(S_train)` (S = (N,D) positive surfaces; internally log → PCA → per-score z-norm; also fits per-dim log-surface mean/std residual scaling), `.transform(S) -> Z (N,k)`, `.inverse(Z) -> S (N,D) strictly positive`, `.to_dict()/.from_dict()` (JSON/torch.save-able), `.n_components_` attr.
- `CondScaler`: z-score fit/transform/to_dict/from_dict for conditions.
- `VelocityMLP(dim, cond_dim, hidden=256, n_blocks=3, dropout=0.1)`: `forward(z_t (B,k), t (B,), cond (B,c)) -> v (B,k)`; sinusoidal t-embedding (64) + FiLM per block.
- `fm_loss(model, z1, cond)`: samples `t~U(0,1)`, `z0~N(0,I)`, `z_t=(1-t)z0+t z1`, returns `MSE(model(z_t,t,cond), z1-z0)`.
- `sample_flow(model, cond, n_steps=50, n_samples=1, generator=None) -> (n_samples, B, k)`: Euler from z0~N(0,I), t: 0→1.
- `EMA(model, decay=0.999)`: `.update(model)`, `.state_dict()`, `.copy_to(model)`.
- `build_dataset(panel, train_end_date, test_start_date)`: pairs consecutive days, target = tomorrow scores, cond = [today scores, scaled market cond]; **excludes from train any pair whose TOMORROW > train_end_date** (fixes boundary leak); chronological 85/15 train/val fallback; returns dict of train/val/test (Z_today, Z_tomorrow, C, S_today_raw, S_tomorrow_raw, dates).

- [ ] **Step 1: failing tests**

```python
def test_preprocessor_roundtrip_and_positivity():
    rng = np.random.default_rng(0)
    S = np.exp(rng.normal(-4.5, 0.4, size=(200, 24)))          # positive, log-normal like tv
    pp = FactorPreprocessor(n_components=None, ev_target=0.99, max_components=12)
    pp.fit(S)
    Z = pp.transform(S); S2 = pp.inverse(Z)
    assert (S2 > 0).all()
    assert np.sqrt(np.mean((np.log(S2) - np.log(S))**2)) < 0.1  # 99% EV reconstruction
    pp2 = FactorPreprocessor.from_dict(pp.to_dict())
    assert np.allclose(pp2.inverse(Z), S2)

def test_fm_learns_conditional_mean():
    """FM on toy: z1 = 2*cond + noise; sampled mean should approach 2*cond."""
    torch.manual_seed(0)
    cond = torch.rand(4096, 1) * 2 - 1
    z1 = 2 * cond + 0.05 * torch.randn(4096, 1)
    model = VelocityMLP(dim=1, cond_dim=1, hidden=64, n_blocks=2, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(400):
        idx = torch.randint(0, 4096, (256,))
        loss = fm_loss(model, z1[idx], cond[idx]); opt.zero_grad(); loss.backward(); opt.step()
    test_c = torch.tensor([[0.5], [-0.5]])
    samp = sample_flow(model, test_c, n_steps=50, n_samples=200,
                       generator=torch.Generator().manual_seed(1))
    means = samp.mean(dim=0)
    assert torch.allclose(means, 2 * test_c, atol=0.15)

def test_boundary_pair_excluded_from_train():
    # panel with dates D-2, D-1(=train_end), D (test); pair (D-1 -> D) must not be in train
    ...build tiny panel dict directly, call build_dataset, assert dates check...

def test_ema_update_math():
    ...decay 0.5, two known params, one update, assert exact averages...
```

- [ ] **Step 2:** FAIL (module missing).
- [ ] **Step 3: implement** src/flow_surface.py (~250 lines) exactly per interfaces above.
- [ ] **Step 4:** `python -m pytest tests/test_flow_surface.py -q` → PASS (FM toy test is the slowest, ~30s CPU).

### Task 8: train_flow_surface.py + config

**Files:**
- Create: `src/train_flow_surface.py`
- Modify: `src/config.ini` — add:

```ini
[flow_surface]
n_tau_grid = 10
n_logm_grid = 20
max_components = 12
ev_target = 0.99
hidden = 256
n_blocks = 3
dropout = 0.1
weight_decay = 0.001
ema_decay = 0.999
n_sample_steps = 50
epochs = 3000
learning_rate = 0.001
batch_size = 128
val_samples = 32
seed_ensemble = 1
```
and `[save_path] flow_model_path = ../models/FlowSurfaceModel.pt`.

Script flow: DataProcessor → Prepare_surface_panel(train_end) → build_dataset → fit FactorPreprocessor + CondScaler on train → train VelocityMLP (AdamW, weight_decay from config, cosine LR, EMA) → per-epoch cheap val: FM loss on val + every 25 epochs sampled val tv-RMSE (32 samples, EMA weights) → early stop patience 40 (on sampled val RMSE) → save checkpoint dict `{'state_dict', 'ema_state_dict', 'preprocessor': pp.to_dict(), 'cond_scaler': cs.to_dict(), 'tau_grid', 'logm_grid', 'cond_names', 'config'}`.

- [ ] Implement; smoke `python train_flow_surface.py --epochs 30` → checkpoint written, val RMSE logged, finite.
- [ ] Full suite still green.

### Task 9: evaluate_surface_forecast.py

**Files:**
- Create: `src/evaluate_surface_forecast.py`

For each test day (243 pairs): RW forecast = today's surface; VAR(1) on train scores (statsmodels VAR, lag 1, fallback ridge lstsq) → inverse to surface; FM: 100 samples (EMA weights, seeded) → point = mean surface, intervals = per-dim 5/95 percentiles. Metrics (all in de-normalized tv space, plus IV space via `iv = sqrt(tv / tau_grid)` broadcast over the tau-major layout):
- tv-RMSE, IV-RMSE, IV-MAPE (epsilon-free, on IV: `mean(|iv_hat - iv|/iv)`) per method.
- Diebold–Mariano FM-vs-RW on daily squared-error differentials, Newey–West (lag 5) variance, two-sided p.
- CRPS per grid cell from the 100-sample empirical CDF (average over cells/days; `crps = mean(|x_i - y|) - 0.5*mean(|x_i - x_j|)`).
- 90% central-interval coverage rate.
- Violation rates for FM samples, RW, and ACTUAL tomorrow surfaces (baseline context): calendar = fraction of adjacent-tau pairs with w decreasing (per logm column); butterfly = corrected finite-difference g(k) < 0 fraction per tau row (reuse fixed src/test.py logic adapted to grids).
Write `logs/flow_surface_eval.json` + figures: `logs/flow_eval_surfaces.png` (worst/median/best day heatmaps: today/actual/FM-mean), `logs/flow_eval_metrics.png` (bar chart), `logs/flow_eval_coverage.png`.

- [ ] Implement; verification = Task 10 run + JSON inspection.

### Task 10: Train + evaluate Model 5

- [ ] `cd src && python train_flow_surface.py --on_gpu` (minutes). Verify checkpoint + training log.
- [ ] `python evaluate_surface_forecast.py` → JSON + figures. Record honestly: literature expectation is FM ≈ RW on point RMSE (surfaces ~0.99 autocorrelated); the FM value-add is calibrated distributions + coherent scenarios. If FM point RMSE > 1.15× RW, say so plainly in the report — do not bury it.

### Task 11: Docs + report + final verification

- [ ] Write `docs/model45_completion_report.md`: decisions (keep HyperIV + why; replace DDPM + why, with citations), all bugs fixed (incl. test.py butterfly-checker bug ⇒ Model 1's 45.69% violation stat needs recomputation — flag, don't recompute here), final metrics tables, hardware notes, deferred list.
- [ ] Update ARCHITECTURE.md + README.md: Model 4/5 sections and status tables; deprecation note on src/diffusion.py + src/train_diffusion.py.
- [ ] Run FULL suite: `python -m pytest tests/ model1_research/tests model2_research/tests -q` → all pass. Run `git status`/`git diff --stat`, ensure no debris. Verification-before-completion checklist against original request.
