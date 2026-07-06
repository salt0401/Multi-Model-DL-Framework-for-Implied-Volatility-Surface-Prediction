"""Evaluation protocol for Model 5 (flow-matching surface forecaster).

Field-standard protocol for IV-surface forecasts:
- Point accuracy: tv-RMSE / IV-RMSE / IV-MAPE vs (1) random walk
  (tomorrow = today) and (2) VAR(1) on PCA factor scores.
- Diebold-Mariano test (squared-error differentials, Newey-West lag 5)
  of the flow model against the random walk.
- Probabilistic quality: CRPS (empirical, 100 samples), 90% central
  interval coverage.
- No-arbitrage: calendar and butterfly violation rates of generated
  surfaces, with the ACTUAL tomorrow surfaces as the empirical reference
  (market surfaces themselves contain violations at grid resolution).

Literature expectation (Goncalves-Guidolin 2006 and successors): daily IV
surfaces are ~0.99 autocorrelated, so beating the random walk at 1-day
horizon on point error is marginal at best. The flow model's value-add is
calibrated predictive distributions and coherent scenario generation.
"""
from dataset import DataProcessor
from flow_surface import (FactorPreprocessor, CondScaler, VelocityMLP,
                          sample_flow, build_dataset, make_ema_model)
from utils import load_config, parse_date, set_seed, setup_logging

from argparse import ArgumentParser

import json
import numpy as np
import torch
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats


# ── Metrics ───────────────────────────────────────────────────────────

def tv_metrics(S_pred, S_true, tau_grid, n_logm):
    """tv-RMSE, IV-RMSE, IV-MAPE for (N, D) surfaces on a tau-major grid."""
    rmse = float(np.sqrt(np.mean((S_pred - S_true) ** 2)))
    tau_flat = np.repeat(tau_grid, n_logm)[None, :]  # (1, D) tau-major
    iv_pred = np.sqrt(np.clip(S_pred, 1e-12, None) / tau_flat)
    iv_true = np.sqrt(np.clip(S_true, 1e-12, None) / tau_flat)
    iv_rmse = float(np.sqrt(np.mean((iv_pred - iv_true) ** 2)))
    iv_mape = float(np.mean(np.abs(iv_pred - iv_true) / np.clip(iv_true, 1e-12, None)))
    return {'tv_rmse': rmse, 'iv_rmse': iv_rmse, 'iv_mape': iv_mape}


def diebold_mariano(err_a, err_b, h_lag=5):
    """DM test on daily mean squared-error differentials d_t = a_t - b_t.

    Newey-West variance with h_lag lags; returns (statistic, two-sided p).
    Negative statistic => method A more accurate.
    """
    d = err_a - err_b
    T = len(d)
    d_bar = d.mean()
    d_c = d - d_bar
    gamma0 = np.mean(d_c ** 2)
    var = gamma0
    for lag in range(1, min(h_lag, T - 1) + 1):
        gamma = np.mean(d_c[lag:] * d_c[:-lag])
        var += 2 * (1 - lag / (h_lag + 1)) * gamma
    var = max(var, 1e-300)
    dm = d_bar / np.sqrt(var / T)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def crps_empirical(samples, y):
    """CRPS from an empirical ensemble.

    samples: (S, N, D), y: (N, D). Returns mean CRPS over all cells.
    crps = E|X - y| - 0.5 E|X - X'|
    """
    term1 = np.mean(np.abs(samples - y[None]), axis=0)          # (N, D)
    S = samples.shape[0]
    # E|X - X'| via sorted-sample identity per cell (O(S log S))
    srt = np.sort(samples, axis=0)
    i = np.arange(1, S + 1)[:, None, None]
    term2 = 2.0 / (S * S) * np.sum((2 * i - S - 1) * srt, axis=0)
    return float(np.mean(term1 - 0.5 * term2))


def coverage_90(samples, y):
    """Fraction of cells where y falls inside the empirical [5%, 95%] band."""
    lo = np.percentile(samples, 5, axis=0)
    hi = np.percentile(samples, 95, axis=0)
    return float(np.mean((y >= lo) & (y <= hi)))


def violation_rates(S, tau_grid, logm_grid):
    """Calendar and butterfly violation rates for (N, D) tau-major surfaces.

    Calendar: fraction of adjacent-tau pairs (per logm column) with w
    decreasing. Butterfly: Gatheral g(k) < 0 rate via finite differences
    along logm (per tau row), using the CORRECTED density formula.
    """
    n_tau, n_logm = len(tau_grid), len(logm_grid)
    W = S.reshape(-1, n_tau, n_logm)

    cal_viol = (np.diff(W, axis=1) < 0).mean()

    k = logm_grid[None, None, :]
    dk = logm_grid[1] - logm_grid[0]
    dw = np.gradient(W, dk, axis=2)
    d2w = np.gradient(dw, dk, axis=2)
    w_safe = np.clip(W, 1e-12, None)
    g = (1 - k * dw / (2 * w_safe)) ** 2 - dw ** 2 / 4 * (1 / w_safe + 0.25) + d2w / 2
    but_viol = (g < 0).mean()

    return float(cal_viol), float(but_viol)


def var1_forecast(Z_train_today, Z_train_tomorrow, Z_test_today):
    """VAR(1) via ridge-regularized least squares: z_{t+1} = A [z_t, 1]."""
    X = np.column_stack([Z_train_today, np.ones(len(Z_train_today))])
    lam = 1e-6
    A = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]),
                        X.T @ Z_train_tomorrow)
    Xt = np.column_stack([Z_test_today, np.ones(len(Z_test_today))])
    return Xt @ A


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    config = load_config('config.ini')
    set_seed(config['training'].getint('seed'))
    log_dir = config['save_path']['log_dir']
    logger = setup_logging(log_dir, 'flow_eval')

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    model_path = config['save_path'].get('flow_model_path',
                                         '../models/FlowSurfaceModel.pt')
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    pp = FactorPreprocessor.from_dict(ckpt['preprocessor'])
    cs = CondScaler.from_dict(ckpt['cond_scaler'])
    tau_grid = np.asarray(ckpt['tau_grid'])
    logm_grid = np.asarray(ckpt['logm_grid'])
    n_logm = len(logm_grid)
    n_steps = ckpt.get('n_sample_steps', 50)

    model = VelocityMLP(**ckpt['model_kwargs']).to(device)
    model = make_ema_model(model, ckpt)
    model.eval()

    # Rebuild the identical data pipeline
    train_end_date = parse_date(config['training']['train_end_date'])
    test_start_date = parse_date(config['training']['test_start_date'])
    fcfg = config['flow_surface']
    dp = DataProcessor(config)
    dp()
    panel = dp.Prepare_surface_panel(train_end_date,
                                     n_tau_grid=fcfg.getint('n_tau_grid'),
                                     n_logm_grid=fcfg.getint('n_logm_grid'))
    splits = build_dataset(panel, train_end_date, test_start_date)
    test = splits['test']
    S_today, S_tomorrow = test['S_today'], test['S_tomorrow']
    N = len(S_today)
    logger.info(f'Test pairs: {N}')

    Z_today = pp.transform(S_today)
    cond = np.concatenate([Z_today, cs.transform(test['C'])], axis=1)

    # ── Forecasts ────────────────────────────────────────────────────
    # 1. Random walk
    S_rw = S_today.copy()

    # 2. VAR(1) on factor scores (train-fit)
    Z_var = var1_forecast(pp.transform(splits['train']['S_today']),
                          pp.transform(splits['train']['S_tomorrow']),
                          Z_today)
    S_var = pp.inverse(Z_var)

    # 3. Flow matching: 100 samples per day. The flow models the z-scored
    # DAILY INCREMENT of factor scores: z1 = z_today + inverse(sample).
    cond_t = torch.as_tensor(cond, dtype=dtype, device=device)
    g = torch.Generator(device=device.type).manual_seed(777)
    raw_samp = sample_flow(model, cond_t, n_steps=n_steps,
                           n_samples=args.n_samples, generator=g)
    raw_samp = raw_samp.cpu().numpy()                            # (S, N, k)
    if ckpt.get('predict_increments'):
        ics = CondScaler.from_dict(ckpt['increment_scaler'])
        Z_samp = Z_today[None] + np.stack(
            [ics.inverse(raw_samp[s]) for s in range(args.n_samples)])
    else:
        Z_samp = raw_samp
    S_samp = np.stack([pp.inverse(Z_samp[s]) for s in range(args.n_samples)])
    S_fm = S_samp.mean(axis=0)

    # ── Point metrics ────────────────────────────────────────────────
    results = {'n_test_pairs': N, 'n_samples': args.n_samples,
               'n_components': pp.n_components_,
               'pca_floor_tv_rmse': float(np.sqrt(np.mean(
                   (pp.inverse(pp.transform(S_tomorrow)) - S_tomorrow) ** 2)))}
    for name, S_pred in [('random_walk', S_rw), ('var1', S_var), ('flow', S_fm)]:
        results[name] = tv_metrics(S_pred, S_tomorrow, tau_grid, n_logm)
        logger.info(f'{name}: {results[name]}')

    # ── DM tests (daily MSE differentials) ───────────────────────────
    daily_mse = {name: np.mean((S_pred - S_tomorrow) ** 2, axis=1)
                 for name, S_pred in [('random_walk', S_rw), ('var1', S_var),
                                      ('flow', S_fm)]}
    dm_fm_rw, p_fm_rw = diebold_mariano(daily_mse['flow'], daily_mse['random_walk'])
    dm_var_rw, p_var_rw = diebold_mariano(daily_mse['var1'], daily_mse['random_walk'])
    results['dm_flow_vs_rw'] = {'stat': dm_fm_rw, 'p_value': p_fm_rw,
                                'note': 'negative stat = flow more accurate'}
    results['dm_var1_vs_rw'] = {'stat': dm_var_rw, 'p_value': p_var_rw}
    logger.info(f'DM flow vs RW: stat={dm_fm_rw:.3f}, p={p_fm_rw:.4f}')

    # ── Probabilistic metrics ────────────────────────────────────────
    results['flow_crps'] = crps_empirical(S_samp, S_tomorrow)
    results['flow_coverage_90'] = coverage_90(S_samp, S_tomorrow)
    logger.info(f"CRPS: {results['flow_crps']:.6f}, "
                f"90% coverage: {results['flow_coverage_90']:.3f}")

    # ── Arbitrage violation rates ────────────────────────────────────
    for name, S_chk in [('actual', S_tomorrow), ('random_walk', S_rw),
                        ('flow_mean', S_fm),
                        ('flow_samples', S_samp.reshape(-1, S_samp.shape[-1]))]:
        cal, but = violation_rates(S_chk, tau_grid, logm_grid)
        results[f'violations_{name}'] = {'calendar': cal, 'butterfly': but}
        logger.info(f'violations {name}: calendar={cal:.4%}, butterfly={but:.4%}')

    with open(os.path.join(log_dir, 'flow_surface_eval.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── Figures ──────────────────────────────────────────────────────
    daily_fm = daily_mse['flow']
    order = np.argsort(daily_fm)
    picks = [order[len(order) // 2], order[0], order[-1]]
    labels = ['median day', 'best day', 'worst day']
    n_tau = len(tau_grid)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for row, (idx, lab) in enumerate(zip(picks, labels)):
        for col, (title, S_x) in enumerate([
                ('today (input)', S_today), ('actual tomorrow', S_tomorrow),
                ('flow mean forecast', S_fm)]):
            ax = axes[row, col]
            im = ax.imshow(S_x[idx].reshape(n_tau, n_logm), aspect='auto',
                           origin='lower', cmap='viridis',
                           extent=[logm_grid[0], logm_grid[-1],
                                   tau_grid[0], tau_grid[-1]])
            ax.set_title(f'{lab}: {title}', fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
            if col == 0:
                ax.set_ylabel('tau')
            if row == 2:
                ax.set_xlabel('logm')
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, 'flow_eval_surfaces.png'), dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    methods = ['random_walk', 'var1', 'flow']
    axes[0].bar(methods, [results[m]['tv_rmse'] for m in methods])
    axes[0].axhline(results['pca_floor_tv_rmse'], color='r', ls='--',
                    label='PCA floor')
    axes[0].set_title('Test tv-RMSE (lower = better)')
    axes[0].legend()
    axes[1].bar(methods, [results[m]['iv_mape'] for m in methods])
    axes[1].set_title('Test IV-MAPE')
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, 'flow_eval_metrics.png'), dpi=110)
    plt.close(fig)

    logger.info('Evaluation complete -> logs/flow_surface_eval.json')


if __name__ == '__main__':
    main()
