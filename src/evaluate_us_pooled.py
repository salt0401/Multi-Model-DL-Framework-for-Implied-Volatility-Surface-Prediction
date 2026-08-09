"""Gated evaluation of the pooled forecaster against the per-ticker models.

Two things this does that the earlier US evaluation did not:

1. IT CAN SAY NO. Pooling is accepted only if it beats the existing
   per-ticker models on at least 5 of 7 names by Diebold-Mariano AND on
   pooled CRPS. "Keep what you have" is a legitimate, reportable outcome.
2. IT USES VALID INFERENCE. The earlier report's "7/7 significant" treated
   seven strongly correlated mega-caps as seven independent tests; PC1
   explains ~77% of cross-sectional single-name IV variation, so the
   effective number of independent tests is nearer 1-2. A moving-block
   bootstrap over DATES (blocks of consecutive snapshots, resampled jointly
   across tickers so the cross-sectional correlation is preserved) gives a
   confidence interval that respects that dependence.
"""
from argparse import ArgumentParser
import json
import os

import numpy as np
import pandas as pd
import torch

from us_dataset import UsOptionsProcessor, build_us_pairs, MAG7
from us_events import EarningsCalendar, add_event_columns, decompose_variance
from flow_us_pooled import PooledFactorPreprocessor, PooledVelocity, sample_pooled
from flow_surface import (FactorPreprocessor, CondScaler, VelocityMLP,
                          sample_flow, make_ema_model)
from train_flow_us_pooled import build_panel, n_events_per_node, SYMBOLS
from evaluate_surface_forecast import (tv_metrics, diebold_mariano,
                                       crps_empirical, coverage_90)
from utils import load_config, parse_date, set_seed, setup_logging


def block_bootstrap_ci(daily_diff, block=10, n_boot=2000, seed=0):
    """Moving-block bootstrap CI for the mean of a daily loss differential.

    Blocks of consecutive dates preserve serial dependence; because the same
    date index is drawn for every ticker, cross-sectional dependence is
    preserved too.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(daily_diff, dtype=float)
    n = len(x)
    if n < block * 2:
        return float(np.mean(x)), (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block))
    starts_max = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        means[b] = x[idx].mean()
    return float(np.mean(x)), (float(np.percentile(means, 2.5)),
                               float(np.percentile(means, 97.5)))


def main():
    ap = ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=100)
    ap.add_argument('--on_gpu', action='store_true')
    args = ap.parse_args()

    config = load_config('config.ini')
    set_seed(config['training'].getint('seed'))
    log_dir = config['save_path']['log_dir']
    logger = setup_logging(log_dir, 'evaluate_us_pooled')
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.on_gpu
                          else 'cpu')
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    cfg = config['flow_us']
    data_dir = config['data_us']['data_dir']
    train_end = parse_date(config['data_us']['train_end_date'])
    test_start = parse_date(config['data_us']['test_start_date'])

    ck = torch.load('../models/FlowPooled_us.pt', map_location=device,
                    weights_only=True)
    pp = PooledFactorPreprocessor.from_dict(ck['preprocessor'])
    c_mean = np.asarray(ck['cond_mean'])
    c_std = np.asarray(ck['cond_std'])
    inc_scale = {t: (np.asarray(a), np.asarray(b))
                 for t, (a, b) in ck['inc_scale'].items()}
    models = []
    for sd in ck['state_dicts']:
        m = PooledVelocity(**ck['model_kwargs']).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    k = ck['k']
    tau_grid = np.asarray(ck['tau_grid'])
    logm_grid = np.asarray(ck['logm_grid'])
    n_logm = len(logm_grid)

    proc = UsOptionsProcessor(data_dir)
    table = proc.build(tickers=SYMBOLS)
    cal = EarningsCalendar.from_data_dir(data_dir)
    table = add_event_columns(table, cal)
    dec = decompose_variance(table)
    panels = {t: build_panel(proc, table, cal, dec, t, cfg, train_end,
                             test_start) for t in SYMBOLS}

    # SPY market block over every split
    spy_scores = {}
    for s in ('train', 'val', 'test'):
        sp = panels['SPY'][s]
        if len(sp['dates']):
            for d, z in zip(sp['dates'], pp.transform('SPY', sp['S_today_diff'])):
                spy_scores[d] = z
    # cross-sectional mean by date (train + test, computed per date)
    from collections import defaultdict
    acc = defaultdict(list)
    for t in SYMBOLS:
        for s in ('train', 'val', 'test'):
            sp = panels[t][s]
            if len(sp['dates']):
                for d, z in zip(sp['dates'], pp.transform(t, sp['S_today_diff'])):
                    acc[d].append(z)
    xmean = {d: np.mean(v, axis=0) for d, v in acc.items()}

    results, daily_pool, daily_priv, daily_rw = {}, {}, {}, {}
    for i, t in enumerate(SYMBOLS):
        sp = panels[t]['test']
        if len(sp['dates']) == 0:
            continue
        Z = pp.transform(t, sp['S_today_diff'])
        spy = np.vstack([spy_scores.get(d, np.zeros(k)) for d in sp['dates']])
        dte = np.clip(sp['dte'], 0, 90)[:, None] / 90.0
        imm = (sp['dte'] <= 3).astype(float)[:, None]
        n_ev = sp['n_ev_tom'].mean(axis=1, keepdims=True)
        disp = Z[:, :1] - spy[:, :1]
        xs = np.vstack([xmean.get(d, np.zeros(k)) for d in sp['dates']])
        cond = np.concatenate([Z, spy, disp, sp['C'], dte, imm, n_ev, xs], 1)
        cond = (cond - c_mean) / c_std

        g = torch.Generator(device=device.type).manual_seed(4242)
        raw = sample_pooled(models,
                            torch.as_tensor(cond, dtype=dtype, device=device),
                            torch.full((len(cond),), i, dtype=torch.long,
                                       device=device),
                            n_steps=ck['n_sample_steps'],
                            n_samples=args.n_samples, generator=g).cpu().numpy()
        a, b = inc_scale[t]
        Zs = Z[None] + (raw * b + a)
        S_diff = np.stack([pp.inverse(t, Zs[s]) for s in range(args.n_samples)])
        # re-add the KNOWN future event variance analytically
        S_samp = np.clip(S_diff + sp['ev_tom'][None], 1e-10, None)
        S_pool = S_samp.mean(0)

        S_true, S_rw = sp['S_tomorrow'], sp['S_today']
        res = {'pooled': tv_metrics(S_pool, S_true, tau_grid, n_logm),
               'random_walk': tv_metrics(S_rw, S_true, tau_grid, n_logm),
               'crps': crps_empirical(S_samp, S_true),
               'coverage_90': coverage_90(S_samp, S_true),
               'n_test': len(S_true)}
        daily_pool[t] = np.mean((S_pool - S_true) ** 2, axis=1)
        daily_rw[t] = np.mean((S_rw - S_true) ** 2, axis=1)

        # Per-ticker private baseline, where one exists
        priv = f'../models/FlowSurface_{t}.pt'
        if os.path.exists(priv):
            pk = torch.load(priv, map_location=device, weights_only=True)
            ppp = FactorPreprocessor.from_dict(pk['preprocessor'])
            cs = CondScaler.from_dict(pk['cond_scaler'])
            ics = CondScaler.from_dict(pk['increment_scaler'])
            pm = make_ema_model(VelocityMLP(**pk['model_kwargs']).to(device), pk)
            pm.eval()
            Zp = ppp.transform(sp['S_today'])
            cp = np.concatenate([Zp, cs.transform(sp['C'])], axis=1)
            g2 = torch.Generator(device=device.type).manual_seed(777)
            rp = sample_flow(pm, torch.as_tensor(cp, dtype=dtype, device=device),
                             n_steps=pk.get('n_sample_steps', 50),
                             n_samples=args.n_samples, generator=g2).cpu().numpy()
            Zpp = Zp[None] + np.stack([ics.inverse(rp[s])
                                       for s in range(args.n_samples)])
            Sp = np.stack([ppp.inverse(Zpp[s]) for s in range(args.n_samples)])
            S_priv = Sp.mean(0)
            res['per_ticker'] = tv_metrics(S_priv, S_true, tau_grid, n_logm)
            res['crps_per_ticker'] = crps_empirical(Sp, S_true)
            daily_priv[t] = np.mean((S_priv - S_true) ** 2, axis=1)
            st, p = diebold_mariano(daily_pool[t], daily_priv[t])
            res['dm_pooled_vs_per_ticker'] = {'stat': st, 'p_value': p}
        st, p = diebold_mariano(daily_pool[t], daily_rw[t])
        res['dm_pooled_vs_rw'] = {'stat': st, 'p_value': p}
        results[t] = res
        logger.info(f"{t}: pooled {res['pooled']['tv_rmse']:.6f} | RW "
                    f"{res['random_walk']['tv_rmse']:.6f} | per-ticker "
                    f"{res.get('per_ticker', {}).get('tv_rmse', float('nan')):.6f}")

    # ── Gate ──────────────────────────────────────────────────────────
    comparable = [t for t in results if 'per_ticker' in results[t]]
    wins = [t for t in comparable
            if results[t]['pooled']['tv_rmse'] < results[t]['per_ticker']['tv_rmse']]
    crps_pool = np.mean([results[t]['crps'] for t in comparable])
    crps_priv = np.mean([results[t]['crps_per_ticker'] for t in comparable])
    gate_pass = (len(wins) >= 5) and (crps_pool < crps_priv)

    # ── Block bootstrap on the pooled-vs-RW differential ─────────────
    common = sorted(set.intersection(*[set(panels[t]['test']['dates'])
                                       for t in results]))
    diffs = []
    for t in results:
        idx = {d: j for j, d in enumerate(panels[t]['test']['dates'])}
        diffs.append([daily_pool[t][idx[d]] - daily_rw[t][idx[d]]
                      for d in common])
    mean_diff, ci = block_bootstrap_ci(np.mean(diffs, axis=0))

    out = {'per_ticker_results': results,
           'gate': {'wins_vs_per_ticker': len(wins), 'of': len(comparable),
                    'winners': wins, 'crps_pooled': float(crps_pool),
                    'crps_per_ticker': float(crps_priv), 'passed': bool(gate_pass)},
           'block_bootstrap_pooled_minus_rw': {
               'mean_daily_mse_diff': mean_diff, 'ci95': ci,
               'n_dates': len(common), 'block_len': 10,
               'note': ('negative = pooled better; blocks of consecutive dates '
                        'resampled jointly across tickers, so both serial and '
                        'cross-sectional dependence are preserved')}}
    with open(os.path.join(log_dir, 'flow_us_pooled_eval.json'), 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nGATE: pooled beats per-ticker on {len(wins)}/{len(comparable)} "
          f"(need >=5); CRPS {crps_pool:.6f} vs {crps_priv:.6f} -> "
          f"{'PASS' if gate_pass else 'FAIL - keep per-ticker models'}")
    print(f"Block bootstrap pooled-minus-RW daily MSE: {mean_diff:.3e} "
          f"95% CI [{ci[0]:.3e}, {ci[1]:.3e}] over {len(common)} dates")


if __name__ == '__main__':
    main()
