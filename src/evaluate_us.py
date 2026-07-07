"""Evaluation for the Mag 7 flow forecasters + TXO comparison.

Per ticker: flow (100 samples) vs next-snapshot random walk vs VAR(1) on
factors — tv-RMSE / IV-RMSE / IV-MAPE, Diebold-Mariano, CRPS, 90% coverage,
violation rates. Reuses the metric functions from evaluate_surface_forecast.

Comparison caveats stated wherever numbers meet: the US branch models the
SHORT-DATED surface (3 expirations, tau <= ~46d) at Mon/Wed/Fri snapshots
(1-2 trading-day horizon), vs TXO's daily full-curve (tau to 2y) setup.
"""
from us_dataset import UsOptionsProcessor, build_us_pairs, MAG7
from flow_surface import (FactorPreprocessor, CondScaler, VelocityMLP,
                          sample_flow, make_ema_model)
from evaluate_surface_forecast import (tv_metrics, diebold_mariano,
                                       crps_empirical, coverage_90,
                                       violation_rates, var1_forecast)
from utils import load_config, parse_date, set_seed, setup_logging

from argparse import ArgumentParser

import json
import numpy as np
import torch
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def evaluate_ticker(ticker, table, proc, config, device, dtype, n_samples,
                    logger):
    cfg = config['flow_us']
    train_end = parse_date(config['data_us']['train_end_date'])
    test_start = parse_date(config['data_us']['test_start_date'])

    ckpt = torch.load(f'../models/FlowSurface_{ticker}.pt',
                      map_location=device, weights_only=True)
    pp = FactorPreprocessor.from_dict(ckpt['preprocessor'])
    cs = CondScaler.from_dict(ckpt['cond_scaler'])
    ics = CondScaler.from_dict(ckpt['increment_scaler'])
    tau_grid = np.asarray(ckpt['tau_grid'])
    logm_grid = np.asarray(ckpt['logm_grid'])

    panel = proc.prepare_surface_panel(table, ticker, train_end,
                                       n_logm=cfg.getint('n_logm'))
    splits = build_us_pairs(panel, train_end, test_start)
    test = splits['test']
    S_today, S_tomorrow = test['S_today'], test['S_tomorrow']
    if len(S_today) == 0:
        return None

    Z_today = pp.transform(S_today)
    cond = np.concatenate([Z_today, cs.transform(test['C'])], axis=1)

    model = VelocityMLP(**ckpt['model_kwargs']).to(device)
    model = make_ema_model(model, ckpt)
    model.eval()

    g = torch.Generator(device=device.type).manual_seed(777)
    raw = sample_flow(model, torch.as_tensor(cond, dtype=dtype, device=device),
                      n_steps=ckpt.get('n_sample_steps', 50),
                      n_samples=n_samples, generator=g).cpu().numpy()
    Z_samp = Z_today[None] + np.stack([ics.inverse(raw[s])
                                       for s in range(n_samples)])
    S_samp = np.stack([pp.inverse(Z_samp[s]) for s in range(n_samples)])
    S_fm = S_samp.mean(axis=0)

    S_var = pp.inverse(var1_forecast(pp.transform(splits['train']['S_today']),
                                     pp.transform(splits['train']['S_tomorrow']),
                                     Z_today))

    n_logm = len(logm_grid)
    out = {'n_test_pairs': len(S_today), 'n_components': pp.n_components_}
    daily = {}
    for name, S_pred in [('random_walk', S_today), ('var1', S_var),
                         ('flow', S_fm)]:
        out[name] = tv_metrics(S_pred, S_tomorrow, tau_grid, n_logm)
        daily[name] = np.mean((S_pred - S_tomorrow) ** 2, axis=1)
    stat, p = diebold_mariano(daily['flow'], daily['random_walk'])
    out['dm_flow_vs_rw'] = {'stat': stat, 'p_value': p}
    out['flow_crps'] = crps_empirical(S_samp, S_tomorrow)
    out['flow_coverage_90'] = coverage_90(S_samp, S_tomorrow)
    for name, S_chk in [('actual', S_tomorrow), ('flow_mean', S_fm)]:
        cal, but = violation_rates(S_chk, tau_grid, logm_grid)
        out[f'violations_{name}'] = {'calendar': cal, 'butterfly': but}
    logger.info(f"{ticker}: flow {out['flow']['tv_rmse']:.6f} vs "
                f"RW {out['random_walk']['tv_rmse']:.6f} "
                f"(DM {stat:.2f}, p={p:.4f}), cov={out['flow_coverage_90']:.2f}")
    return out


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    config = load_config('config.ini')
    set_seed(config['training'].getint('seed'))
    log_dir = config['save_path']['log_dir']
    logger = setup_logging(log_dir, 'evaluate_us')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.on_gpu
                          else "cpu")
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    proc = UsOptionsProcessor(config['data_us']['data_dir'])
    table = proc.build()

    results = {}
    for t in MAG7:
        try:
            r = evaluate_ticker(t, table, proc, config, device, dtype,
                                args.n_samples, logger)
            if r:
                results[t] = r
        except FileNotFoundError:
            logger.info(f'{t}: no checkpoint, skipped')

    # Cross-ticker aggregate + wins-vs-RW count
    wins = sum(1 for r in results.values()
               if r['flow']['tv_rmse'] < r['random_walk']['tv_rmse'])
    sig_wins = sum(1 for r in results.values()
                   if r['flow']['tv_rmse'] < r['random_walk']['tv_rmse']
                   and r['dm_flow_vs_rw']['p_value'] < 0.05)
    summary = {'tickers': results, 'flow_beats_rw': wins,
               'flow_beats_rw_significant': sig_wins, 'n_tickers': len(results)}

    # TXO reference block if available
    txo_path = os.path.join(log_dir, 'flow_surface_eval.json')
    if os.path.exists(txo_path):
        txo = json.load(open(txo_path))
        summary['txo_reference'] = {
            'flow_tv_rmse': txo['flow']['tv_rmse'],
            'rw_tv_rmse': txo['random_walk']['tv_rmse'],
            'dm_p': txo['dm_flow_vs_rw']['p_value'],
            'flow_iv_mape': txo['flow']['iv_mape'],
            'note': ('TXO: daily full-curve (tau to 2y); US: Mon/Wed/Fri '
                     'short-dated (tau<=46d) — horizons and curve scopes differ')}

    with open(os.path.join(log_dir, 'flow_us_eval.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Summary figure: per-ticker RMSE ratio flow/RW
    if results:
        names = list(results)
        ratios = [results[t]['flow']['tv_rmse'] /
                  results[t]['random_walk']['tv_rmse'] for t in names]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ['tab:green' if r < 1 else 'tab:red' for r in ratios]
        ax.bar(names, ratios, color=colors)
        ax.axhline(1.0, color='k', lw=1, ls='--', label='random walk parity')
        ax.set_ylabel('flow tv-RMSE / RW tv-RMSE (lower = better)')
        ax.set_title('Mag 7: flow forecaster vs next-snapshot random walk')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(log_dir, 'flow_us_vs_rw.png'), dpi=110)
        plt.close(fig)

    logger.info(f'US evaluation complete: flow beats RW on {wins}/{len(results)} '
                f'tickers ({sig_wins} significant at 5%).')


if __name__ == '__main__':
    main()
