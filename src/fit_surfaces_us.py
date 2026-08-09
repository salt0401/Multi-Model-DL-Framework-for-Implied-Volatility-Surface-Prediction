"""Fit Global eSSVI (Model 1, US) across the Mag 7 + SPY panel.

Pipeline per (ticker, snapshot):
  quotes -> parity forward -> de-event (subtract n_events * sigma_j^2)
         -> Global eSSVI fit (arbitrage-free by construction)
         -> validation gates + arbitrage report

The A/B gate that matters: this is compared against the TXO-era frozen
rho_0 = -0.95 parameterization on the SAME slices, cross-validated on the
strike axis, in implied-vol points.
"""
from argparse import ArgumentParser
import json
import os

import numpy as np
import pandas as pd

from us_dataset import UsOptionsProcessor, MAG7
from us_events import (EarningsCalendar, add_event_columns, decompose_variance,
                       attach_decomposition)
from svi_us import (fit_snapshot, cv_iv_rmse, iv_rmse, arbitrage_report,
                    essvi_w, unpack)
from utils import load_config, setup_logging, set_seed

from scipy.optimize import least_squares


def build_slices(group, min_quotes=6):
    """One (ticker, date) -> ordered per-expiry slices of de-evented variance."""
    slices = []
    for exp, g in group.groupby('expiration'):
        g = g.sort_values('logm')
        if len(g) < min_quotes:
            continue
        w = g['w_diff'].to_numpy(float)
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            continue
        spread = g['spread'].to_numpy(float)
        weight = 1.0 / np.maximum(spread, 0.01)
        weight = weight / np.mean(weight)
        slices.append({'k': g['logm'].to_numpy(float), 'w': w,
                       'tau': float(g['tau'].iloc[0]), 'weight': weight,
                       'expiration': exp})
    return slices


# ── Frozen-rho baseline (the TXO-era parameterization) ────────────────

def _frozen_resid(p, slices, rho0=-0.95):
    """Same eSSVI family but with rho FROZEN at the TXO value, theta monotone."""
    n = len(slices)
    theta = np.cumsum(np.logaddexp(0.0, p[:n])) + 1e-8
    psi = 4.0 / (1.0 + abs(rho0)) * (0.5 * (1 + np.tanh(0.5 * p[n:2 * n])))
    out = []
    for i, s in enumerate(slices):
        out.append(s['weight'] * (essvi_w(s['k'], theta[i], rho0, psi[i]) - s['w']))
    return np.concatenate(out)


def fit_frozen(slices, rho0=-0.95):
    n = len(slices)
    theta0 = np.maximum.accumulate(
        [max(s['w'][np.argmin(np.abs(s['k']))], 1e-6) for s in slices])
    inc = np.diff(np.concatenate([[0.0], theta0]))
    x0 = np.concatenate([np.log(np.expm1(np.maximum(inc, 1e-6))), np.zeros(n)])
    sol = least_squares(_frozen_resid, x0, args=(slices, rho0), method='trf',
                        max_nfev=400)
    theta = np.cumsum(np.logaddexp(0.0, sol.x[:n])) + 1e-8
    psi = 4.0 / (1.0 + abs(rho0)) * (0.5 * (1 + np.tanh(0.5 * sol.x[n:2 * n])))
    return {'theta': theta, 'rho': np.full(n, rho0), 'psi': psi,
            'tau': np.array([s['tau'] for s in slices])}


def cv_frozen(slices, n_folds=4, rho0=-0.95):
    errs = []
    for f in range(n_folds):
        tr, te, ok = [], [], True
        for s in slices:
            m = np.arange(len(s['k'])) % n_folds != f
            if m.sum() < 5 or (~m).sum() < 1:
                ok = False
                break
            tr.append({'k': s['k'][m], 'w': s['w'][m], 'tau': s['tau'],
                       'weight': s['weight'][m]})
            te.append({'k': s['k'][~m], 'w': s['w'][~m], 'tau': s['tau'],
                       'weight': s['weight'][~m]})
        if not ok:
            continue
        fit = fit_frozen(tr, rho0)
        for i, s in enumerate(te):
            w_hat = np.maximum(essvi_w(s['k'], fit['theta'][i], rho0,
                                       fit['psi'][i]), 1e-12)
            errs.append(np.sqrt(w_hat / s['tau'])
                        - np.sqrt(np.maximum(s['w'], 1e-12) / s['tau']))
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2))) if errs else np.nan


def main():
    ap = ArgumentParser()
    ap.add_argument('--tickers', default=','.join(MAG7 + ['SPY']))
    ap.add_argument('--max-snapshots', type=int, default=200,
                    help='per ticker, for the A/B report (0 = all)')
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    config = load_config('config.ini')
    set_seed(config['training'].getint('seed'))
    log_dir = config['save_path']['log_dir']
    logger = setup_logging(log_dir, 'fit_surfaces_us')
    data_dir = config['data_us']['data_dir']

    tickers = args.tickers.split(',')
    proc = UsOptionsProcessor(data_dir)
    table = proc.build(tickers=tickers)
    cal = EarningsCalendar(os.path.join(data_dir, 'earnings_dates_v2.csv'))
    table = add_event_columns(table, cal)
    dec = decompose_variance(table)
    table = attach_decomposition(table, dec)

    # De-evented total variance: remove the scheduled announcement variance
    table['w_diff'] = (table['total_var']
                       - table['n_earnings'] * table['jump_var'].fillna(0.0))
    table = table[table['w_diff'] > 1e-8]
    logger.info(f'{len(table)} quotes after de-eventing, '
                f'{table.groupby(["ticker", "date"]).ngroups} snapshots')

    rows = []
    for (t, d), g in table.groupby(['ticker', 'date']):
        slices = build_slices(g)
        if len(slices) < 2:
            continue
        rows.append((t, d, slices))

    # Subsample per ticker for the A/B (full-panel fitting is done below)
    by_ticker = {}
    for t, d, s in rows:
        by_ticker.setdefault(t, []).append((d, s))

    results = []
    for t, items in by_ticker.items():
        items.sort(key=lambda x: x[0])
        sel = items if args.max_snapshots == 0 else items[::max(
            1, len(items) // args.max_snapshots)]
        for d, slices in sel:
            fit = fit_snapshot(slices)
            rep = arbitrage_report(fit)
            rec = {'ticker': t, 'date': d, 'n_slices': len(slices),
                   'iv_rmse_in': iv_rmse(fit, slices),
                   'cv_iv_rmse': cv_iv_rmse(slices),
                   'cv_iv_rmse_frozen': cv_frozen(slices),
                   **rep}
            results.append(rec)
        logger.info(f'{t}: fitted {len(sel)} snapshots')

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(log_dir, 'svi_us_fits.csv'), index=False)

    summary = (res.groupby('ticker')
                  .agg(n=('date', 'size'),
                       cv_rmse_volpts=('cv_iv_rmse', lambda s: 100 * s.median()),
                       cv_rmse_frozen_volpts=('cv_iv_rmse_frozen',
                                              lambda s: 100 * s.median()),
                       rho_med=('rho_min', 'median'),
                       wing_ratio=('wing_ratio', 'median'),
                       butterfly_ok=('butterfly_ok', 'mean'),
                       theta_incr=('theta_increasing', 'mean'),
                       cal_viol=('calendar_violation_rate', 'mean'))
                  .round(4))
    summary['improvement_x'] = (summary['cv_rmse_frozen_volpts']
                                / summary['cv_rmse_volpts']).round(2)

    print('\nGLOBAL eSSVI vs FROZEN rho_0=-0.95 (strike-axis CV, vol points)')
    print(summary.to_string())
    print('\nGates: butterfly_ok / theta_incr should be 1.0 (by construction);')
    print('cal_viol should be 0.0; bid-ask noise floor is ~0.39 vol points.')

    out = {'per_ticker': json.loads(summary.reset_index().to_json(orient='records')),
           'pooled': {
               'cv_rmse_volpts': float(100 * res['cv_iv_rmse'].median()),
               'cv_rmse_frozen_volpts': float(100 * res['cv_iv_rmse_frozen'].median()),
               'butterfly_ok_rate': float(res['butterfly_ok'].mean()),
               'theta_increasing_rate': float(res['theta_increasing'].mean()),
               'calendar_violation_rate': float(res['calendar_violation_rate'].mean()),
               'n_snapshots': int(len(res))}}
    with open(os.path.join(log_dir, 'svi_us_report.json'), 'w') as f:
        json.dump(out, f, indent=2)
    logger.info(f"pooled CV {out['pooled']['cv_rmse_volpts']:.3f} vs frozen "
                f"{out['pooled']['cv_rmse_frozen_volpts']:.3f} vol points")


if __name__ == '__main__':
    main()
