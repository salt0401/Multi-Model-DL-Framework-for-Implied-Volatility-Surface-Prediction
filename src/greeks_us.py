"""Analytic local variance and Greeks from the fitted eSSVI surface (M2).

Replaces the ICNN Dupire PINN. Two reasons the PINN is the wrong object here:

1. Its stated motivation was repairing Model 1's butterfly violations so the
   Dupire denominator could not go negative. Under the Global eSSVI backbone
   the butterfly density is positive BY CONSTRUCTION, so the denominator is
   structurally safe and there is nothing left to repair.
2. Local volatility is only defined where total variance is differentiable in
   time. Across a scheduled earnings announcement it is not — w steps by
   sigma_j^2 — and the implied event-day local variance measured on this data
   is 0.77-3.03 /yr (an 88-174% one-day local vol, 8-30x the diffusive level).
   Any network with a smoothness prior averages that spike away, and vanna,
   volga and dsigma_LV/dK inherit the error.

So: local variance is computed in CLOSED FORM on the DE-EVENTED surface,
where it is well defined, and the discrete event jump is reported separately
rather than smeared into it.

Gatheral's local variance in terms of total variance w(k, T):

    sigma_LV^2 = (dw/dT) / g(k),
    g(k) = 1 - (k/w)(dw/dk) + (1/4)(-1/4 - 1/w + k^2/w^2)(dw/dk)^2 + (1/2)(d2w/dk2)

g(k) is exactly the butterfly density, which the parameterization keeps > 0.
"""
from argparse import ArgumentParser
import json
import os

import numpy as np
import pandas as pd

from svi_us import essvi_w


# ── Closed-form eSSVI derivatives ─────────────────────────────────────

def essvi_derivs(k, theta, rho, psi):
    """w, dw/dk, d2w/dk2 for one eSSVI slice, in closed form.

    With phi = psi/theta and D = sqrt((phi k + rho)^2 + 1 - rho^2):
        w      = theta/2 * (1 + rho*phi*k + D)
        dw/dk  = (theta*phi/2) * (rho + (phi k + rho)/D)
        d2w/dk2= (theta*phi^2/2) * (1 - rho^2) / D^3
    """
    k = np.asarray(k, dtype=float)
    theta = max(float(theta), 1e-12)
    phi = psi / theta
    x = phi * k
    D = np.sqrt(np.maximum((x + rho) ** 2 + 1.0 - rho ** 2, 1e-16))
    w = 0.5 * theta * (1.0 + rho * x + D)
    dw = 0.5 * theta * phi * (rho + (x + rho) / D)
    d2w = 0.5 * theta * phi ** 2 * (1.0 - rho ** 2) / D ** 3
    return w, dw, d2w


def butterfly_density(k, theta, rho, psi):
    """g(k) — the Dupire denominator and the risk-neutral density factor."""
    w, dw, d2w = essvi_derivs(k, theta, rho, psi)
    w = np.maximum(w, 1e-12)
    return (1.0 - (k / w) * dw
            + 0.25 * (-0.25 - 1.0 / w + k ** 2 / w ** 2) * dw ** 2
            + 0.5 * d2w)


def local_variance(fit, k, slice_idx):
    """Dupire local variance on the DE-EVENTED surface at one slice.

    dw/dT is taken from the fitted theta term structure (a forward difference
    between adjacent slices, which is positive because theta is constructed
    strictly increasing), evaluated at fixed log-moneyness.
    """
    theta, rho, psi, tau = (fit['theta'], fit['rho'], fit['psi'], fit['tau'])
    i = slice_idx
    j = min(i + 1, len(theta) - 1)
    if j == i:
        j, i2 = i, max(i - 1, 0)
        dT = max(tau[j] - tau[i2], 1e-8)
        dwdT = (essvi_w(k, theta[j], rho[j], psi[j])
                - essvi_w(k, theta[i2], rho[i2], psi[i2])) / dT
    else:
        dT = max(tau[j] - tau[i], 1e-8)
        dwdT = (essvi_w(k, theta[j], rho[j], psi[j])
                - essvi_w(k, theta[i], rho[i], psi[i])) / dT
    g = butterfly_density(k, theta[slice_idx], rho[slice_idx], psi[slice_idx])
    return np.maximum(dwdT, 0.0) / np.maximum(g, 1e-8)


# ── Black-Scholes higher-order Greeks ─────────────────────────────────

def bs_greeks(k, tau, w):
    """Vanna and volga per unit forward from log-moneyness and total variance.

    sigma = sqrt(w/tau);  d1 = (-k + w/2)/sqrt(w);  d2 = d1 - sqrt(w)
    vega  = F*phi(d1)*sqrt(tau)
    vanna = -phi(d1)*d2/sigma          (d vega / d spot, per unit forward)
    volga = vega*d1*d2/sigma
    """
    from scipy.stats import norm
    w = np.maximum(np.asarray(w, dtype=float), 1e-12)
    tau = max(float(tau), 1e-12)
    sigma = np.sqrt(w / tau)
    sw = np.sqrt(w)
    d1 = (-np.asarray(k, dtype=float) + 0.5 * w) / sw
    d2 = d1 - sw
    pdf = norm.pdf(d1)
    vega = pdf * np.sqrt(tau)
    return {'vega': vega,
            'vanna': -pdf * d2 / np.maximum(sigma, 1e-12),
            'volga': vega * d1 * d2 / np.maximum(sigma, 1e-12),
            'sigma': sigma}


def extract_features(fit, jump_var, n_events_by_slice, k_grid=None):
    """Downstream feature block replacing Module D's ICNN Greeks.

    Returns per-slice: ATM local volatility, ATM skew dsigma/dk, vanna, volga,
    minimum butterfly density (a surface-health check), plus the DISCRETE
    event jump kept separate from the diffusive local vol.
    """
    if k_grid is None:
        k_grid = np.linspace(-0.3, 0.3, 61)
    j0 = int(np.argmin(np.abs(k_grid)))
    out = []
    for i in range(len(fit['theta'])):
        w, dw, _ = essvi_derivs(k_grid, fit['theta'][i], fit['rho'][i],
                                fit['psi'][i])
        lv = local_variance(fit, k_grid, i)
        tau = float(fit['tau'][i])
        g = bs_greeks(k_grid, tau, w)
        dens = butterfly_density(k_grid, fit['theta'][i], fit['rho'][i],
                                 fit['psi'][i])
        sigma_atm = float(np.sqrt(max(w[j0], 1e-12) / max(tau, 1e-12)))
        out.append({
            'slice': i, 'tau': tau,
            'local_vol_atm': float(np.sqrt(max(lv[j0], 0.0))),
            'iv_atm': sigma_atm,
            'skew_dsigma_dk': float(dw[j0] / (2 * sigma_atm * tau)),
            'vanna_atm': float(g['vanna'][j0]),
            'volga_atm': float(g['volga'][j0]),
            'min_density': float(np.min(dens)),
            'n_events': int(n_events_by_slice[i]),
            'event_jump_var': float(jump_var) if np.isfinite(jump_var) else 0.0,
            'event_implied_move': (float(np.sqrt(jump_var))
                                   if np.isfinite(jump_var) and jump_var > 0
                                   else 0.0),
        })
    return out


def main():
    ap = ArgumentParser()
    ap.add_argument('--tickers', default='AAPL,NVDA,TSLA,SPY')
    ap.add_argument('--max-snapshots', type=int, default=60)
    ap.add_argument('--report', action='store_true')
    args = ap.parse_args()

    from us_dataset import UsOptionsProcessor
    from us_events import (EarningsCalendar, add_event_columns,
                           decompose_variance, attach_decomposition)
    from fit_surfaces_us import build_slices
    from svi_us import fit_snapshot
    from utils import load_config, setup_logging

    config = load_config('config.ini')
    log_dir = config['save_path']['log_dir']
    logger = setup_logging(log_dir, 'greeks_us')
    data_dir = config['data_us']['data_dir']

    proc = UsOptionsProcessor(data_dir)
    table = proc.build(tickers=args.tickers.split(','))
    cal = EarningsCalendar.from_data_dir(data_dir)
    table = add_event_columns(table, cal)
    dec = decompose_variance(table)
    table = attach_decomposition(table, dec)
    table['w_diff'] = (table['total_var']
                       - table['n_earnings'] * table['jump_var'].fillna(0.0))
    table = table[table['w_diff'] > 1e-8]

    rows = []
    for t, gt in table.groupby('ticker'):
        snaps = sorted(gt['date'].unique())
        step = max(1, len(snaps) // args.max_snapshots)
        for d in snaps[::step]:
            g = gt[gt['date'] == d]
            slices = build_slices(g)
            if len(slices) < 2:
                continue
            fit = fit_snapshot(slices)
            n_ev = [int(g[g['expiration'] == s['expiration']]['n_earnings'].iloc[0])
                    for s in slices]
            jv = float(g['jump_var'].iloc[0])
            for f in extract_features(fit, jv, n_ev):
                f.update({'ticker': t, 'date': d})
                rows.append(f)

    out = pd.DataFrame(rows)
    path = os.path.join(log_dir, 'greeks_us_features.csv')
    out.to_csv(path, index=False)

    summ = (out.groupby('ticker')
               .agg(n=('slice', 'size'),
                    local_vol_atm=('local_vol_atm', 'median'),
                    iv_atm=('iv_atm', 'median'),
                    skew=('skew_dsigma_dk', 'median'),
                    vanna=('vanna_atm', 'median'),
                    volga=('volga_atm', 'median'),
                    min_density=('min_density', 'min'),
                    implied_move=('event_implied_move', 'median'))
               .round(4))
    print('\nANALYTIC GREEKS FROM eSSVI (M2 replacement)')
    print(summ.to_string())
    worst = float(out['min_density'].min())
    print(f'\nworst risk-neutral density across all slices: {worst:.4f} — '
          + ('POSITIVE, so the Dupire denominator is never degenerate.'
             if worst > 0 else
             'NEGATIVE: butterfly arbitrage present, local vol is unusable.'))
    with open(os.path.join(log_dir, 'greeks_us_report.json'), 'w') as f:
        json.dump(json.loads(summ.reset_index().to_json(orient='records')),
                  f, indent=2)
    logger.info(f'{len(out)} slice-features -> {path}')


if __name__ == '__main__':
    main()
