"""Train the pooled cross-sectional flow forecaster over Mag 7 + SPY.

Forecasts DE-EVENTED factor scores and re-adds the known future earnings
variance analytically, so the deterministic quarterly sawtooth never has to be
learned. Trains a seed ensemble; the velocity fields are averaged at sampling.
"""
from argparse import ArgumentParser
import json
import os

import numpy as np
import pandas as pd
import torch
from torch import optim

from us_dataset import UsOptionsProcessor, build_us_pairs, MAG7
from us_events import EarningsCalendar, add_event_columns, decompose_variance
from flow_us_pooled import (PooledFactorPreprocessor, PooledVelocity,
                            pooled_fm_loss, sample_pooled)
from flow_surface import EMA
from utils import load_config, parse_date, set_seed, setup_logging

SYMBOLS = MAG7 + ['SPY']


def n_events_per_node(cal, ticker, dates, tau_grid):
    """(N, n_tau) count of announcements in (date, date + tau] per grid node."""
    d = np.asarray(dates, dtype='datetime64[ns]')
    out = np.zeros((len(d), len(tau_grid)))
    for j, tau in enumerate(tau_grid):
        end = d + (tau * 365.25).astype('timedelta64[D]') if hasattr(
            tau * 365.25, 'astype') else d + np.timedelta64(
                int(round(float(tau) * 365.25)), 'D')
        out[:, j] = cal.count_in(ticker, d, end)
    return out


def build_panel(proc, table, cal, dec, ticker, cfg, train_end, test_start):
    """Panel + de-evented surfaces + per-node event counts for one symbol."""
    panel = proc.prepare_surface_panel(table, ticker, train_end,
                                       n_logm=cfg.getint('n_logm'))
    splits = build_us_pairs(panel, train_end, test_start)
    tau_grid = panel['tau_grid']
    n_tau, n_lg = len(tau_grid), len(panel['logm_grid'])

    jv = (dec[dec.ticker == ticker].set_index('date')['jump_var']
          .fillna(0.0).clip(lower=0.0))

    def deevent(S, dates):
        n_ev = n_events_per_node(cal, ticker, dates, tau_grid)   # (N, n_tau)
        j = jv.reindex(pd.DatetimeIndex(dates)).ffill().fillna(0.0).to_numpy()
        ev = (n_ev * j[:, None])[:, :, None] * np.ones((1, 1, n_lg))
        return np.clip(S.reshape(-1, n_tau, n_lg) - ev, 1e-8, None).reshape(
            len(dates), -1), ev.reshape(len(dates), -1)

    for name, sp in splits.items():
        if len(sp['dates']) == 0:
            continue
        tom = [panel['dates'][panel['dates'].index(d) + 1] for d in sp['dates']]
        sp['S_today_diff'], _ = deevent(sp['S_today'], sp['dates'])
        sp['S_tom_diff'], sp['ev_tom'] = deevent(sp['S_tomorrow'], tom)
        sp['n_ev_tom'] = n_events_per_node(cal, ticker, tom, tau_grid)
        sp['dte'] = cal.days_to_next(ticker, np.asarray(sp['dates'],
                                                        dtype='datetime64[ns]'))
    splits['_panel'] = panel
    return splits


def main():
    ap = ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--on_gpu', action='store_true')
    args = ap.parse_args()

    config = load_config('config.ini')
    set_seed(config['training'].getint('seed'))
    log_dir = config['save_path']['log_dir']
    logger = setup_logging(log_dir, 'flow_us_pooled')
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.on_gpu
                          else 'cpu')
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    cfg = config['flow_us']
    data_dir = config['data_us']['data_dir']
    train_end = parse_date(config['data_us']['train_end_date'])
    test_start = parse_date(config['data_us']['test_start_date'])
    epochs = args.epochs if args.epochs is not None else cfg.getint('epochs')

    proc = UsOptionsProcessor(data_dir)
    table = proc.build(tickers=SYMBOLS)
    cal = EarningsCalendar.from_data_dir(data_dir)
    table = add_event_columns(table, cal)
    dec = decompose_variance(table)
    logger.info(f'{len(table)} quotes, {dec.ticker.nunique()} symbols decomposed')

    panels = {t: build_panel(proc, table, cal, dec, t, cfg, train_end,
                             test_start) for t in SYMBOLS}

    # Shared basis over per-ticker-centred DE-EVENTED surfaces
    pp = PooledFactorPreprocessor(n_components=cfg.getint('max_components',
                                                          fallback=6))
    pp.fit({t: panels[t]['train']['S_today_diff'] for t in SYMBOLS})
    logger.info(f'pooled basis: k={pp.n_components_}, '
                f'EV={pp.explained_variance_ratio_.sum():.4f}')

    # Gate the shared basis: per-ticker reconstruction must not degrade badly
    for t in SYMBOLS:
        logger.info(f'  {t} recon MSE {pp.reconstruction_mse(t, panels[t]["train"]["S_today_diff"]):.3e}')

    # SPY market block, aligned by date
    spy_scores = {d: z for d, z in zip(
        panels['SPY']['train']['dates']
        + panels['SPY']['val']['dates'] + panels['SPY']['test']['dates'],
        np.vstack([pp.transform('SPY', panels['SPY'][s]['S_today_diff'])
                   for s in ('train', 'val', 'test')
                   if len(panels['SPY'][s]['dates'])]))}

    k = pp.n_components_

    def features(t, split):
        sp = panels[t][split]
        if len(sp['dates']) == 0:
            return None
        Z = pp.transform(t, sp['S_today_diff'])
        Zt = pp.transform(t, sp['S_tom_diff'])
        spy = np.vstack([spy_scores.get(d, np.zeros(k)) for d in sp['dates']])
        C = sp['C']                                  # market/vol features + gap
        dte = np.clip(sp['dte'], 0, 90)[:, None] / 90.0
        imminent = (sp['dte'] <= 3).astype(float)[:, None]
        n_ev = sp['n_ev_tom'].mean(axis=1, keepdims=True)
        disp = (Z[:, :1] - spy[:, :1])               # dispersion proxy
        return {'Z': Z, 'Zt': Zt, 'dates': sp['dates'],
                'cond_parts': [Z, spy, disp, C, dte, imminent, n_ev],
                'split': sp}

    feats = {t: {s: features(t, s) for s in ('train', 'val', 'test')}
             for t in SYMBOLS}

    # Cross-sectional mean score by date (zero-parameter block)
    from collections import defaultdict
    acc = defaultdict(list)
    for t in SYMBOLS:
        f = feats[t]['train']
        if f:
            for d, z in zip(f['dates'], f['Z']):
                acc[d].append(z)
    xmean = {d: np.mean(v, axis=0) for d, v in acc.items()}

    def assemble(t, s):
        f = feats[t][s]
        if f is None:
            return None
        xs = np.vstack([xmean.get(d, np.zeros(k)) for d in f['dates']])
        cond = np.concatenate(f['cond_parts'] + [xs], axis=1)
        return cond, f

    # Standardize conditioning on TRAIN only
    tr_conds = [assemble(t, 'train')[0] for t in SYMBOLS]
    all_tr = np.vstack(tr_conds)
    c_mean, c_std = all_tr.mean(0), np.clip(all_tr.std(0), 1e-9, None)

    # Increment targets, z-scored on train
    inc_scale = {}
    X, Y, T = [], [], []
    for i, t in enumerate(SYMBOLS):
        cond, f = assemble(t, 'train')
        inc = f['Zt'] - f['Z']
        inc_scale[t] = (inc.mean(0), np.clip(inc.std(0), 1e-9, None))
        Y.append((inc - inc_scale[t][0]) / inc_scale[t][1])
        X.append((cond - c_mean) / c_std)
        T.append(np.full(len(inc), i))
    Xnp, Ynp = np.vstack(X), np.vstack(Y)
    # Fail loudly: a single NaN column silently turns the whole FM loss to NaN.
    if not np.isfinite(Xnp).all() or not np.isfinite(Ynp).all():
        bad = [i for i in range(Xnp.shape[1])
               if not np.isfinite(Xnp[:, i]).all()]
        raise ValueError(f'non-finite training inputs; cond columns {bad}, '
                         f'targets finite={np.isfinite(Ynp).all()}')
    Xtr = torch.as_tensor(Xnp, dtype=dtype, device=device)
    Ytr = torch.as_tensor(Ynp, dtype=dtype, device=device)
    Ttr = torch.as_tensor(np.concatenate(T), dtype=torch.long, device=device)
    logger.info(f'pooled training pairs: {len(Ytr)}, cond_dim={Xtr.shape[1]}')

    models = []
    for seed in range(args.seeds):
        torch.manual_seed(1000 + seed)
        m = PooledVelocity(dim=k, cond_dim=Xtr.shape[1], n_tickers=len(SYMBOLS),
                           hidden=cfg.getint('hidden'),
                           n_blocks=cfg.getint('n_blocks'),
                           dropout=cfg.getfloat('dropout')).to(device)
        opt = optim.AdamW(m.parameters(), lr=cfg.getfloat('learning_rate'),
                          weight_decay=cfg.getfloat('weight_decay'))
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        ema = EMA(m, decay=cfg.getfloat('ema_decay'))
        bs = cfg.getint('batch_size')
        for ep in range(epochs):
            m.train()
            perm = torch.randperm(len(Ytr), device=device)
            for s in range(0, len(Ytr), bs):
                idx = perm[s:s + bs]
                loss = pooled_fm_loss(m, Ytr[idx], Xtr[idx], Ttr[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                ema.update(m)
            sch.step()
        ema.copy_to(m)
        m.eval()
        models.append(m)
        logger.info(f'seed {seed}: final FM loss {loss.item():.5f}')

    path = '../models/FlowPooled_us.pt'
    torch.save({
        'state_dicts': [m.state_dict() for m in models],
        'preprocessor': pp.to_dict(),
        'cond_mean': c_mean.tolist(), 'cond_std': c_std.tolist(),
        'inc_scale': {t: [a.tolist(), b.tolist()]
                      for t, (a, b) in inc_scale.items()},
        'symbols': SYMBOLS, 'k': k,
        'model_kwargs': {'dim': k, 'cond_dim': int(Xtr.shape[1]),
                         'n_tickers': len(SYMBOLS),
                         'hidden': cfg.getint('hidden'),
                         'n_blocks': cfg.getint('n_blocks'),
                         'dropout': cfg.getfloat('dropout')},
        'n_sample_steps': cfg.getint('n_sample_steps'),
        'tau_grid': panels['AAPL']['_panel']['tau_grid'].tolist(),
        'logm_grid': panels['AAPL']['_panel']['logm_grid'].tolist(),
    }, path)
    with open(os.path.join(log_dir, 'flow_us_pooled_train.json'), 'w') as f:
        json.dump({'k': k, 'seeds': args.seeds, 'n_pairs': int(len(Ytr)),
                   'cond_dim': int(Xtr.shape[1]),
                   'explained_variance': float(pp.explained_variance_ratio_.sum())},
                  f, indent=2)
    logger.info(f'saved {path}')


if __name__ == '__main__':
    main()
