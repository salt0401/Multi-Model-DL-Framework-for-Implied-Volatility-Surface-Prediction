"""Per-ticker flow-matching surface forecasters for the Mag 7 branch.

One VelocityMLP per ticker over PCA factors of the short-dated (tau <= ~46d)
log total-variance surface; increments formulation (established on TXO: the
levels flow lost to the random walk); gap_days to the next snapshot is a
condition (Mon/Wed/Fri cadence makes horizons irregular but known ahead).
"""
from us_dataset import UsOptionsProcessor, build_us_pairs, MAG7
from flow_surface import (FactorPreprocessor, CondScaler, VelocityMLP,
                          fm_loss, sample_flow, EMA)
from utils import load_config, parse_date, set_seed, setup_logging, EarlyStopping

from argparse import ArgumentParser

import json
import numpy as np
import torch
from torch import optim
import os


def train_one_ticker(ticker, table, proc, config, device, dtype, logger):
    cfg = config['flow_us']
    train_end = parse_date(config['data_us']['train_end_date'])
    test_start = parse_date(config['data_us']['test_start_date'])

    panel = proc.prepare_surface_panel(table, ticker, train_end,
                                       n_logm=cfg.getint('n_logm'))
    splits = build_us_pairs(panel, train_end, test_start)
    n_tr, n_va, n_te = (len(splits[k]['dates']) for k in ('train', 'val', 'test'))
    logger.info(f'{ticker}: pairs train={n_tr} val={n_va} test={n_te}')

    pp = FactorPreprocessor(ev_target=cfg.getfloat('ev_target'),
                            max_components=cfg.getint('max_components'))
    pp.fit(splits['train']['S_today'])
    cs = CondScaler().fit(splits['train']['C'])

    def prep(sp):
        Z_today = pp.transform(sp['S_today'])
        Z_tom = pp.transform(sp['S_tomorrow'])
        cond = np.concatenate([Z_today, cs.transform(sp['C'])], axis=1)
        return Z_today, Z_tom, cond

    Z_tr_today, Z_tr_tom, cond_tr = prep(splits['train'])
    Z_va_today, _, cond_va = prep(splits['val'])

    ics = CondScaler().fit(Z_tr_tom - Z_tr_today)
    inc_tr = ics.transform(Z_tr_tom - Z_tr_today)

    k = pp.n_components_
    model = VelocityMLP(dim=k, cond_dim=cond_tr.shape[1],
                        hidden=cfg.getint('hidden'),
                        n_blocks=cfg.getint('n_blocks'),
                        dropout=cfg.getfloat('dropout')).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.getfloat('learning_rate'),
                            weight_decay=cfg.getfloat('weight_decay'))
    epochs = cfg.getint('epochs')
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = EMA(model, decay=cfg.getfloat('ema_decay'))
    early_stopping = EarlyStopping(patience=cfg.getint('patience', fallback=40))

    Z1 = torch.as_tensor(inc_tr, dtype=dtype, device=device)
    COND = torch.as_tensor(cond_tr, dtype=dtype, device=device)
    batch_size = cfg.getint('batch_size')
    n_steps = cfg.getint('n_sample_steps')
    val_samples = cfg.getint('val_samples')
    val_every = cfg.getint('val_every', fallback=25)

    model_path = f'../models/FlowSurface_{ticker}.pt'
    best_val = float('inf')

    def save_ckpt():
        torch.save({'state_dict': model.state_dict(),
                    'ema_shadow': ema.state_dict(),
                    'preprocessor': pp.to_dict(), 'cond_scaler': cs.to_dict(),
                    'increment_scaler': ics.to_dict(), 'predict_increments': True,
                    'tau_grid': panel['tau_grid'].tolist(),
                    'logm_grid': panel['logm_grid'].tolist(),
                    'cond_names': panel['cond_names'] + ['gap_days'],
                    'ticker': ticker,
                    'model_kwargs': {'dim': k, 'cond_dim': cond_tr.shape[1],
                                     'hidden': cfg.getint('hidden'),
                                     'n_blocks': cfg.getint('n_blocks'),
                                     'dropout': cfg.getfloat('dropout')},
                    'n_sample_steps': n_steps}, model_path)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(Z1.shape[0], device=device)
        for start in range(0, Z1.shape[0], batch_size):
            idx = perm[start:start + batch_size]
            loss = fm_loss(model, Z1[idx], COND[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
        scheduler.step()

        if (epoch + 1) % val_every == 0:
            backup = {n: p.detach().clone() for n, p in model.named_parameters()}
            ema.copy_to(model)
            g = torch.Generator(device=device.type).manual_seed(1234)
            samp = sample_flow(model, torch.as_tensor(cond_va, dtype=dtype,
                                                      device=device),
                               n_steps=n_steps, n_samples=val_samples,
                               generator=g)
            inc = ics.inverse(samp.mean(dim=0).cpu().numpy())
            S_pred = pp.inverse(Z_va_today + inc)
            val_rmse = float(np.sqrt(np.mean(
                (S_pred - splits['val']['S_tomorrow']) ** 2)))
            with torch.no_grad():
                for n, p in model.named_parameters():
                    p.copy_(backup[n])
            if val_rmse < best_val:
                best_val = val_rmse
                save_ckpt()
            if early_stopping.step(val_rmse):
                logger.info(f'{ticker}: early stop at {epoch+1}, '
                            f'best val tv-RMSE {best_val:.6f}')
                break
    if best_val == float('inf'):
        save_ckpt()
    logger.info(f'{ticker}: done, best val tv-RMSE {best_val:.6f} (k={k})')
    return {'ticker': ticker, 'n_components': k, 'best_val_rmse': best_val,
            'n_train_pairs': n_tr}


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--tickers", type=str, default=','.join(MAG7))
    args = parser.parse_args()

    config = load_config('config.ini')
    set_seed(config['training'].getint('seed'))
    log_dir = config['save_path']['log_dir']
    logger = setup_logging(log_dir, 'flow_us')
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.on_gpu
                          else "cpu")
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    proc = UsOptionsProcessor(config['data_us']['data_dir'])
    table = proc.build()
    logger.info(f'US option table: {len(table)} rows')

    summary = []
    for ticker in args.tickers.split(','):
        summary.append(train_one_ticker(ticker, table, proc, config,
                                        device, dtype, logger))
    with open(os.path.join(log_dir, 'flow_us_train.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info('All tickers done.')


if __name__ == '__main__':
    main()
