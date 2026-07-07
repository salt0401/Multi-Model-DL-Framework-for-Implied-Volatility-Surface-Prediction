"""Pooled cross-ticker HyperIV training on Mag 7 US option chains.

One model for all seven underlyings: the reference set identifies the surface
(HyperIV's own cross-asset transfer results motivate pooling), which also
multiplies the effective training set — the point of the "deeper market"
experiment. Reuses HyperIVModel / HyperIVLoss / the training machinery from
train_hyperiv.py unchanged.
"""
from hyperiv import HyperIVModel, HyperIVLoss
from train_hyperiv import (collate_surfaces, train_one_epoch, evaluate,
                           compute_violation_rates, load_checkpoint_state)
from us_dataset import UsOptionsProcessor, MAG7
from utils import (load_config, parse_date, set_seed, setup_logging,
                   MetricsTracker, EarlyStopping)

from argparse import ArgumentParser

import copy
import json
import torch
from torch import optim
import os


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config('config.ini')
    set_seed(config['training'].getint('seed'))
    log_dir = config['save_path']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir, 'hyperiv_us')

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")

    cfg = config['hyperiv_us']
    dtype = torch.float64 if cfg.get('dtype', 'float32') == 'float64' else torch.float32
    torch.set_default_dtype(dtype)

    n_reference = cfg.getint('n_reference')
    epochs = args.epochs if args.epochs is not None else cfg.getint('epochs')
    batch_size = cfg.getint('batch_size')
    lr = cfg.getfloat('learning_rate')
    gradient_clip = config['training'].getfloat('gradient_clip', fallback=1.0)

    train_end = parse_date(config['data_us']['train_end_date'])
    test_start = parse_date(config['data_us']['test_start_date'])

    logger.info('Building US option table...')
    proc = UsOptionsProcessor(config['data_us']['data_dir'])
    table = proc.build()
    logger.info(f'{len(table)} option rows, '
                f'{table.groupby("ticker").size().to_dict()}')
    pooled = proc.prepare_hyperiv_surfaces(table)

    train_surfaces, test_by_ticker = [], {t: [] for t in MAG7}
    for d, t, tens in pooled:
        tens = tuple(x.to(dtype) for x in tens)
        if d <= train_end:
            train_surfaces.append(tens)
        elif d >= test_start:
            test_by_ticker[t].append(tens)

    n_train = int(len(train_surfaces) * 0.85)
    val_surfaces = train_surfaces[n_train:]
    train_surfaces = train_surfaces[:n_train]
    n_test = sum(len(v) for v in test_by_ticker.values())
    logger.info(f'Surfaces - train: {len(train_surfaces)}, '
                f'val: {len(val_surfaces)}, test: {n_test}')

    all_feats = torch.cat([torch.cat([a, b, c], dim=-1)
                           for a, b, c, _ in train_surfaces], dim=0)
    feat_mean, feat_std = all_feats.mean(dim=0), all_feats.std(dim=0)

    model = HyperIVModel(
        embed_dim=cfg.getint('embed_dim'),
        n_heads=cfg.getint('transformer_heads'),
        n_transformer_layers=cfg.getint('transformer_layers'),
        target_hidden_dims=tuple(int(x) for x in
                                 cfg['target_hidden_dims'].split(',')),
    ).to(dtype).to(device)
    model.set_normalization(feat_mean, feat_std)

    w_mse = cfg.getfloat('w_mse', fallback=1.0)
    w_cal = cfg.getfloat('w_calendar', fallback=1.0)
    w_but = cfg.getfloat('w_butterfly', fallback=10.0)
    w_price = cfg.getfloat('w_price', fallback=0.1)
    warmup = cfg.getint('penalty_warmup_epochs', fallback=15)
    ramp = cfg.getint('penalty_ramp_epochs', fallback=25)

    # Penalty warmup: measured on this data, the calendar penalty at full
    # weight dominates the early loss (batch maxima 50x the MSE — short-dated
    # single-name term structures brush the calendar bound around earnings),
    # and w~0 is the global penalty attractor that the saturated softplus
    # then locks in. Fit first, ramp the penalties in.
    def loss_at_epoch(epoch):
        s = 0.0 if epoch < warmup else min(1.0, (epoch - warmup) / max(ramp, 1))
        return HyperIVLoss(w_mse=w_mse, w_calendar=s * w_cal,
                           w_butterfly=s * w_but, w_price=w_price)

    # Validation/early-stopping always uses the FINAL weights
    loss_fn = HyperIVLoss(w_mse=w_mse, w_calendar=w_cal,
                          w_butterfly=w_but, w_price=w_price)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    metrics = MetricsTracker()
    early_stopping = EarlyStopping(patience=50)
    model_path = '../models/HyperIVModel_us.pt'

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    stable_state = None

    logger.info(f'Training pooled US HyperIV ({device}, penalty warmup '
                f'{warmup}+{ramp} epochs)...')
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_surfaces, loss_at_epoch(epoch),
                                     optimizer, n_reference, batch_size, device,
                                     gradient_clip=gradient_clip)
        # Spike guard only meaningful once weights stop ramping (loss scale
        # changes across the warmup by construction)
        if epoch < warmup + ramp:
            best_train_loss = float('inf')
        if stable_state is not None and train_loss > 3 * best_train_loss:
            model.load_state_dict(stable_state)
            for g in optimizer.param_groups:
                g['lr'] *= 0.5
            logger.info(f'Epoch {epoch+1}: spike guard fired '
                        f'({train_loss:.2e} > 3x{best_train_loss:.2e})')
            continue
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            stable_state = copy.deepcopy(model.state_dict())

        val_loss, val_rmse, val_mape, val_iv_rmse = evaluate(
            model, val_surfaces, loss_fn, n_reference, batch_size, device)
        scheduler.step()

        if metrics.update(epoch, train_loss, val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'state_dict': model.state_dict(),
                        'feat_mean': feat_mean, 'feat_std': feat_std,
                        'config': dict(cfg)}, model_path)

        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - '
                        f'Val: {val_loss:.6f} (RMSE={val_rmse:.6f}, '
                        f'MAPE={val_mape:.4f}, IV-RMSE={val_iv_rmse:.6f})')
        if early_stopping.step(val_loss):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    model.load_state_dict(load_checkpoint_state(model_path, device))
    results = {'n_train': len(train_surfaces), 'n_val': len(val_surfaces),
               'n_test': n_test, 'best_epoch': metrics.best_epoch,
               'best_val_loss': best_val_loss, 'per_ticker': {}}
    for t in MAG7:
        if not test_by_ticker[t]:
            continue
        loss, rmse, mape, iv_rmse = evaluate(
            model, test_by_ticker[t], loss_fn, n_reference, batch_size, device)
        cal, but, n_grid = compute_violation_rates(
            model, test_by_ticker[t], n_reference, device)
        results['per_ticker'][t] = {
            'n_surfaces': len(test_by_ticker[t]), 'test_rmse': float(rmse),
            'test_mape': float(mape), 'test_iv_rmse': float(iv_rmse),
            'calendar_violation_rate': cal, 'butterfly_violation_rate': but}
        logger.info(f'{t}: RMSE={rmse:.6f} MAPE={mape:.4f} '
                    f'IV-RMSE={iv_rmse:.6f} viol cal/but={cal:.4%}/{but:.4%}')

    all_test = [s for v in test_by_ticker.values() for s in v]
    loss, rmse, mape, iv_rmse = evaluate(model, all_test, loss_fn,
                                         n_reference, batch_size, device)
    results['pooled_test'] = {'test_rmse': float(rmse), 'test_mape': float(mape),
                              'test_iv_rmse': float(iv_rmse)}
    logger.info(f'POOLED test: RMSE={rmse:.6f} MAPE={mape:.4f} IV-RMSE={iv_rmse:.6f}')

    with open(os.path.join(log_dir, 'hyperiv_us_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    logger.info('US HyperIV training complete.')


if __name__ == '__main__':
    main()
