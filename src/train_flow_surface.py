"""Training script for Model 5 (replacement): conditional flow matching over
PCA factors of the log total-variance surface.

Supersedes src/train_diffusion.py (grid-space DDPM draft, deprecated). See
src/flow_surface.py for the rationale and docs/model45_completion_report.md
for the decision record.

Anti-memorization levers for the ~1,450-pair regime: small residual MLP,
dropout, weight decay 1e-3, EMA weights, early stopping on sampled val RMSE.
"""
from dataset import DataProcessor
from flow_surface import (FactorPreprocessor, CondScaler, VelocityMLP,
                          fm_loss, sample_flow, EMA, build_dataset)
from utils import (load_config, parse_date, set_seed, setup_logging,
                   EarlyStopping)

from argparse import ArgumentParser

import json
import numpy as np
import torch
from torch import optim
import os


def make_condition(Z_today, C_scaled):
    """Model condition = [today's factor scores, scaled market features]."""
    return np.concatenate([Z_today, C_scaled], axis=1)


def sampled_val_rmse(model, pp, ics, Z_today_val, cond_val, S_tomorrow_val,
                     device, dtype, n_samples=32, n_steps=50, seed=1234):
    """De-normalized total-variance RMSE of the mean of n_samples draws.

    The flow models the SCALED DAILY INCREMENT of factor scores; forecasts
    are z_today + inverse-scaled sampled increments (persistence is exact
    by construction — daily surfaces are ~0.99 autocorrelated, so a levels
    flow was measurably worse than the random walk)."""
    cond_t = torch.as_tensor(cond_val, dtype=dtype, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    samples = sample_flow(model, cond_t, n_steps=n_steps, n_samples=n_samples,
                          generator=g)                      # (S, B, k)
    inc_mean = ics.inverse(samples.mean(dim=0).cpu().numpy())
    S_pred = pp.inverse(Z_today_val + inc_mean)
    return float(np.sqrt(np.mean((S_pred - S_tomorrow_val) ** 2)))


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config('config.ini')
    seed = config['training'].getint('seed')
    set_seed(seed)

    log_dir = config['save_path']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir, 'flow_surface')

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")
    dtype = torch.float32
    torch.set_default_dtype(dtype)

    cfg = config['flow_surface']
    n_tau_grid = cfg.getint('n_tau_grid')
    n_logm_grid = cfg.getint('n_logm_grid')
    max_components = cfg.getint('max_components')
    ev_target = cfg.getfloat('ev_target')
    hidden = cfg.getint('hidden')
    n_blocks = cfg.getint('n_blocks')
    dropout = cfg.getfloat('dropout')
    weight_decay = cfg.getfloat('weight_decay')
    ema_decay = cfg.getfloat('ema_decay')
    n_sample_steps = cfg.getint('n_sample_steps')
    epochs = args.epochs if args.epochs is not None else cfg.getint('epochs')
    lr = cfg.getfloat('learning_rate')
    batch_size = cfg.getint('batch_size')
    val_samples = cfg.getint('val_samples')
    val_every = cfg.getint('val_every', fallback=25)
    patience = cfg.getint('patience', fallback=40)

    train_end_date = parse_date(config['training']['train_end_date'])
    test_start_date = parse_date(config['training']['test_start_date'])

    # Data
    logger.info('Preparing surface panel...')
    dp = DataProcessor(config)
    dp()
    panel = dp.Prepare_surface_panel(train_end_date,
                                     n_tau_grid=n_tau_grid,
                                     n_logm_grid=n_logm_grid)
    splits = build_dataset(panel, train_end_date, test_start_date)
    logger.info(f"Pairs - train: {len(splits['train']['dates'])}, "
                f"val: {len(splits['val']['dates'])}, "
                f"test: {len(splits['test']['dates'])}")

    # Fit preprocessing on TRAIN only
    pp = FactorPreprocessor(ev_target=ev_target, max_components=max_components)
    pp.fit(splits['train']['S_today'])
    logger.info(f'PCA components: {pp.n_components_} '
                f'(EV: {pp.explained_variance_ratio_.sum():.4f})')

    cs = CondScaler().fit(splits['train']['C'])

    def prep(split):
        Z_today = pp.transform(split['S_today'])
        Z_tomorrow = pp.transform(split['S_tomorrow'])
        cond = make_condition(Z_today, cs.transform(split['C']))
        return Z_today, Z_tomorrow, cond

    Z_tr_today, Z_tr_tom, cond_tr = prep(splits['train'])
    Z_va_today, Z_va_tom, cond_va = prep(splits['val'])

    # Flow target = z-scored daily increment of factor scores (train stats)
    ics = CondScaler().fit(Z_tr_tom - Z_tr_today)
    inc_tr = ics.transform(Z_tr_tom - Z_tr_today)

    k = pp.n_components_
    cond_dim = cond_tr.shape[1]

    model = VelocityMLP(dim=k, cond_dim=cond_dim, hidden=hidden,
                        n_blocks=n_blocks, dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'VelocityMLP: dim={k}, cond_dim={cond_dim}, {n_params:,} params')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = EMA(model, decay=ema_decay)
    early_stopping = EarlyStopping(patience=patience)

    Z1 = torch.as_tensor(inc_tr, dtype=dtype, device=device)
    COND = torch.as_tensor(cond_tr, dtype=dtype, device=device)
    n_train = Z1.shape[0]

    model_path = config['save_path'].get('flow_model_path',
                                         '../models/FlowSurfaceModel.pt')

    def save_checkpoint():
        torch.save({
            'state_dict': model.state_dict(),
            'ema_shadow': ema.state_dict(),
            'preprocessor': pp.to_dict(),
            'cond_scaler': cs.to_dict(),
            'increment_scaler': ics.to_dict(),
            'predict_increments': True,
            'tau_grid': panel['tau_grid'].tolist(),
            'logm_grid': panel['logm_grid'].tolist(),
            'cond_names': panel['cond_names'],
            'model_kwargs': {'dim': k, 'cond_dim': cond_dim, 'hidden': hidden,
                             'n_blocks': n_blocks, 'dropout': dropout},
            'n_sample_steps': n_sample_steps,
            'config': dict(cfg),
        }, model_path)

    best_val_rmse = float('inf')
    train_losses = []
    val_history = []

    logger.info(f'Training flow matching on {device} ({epochs} epochs max)...')
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            loss = fm_loss(model, Z1[idx], COND[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update(model)
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_losses.append(epoch_loss / max(n_batches, 1))

        if (epoch + 1) % val_every == 0:
            # Validate with EMA weights
            backup = {name: p.detach().clone()
                      for name, p in model.named_parameters()}
            ema.copy_to(model)
            val_rmse = sampled_val_rmse(model, pp, ics, Z_va_today, cond_va,
                                        splits['val']['S_tomorrow'],
                                        device, dtype,
                                        n_samples=val_samples,
                                        n_steps=n_sample_steps)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    p.copy_(backup[name])

            val_history.append({'epoch': epoch + 1, 'val_rmse': val_rmse})
            logger.info(f'Epoch {epoch+1}/{epochs} - FM loss: {train_losses[-1]:.5f} '
                        f'- val tv-RMSE (EMA, {val_samples} samples): {val_rmse:.6f}')

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                save_checkpoint()
                logger.info('  New best checkpoint saved')

            if early_stopping.step(val_rmse):
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

    if best_val_rmse == float('inf'):
        save_checkpoint()

    with open(os.path.join(log_dir, 'flow_surface_train.json'), 'w') as f:
        json.dump({
            'n_components': k,
            'explained_variance': float(pp.explained_variance_ratio_.sum()),
            'n_params': n_params,
            'best_val_rmse': best_val_rmse,
            'val_history': val_history,
            'final_train_loss': train_losses[-1] if train_losses else None,
        }, f, indent=2)
    logger.info(f'Flow-surface training complete. Best val tv-RMSE: {best_val_rmse:.6f}')


if __name__ == '__main__':
    main()
