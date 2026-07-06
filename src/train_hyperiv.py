"""Training script for HyperIV model.

Each training batch = one day's observed options surface.
Reference set = random subset of observed options (n_reference).
Target set = remaining options from same day.

Upgrades over the original draft:
- float32 by default (config [hyperiv] dtype) — FP64 runs at 1/64 rate on Ada GPUs
- input standardization from TRAIN-set stats stored in the checkpoint
- seeded reference sampling at eval (the draft used the first-50 options of
  surfaces sorted by (tau, logm), i.e. always the short-maturity corner)
- masked losses (padding excluded from reduction)
- PIVOT price-space auxiliary loss (weights wired to config)
- test-set arbitrage-violation rates + JSON results export + fit plot
"""
from hyperiv import HyperIVModel, HyperIVLoss
from dataset import DataProcessor
from utils import (load_config, parse_date, set_seed,
                   setup_logging, MetricsTracker, EarlyStopping,
                   compute_rmse, compute_mape, compute_iv_rmse)

from argparse import ArgumentParser

import json
import torch
from torch import optim
import numpy as np
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def collate_surfaces(batch, n_reference, device, generator=None):
    """Collate a list of per-day tensors into padded batches.

    Args:
        batch: list of (tau, logm, total_var, y_atm) tensors, each (n_options, 1)
        n_reference: number of reference options to sample per surface
        device: torch device
        generator: optional torch.Generator for deterministic reference sampling

    Returns:
        ref_set:     (B, n_ref, 3)
        ref_mask:    (B, n_ref) bool — True where padded
        target_tau:  (B, max_target, 1)
        target_logm: (B, max_target, 1)
        target_yATM: (B, max_target, 1)
        target_tv:   (B, max_target, 1)
        target_mask: (B, max_target) bool — True where padded
    """
    ref_sets = []
    target_taus = []
    target_logms = []
    target_yATMs = []
    target_tvs = []
    target_sizes = []

    for tau, logm, tv, yatm in batch:
        n = tau.shape[0]
        if n <= n_reference:
            # Too few options — use all as reference, duplicate some as target
            ref_idx = list(range(n))
            target_idx = list(range(n))
        else:
            perm = torch.randperm(n, generator=generator)
            ref_idx = perm[:n_reference]
            target_idx = perm[n_reference:]

        ref_data = torch.cat([tau[ref_idx], logm[ref_idx], tv[ref_idx]], dim=-1)  # (n_ref, 3)
        ref_sets.append(ref_data)
        target_taus.append(tau[target_idx])
        target_logms.append(logm[target_idx])
        target_yATMs.append(yatm[target_idx])
        target_tvs.append(tv[target_idx])
        target_sizes.append(len(target_idx))

    # Pad reference sets
    max_ref = max(r.shape[0] for r in ref_sets)
    padded_refs = []
    ref_masks = []
    for r in ref_sets:
        pad_len = max_ref - r.shape[0]
        if pad_len > 0:
            padded_refs.append(torch.cat([r, torch.zeros(pad_len, 3, dtype=r.dtype)], dim=0))
            ref_masks.append(torch.cat([torch.zeros(r.shape[0], dtype=torch.bool),
                                         torch.ones(pad_len, dtype=torch.bool)]))
        else:
            padded_refs.append(r)
            ref_masks.append(torch.zeros(r.shape[0], dtype=torch.bool))

    # Pad targets
    max_target = max(target_sizes)
    padded_taus = []
    padded_logms = []
    padded_yATMs = []
    padded_tvs = []
    target_masks = []

    for i in range(len(batch)):
        pad_len = max_target - target_sizes[i]
        t_tau = target_taus[i]
        t_logm = target_logms[i]
        t_yatm = target_yATMs[i]
        t_tv = target_tvs[i]

        if pad_len > 0:
            padded_taus.append(torch.cat([t_tau, torch.zeros(pad_len, 1, dtype=t_tau.dtype)]))
            padded_logms.append(torch.cat([t_logm, torch.zeros(pad_len, 1, dtype=t_logm.dtype)]))
            padded_yATMs.append(torch.cat([t_yatm, torch.zeros(pad_len, 1, dtype=t_yatm.dtype)]))
            padded_tvs.append(torch.cat([t_tv, torch.zeros(pad_len, 1, dtype=t_tv.dtype)]))
            target_masks.append(torch.cat([torch.zeros(target_sizes[i], dtype=torch.bool),
                                            torch.ones(pad_len, dtype=torch.bool)]))
        else:
            padded_taus.append(t_tau)
            padded_logms.append(t_logm)
            padded_yATMs.append(t_yatm)
            padded_tvs.append(t_tv)
            target_masks.append(torch.zeros(target_sizes[i], dtype=torch.bool))

    return (
        torch.stack(padded_refs).to(device),
        torch.stack(ref_masks).to(device),
        torch.stack(padded_taus).to(device),
        torch.stack(padded_logms).to(device),
        torch.stack(padded_yATMs).to(device),
        torch.stack(padded_tvs).to(device),
        torch.stack(target_masks).to(device),
    )


def train_one_epoch(model, surfaces, loss_fn, optimizer, n_reference, batch_size, device, gradient_clip=1.0):
    """Train for one epoch over daily surfaces."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    indices = list(range(len(surfaces)))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch = [surfaces[i] for i in batch_idx]

        ref_set, ref_mask, t_tau, t_logm, t_yATM, t_tv, t_mask = collate_surfaces(
            batch, n_reference, device
        )

        tv_pred, grad_tau, grad_logm, grad_logm2 = model(
            ref_set, t_tau, t_logm, t_yATM, ref_mask=ref_mask
        )

        loss_total, mse, cal, but, price = loss_fn(
            tv_pred, t_tv, t_logm, grad_tau, grad_logm, grad_logm2,
            valid_mask=~t_mask
        )

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss_total.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, surfaces, loss_fn, n_reference, batch_size, device, seed=9000):
    """Evaluate on held-out surfaces with deterministic, unbiased reference
    sampling (seeded randperm — matches the training distribution instead of
    the draft's first-50 short-maturity corner)."""
    model.eval()
    all_preds = []
    all_true = []
    all_taus = []
    total_loss = 0.0
    n_batches = 0

    generator = torch.Generator().manual_seed(seed)

    for start in range(0, len(surfaces), batch_size):
        batch = surfaces[start:start + batch_size]
        ref_set, ref_mask, t_tau, t_logm, t_yATM, t_tv, t_mask = collate_surfaces(
            batch, n_reference, device, generator=generator
        )

        tv_pred, grad_tau, grad_logm, grad_logm2 = model(
            ref_set, t_tau, t_logm, t_yATM, ref_mask=ref_mask
        )

        loss_total, _, _, _, _ = loss_fn(
            tv_pred, t_tv, t_logm, grad_tau, grad_logm, grad_logm2,
            valid_mask=~t_mask
        )
        total_loss += loss_total.item()
        n_batches += 1

        valid = ~t_mask  # (B, max_target)
        all_preds.append(tv_pred.detach()[valid].cpu().numpy().flatten())
        all_true.append(t_tv.detach()[valid].cpu().numpy().flatten())
        all_taus.append(t_tau.detach()[valid].cpu().numpy().flatten())

    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    taus = np.concatenate(all_taus)

    rmse = compute_rmse(preds, true)
    mape = compute_mape(preds, true)
    iv_rmse = compute_iv_rmse(preds, true, np.clip(taus, 1e-6, None))
    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, rmse, mape, iv_rmse


def compute_violation_rates(model, surfaces, n_reference, device,
                            n_logm=41, seed=9100):
    """Arbitrage-violation rates of the model's surfaces on dense query grids.

    For each day: reference = seeded random n_reference options; queries =
    (day's unique taus x linspace over day's observed logm range). Counts
    calendar (dw/dtau < 0) and butterfly (Gatheral g(k) < 0) violations.
    """
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    cal_viol = 0
    but_viol = 0
    total = 0

    for tau, logm, tv, yatm in surfaces:
        n = tau.shape[0]
        perm = torch.randperm(n, generator=generator)
        ref_idx = perm[:min(n_reference, n)]
        ref_data = torch.cat([tau[ref_idx], logm[ref_idx], tv[ref_idx]],
                             dim=-1).unsqueeze(0).to(device)

        taus_u = torch.unique(tau.flatten())
        if taus_u.numel() > 6:
            idx = torch.linspace(0, taus_u.numel() - 1, 6).long()
            taus_u = taus_u[idx]
        logm_grid = torch.linspace(logm.min().item(), logm.max().item(), n_logm,
                                   dtype=tau.dtype)

        tt, ll = torch.meshgrid(taus_u, logm_grid, indexing='ij')
        q_tau = tt.reshape(-1, 1).unsqueeze(0).to(device)
        q_logm = ll.reshape(-1, 1).unsqueeze(0).to(device)
        # yATM: interpolate day's y_atm at query taus (nearest observed tau)
        flat_tau = tau.flatten()
        near_idx = (flat_tau.unsqueeze(0) - tt.reshape(-1, 1)).abs().argmin(dim=1)
        q_yatm = yatm[near_idx].reshape(1, -1, 1).to(device)

        tv_pred, grad_tau, grad_logm, grad_logm2 = model(ref_data, q_tau, q_logm, q_yatm)

        w_safe = tv_pred.clamp(min=1e-8)
        g_k = (1 - (q_logm * grad_logm) / (2 * w_safe)) ** 2 \
              - grad_logm ** 2 / 4 * (1 / w_safe + 0.25) \
              + grad_logm2 / 2

        cal_viol += (grad_tau < 0).sum().item()
        but_viol += (g_k < 0).sum().item()
        total += tv_pred.numel()

    return cal_viol / max(total, 1), but_viol / max(total, 1), total


def save_fit_plot(model, surfaces, n_reference, device, path, seed=9200):
    """Scatter of predicted vs true total variance + one day's smile fit."""
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    batch = surfaces[:64]
    ref_set, ref_mask, t_tau, t_logm, t_yATM, t_tv, t_mask = collate_surfaces(
        batch, n_reference, device, generator=generator)
    tv_pred, _, _, _ = model(ref_set, t_tau, t_logm, t_yATM, ref_mask=ref_mask)
    valid = ~t_mask
    p = tv_pred.detach()[valid].cpu().numpy()
    t = t_tv.detach()[valid].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(t, p, s=2, alpha=0.3)
    lim = [0, max(t.max(), p.max()) * 1.05]
    axes[0].plot(lim, lim, 'r--', lw=1)
    axes[0].set_xlabel('true total variance')
    axes[0].set_ylabel('predicted total variance')
    axes[0].set_title('HyperIV test fit')

    # One day's smile: shortest-tau slice of the first surface
    tau0, logm0, tv0, yatm0 = batch[0]
    t_min = tau0.min()
    m = (tau0 == t_min).flatten()
    ref0 = torch.cat([tau0[:n_reference], logm0[:n_reference], tv0[:n_reference]],
                     dim=-1).unsqueeze(0).to(device)
    q_logm = torch.linspace(logm0.min().item(), logm0.max().item(), 81,
                            dtype=tau0.dtype).reshape(1, -1, 1).to(device)
    q_tau = torch.full_like(q_logm, t_min.item())
    q_yatm = torch.full_like(q_logm, yatm0[m][0].item())
    smile, _, _, _ = model(ref0, q_tau, q_logm, q_yatm)
    axes[1].scatter(logm0[m].numpy(), tv0[m].numpy(), s=12, label='observed')
    axes[1].plot(q_logm.flatten().cpu().numpy(),
                 smile.detach().flatten().cpu().numpy(), 'r-', label='HyperIV')
    axes[1].set_xlabel('log-moneyness')
    axes[1].set_ylabel('total variance')
    axes[1].set_title(f'Smile fit, shortest tau = {t_min.item():.3f}')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_checkpoint_state(path, device):
    """Load either the new dict format or a legacy raw state_dict."""
    obj = torch.load(path, map_location=device, weights_only=True)
    if isinstance(obj, dict) and 'state_dict' in obj:
        return obj['state_dict']
    return obj


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--finetune", type=str, default=None,
                        help='Path to checkpoint for fine-tuning (transfer learning)')
    args = parser.parse_args()

    config = load_config('config.ini')
    seed = config['training'].getint('seed')
    set_seed(seed)

    log_dir = config['save_path']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir, 'hyperiv')

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")

    hiv_cfg = config['hyperiv']
    dtype_name = hiv_cfg.get('dtype', 'float32')
    dtype = torch.float64 if dtype_name == 'float64' else torch.float32
    torch.set_default_dtype(dtype)

    embed_dim = hiv_cfg.getint('embed_dim')
    n_heads = hiv_cfg.getint('transformer_heads')
    n_layers = hiv_cfg.getint('transformer_layers')
    target_hidden_dims = tuple(int(x) for x in hiv_cfg['target_hidden_dims'].split(','))
    n_reference = hiv_cfg.getint('n_reference')
    epochs = args.epochs if args.epochs is not None else hiv_cfg.getint('epochs')
    lr = hiv_cfg.getfloat('learning_rate')
    batch_size = hiv_cfg.getint('batch_size')
    gradient_clip = config['training'].getfloat('gradient_clip', fallback=1.0)

    w_mse = hiv_cfg.getfloat('w_mse', fallback=1.0)
    w_calendar = hiv_cfg.getfloat('w_calendar', fallback=10.0)
    w_butterfly = hiv_cfg.getfloat('w_butterfly', fallback=10.0)
    w_price = hiv_cfg.getfloat('w_price', fallback=0.1)

    # Data
    logger.info('Preprocessing data...')
    dp = DataProcessor(config)
    dp()

    train_end_date = parse_date(config['training']['train_end_date'])
    test_start_date = parse_date(config['training']['test_start_date'])

    surfaces = dp.Prepare_hyperiv_data()

    # Split: surfaces before train_end_date for train, after for test
    train_surfaces = []
    test_surfaces = []
    for date, surface in surfaces:
        surface = tuple(x.to(dtype) for x in surface)
        if date <= train_end_date:
            train_surfaces.append(surface)
        elif date >= test_start_date:
            test_surfaces.append(surface)

    # Further split train into train/val (chronological: last 20%)
    n_train = int(len(train_surfaces) * 0.8)
    val_surfaces = train_surfaces[n_train:]
    train_surfaces = train_surfaces[:n_train]

    logger.info(f'Train surfaces: {len(train_surfaces)}, Val: {len(val_surfaces)}, Test: {len(test_surfaces)}')

    # Input standardization stats from TRAIN surfaces only
    all_feats = torch.cat([
        torch.cat([tau, logm, tv], dim=-1)
        for tau, logm, tv, _ in train_surfaces
    ], dim=0)
    feat_mean = all_feats.mean(dim=0)
    feat_std = all_feats.std(dim=0)
    logger.info(f'Feature stats (tau, logm, tv): mean={feat_mean.tolist()}, std={feat_std.tolist()}')

    # Model
    model = HyperIVModel(
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_transformer_layers=n_layers,
        target_hidden_dims=target_hidden_dims,
    ).to(dtype).to(device)
    model.set_normalization(feat_mean, feat_std)

    loss_fn = HyperIVLoss(w_mse=w_mse, w_calendar=w_calendar,
                          w_butterfly=w_butterfly, w_price=w_price)

    # Fine-tuning: load pretrained weights
    if args.finetune:
        if not os.path.exists(args.finetune):
            raise FileNotFoundError(f'--finetune checkpoint not found: {args.finetune}')
        from transfer import load_finetune_weights, setup_finetune_optimizer
        logger.info(f'Fine-tuning from: {args.finetune}')
        transferred, reinitialized = load_finetune_weights(model, args.finetune, device)
        optimizer = setup_finetune_optimizer(model, transferred, reinitialized,
                                             base_lr=lr * 0.1, new_lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    metrics = MetricsTracker()
    early_stopping = EarlyStopping(patience=50)
    model_path = config['save_path'].get('hyperiv_model_path', '../models/HyperIVModel.pt')
    best_val_loss = float('inf')

    def save_checkpoint():
        torch.save({
            'state_dict': model.state_dict(),
            'feat_mean': feat_mean,
            'feat_std': feat_std,
            'config': dict(hiv_cfg),
        }, model_path)

    # Spike guard (Model 1's documented gradient-explosion practice): a single
    # violent penalty batch was measured to collapse the model irrecoverably
    # (softplus saturation). On a >3x train-loss spike, restore the last
    # stable weights and halve the learning rate.
    import copy
    best_train_loss = float('inf')
    stable_state = None

    logger.info(f'Training HyperIV ({dtype_name} on {device})...')
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_surfaces, loss_fn, optimizer,
            n_reference, batch_size, device, gradient_clip=gradient_clip
        )

        if stable_state is not None and train_loss > 3 * best_train_loss:
            model.load_state_dict(stable_state)
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
            logger.info(f'Epoch {epoch+1}: train-loss spike '
                        f'({train_loss:.2e} > 3x{best_train_loss:.2e}) — '
                        f'restored stable weights, halved LR')
            continue
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            stable_state = copy.deepcopy(model.state_dict())

        val_loss, val_rmse, val_mape, val_iv_rmse = evaluate(
            model, val_surfaces, loss_fn, n_reference, batch_size, device
        )
        scheduler.step()

        is_best = metrics.update(epoch, train_loss, val_loss)
        if is_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint()

        if (epoch + 1) % 10 == 0:
            logger.info(
                f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - '
                f'Val: {val_loss:.6f} (RMSE={val_rmse:.6f}, MAPE={val_mape:.4f}, IV-RMSE={val_iv_rmse:.6f})'
            )

        if early_stopping.step(val_loss):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    # Test evaluation
    results = {
        'dtype': dtype_name,
        'n_train': len(train_surfaces),
        'n_val': len(val_surfaces),
        'n_test': len(test_surfaces),
        'loss_weights': {'w_mse': w_mse, 'w_calendar': w_calendar,
                         'w_butterfly': w_butterfly, 'w_price': w_price},
        'best_val_loss': best_val_loss,
        'best_epoch': metrics.best_epoch,
    }
    if test_surfaces:
        logger.info('Evaluating on test set...')
        model.load_state_dict(load_checkpoint_state(model_path, device))
        test_loss, test_rmse, test_mape, test_iv_rmse = evaluate(
            model, test_surfaces, loss_fn, n_reference, batch_size, device
        )
        logger.info(f'Test - Loss: {test_loss:.6f}, RMSE: {test_rmse:.6f}, '
                     f'MAPE: {test_mape:.4f}, IV-RMSE: {test_iv_rmse:.6f}')

        cal_rate, but_rate, n_grid = compute_violation_rates(
            model, test_surfaces, n_reference, device)
        logger.info(f'Test violation rates over {n_grid} grid points - '
                     f'calendar: {cal_rate:.4%}, butterfly: {but_rate:.4%}')

        save_fit_plot(model, test_surfaces, n_reference, device,
                      os.path.join(log_dir, 'hyperiv_test_fit.png'))

        results.update({
            'test_loss': test_loss,
            'test_rmse': float(test_rmse),
            'test_mape': float(test_mape),
            'test_iv_rmse': float(test_iv_rmse),
            'calendar_violation_rate': cal_rate,
            'butterfly_violation_rate': but_rate,
            'violation_grid_points': n_grid,
        })

    with open(os.path.join(log_dir, 'hyperiv_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    metrics.save(os.path.join(log_dir, 'hyperiv_metrics.json'))
    logger.info('HyperIV training complete.')


if __name__ == '__main__':
    main()
