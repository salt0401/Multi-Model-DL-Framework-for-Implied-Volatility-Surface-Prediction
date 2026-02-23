"""Training script for HyperIV model.

Each training batch = one day's observed options surface.
Reference set = random subset of observed options (n_reference).
Target set = remaining options from same day.
"""
from hyperiv import HyperIVModel, HyperIVLoss
from dataset import DataProcessor
from utils import (load_config, parse_list_config, parse_date, set_seed,
                   setup_logging, MetricsTracker, EarlyStopping,
                   compute_rmse, compute_mape, compute_iv_rmse)

from argparse import ArgumentParser

import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import os


def collate_surfaces(batch, n_reference, device):
    """Collate a list of per-day tensors into padded batches.

    Args:
        batch: list of (tau, logm, total_var, y_atm) tensors, each (n_options, 1)
        n_reference: number of reference options to sample per surface
        device: torch device

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
            perm = torch.randperm(n)
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

        # Zero out padded targets in loss
        valid = (~t_mask).unsqueeze(-1).float()
        loss_total, mse, cal, but = loss_fn(
            tv_pred * valid, t_tv * valid, t_logm * valid,
            grad_tau * valid, grad_logm * valid, grad_logm2 * valid
        )

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        total_loss += loss_total.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, surfaces, loss_fn, n_reference, device):
    """Evaluate on held-out surfaces."""
    model.eval()
    all_preds = []
    all_true = []
    all_taus = []
    total_loss = 0.0
    n_batches = 0

    for surface in surfaces:
        tau, logm, tv, yatm = surface
        n = tau.shape[0]
        if n <= n_reference:
            ref_idx = list(range(n))
            target_idx = list(range(n))
        else:
            ref_idx = list(range(n_reference))
            target_idx = list(range(n_reference, n))

        ref_data = torch.cat([tau[ref_idx], logm[ref_idx], tv[ref_idx]], dim=-1).unsqueeze(0).to(device)
        t_tau = tau[target_idx].unsqueeze(0).to(device)
        t_logm = logm[target_idx].unsqueeze(0).to(device)
        t_yATM = yatm[target_idx].unsqueeze(0).to(device)
        t_tv = tv[target_idx].unsqueeze(0).to(device)

        tv_pred, grad_tau, grad_logm, grad_logm2 = model(ref_data, t_tau, t_logm, t_yATM)

        loss_total, _, _, _ = loss_fn(tv_pred, t_tv, t_logm, grad_tau, grad_logm, grad_logm2)
        total_loss += loss_total.item()
        n_batches += 1

        all_preds.append(tv_pred.detach().cpu().numpy().flatten())
        all_true.append(t_tv.detach().cpu().numpy().flatten())
        all_taus.append(t_tau.detach().cpu().numpy().flatten())

    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    taus = np.concatenate(all_taus)

    rmse = compute_rmse(preds, true)
    mape = compute_mape(preds, true)
    iv_rmse = compute_iv_rmse(preds, true, taus)
    avg_loss = total_loss / max(n_batches, 1)

    return avg_loss, rmse, mape, iv_rmse


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
    torch.set_default_dtype(torch.float64)

    hiv_cfg = config['hyperiv']
    embed_dim = hiv_cfg.getint('embed_dim')
    n_heads = hiv_cfg.getint('transformer_heads')
    n_layers = hiv_cfg.getint('transformer_layers')
    target_hidden_dims = tuple(int(x) for x in hiv_cfg['target_hidden_dims'].split(','))
    n_reference = hiv_cfg.getint('n_reference')
    epochs = args.epochs if args.epochs is not None else hiv_cfg.getint('epochs')
    lr = hiv_cfg.getfloat('learning_rate')
    batch_size = hiv_cfg.getint('batch_size')

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
        if date <= train_end_date:
            train_surfaces.append(surface)
        elif date >= test_start_date:
            test_surfaces.append(surface)

    # Further split train into train/val (chronological: last 20%)
    n_train = int(len(train_surfaces) * 0.8)
    val_surfaces = train_surfaces[n_train:]
    train_surfaces = train_surfaces[:n_train]

    logger.info(f'Train surfaces: {len(train_surfaces)}, Val: {len(val_surfaces)}, Test: {len(test_surfaces)}')

    # Model
    model = HyperIVModel(
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_transformer_layers=n_layers,
        target_hidden_dims=target_hidden_dims,
    ).double().to(device)

    loss_fn = HyperIVLoss()

    # Fine-tuning: load pretrained weights
    if args.finetune and os.path.exists(args.finetune):
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

    logger.info('Training HyperIV...')
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_surfaces, loss_fn, optimizer,
            n_reference, batch_size, device
        )
        val_loss, val_rmse, val_mape, val_iv_rmse = evaluate(
            model, val_surfaces, loss_fn, n_reference, device
        )
        scheduler.step()

        is_best = metrics.update(epoch, train_loss, val_loss)
        if is_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - '
                f'Val: {val_loss:.6f} (RMSE={val_rmse:.6f}, MAPE={val_mape:.4f}, IV-RMSE={val_iv_rmse:.6f})'
            )

        if early_stopping.step(val_loss):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    # Test evaluation
    if test_surfaces:
        logger.info('Evaluating on test set...')
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        test_loss, test_rmse, test_mape, test_iv_rmse = evaluate(
            model, test_surfaces, loss_fn, n_reference, device
        )
        logger.info(f'Test - Loss: {test_loss:.6f}, RMSE: {test_rmse:.6f}, '
                     f'MAPE: {test_mape:.4f}, IV-RMSE: {test_iv_rmse:.6f}')

    metrics.save(os.path.join(log_dir, 'hyperiv_metrics.json'))
    logger.info('HyperIV training complete.')


if __name__ == '__main__':
    main()
