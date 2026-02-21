"""Training script for experimental adjustment models (xLSTM, TFT, baseline GRU).

Trains models on the same data pipeline as the original adjustment model,
using identical chronological split and KDE-weighted loss for fair comparison.

Usage (run from model2_research/ directory):
    python train_models.py --model xlstm
    python train_models.py --model tft
    python train_models.py --model baseline   # original GRU for comparison

GPU is required by default. Use --allow_cpu only if explicitly approved.

Models and logs are saved under model2_research/ to avoid mixing
with the main src/ directory's production models.
"""
import sys
import os

# Add src/ to import path for shared modules
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, _src_dir)

from model import MultiModel
from dataset import DataProcessor
from adjustment import AdjustmentLoss, TVAdjustmentModel
from utils import (load_config, parse_list_config, set_seed,
                   setup_logging, MetricsTracker, EarlyStopping,
                   compute_rmse, compute_mape)

from xlstm_adjustment import xLSTMAdjustmentModel
from tft_adjustment import TFTAdjustmentModel
from optimizers import CautiousAdamW, AdamCPR

from argparse import ArgumentParser
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from datetime import datetime


MODEL_REGISTRY = {
    'xlstm': xLSTMAdjustmentModel,
    'tft': TFTAdjustmentModel,
    'baseline': TVAdjustmentModel,
}


def build_model(model_type, input_dim, adj_cfg):
    """Factory: create model by type with config-driven hyperparameters.

    Args:
        model_type: 'xlstm', 'tft', or 'baseline'
        input_dim: number of input features (6 base or 12 with enhancement)
        adj_cfg: config section with hyperparameters

    Returns:
        nn.Module instance
    """
    hidden_dim = adj_cfg.getint('gru_hidden_dim')
    dropout = adj_cfg.getfloat('dropout')
    prediction_target = adj_cfg.get('prediction_target', 'ratio')
    attention_heads = adj_cfg.getint('attention_heads')

    if model_type == 'xlstm':
        return xLSTMAdjustmentModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=adj_cfg.getint('gru_layers'),
            attention_heads=attention_heads,
            dropout=dropout,
            prediction_target=prediction_target,
        )
    elif model_type == 'tft':
        return TFTAdjustmentModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            lstm_layers=adj_cfg.getint('gru_layers'),
            attention_heads=attention_heads,
            dropout=max(dropout - 0.1, 0.05),  # TFT uses slightly lower dropout
            prediction_target=prediction_target,
        )
    elif model_type == 'baseline':
        return TVAdjustmentModel(
            input_dim=input_dim,
            gru_hidden_dim=hidden_dim,
            gru_layers=adj_cfg.getint('gru_layers'),
            attention_heads=attention_heads,
            dropout=dropout,
            prediction_target=prediction_target,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use: {list(MODEL_REGISTRY.keys())}")


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = ArgumentParser(description='Train experimental adjustment models')
    parser.add_argument('--model', type=str, required=True,
                        choices=['xlstm', 'tft', 'baseline'],
                        help='Model architecture to train')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'cwd', 'cpr'],
                        help='Optimizer: adam (baseline), adamw, cwd (CautiousAdamW), cpr (AdamCPR)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for adamw/cwd (default: 0.01)')
    parser.add_argument('--allow_cpu', action='store_true',
                        help='Allow CPU training (GPU required by default)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs from config')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (default: 256 for GPU)')
    args = parser.parse_args()

    # Default to larger batch size for GPU (better utilization)
    if args.batch_size is None:
        args.batch_size = 256

    # --- GPU enforcement ---
    if not torch.cuda.is_available() and not args.allow_cpu:
        print("ERROR: No GPU detected. GPU is required for training.")
        print("       If you have explicit approval to use CPU, add --allow_cpu flag.")
        sys.exit(1)

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    torch.set_default_dtype(torch.float64)

    # --- Config ---
    config_path = os.path.join(_src_dir, 'config.ini')
    config = load_config(config_path)
    seed = config['training'].getint('seed')
    set_seed(seed)

    adj_cfg = config['adjustment']
    epochs = args.epochs or adj_cfg.getint('epochs')
    lr = args.lr or adj_cfg.getfloat('learning_rate')
    batch_size = args.batch_size

    # --- Logging ---
    research_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(research_dir, 'logs')
    model_dir = os.path.join(research_dir, 'models')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger = setup_logging(log_dir, f'{args.model}_adjustment')
    logger.info(f'=== Training {args.model.upper()} Adjustment Model ===')
    logger.info(f'Device: {device} (GPU: {torch.cuda.get_device_name(0) if use_gpu else "N/A"})')

    # --- Step 1: Load base model ---
    logger.info('Loading base model (Model 1 SSVI+NN)...')
    hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
    ensemble_num = config['model_sett'].getint('ensemble_num')
    base_model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)

    model_path = os.path.join(_src_dir, '..', config['save_path']['model_path'])
    if not os.path.exists(model_path):
        # Try relative to src dir
        model_path = os.path.join(_src_dir, config['save_path']['model_path'])
    if not os.path.exists(model_path):
        logger.error(f'Base model not found. Searched: {model_path}')
        sys.exit(1)

    base_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    base_model.train(False)
    logger.info(f'Base model loaded from {model_path}')

    # --- Step 2: Prepare data ---
    logger.info('Preparing adjustment data...')
    # DataProcessor expects to be run from src/ dir for relative paths
    original_cwd = os.getcwd()
    os.chdir(_src_dir)
    dp = DataProcessor(config)
    dp()

    sequence_length = adj_cfg.getint('sequence_length')
    sequences, targets, masks, seq_dates = dp.prepare_adjustment_data(
        base_model, device, sequence_length
    )
    os.chdir(original_cwd)

    if sequences is None:
        logger.error('Failed to prepare adjustment data')
        sys.exit(1)

    input_dim = sequences.shape[-1]
    logger.info(f'Data shape: sequences={sequences.shape}, targets={targets.shape}, '
                f'input_dim={input_dim}')

    # --- Step 3: Chronological split ---
    unique_dates = np.sort(np.unique(seq_dates))
    split_idx = int(len(unique_dates) * 0.8)
    val_start_date = unique_dates[split_idx]

    train_mask = seq_dates < val_start_date
    val_mask = seq_dates >= val_start_date

    logger.info(f'Chronological split: train < {val_start_date}, '
                f'train={train_mask.sum()}, val={val_mask.sum()}')

    # --- Step 4: KDE weights (train-only) ---
    logger.info('Fitting KDE weights...')
    loss_fn = AdjustmentLoss(
        kde_bandwidth=adj_cfg.getfloat('kde_bandwidth'),
        mape_weight=0.5,
    )
    train_kde_weights = loss_fn.fit_kde_weights(targets[train_mask].numpy())
    val_kde_weights = loss_fn.eval_kde_weights(targets[val_mask].numpy())

    # --- Step 5: DataLoaders ---
    train_ds = TensorDataset(
        sequences[train_mask], targets[train_mask], masks[train_mask],
        torch.tensor(train_kde_weights, dtype=torch.float64),
    )
    val_ds = TensorDataset(
        sequences[val_mask], targets[val_mask], masks[val_mask],
        torch.tensor(val_kde_weights, dtype=torch.float64),
    )
    loader_kwargs = {'pin_memory': True, 'num_workers': 0} if use_gpu else {}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    # --- Step 6: Build model ---
    logger.info(f'Building {args.model} model...')
    model = build_model(args.model, input_dim, adj_cfg)
    model = model.double().to(device)

    total_params, trainable_params = count_parameters(model)
    logger.info(f'Model parameters: {total_params:,} total, {trainable_params:,} trainable')

    # --- Build optimizer ---
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'cwd':
        optimizer = CautiousAdamW(model.parameters(), lr=lr,
                                  weight_decay=args.weight_decay, cautious=True)
    elif args.optimizer == 'cpr':
        optimizer = AdamCPR(model.parameters(), lr=lr,
                            kappa_init_method='warm_start', kappa_init_param=1.0,
                            mu=0.01, lam_lr=0.1)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    logger.info(f'Optimizer: {args.optimizer} (wd={args.weight_decay if args.optimizer in ["adamw","cwd"] else "N/A"})')

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    metrics = MetricsTracker()
    early_stopping = EarlyStopping(patience=100)
    opt_suffix = f'_{args.optimizer}' if args.optimizer != 'adam' else ''
    save_path = os.path.join(model_dir, f'{args.model}{opt_suffix}_AdjustmentModel.pt')
    best_val_loss = float('inf')

    # --- Step 7: Train ---
    logger.info(f'Training: epochs={epochs}, lr={lr}, batch_size={batch_size}')
    logger.info(f'Save path: {save_path}')

    train_start_time = datetime.now()

    for epoch in range(epochs):
        # --- Train phase ---
        model.train()
        train_loss_sum = 0.0
        n_train = 0

        for seq_batch, target_batch, mask_batch, weight_batch in train_loader:
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            weight_batch = weight_batch.to(device)

            pred = model(seq_batch, mask_batch)
            loss = loss_fn(pred, target_batch, weight_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(seq_batch)
            n_train += len(seq_batch)

        train_loss = train_loss_sum / max(n_train, 1)

        # --- Validation phase ---
        model.train(False)
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for seq_batch, target_batch, mask_batch, weight_batch in val_loader:
                seq_batch = seq_batch.to(device)
                target_batch = target_batch.to(device)
                mask_batch = mask_batch.to(device)

                pred = model(seq_batch, mask_batch)
                loss = loss_fn(pred, target_batch)
                val_loss_sum += loss.item() * len(seq_batch)
                n_val += len(seq_batch)

        val_loss = val_loss_sum / max(n_val, 1)
        scheduler.step(val_loss)

        is_best = metrics.update(epoch, train_loss, val_loss)
        if is_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0:
            elapsed = (datetime.now() - train_start_time).total_seconds() / 60
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f'Epoch {epoch+1}/{epochs} - '
                f'Train: {train_loss:.6f} - Val: {val_loss:.6f} - '
                f'Best: {best_val_loss:.6f} - LR: {current_lr:.2e} - '
                f'Time: {elapsed:.1f}min'
            )

        if early_stopping.step(val_loss):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    total_time = (datetime.now() - train_start_time).total_seconds() / 60
    logger.info(f'Training complete: {total_time:.1f} minutes, best val loss: {best_val_loss:.6f}')

    # --- Step 8: Final evaluation ---
    logger.info('Final evaluation on validation set...')
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.train(False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for seq_batch, target_batch, mask_batch, _ in val_loader:
            seq_batch = seq_batch.to(device)
            mask_batch = mask_batch.to(device)
            pred = model(seq_batch, mask_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target_batch.numpy())

    preds = np.concatenate(all_preds).flatten()
    true = np.concatenate(all_targets).flatten()

    rmse = compute_rmse(preds, true)
    mape = compute_mape(preds, true)
    logger.info(f'{args.model.upper()} model - RMSE: {rmse:.6f}, MAPE: {mape:.4f}')

    # --- Step 9: Save results ---
    results = {
        'model_type': args.model,
        'optimizer': args.optimizer,
        'weight_decay': args.weight_decay if args.optimizer in ['adamw', 'cwd'] else None,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'best_val_loss': best_val_loss,
        'best_epoch': metrics.best_epoch,
        'final_rmse': float(rmse),
        'final_mape': float(mape),
        'training_minutes': round(total_time, 1),
        'device': str(device),
        'input_dim': input_dim,
        'hidden_dim': adj_cfg.getint('gru_hidden_dim'),
        'epochs_run': len(metrics.train_losses),
        'learning_rate': lr,
        'batch_size': batch_size,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(log_dir, f'{args.model}{opt_suffix}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Results saved to {results_path}')

    # Save training curves
    metrics.save(os.path.join(log_dir, f'{args.model}{opt_suffix}_metrics.json'))

    # --- Step 10: TFT interpretability analysis ---
    if args.model == 'tft':
        logger.info('Running TFT interpretability analysis...')
        feature_names = ['vix_change', 'underlying_return', 'logm', 'tau', 'tv_pred', 'itm_otm']
        if input_dim > 6:
            feature_names += ['sp500_return', 'iv_term_slope', 'iv_skew',
                              'vrp_20d', 'futures_basis_pct', 'rv_20d'][:input_dim - 6]

        # Sample a batch for analysis
        sample_seq = sequences[val_mask][:256].to(device)
        sample_mask = masks[val_mask][:256].to(device)

        avg_var, avg_attn = model.get_feature_importance(sample_seq, sample_mask)
        avg_var = avg_var.cpu().numpy()   # (T, J)
        avg_attn = avg_attn.cpu().numpy() # (H, T)

        # Log feature importance (average across time)
        mean_importance = avg_var.mean(axis=0)  # (J,)
        logger.info('Feature importance (avg across timesteps):')
        for name, imp in sorted(zip(feature_names, mean_importance),
                                 key=lambda x: -x[1]):
            logger.info(f'  {name}: {imp:.4f}')

        # Log temporal attention pattern
        mean_attn = avg_attn.mean(axis=0)  # (T,)
        logger.info(f'Temporal attention (last 5 steps): '
                    f'{mean_attn[-5:].tolist()}')

        # Save interpretability data
        interp_data = {
            'feature_names': feature_names,
            'variable_importance_per_timestep': avg_var.tolist(),
            'mean_variable_importance': mean_importance.tolist(),
            'attention_per_head': avg_attn.tolist(),
            'mean_temporal_attention': mean_attn.tolist(),
        }
        interp_path = os.path.join(log_dir, 'tft_interpretability.json')
        with open(interp_path, 'w') as f:
            json.dump(interp_data, f, indent=2)
        logger.info(f'Interpretability data saved to {interp_path}')

    logger.info('Done.')


if __name__ == '__main__':
    main()
