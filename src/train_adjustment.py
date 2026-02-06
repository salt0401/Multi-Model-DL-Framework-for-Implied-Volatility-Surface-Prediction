"""Adjustment model training pipeline.

Full pipeline:
1. Load trained base model, compute tv_pred on all data
2. Load VIX data, compute features
3. Build padded time-series sequences per option
4. Oversample crisis periods
5. Fit KDE on targets for loss weighting
6. Train GRU+Attention model on crisis periods
7. Evaluate on held-out crisis (e.g., 2020/03 COVID)
8. Integration: detect structural break -> apply adjustment only when break detected
"""
from model import MultiModel
from dataset import DataProcessor
from adjustment import TVAdjustmentModel, AdjustmentLoss
from structural_break import detect_structural_break
from utils import (load_config, parse_list_config, parse_date, set_seed,
                   setup_logging, MetricsTracker, EarlyStopping, compute_rmse, compute_mape)

from argparse import ArgumentParser
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os


def oversample_crisis_indices(dates, event_dates, factor=5):
    """Return indices with crisis periods oversampled by factor.

    Args:
        dates: array of date strings (YYYYMM format check)
        event_dates: list of YYYYMM strings for crisis periods
        factor: oversampling multiplier

    Returns:
        numpy array of indices (with crisis indices repeated)
    """
    indices = list(range(len(dates)))
    crisis_mask = np.zeros(len(dates), dtype=bool)

    for event in event_dates:
        year = int(event[:4])
        month = int(event[4:6])
        for i, d in enumerate(dates):
            if hasattr(d, 'year'):
                if d.year == year and d.month == month:
                    crisis_mask[i] = True

    crisis_indices = np.where(crisis_mask)[0]
    oversampled = np.concatenate([
        np.array(indices),
        np.tile(crisis_indices, factor - 1)
    ])
    return oversampled


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    args = parser.parse_args()

    config = load_config('config.ini')
    seed = config['training'].getint('seed')
    set_seed(seed)

    log_dir = config['save_path']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir, 'adjustment')

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")
    torch.set_default_dtype(torch.float64)

    adj_cfg = config['adjustment']

    # Step 1: Load trained base model
    logger.info('Loading base model...')
    hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
    ensemble_num = config['model_sett'].getint('ensemble_num')
    base_model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)

    model_path = config['save_path']['model_path']
    if not os.path.exists(model_path):
        logger.error(f'Base model not found at {model_path}. Train it first.')
        return
    base_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    base_model.eval()

    # Step 2: Load data and compute features
    logger.info('Preparing data...')
    dp = DataProcessor(config)
    dp()

    sequence_length = adj_cfg.getint('sequence_length')
    sequences, targets, masks = dp.prepare_adjustment_data(base_model, device, sequence_length)

    if sequences is None:
        logger.error('Failed to prepare adjustment data (VIX file missing?)')
        return

    logger.info(f'Adjustment data shape: sequences={sequences.shape}, targets={targets.shape}')

    # Step 4: Fit KDE weights
    logger.info('Fitting KDE weights...')
    loss_fn = AdjustmentLoss(
        kde_bandwidth=adj_cfg.getfloat('kde_bandwidth'),
        mape_weight=0.5
    )
    kde_weights = loss_fn.fit_kde_weights(targets.numpy())

    # Step 5: Create DataLoader
    batch_size = adj_cfg.getint('batch_size')
    weight_tensor = torch.tensor(kde_weights, dtype=torch.float64)

    dataset = TensorDataset(sequences, targets, masks, weight_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Step 6: Build model
    logger.info('Building adjustment model...')
    prediction_target = adj_cfg.get('prediction_target', 'ratio')
    adj_model = TVAdjustmentModel(
        input_dim=sequences.shape[-1],
        gru_hidden_dim=adj_cfg.getint('gru_hidden_dim'),
        gru_layers=adj_cfg.getint('gru_layers'),
        attention_heads=adj_cfg.getint('attention_heads'),
        dropout=adj_cfg.getfloat('dropout'),
        prediction_target=prediction_target,
    ).double().to(device)

    optimizer = optim.Adam(adj_model.parameters(), lr=adj_cfg.getfloat('learning_rate'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
    epochs = adj_cfg.getint('epochs')

    metrics = MetricsTracker()
    early_stopping = EarlyStopping(patience=100)
    adj_model_path = config['save_path']['adjustment_model_path']
    best_val_loss = float('inf')

    # Step 7: Train
    logger.info('Training adjustment model...')
    for epoch in range(epochs):
        adj_model.train()
        train_loss_sum = 0.0
        n_train = 0

        for seq_batch, target_batch, mask_batch, weight_batch in train_loader:
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)
            mask_batch = mask_batch.to(device)
            weight_batch = weight_batch.to(device)

            pred = adj_model(seq_batch, mask_batch)
            loss = loss_fn(pred, target_batch, weight_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adj_model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(seq_batch)
            n_train += len(seq_batch)

        train_loss = train_loss_sum / max(n_train, 1)

        # Validation
        adj_model.eval()
        val_loss_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for seq_batch, target_batch, mask_batch, weight_batch in val_loader:
                seq_batch = seq_batch.to(device)
                target_batch = target_batch.to(device)
                mask_batch = mask_batch.to(device)

                pred = adj_model(seq_batch, mask_batch)
                loss = loss_fn(pred, target_batch)
                val_loss_sum += loss.item() * len(seq_batch)
                n_val += len(seq_batch)

        val_loss = val_loss_sum / max(n_val, 1)
        scheduler.step(val_loss)

        is_best = metrics.update(epoch, train_loss, val_loss)
        if is_best and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(adj_model.state_dict(), adj_model_path)

        if (epoch + 1) % 50 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - Val: {val_loss:.6f}')

        if early_stopping.step(val_loss):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    logger.info(f'Best validation loss: {best_val_loss:.6f}')

    # Step 8: Final evaluation
    logger.info('Evaluating adjustment model...')
    adj_model.load_state_dict(torch.load(adj_model_path, map_location=device, weights_only=True))
    adj_model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for seq_batch, target_batch, mask_batch, _ in val_loader:
            seq_batch = seq_batch.to(device)
            mask_batch = mask_batch.to(device)
            pred = adj_model(seq_batch, mask_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target_batch.numpy())

    preds = np.concatenate(all_preds).flatten()
    true = np.concatenate(all_targets).flatten()

    rmse = compute_rmse(preds, true)
    mape = compute_mape(preds, true)
    logger.info(f'Adjustment model - RMSE: {rmse:.6f}, MAPE: {mape:.4f}')

    # Save metrics
    metrics.save(os.path.join(log_dir, 'adjustment_metrics.json'))
    logger.info('Adjustment model training complete.')


if __name__ == '__main__':
    main()
