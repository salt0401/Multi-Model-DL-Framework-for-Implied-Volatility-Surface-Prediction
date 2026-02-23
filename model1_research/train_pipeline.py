import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

"""End-to-end training pipeline: SSVI+NN (Stage 1) -> Adjustment Model (Stage 2).

Usage:
    python train_pipeline.py --on_gpu [--stage1_epochs 3] [--stage2_epochs 1000] [--skip_stage1]

Reuses existing training functions from train.py, dataset.py, model.py, adjustment.py.
Adds SSVI diagnostics and transition validation between stages.
"""
from model import MultiModel, WeightedSumLoss
from dataset import DataProcessor
from train import train_one_epoch, validate
from adjustment import TVAdjustmentModel, AdjustmentLoss
from utils import (load_config, parse_list_config, parse_date, set_seed,
                   setup_logging, MetricsTracker, EarlyStopping, compute_rmse, compute_mape)

from argparse import ArgumentParser
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import os


# ---------------------------------------------------------------------------
# JSON encoder for numpy types
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# Step 1: SSVI Diagnostic Functions
# ---------------------------------------------------------------------------
def extract_ssvi_params(model):
    """Extract rho, eta, gamma from all ensemble members.

    Returns list of dicts with raw values, transformed values, and
    Gatheral-Jacquier butterfly arbitrage check.
    """
    params_list = []
    for i, single in enumerate(model.ensemble_list):
        ssvi = single.Prior
        entry = {'member': i}

        raw_rho = ssvi.raw_rho.item()
        rho = -torch.sigmoid(ssvi.raw_rho).item()
        entry['raw_rho'] = raw_rho
        entry['rho'] = rho

        if hasattr(ssvi, 'raw_eta'):
            raw_eta = ssvi.raw_eta.item()
            raw_gamma = ssvi.raw_gamma.item()
            eta = 2 * torch.sigmoid(ssvi.raw_eta).item()
            gamma = torch.sigmoid(ssvi.raw_gamma).item()
            gj = eta * (1 + abs(rho))

            entry.update({
                'raw_eta': raw_eta, 'eta': eta,
                'raw_gamma': raw_gamma, 'gamma': gamma,
                'gj_value': gj,
                'gj_satisfied': gj < 2,
            })
        elif hasattr(ssvi, 'raw_lambda'):
            raw_lam = ssvi.raw_lambda.item()
            lam = torch.exp(ssvi.raw_lambda).item()
            entry.update({'raw_lambda': raw_lam, 'lambda': lam})

        params_list.append(entry)
    return params_list


def check_ssvi_health(params_list, logger):
    """Validate SSVI params against known failure modes. Returns True if healthy."""
    healthy = True

    # Check all rho < 0 (equity left-skew)
    rhos = [p['rho'] for p in params_list]
    if not all(r < 0 for r in rhos):
        logger.warning(f'SSVI: Not all rho negative: {rhos}')
        healthy = False
    else:
        logger.info(f'SSVI: All rho negative (left-skew) - OK. Mean={np.mean(rhos):.4f}')

    # Check Gatheral-Jacquier constraint
    for p in params_list:
        if 'gj_value' in p:
            if not p['gj_satisfied']:
                logger.warning(f'SSVI member {p["member"]}: GJ violation! '
                               f'eta*(1+|rho|)={p["gj_value"]:.4f} >= 2')
                healthy = False

    # Check eta not near upper bound 2.0 (previous exp() explosion bug)
    for p in params_list:
        if 'eta' in p and p['eta'] > 1.95:
            logger.warning(f'SSVI member {p["member"]}: eta={p["eta"]:.4f} near upper bound 2.0')
            healthy = False

    # Check parameters moved from initialization
    init_rho = -torch.sigmoid(torch.tensor(0.0)).item()  # -0.5
    unmoved = sum(1 for r in rhos if abs(r - init_rho) < 1e-4)
    if unmoved == len(rhos):
        logger.warning('SSVI: All rho values at initialization (-0.5) - model may be untrained')
        healthy = False

    return healthy


def run_iv_surface_sanity_check(model, device, logger):
    """Quick surface shape test: verify left-skew at tau=0.5, yATM=0.05.

    Note: Cannot use torch.no_grad() here because SmileModel.forward() uses
    torch.autograd.grad(create_graph=True) internally for gradient computation.
    """
    model.train(False)
    tau = torch.tensor([[0.5]], dtype=torch.float64, device=device)
    yATM = torch.tensor([[0.05]], dtype=torch.float64, device=device)

    logm_values = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    results = []
    for lm in logm_values:
        logm_t = torch.tensor([[lm]], dtype=torch.float64, device=device)
        tv, _, _, _ = model(tau, logm_t, yATM)
        tv_val = tv.detach().item()
        iv = np.sqrt(tv_val / 0.5) if tv_val > 0 else float('nan')
        results.append({'logm': lm, 'total_var': tv_val, 'implied_vol': iv})

    # Log the surface
    logger.info('IV surface check (tau=0.5, yATM=0.05):')
    for r in results:
        logger.info(f'  logm={r["logm"]:+.1f}  TV={r["total_var"]:.6f}  IV={r["implied_vol"]:.4f}')

    # Verify left-skew: OTM put IV > OTM call IV
    tv_put = results[0]['total_var']   # logm=-0.3
    tv_call = results[-1]['total_var']  # logm=+0.3
    skew_ok = tv_put > tv_call
    if skew_ok:
        logger.info('IV surface: Left-skew confirmed (OTM put IV > OTM call IV)')
    else:
        logger.warning('IV surface: No left-skew detected!')

    return {'points': results, 'left_skew': skew_ok}


def validate_base_model_predictions(model, val_loader, device, logger):
    """Run base model on val set, check for NaN/Inf/negative predictions.

    Returns dict with statistics and whether predictions are valid.
    """
    model.train(False)
    all_preds = []

    for tau, logm, y, yATM in val_loader:
        tau = tau.to(device)
        logm = logm.to(device)
        yATM = yATM.to(device)
        pred, _, _, _ = model(tau, logm, yATM)
        all_preds.append(pred.detach().cpu().numpy())

    preds = np.concatenate(all_preds).flatten()

    stats = {
        'count': len(preds),
        'mean': float(np.mean(preds)),
        'std': float(np.std(preds)),
        'min': float(np.min(preds)),
        'max': float(np.max(preds)),
        'nan_count': int(np.isnan(preds).sum()),
        'inf_count': int(np.isinf(preds).sum()),
        'negative_count': int((preds < 0).sum()),
    }

    valid = stats['nan_count'] == 0 and stats['inf_count'] == 0
    stats['valid'] = valid

    logger.info(f'Base model predictions: mean={stats["mean"]:.6f}, std={stats["std"]:.6f}, '
                f'min={stats["min"]:.6f}, max={stats["max"]:.6f}')
    if stats['nan_count'] > 0:
        logger.error(f'  {stats["nan_count"]} NaN predictions!')
    if stats['inf_count'] > 0:
        logger.error(f'  {stats["inf_count"]} Inf predictions!')
    if stats['negative_count'] > 0:
        logger.warning(f'  {stats["negative_count"]} negative predictions')

    return stats


# ---------------------------------------------------------------------------
# Step 2: Stage 1 - Train SSVI+NN
# ---------------------------------------------------------------------------
def train_stage1(config, device, logger, epochs=3, dp=None):
    """Train SSVI+NN base model. Returns (model, val_loader, report_dict)."""
    logger.info('=' * 60)
    logger.info('STAGE 1: Training SSVI+NN Base Model')
    logger.info('=' * 60)

    train_start_date = parse_date(config['training']['train_start_date'])
    train_end_date = parse_date(config['training']['train_end_date'])
    batch_size = config['training'].getint('batch_size')

    train_loader, val_loader, c6_loader = dp.Prepare_train_data(
        train_start_date, train_end_date, batch_size)

    # Model setup
    learning_rate = config['model_sett'].getfloat('learning_rate')
    ensemble_num = config['model_sett'].getint('ensemble_num')
    hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
    loss_weights = parse_list_config(config['model_sett']['loss_weights'])

    model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)
    loss_function = WeightedSumLoss(weights=loss_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    gradient_clip = config['training'].getfloat('gradient_clip')

    # Scheduler (adapted for short runs)
    milestone_start = config['training'].getint('scheduler_milestones_start')
    milestone_step = config['training'].getint('scheduler_milestones_step')
    scheduler_gamma = config['training'].getfloat('scheduler_gamma')
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[i for i in range(milestone_start, epochs, milestone_step)],
        gamma=scheduler_gamma
    )

    metrics = MetricsTracker()
    patience = config['training'].getint('early_stopping_patience')
    early_stopping = EarlyStopping(patience=patience)
    model_path = config['save_path']['model_path']
    best_loss = float('inf')

    # SSVI parameter trajectory
    ssvi_trajectory = []

    logger.info(f'Training for {epochs} epochs...')
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, c6_loader,
                                     loss_function, optimizer, device, gradient_clip)
        val_loss = validate(model, val_loader, c6_loader, loss_function, device)
        scheduler.step()

        is_best = metrics.update(epoch, train_loss, val_loss)
        logger.info(f'Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f} - Val: {val_loss:.6f} '
                    f'- LR: {scheduler.get_last_lr()[0]:.2e}')

        # Extract SSVI params each epoch
        params = extract_ssvi_params(model)
        ssvi_trajectory.append({'epoch': epoch + 1, 'params': params})

        for p in params:
            if 'eta' in p:
                logger.info(f'  Member {p["member"]}: rho={p["rho"]:.4f}, eta={p["eta"]:.4f}, '
                            f'gamma={p["gamma"]:.4f}, GJ={p["gj_value"]:.4f} '
                            f'{"OK" if p["gj_satisfied"] else "VIOLATION"}')

        if is_best and val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            logger.info(f'  New best model saved (val_loss={val_loss:.6f})')

        if early_stopping.step(val_loss):
            logger.info(f'Early stopping at epoch {epoch+1}')
            break

    # Post-training diagnostics
    logger.info('Running post-training SSVI diagnostics...')
    final_params = extract_ssvi_params(model)
    ssvi_healthy = check_ssvi_health(final_params, logger)
    surface_check = run_iv_surface_sanity_check(model, device, logger)

    # Transition validation
    logger.info('Validating base model predictions on val set...')
    pred_stats = validate_base_model_predictions(model, val_loader, device, logger)

    report = {
        'epochs_run': len(metrics.train_losses),
        'best_val_loss': best_loss,
        'final_train_loss': metrics.train_losses[-1],
        'final_val_loss': metrics.val_losses[-1],
        'ssvi_params_final': final_params,
        'ssvi_healthy': ssvi_healthy,
        'surface_check': surface_check,
        'prediction_stats': pred_stats,
    }

    # Save SSVI trajectory and stage metrics
    log_dir = config['save_path']['log_dir']
    _save_json(ssvi_trajectory, os.path.join(log_dir, 'pipeline_ssvi_params.json'))
    metrics.save(os.path.join(log_dir, 'pipeline_stage1_metrics.json'))

    return model, val_loader, report


# ---------------------------------------------------------------------------
# Step 4: Stage 2 - Train Adjustment Model
# ---------------------------------------------------------------------------
def train_stage2(config, device, logger, base_model, dp, epochs=None):
    """Train GRU+Attention adjustment model. Returns report_dict."""
    logger.info('=' * 60)
    logger.info('STAGE 2: Training Adjustment Model')
    logger.info('=' * 60)

    adj_cfg = config['adjustment']
    if epochs is None:
        epochs = adj_cfg.getint('epochs')
    sequence_length = adj_cfg.getint('sequence_length')

    # Prepare adjustment data (chunked inference to avoid GPU OOM)
    logger.info('Preparing adjustment data (chunked inference)...')
    result = dp.prepare_adjustment_data(base_model, device, sequence_length)
    sequences, targets, masks, seq_dates = result

    if sequences is None:
        logger.error('Failed to prepare adjustment data (VIX file missing?)')
        return {'error': 'data_preparation_failed'}

    logger.info(f'Adjustment data: sequences={sequences.shape}, targets={targets.shape}')

    # Log input feature statistics
    feat_names = ['vix_change', 'underlying_return', 'logm', 'tau', 'tv_pred', 'itm_otm']
    # The actual feature count may be larger if enhancement features are present
    n_features = sequences.shape[-1]
    if n_features > len(feat_names):
        feat_names += [f'enhance_{i}' for i in range(n_features - len(feat_names))]

    # Use last timestep of each sequence for statistics (most meaningful)
    last_step = sequences[:, -1, :].numpy()
    feat_stats = {}
    for i, name in enumerate(feat_names[:n_features]):
        col = last_step[:, i]
        feat_stats[name] = {
            'mean': float(np.mean(col)), 'std': float(np.std(col)),
            'min': float(np.min(col)), 'max': float(np.max(col)),
            'nan_count': int(np.isnan(col).sum()),
        }
        logger.info(f'  Feature {name}: mean={feat_stats[name]["mean"]:.4f}, '
                    f'std={feat_stats[name]["std"]:.4f}, '
                    f'range=[{feat_stats[name]["min"]:.4f}, {feat_stats[name]["max"]:.4f}]')

    # Log target distribution
    tgt = targets.numpy().flatten()
    target_stats = {
        'mean': float(np.mean(tgt)), 'std': float(np.std(tgt)),
        'min': float(np.min(tgt)), 'max': float(np.max(tgt)),
        'p5': float(np.percentile(tgt, 5)), 'p50': float(np.percentile(tgt, 50)),
        'p95': float(np.percentile(tgt, 95)),
    }
    logger.info(f'  Target (tv_ratio): mean={target_stats["mean"]:.4f}, '
                f'std={target_stats["std"]:.4f}, '
                f'p5={target_stats["p5"]:.4f}, p50={target_stats["p50"]:.4f}, '
                f'p95={target_stats["p95"]:.4f}')

    # Chronological split: first 80% dates -> train, last 20% -> val
    unique_dates = np.sort(np.unique(seq_dates))
    split_idx = int(len(unique_dates) * 0.8)
    val_start_date = unique_dates[split_idx]

    train_mask = seq_dates < val_start_date
    val_mask = seq_dates >= val_start_date
    logger.info(f'Chronological split: train={train_mask.sum()}, val={val_mask.sum()}, '
                f'val starts {val_start_date}')

    # KDE fit on train targets only (preventing leakage)
    logger.info('Fitting KDE weights (train-only)...')
    loss_fn = AdjustmentLoss(
        kde_bandwidth=adj_cfg.getfloat('kde_bandwidth'), mape_weight=0.5)
    train_kde_weights = loss_fn.fit_kde_weights(targets[train_mask].numpy())
    val_kde_weights = loss_fn.eval_kde_weights(targets[val_mask].numpy())

    # DataLoaders
    batch_size = adj_cfg.getint('batch_size')
    train_ds = TensorDataset(
        sequences[train_mask], targets[train_mask], masks[train_mask],
        torch.tensor(train_kde_weights, dtype=torch.float64))
    val_ds = TensorDataset(
        sequences[val_mask], targets[val_mask], masks[val_mask],
        torch.tensor(val_kde_weights, dtype=torch.float64))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model with dynamic input_dim from data
    input_dim = sequences.shape[-1]
    prediction_target = adj_cfg.get('prediction_target', 'ratio')
    adj_model = TVAdjustmentModel(
        input_dim=input_dim,
        gru_hidden_dim=adj_cfg.getint('gru_hidden_dim'),
        gru_layers=adj_cfg.getint('gru_layers'),
        attention_heads=adj_cfg.getint('attention_heads'),
        dropout=adj_cfg.getfloat('dropout'),
        prediction_target=prediction_target,
    ).double().to(device)

    optimizer = optim.Adam(adj_model.parameters(), lr=adj_cfg.getfloat('learning_rate'))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    metrics = MetricsTracker()
    early_stopping = EarlyStopping(patience=100)
    adj_model_path = config['save_path']['adjustment_model_path']
    best_val_loss = float('inf')

    # Training loop
    logger.info(f'Training adjustment model for up to {epochs} epochs...')
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
        adj_model.train(False)
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

    # Final assessment
    logger.info('Running final assessment on val set...')
    adj_model.load_state_dict(torch.load(adj_model_path, map_location=device, weights_only=True))
    adj_model.train(False)

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

    # Save stage metrics
    log_dir = config['save_path']['log_dir']
    metrics.save(os.path.join(log_dir, 'pipeline_stage2_metrics.json'))

    report = {
        'epochs_run': len(metrics.train_losses),
        'best_val_loss': best_val_loss,
        'rmse': rmse,
        'mape': mape,
        'input_dim': input_dim,
        'train_samples': int(train_mask.sum()),
        'val_samples': int(val_mask.sum()),
        'feature_stats': feat_stats,
        'target_stats': target_stats,
    }
    return report


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = ArgumentParser(description='SSVI+NN -> Adjustment Model pipeline')
    parser.add_argument('--on_gpu', action='store_true')
    parser.add_argument('--stage1_epochs', type=int, default=3)
    parser.add_argument('--stage2_epochs', type=int, default=None,
                        help='Override adjustment model epochs (default: from config)')
    parser.add_argument('--skip_stage1', action='store_true',
                        help='Skip Stage 1, load existing checkpoint for Stage 2')
    args = parser.parse_args()

    config = load_config('config.ini')
    seed = config['training'].getint('seed')
    set_seed(seed)

    log_dir = config['save_path']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir, 'pipeline')

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device('cuda:0' if use_gpu else 'cpu')
    torch.set_default_dtype(torch.float64)

    logger.info('=' * 60)
    logger.info('PIPELINE START')
    logger.info(f'  Device: {device}')
    logger.info(f'  Stage 1 epochs: {args.stage1_epochs}')
    logger.info(f'  Stage 2 epochs: {args.stage2_epochs or "config default"}')
    logger.info(f'  Skip Stage 1: {args.skip_stage1}')
    logger.info('=' * 60)

    # Shared data processor (loaded once)
    logger.info('Loading and preprocessing data...')
    dp = DataProcessor(config)
    dp()
    logger.info('Data preprocessing complete.')

    pipeline_report = {}

    # --- Stage 1 ---
    if not args.skip_stage1:
        model, val_loader, stage1_report = train_stage1(
            config, device, logger, epochs=args.stage1_epochs, dp=dp)
        pipeline_report['stage1'] = stage1_report

        # Transition gate: abort if predictions are invalid
        if not stage1_report['prediction_stats']['valid']:
            logger.error('PIPELINE ABORTED: Stage 1 predictions contain NaN/Inf')
            pipeline_report['aborted'] = True
            _save_json(pipeline_report, os.path.join(log_dir, 'pipeline_report.json'))
            return
    else:
        # Load existing checkpoint
        logger.info('Skipping Stage 1, loading existing checkpoint...')
        hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
        ensemble_num = config['model_sett'].getint('ensemble_num')
        model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)

        model_path = config['save_path']['model_path']
        if not os.path.exists(model_path):
            logger.error(f'No checkpoint found at {model_path}. Run Stage 1 first.')
            return
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.train(False)

        # Still run diagnostics on loaded model
        final_params = extract_ssvi_params(model)
        ssvi_healthy = check_ssvi_health(final_params, logger)
        surface_check = run_iv_surface_sanity_check(model, device, logger)
        pipeline_report['stage1'] = {
            'skipped': True,
            'ssvi_params_final': final_params,
            'ssvi_healthy': ssvi_healthy,
            'surface_check': surface_check,
        }

    # --- Stage 2 ---
    stage2_report = train_stage2(
        config, device, logger, model, dp, epochs=args.stage2_epochs)
    pipeline_report['stage2'] = stage2_report

    # --- Pipeline Report ---
    _save_json(pipeline_report, os.path.join(log_dir, 'pipeline_report.json'))

    logger.info('=' * 60)
    logger.info('PIPELINE COMPLETE')
    if 'stage1' in pipeline_report and not pipeline_report['stage1'].get('skipped'):
        s1 = pipeline_report['stage1']
        logger.info(f'  Stage 1: best_val={s1["best_val_loss"]:.6f}, '
                    f'SSVI healthy={s1["ssvi_healthy"]}')
    if 'stage2' in pipeline_report and 'rmse' in pipeline_report['stage2']:
        s2 = pipeline_report['stage2']
        logger.info(f'  Stage 2: RMSE={s2["rmse"]:.6f}, MAPE={s2["mape"]:.4f}')
    logger.info(f'  Report saved to {os.path.join(log_dir, "pipeline_report.json")}')
    logger.info('=' * 60)


if __name__ == '__main__':
    main()
