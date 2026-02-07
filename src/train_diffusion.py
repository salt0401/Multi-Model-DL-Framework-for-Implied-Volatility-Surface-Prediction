"""Training script for Conditional DDPM IV Surface Forecasting.

Trains a 1D U-Net to denoise IV surfaces conditioned on current day features.
Generates next-day IV surface predictions via reverse diffusion.
"""
from diffusion import UNet1D, CosineNoiseSchedule, DiffusionTrainer, DiffusionSampler
from dataset import DataProcessor
from utils import (load_config, parse_date, set_seed, setup_logging,
                   MetricsTracker, EarlyStopping, compute_rmse)

from argparse import ArgumentParser

import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import os


def train_one_epoch(trainer, data_loader):
    """Train for one epoch. Returns average loss."""
    total_loss = 0.0
    n_batches = 0

    for surface_today, surface_tomorrow, cond_features in data_loader:
        surface_today = surface_today.to(trainer.device)
        surface_tomorrow = surface_tomorrow.to(trainer.device)
        cond_features = cond_features.to(trainer.device)

        loss = trainer.train_step(surface_tomorrow, surface_today, cond_features)
        total_loss += loss
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, schedule, data_loader, device):
    """Evaluate by sampling and comparing to ground truth."""
    model.eval()
    sampler = DiffusionSampler(model, schedule, device)

    all_preds = []
    all_true = []

    with torch.no_grad():
        for surface_today, surface_tomorrow, cond_features in data_loader:
            surface_today = surface_today.to(device)
            surface_tomorrow = surface_tomorrow.to(device)
            cond_features = cond_features.to(device)

            # Sample one prediction per input
            pred = sampler.sample(surface_today, cond_features, n_samples=surface_today.shape[0])

            all_preds.append(pred.cpu().numpy())
            all_true.append(surface_tomorrow.cpu().numpy())

    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    rmse = compute_rmse(preds.flatten(), true.flatten())

    return rmse


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
    logger = setup_logging(log_dir, 'diffusion')

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")
    torch.set_default_dtype(torch.float64)

    diff_cfg = config['diffusion']
    n_tau_grid = diff_cfg.getint('n_tau_grid')
    n_logm_grid = diff_cfg.getint('n_logm_grid')
    surface_dim = n_tau_grid * n_logm_grid
    unet_channels = tuple(int(x) for x in diff_cfg['unet_channels'].split(','))
    timesteps = diff_cfg.getint('timesteps')
    epochs = args.epochs if args.epochs is not None else diff_cfg.getint('epochs')
    lr = diff_cfg.getfloat('learning_rate')
    batch_size = diff_cfg.getint('batch_size')
    condition_dim = diff_cfg.getint('condition_dim')

    # Data
    logger.info('Preparing diffusion data...')
    dp = DataProcessor(config)
    dp()

    train_end_date = parse_date(config['training']['train_end_date'])
    test_start_date = parse_date(config['training']['test_start_date'])

    train_loader, val_loader, test_loader = dp.Prepare_diffusion_data(
        train_end_date, test_start_date,
        n_tau_grid=n_tau_grid, n_logm_grid=n_logm_grid,
        batch_size=batch_size
    )

    logger.info(f'Surface dim: {surface_dim}, Condition dim: {condition_dim}')

    # Model
    model = UNet1D(
        surface_dim=surface_dim,
        channels=unet_channels,
        condition_dim=condition_dim,
        time_dim=128,
    ).double().to(device)

    schedule = CosineNoiseSchedule(T=timesteps).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trainer = DiffusionTrainer(model, schedule, optimizer, device)

    metrics = MetricsTracker()
    early_stopping = EarlyStopping(patience=100)
    model_path = config['save_path'].get('diffusion_model_path', '../models/DiffusionModel.pt')
    best_val_rmse = float('inf')

    logger.info('Training DDPM...')
    for epoch in range(epochs):
        train_loss = train_one_epoch(trainer, train_loader)
        scheduler.step()

        metrics.update(epoch, train_loss)

        if (epoch + 1) % 50 == 0:
            logger.info(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}')

        # Periodic validation (sampling is expensive)
        if (epoch + 1) % 100 == 0 and val_loader is not None:
            val_rmse = evaluate(model, schedule, val_loader, device)
            logger.info(f'  Val RMSE: {val_rmse:.6f}')

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(model.state_dict(), model_path)
                logger.info(f'  New best model saved')

            if early_stopping.step(val_rmse):
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

    # Save final model if no validation was done
    if best_val_rmse == float('inf'):
        torch.save(model.state_dict(), model_path)

    # Test evaluation
    if test_loader is not None:
        logger.info('Evaluating on test set...')
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        test_rmse = evaluate(model, schedule, test_loader, device)
        logger.info(f'Test RMSE: {test_rmse:.6f}')

    metrics.save(os.path.join(log_dir, 'diffusion_metrics.json'))
    logger.info('DDPM training complete.')


if __name__ == '__main__':
    main()
