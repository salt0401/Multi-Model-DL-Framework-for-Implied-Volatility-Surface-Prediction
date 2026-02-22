"""Dupire PINN training with collocation resampling and autograd PDE.

Trains dual-network (PriceNetwork + LocalVolNetwork) on Dupire PDE constraint.
V1 prototype uses Black-Scholes synthetic data for validation.
"""
from dupire_pinn import (
    PriceNetwork, ICNNPriceNetwork, LocalVolNetwork, DupirePINNLoss, DupireSampler, bs_call_price
)
from utils import load_config, set_seed, setup_logging, MetricsTracker

from argparse import ArgumentParser
import gc
import torch
from torch import optim
import numpy as np
import os


def train_dupire(config, device, logger):
    """Train Dupire PINN local vol extractor.

    Args:
        config: ConfigParser object
        device: torch device
        logger: logging.Logger
    """
    cfg = config['dupire']
    hidden_dim = cfg.getint('hidden_dim')
    n_layers = cfg.getint('n_layers')
    K_min = cfg.getfloat('K_min')
    K_max = cfg.getfloat('K_max')
    tau_min = cfg.getfloat('tau_min')
    tau_max = cfg.getfloat('tau_max')
    n_interior = cfg.getint('n_interior')
    n_boundary = cfg.getint('n_boundary')
    resample_every = cfg.getint('resample_every')
    lambda_fit = cfg.getfloat('lambda_fit')
    lambda_pde = cfg.getfloat('lambda_pde')
    lambda_cal = cfg.getfloat('lambda_cal')
    lambda_but = cfg.getfloat('lambda_but')
    lambda_smooth = cfg.getfloat('lambda_smooth')
    epochs = cfg.getint('epochs')
    lr = cfg.getfloat('learning_rate')
    grad_clip = cfg.getfloat('gradient_clip')
    sigma_bs = cfg.getfloat('sigma_bs')

    # Models
    if config.getboolean('dupire', 'use_icnn', fallback=getattr(config, 'use_icnn', False)):
        logger.info("Using ICNN Price Network (V2) for hard convexity guarantee.")
        price_net = ICNNPriceNetwork(
            hidden_dim=hidden_dim, n_layers=n_layers
        ).double().to(device)
    else:
        logger.info("Using standard MLP Price Network (V1).")
        price_net = PriceNetwork(
            hidden_dim=hidden_dim, n_layers=n_layers
        ).double().to(device)
        
    localvol_net = LocalVolNetwork(
        hidden_dim=hidden_dim, n_layers=n_layers
    ).double().to(device)

    # Count parameters
    n_price_params = sum(p.numel() for p in price_net.parameters())
    n_lv_params = sum(p.numel() for p in localvol_net.parameters())
    logger.info(f'PriceNetwork params: {n_price_params:,}')
    logger.info(f'LocalVolNetwork params: {n_lv_params:,}')
    logger.info(f'Total params: {n_price_params + n_lv_params:,}')

    # Loss
    loss_fn = DupirePINNLoss(
        lambda_fit=lambda_fit, lambda_pde=lambda_pde,
        lambda_cal=lambda_cal, lambda_but=lambda_but,
        lambda_smooth=lambda_smooth
    )

    # Sampler
    sampler = DupireSampler(
        K_min=K_min, K_max=K_max, tau_min=tau_min, tau_max=tau_max,
        n_interior=n_interior, n_boundary=n_boundary,
        strike=1.0, sigma_bs=sigma_bs
    )

    # Optimizer (joint over both networks)
    all_params = list(price_net.parameters()) + list(localvol_net.parameters())
    optimizer = optim.AdamW(all_params, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5
    )

    metrics = MetricsTracker()
    best_total_loss = float('inf')
    model_path = config['save_path']['dupire_model_path']

    for epoch in range(epochs):
        price_net.train()
        localvol_net.train()

        # Re-sample collocation points periodically
        if epoch % resample_every == 0:
            data = sampler.sample(device=device)

        # Forward + loss (interior points)
        total_loss, loss_fit, loss_pde, loss_cal, loss_but, loss_smooth = loss_fn(
            price_net, localvol_net,
            data['K_interior'], data['tau_interior'], data['C_target']
        )

        # Boundary loss: tau → 0 payoff
        C_bnd_pred = price_net(data['K_boundary'], data['tau_boundary'])
        loss_bnd = torch.mean((C_bnd_pred - data['C_boundary']) ** 2)
        total_loss = total_loss + loss_bnd

        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(all_params, grad_clip)

        optimizer.step()

        total_val = total_loss.item()
        scheduler.step(total_val)
        metrics.update(epoch, total_val, total_val)

        if total_val < best_total_loss:
            best_total_loss = total_val
            torch.save({
                'price_net': price_net.state_dict(),
                'localvol_net': localvol_net.state_dict(),
                'epoch': epoch,
                'total_loss': total_val,
            }, model_path)

        if (epoch + 1) % 100 == 0:
            logger.info(
                f'Epoch {epoch+1}/{epochs} - Total: {total_val:.6f} - '
                f'Fit: {loss_fit.item():.6f} - PDE: {loss_pde.item():.6f} - '
                f'Cal: {loss_cal.item():.6f} - But: {loss_but.item():.6f} - '
                f'Smooth: {loss_smooth.item():.6f} - Bnd: {loss_bnd.item():.6f}'
            )

        # Prevent VRAM fragmentation
        if (epoch + 1) % 50 == 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    logger.info(f'Best total loss: {best_total_loss:.6f}')

    # Validation: compute local vol on a grid and compare with known sigma
    price_net.eval()
    localvol_net.eval()
    with torch.no_grad():
        K_val = torch.linspace(K_min, K_max, 20, device=device).reshape(-1, 1)
        tau_val = torch.full_like(K_val, (tau_min + tau_max) / 2)
        sigma2_pred = localvol_net(K_val, tau_val).cpu().numpy().flatten()
        sigma_pred = np.sqrt(sigma2_pred)

    logger.info(f'Validation local vol (mean): {sigma_pred.mean():.4f} '
                f'(expected ~{sigma_bs:.4f})')
    logger.info(f'Validation local vol (std): {sigma_pred.std():.6f}')

    return price_net, localvol_net


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--use_icnn", action='store_true', help="Use ICNN architecture for PriceNetwork")
    parser.add_argument("--finetune", type=str, default=None,
                        help='Path to pretrained checkpoint for transfer learning')
    args = parser.parse_args()

    config = load_config('config.ini')
    config.use_icnn = args.use_icnn  # Pass down the CLI argument
    seed = config['training'].getint('seed')
    set_seed(seed)

    log_dir = config['save_path']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir, 'dupire')

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")
    torch.set_default_dtype(torch.float64)

    logger.info(f'Device: {device}')
    logger.info(f'Float64 mode enabled')

    price_net, localvol_net = train_dupire(config, device, logger)
    logger.info('Dupire PINN training complete.')


if __name__ == '__main__':
    main()
