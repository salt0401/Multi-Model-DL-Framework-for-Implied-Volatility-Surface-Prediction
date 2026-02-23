"""DGM training with re-sampling, transfer learning, and L2 regularization.

Trains DGM on GBM backward Kolmogorov PDE with configurable sigma/T ranges.
Supports transfer learning: fine-tune pretrained model on new tau range.
"""
from dgm import DGMNetwork, DGMLoss, DGMSampler, bs_call_price
from utils import load_config, set_seed, setup_logging, MetricsTracker, EarlyStopping

from argparse import ArgumentParser
import copy

import torch
from torch import optim
import numpy as np
import os


def train_dgm(config, device, logger, pretrained_model=None):
    """Train DGM PDE solver.

    Args:
        config: ConfigParser object
        device: torch device
        logger: logging.Logger
        pretrained_model: optional pretrained DGMNetwork for transfer learning
    """
    dgm_cfg = config['dgm']
    hidden_dim = dgm_cfg.getint('hidden_dim')
    n_layers = dgm_cfg.getint('n_layers')
    sigma_min = dgm_cfg.getfloat('sigma_min')
    sigma_max = dgm_cfg.getfloat('sigma_max')
    S_min = dgm_cfg.getfloat('S_min')
    S_max = dgm_cfg.getfloat('S_max')
    T_min = dgm_cfg.getfloat('T_min')
    T_max = dgm_cfg.getfloat('T_max')
    lambda_pde = dgm_cfg.getfloat('lambda_pde')
    lambda_bc = dgm_cfg.getfloat('lambda_bc')
    lambda_tc = dgm_cfg.getfloat('lambda_tc')
    n_interior = dgm_cfg.getint('n_interior')
    n_boundary = dgm_cfg.getint('n_boundary')
    n_terminal = dgm_cfg.getint('n_terminal')
    resample_every = dgm_cfg.getint('resample_every')
    epochs = dgm_cfg.getint('epochs')

    # Use midpoint sigma for training
    sigma = (sigma_min + sigma_max) / 2

    # Model
    model = DGMNetwork(input_dim=2, hidden_dim=hidden_dim, n_layers=n_layers).double().to(device)

    if pretrained_model is not None:
        logger.info('Transfer learning: loading pretrained weights')
        model.load_state_dict(pretrained_model.state_dict())
        pretrained_params = {name: p.clone() for name, p in model.named_parameters()}
        lr = dgm_cfg.getfloat('transfer_lr')
        l2_lambda = dgm_cfg.getfloat('transfer_l2_lambda')
    else:
        pretrained_params = None
        lr = 0.001
        l2_lambda = 0.0

    loss_fn = DGMLoss(sigma=sigma, r=0.0, lambda_pde=lambda_pde, lambda_bc=lambda_bc, lambda_tc=lambda_tc)

    sampler = DGMSampler(
        S_min=S_min, S_max=S_max, T_min=T_min, T_max=T_max,
        n_interior=n_interior, n_boundary=n_boundary, n_terminal=n_terminal,
        strike=1.0, sigma=sigma
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)

    metrics = MetricsTracker()
    best_pde_loss = float('inf')
    model_path = config['save_path']['dgm_model_path']

    for epoch in range(epochs):
        model.train()

        # Re-sample domain points every N epochs
        if epoch % resample_every == 0:
            data = sampler.sample(device=device)

        total_loss, loss_pde, loss_bc, loss_tc = loss_fn(
            model,
            data['x_interior'], data['t_interior'],
            data['x_boundary'], data['t_boundary'], data['u_boundary'],
            data['x_terminal'], data['u_terminal']
        )

        # Transfer learning: L2 regularization toward pretrained weights
        if pretrained_params is not None:
            l2_reg = sum(
                torch.sum((p - pretrained_params[name]) ** 2)
                for name, p in model.named_parameters()
            )
            total_loss = total_loss + l2_lambda * l2_reg

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        pde_val = loss_pde.item()
        scheduler.step(pde_val)
        metrics.update(epoch, total_loss.item(), pde_val)

        if pde_val < best_pde_loss:
            best_pde_loss = pde_val
            torch.save(model.state_dict(), model_path)

        if (epoch + 1) % 100 == 0:
            logger.info(
                f'Epoch {epoch+1}/{epochs} - Total: {total_loss.item():.6f} - '
                f'PDE: {pde_val:.6f} - BC: {loss_bc.item():.6f} - TC: {loss_tc.item():.6f}'
            )

    logger.info(f'Best PDE residual loss: {best_pde_loss:.6f}')

    # Validation against Black-Scholes
    model.eval()
    test_S = np.linspace(S_min, S_max, 20)
    test_T = (T_min + T_max) / 2
    with torch.no_grad():
        x_test = torch.tensor(test_S, dtype=torch.float64, device=device).reshape(-1, 1)
        t_test = torch.full_like(x_test, test_T)
        dgm_prices = model(x_test, t_test).cpu().numpy().flatten()

    bs_prices = bs_call_price(test_S, 1.0, test_T, 0.0, sigma)
    price_rmse = np.sqrt(np.mean((dgm_prices - bs_prices) ** 2))
    logger.info(f'DGM vs BS price RMSE: {price_rmse:.6f}')

    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--transfer_from", type=str, default=None,
                        help='Path to pretrained DGM model for transfer learning')
    args = parser.parse_args()

    config = load_config('config.ini')
    seed = config['training'].getint('seed')
    set_seed(seed)

    log_dir = config['save_path']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logging(log_dir, 'dgm')

    use_gpu = torch.cuda.is_available() and args.on_gpu
    device = torch.device("cuda:0" if use_gpu else "cpu")
    torch.set_default_dtype(torch.float64)

    pretrained_model = None
    if args.transfer_from and os.path.exists(args.transfer_from):
        logger.info(f'Loading pretrained model from {args.transfer_from}')
        dgm_cfg = config['dgm']
        pretrained_model = DGMNetwork(
            input_dim=2,
            hidden_dim=dgm_cfg.getint('hidden_dim'),
            n_layers=dgm_cfg.getint('n_layers')
        ).double().to(device)
        pretrained_model.load_state_dict(torch.load(args.transfer_from, map_location=device, weights_only=True))

    model = train_dgm(config, device, logger, pretrained_model)
    logger.info('DGM training complete.')


if __name__ == '__main__':
    main()
