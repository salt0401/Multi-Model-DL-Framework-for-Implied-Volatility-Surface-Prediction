"""Dupire PINN training with collocation resampling and autograd PDE.

Trains dual-network (PriceNetwork + LocalVolNetwork) on Dupire PDE constraint.
Supports two data source modes:
  - Synthetic BS mode (default): uses Black-Scholes formula for pipeline validation
  - Model 1 mode (--use_model1): queries pre-trained MultiModel for real IV surface targets
"""
import sys
import os

# Add paths for cross-module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model1_research'))

from dupire_pinn import (
    PriceNetwork, ICNNPriceNetwork, LocalVolNetwork, DupirePINNLoss, DupireSampler, bs_call_price
)
from utils import load_config, set_seed, setup_logging, MetricsTracker

from argparse import ArgumentParser
import gc
import torch
from torch import optim
import numpy as np


def train_dupire(config, device, logger, base_model=None, yATM=None):
    """Train Dupire PINN local vol extractor.

    Args:
        config: ConfigParser object
        device: torch device
        logger: logging.Logger
        base_model: (optional) pre-trained Model 1 MultiModel. If provided,
                    the sampler queries Model 1 for real targets instead of BS.
        yATM: (optional) ATM total variance scalar. Required if base_model is set.
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

    # Sampler (Model 1 mode or synthetic BS mode)
    if base_model is not None:
        logger.info(f"Using Model 1 mode: querying MultiModel for targets (yATM={yATM:.6f})")
        sampler = DupireSampler(
            K_min=K_min, K_max=K_max, tau_min=tau_min, tau_max=tau_max,
            n_interior=n_interior, n_boundary=n_boundary,
            strike=1.0, base_model=base_model, yATM=yATM
        )
    else:
        logger.info(f"Using synthetic BS mode: sigma_bs={sigma_bs:.4f}")
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

    # Validation: compute local vol on a grid
    price_net.eval()
    localvol_net.eval()
    with torch.no_grad():
        K_val = torch.linspace(K_min, K_max, 20, device=device).reshape(-1, 1)
        tau_val = torch.full_like(K_val, (tau_min + tau_max) / 2)
        sigma2_pred = localvol_net(K_val, tau_val).cpu().numpy().flatten()
        sigma_pred = np.sqrt(sigma2_pred)

    logger.info(f'Validation local vol (mean): {sigma_pred.mean():.4f}')
    logger.info(f'Validation local vol (std):  {sigma_pred.std():.6f}')
    if base_model is None:
        logger.info(f'  (synthetic BS expected ~{sigma_bs:.4f})')
    else:
        logger.info(f'  (Model 1 mode — no single expected value)')

    return price_net, localvol_net


def _load_model1(config, device, logger):
    """Load pre-trained Model 1 (MultiModel) and compute yATM from dataset.

    Args:
        config: ConfigParser with [model_sett] and [save_path] sections
        device: torch device
        logger: logging.Logger

    Returns:
        (base_model, yATM): loaded MultiModel in eval mode and mean ATM total variance
    """
    from model import MultiModel
    from dataset import DataProcessor

    # Build MultiModel with the same architecture as training
    hidden_sizes = [int(x) for x in config['model_sett']['hidden_sizes'].split(',')]
    ensemble_num = config['model_sett'].getint('ensemble_num')
    epsilon = config['model_sett'].getfloat('epsilon', fallback=0.01)

    base_model = MultiModel(
        hidden_sizes=hidden_sizes, ensemble_num=ensemble_num, epsilon=epsilon
    ).double().to(device)

    # Load checkpoint
    model_path = config['save_path']['model_path']
    logger.info(f'Loading Model 1 from: {model_path}')
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    base_model.load_state_dict(state_dict)
    base_model.eval()
    logger.info(f'Model 1 loaded successfully ({sum(p.numel() for p in base_model.parameters()):,} params)')

    # Compute yATM from the dataset
    # yATM is the ATM total variance per day. We use the mean across the training set
    # as a representative scalar for the surface calibration.
    logger.info('Computing mean yATM from dataset...')
    dp = DataProcessor(config)
    dp()  # triggers preprocess() + synthesize() + getYATM()
    # DataProcessor stores its data in self.prs_dataset, column name is 'y_atm'
    df = dp.prs_dataset
    yATM_values = df['y_atm'].values
    yATM_mean = float(np.mean(yATM_values))
    logger.info(f'Dataset y_atm: mean={yATM_mean:.6f}, std={np.std(yATM_values):.6f}, '
                f'min={np.min(yATM_values):.6f}, max={np.max(yATM_values):.6f}')

    return base_model, yATM_mean


def main():
    parser = ArgumentParser()
    parser.add_argument("--on_gpu", action='store_true')
    parser.add_argument("--use_icnn", action='store_true', help="Use ICNN architecture for PriceNetwork")
    parser.add_argument("--use_model1", action='store_true',
                        help="Use pre-trained Model 1 as target source (production mode)")
    parser.add_argument("--yATM", type=float, default=None,
                        help="Override yATM value (default: computed from dataset mean)")
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

    # Load Model 1 if requested
    base_model = None
    yATM = None
    if args.use_model1:
        logger.info('='*60)
        logger.info('Model 1 mode: loading pre-trained MultiModel...')
        logger.info('='*60)
        base_model, yATM_computed = _load_model1(config, device, logger)
        yATM = args.yATM if args.yATM is not None else yATM_computed
        logger.info(f'Final yATM for training: {yATM:.6f}')
    else:
        logger.info('Synthetic BS mode (use --use_model1 for production training)')

    price_net, localvol_net = train_dupire(config, device, logger,
                                           base_model=base_model, yATM=yATM)
    logger.info('Dupire PINN training complete.')


if __name__ == '__main__':
    main()
