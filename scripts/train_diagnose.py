"""Short diagnostic training: 4 epochs, log SSVI params + individual losses per epoch."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from torch import optim
from tqdm import tqdm
import numpy as np

from model import MultiModel, WeightedSumLoss
from dataset import DataProcessor
from utils import load_config, parse_list_config, parse_date, set_seed
from train import train_one_epoch, validate

set_seed(42)
torch.set_default_dtype(torch.float64)

# ── Setup ───────────────────────────────────────────────────────────
config = load_config(os.path.join(os.path.dirname(__file__), '..', 'src', 'config.ini'))

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
print(f"Device: {device} (GPU: {use_gpu})")

# Data
dp = DataProcessor(config)
dp()
train_start = parse_date(config['training']['train_start_date'])
train_end = parse_date(config['training']['train_end_date'])
batch_size = config['training'].getint('batch_size')
train_loader, val_loader, c6_loader = dp.Prepare_train_data(train_start, train_end, batch_size)

# Model (fresh, with -sigmoid rho constraint)
hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
ensemble_num = config['model_sett'].getint('ensemble_num')
loss_weights = parse_list_config(config['model_sett']['loss_weights'])
model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)
loss_function = WeightedSumLoss(weights=loss_weights).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config['model_sett'].getfloat('learning_rate'))
gradient_clip = config['training'].getfloat('gradient_clip')


def get_ssvi_params(model):
    """Extract SSVI parameters from all ensemble members."""
    params = []
    for i, single in enumerate(model.ensemble_list):
        ssvi = single.Prior
        rho = -torch.sigmoid(ssvi.raw_rho).item()
        eta = 2 * torch.sigmoid(ssvi.raw_eta).item()
        gamma = torch.sigmoid(ssvi.raw_gamma).item()
        params.append({'rho': rho, 'eta': eta, 'gamma': gamma,
                       'raw_rho': ssvi.raw_rho.item()})
    return params


def get_individual_losses(model, loader, c6_loader, loss_fn, device):
    """Run one validation pass and return individual loss components.
    Note: SmileModel needs autograd.grad, so no torch.no_grad() here.
    """
    model.train(False)
    c6_iter = iter(c6_loader)
    all_individual = []

    for tau, logm, y, yATM in loader:
        try:
            tau_c6, logm_c6, yATM_c6 = next(c6_iter)
        except StopIteration:
            c6_iter = iter(c6_loader)
            tau_c6, logm_c6, yATM_c6 = next(c6_iter)

        tau, logm, y, yATM = tau.to(device), logm.to(device), y.to(device), yATM.to(device)
        tau_c6, logm_c6, yATM_c6 = tau_c6.to(device), logm_c6.to(device), yATM_c6.to(device)

        out, gt, gl, gl2 = model(tau, logm, yATM)
        out_c6, _, _, gl2_c6 = model(tau_c6, logm_c6, yATM_c6)

        _ = loss_fn(out, y, logm, gt, gl, gl2, out_c6, logm_c6, gl2_c6)
        all_individual.append(loss_fn.individual_losses.cpu())

    avg = torch.stack(all_individual).mean(dim=0)
    return avg


# ── Log initial state ───────────────────────────────────────────────
loss_names = ['RMSE', 'MAPE', 'Calendar', 'Butterfly', 'Linear', 'UpperBound']

print("\n" + "=" * 80)
print("INITIAL STATE (before training)")
print("=" * 80)

init_params = get_ssvi_params(model)
for i, p in enumerate(init_params):
    print(f"  Ensemble[{i}]: rho={p['rho']:.6f}, eta={p['eta']:.6f}, gamma={p['gamma']:.6f}")

init_losses = get_individual_losses(model, val_loader, c6_loader, loss_function, device)
print(f"\n  Individual losses (val):")
for name, val, w in zip(loss_names, init_losses, loss_weights):
    print(f"    {name:>12}: {val:.6f}  (weighted: {val*w:.6f})")
print(f"  Total weighted: {sum(v*w for v, w in zip(init_losses, loss_weights)):.6f}")

# ── Train 10 epochs ─────────────────────────────────────────────────
NUM_EPOCHS = 10
print("\n" + "=" * 80)
print(f"TRAINING ({NUM_EPOCHS} epochs)")
print("=" * 80)

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, c6_loader, loss_function, optimizer, device, gradient_clip)
    val_loss = validate(model, val_loader, c6_loader, loss_function, device)

    print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} --- Train: {train_loss:.4f}, Val: {val_loss:.4f}")

    # SSVI params
    params = get_ssvi_params(model)
    rhos = [p['rho'] for p in params]
    etas = [p['eta'] for p in params]
    gammas = [p['gamma'] for p in params]
    print(f"  rho:   {['%.4f' % r for r in rhos]}  mean={np.mean(rhos):.4f}")
    print(f"  eta:   {['%.4f' % e for e in etas]}  mean={np.mean(etas):.4f}")
    print(f"  gamma: {['%.4f' % g for g in gammas]}  mean={np.mean(gammas):.4f}")

    # Individual losses
    ind_losses = get_individual_losses(model, val_loader, c6_loader, loss_function, device)
    for name, val, w in zip(loss_names, ind_losses, loss_weights):
        print(f"    {name:>12}: {val:.6f}  (weighted: {val*w:.6f})")

# ── Final comparison ────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PARAMETER CHANGE SUMMARY (init -> final)")
print("=" * 80)

final_params = get_ssvi_params(model)
for i in range(ensemble_num):
    ip = init_params[i]
    fp = final_params[i]
    drho = fp['rho'] - ip['rho']
    deta = fp['eta'] - ip['eta']
    dgamma = fp['gamma'] - ip['gamma']
    print(f"  Ensemble[{i}]: rho {ip['rho']:.4f} -> {fp['rho']:.4f} ({drho:+.4f})"
          f"  eta {ip['eta']:.4f} -> {fp['eta']:.4f} ({deta:+.4f})"
          f"  gamma {ip['gamma']:.4f} -> {fp['gamma']:.4f} ({dgamma:+.4f})")

rho_changes = [final_params[i]['rho'] - init_params[i]['rho'] for i in range(ensemble_num)]
mean_change = np.mean(rho_changes)
print(f"\n  Mean rho change: {mean_change:+.6f}")
if mean_change < 0:
    print("  -> rho moving MORE negative (correct direction for left-skew)")
else:
    print("  -> rho moving LESS negative (still problematic)")

print()
