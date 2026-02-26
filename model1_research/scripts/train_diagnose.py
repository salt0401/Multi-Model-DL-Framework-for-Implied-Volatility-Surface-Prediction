import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

"""Short diagnostic training: 4 epochs, log SSVI params + individual losses per epoch."""
import torch
from torch import optim
from tqdm import tqdm
import numpy as np

from model1_research.model import MultiModel, WeightedSumLoss
from src.dataset import DataProcessor
from src.utils import load_config, parse_list_config, parse_date, set_seed
from model1_research.train import train_one_epoch, validate

set_seed(42)
torch.set_default_dtype(torch.float64)

# ── Setup ───────────────────────────────────────────────────────────
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'config.ini')
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config missing at {config_path}")
config = load_config(config_path)

# Fix relative paths dynamically
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'dataset'))
config['data']['data_folder'] = data_dir + os.sep

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
    """Extract eSSVI parameters from all ensemble members."""
    params = []
    for i, single in enumerate(model.ensemble_list):
        ssvi = getattr(single, 'Prior', None)
        if ssvi is None: continue
        rho_0 = torch.clamp(ssvi.raw_rho_0, -0.999, 0.999).item()
        rho_inf = torch.clamp(ssvi.raw_rho_inf, -0.999, 0.999).item()
        decay = torch.abs(ssvi.raw_decay).item()
        eta = torch.abs(ssvi.raw_eta).item()
        gamma = torch.sigmoid(ssvi.raw_gamma).item()
        params.append({'rho_0': rho_0, 'rho_inf': rho_inf, 'decay': decay, 'eta': eta, 'gamma': gamma})
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
    print(f"  Ensemble[{i}]: rho_0={p['rho_0']:.6f}, rho_inf={p['rho_inf']:.6f}, decay={p['decay']:.6f}, eta={p['eta']:.6f}")

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

    # eSSVI params
    params = get_ssvi_params(model)
    rho_0 = [p['rho_0'] for p in params]
    rho_inf = [p['rho_inf'] for p in params]
    decays = [p['decay'] for p in params]
    etas = [p['eta'] for p in params]
    print(f"  rho_0:   {['%.4f' % r for r in rho_0]}  mean={np.mean(rho_0):.4f}")
    print(f"  rho_inf: {['%.4f' % r for r in rho_inf]}  mean={np.mean(rho_inf):.4f}")
    print(f"  decay:   {['%.4f' % d for d in decays]}  mean={np.mean(decays):.4f}")
    print(f"  eta:     {['%.4f' % e for e in etas]}  mean={np.mean(etas):.4f}")

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
    drho_0 = fp['rho_0'] - ip['rho_0']
    drho_inf = fp['rho_inf'] - ip['rho_inf']
    ddecay = fp['decay'] - ip['decay']
    deta = fp['eta'] - ip['eta']
    print(f"  Ensemble[{i}]: rho_0 {ip['rho_0']:.4f} -> {fp['rho_0']:.4f} ({drho_0:+.4f}) "
          f"rho_inf {ip['rho_inf']:.4f} -> {fp['rho_inf']:.4f} ({drho_inf:+.4f}) "
          f"decay {ip['decay']:.4f} -> {fp['decay']:.4f} ({ddecay:+.4f}) "
          f"eta {ip['eta']:.4f} -> {fp['eta']:.4f} ({deta:+.4f})")

rho0_changes = [final_params[i]['rho_0'] - init_params[i]['rho_0'] for i in range(ensemble_num)]
mean_change = np.mean(rho0_changes)
print(f"\n  Mean rho_0 change: {mean_change:+.6f}")
if mean_change < 0:
    print("  -> rho_0 moving MORE negative (correct direction for left-skew)")
else:
    print("  -> rho_0 moving LESS negative (still problematic)")

print()
