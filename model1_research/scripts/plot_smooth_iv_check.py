import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

"""Verify: is the zigzag from mixed yATM or from model overfitting?

Method: Fix yATM to specific values, sweep logm on a smooth grid.
If the model learned a proper IV surface, each fixed-yATM curve should be SMOOTH.
If it's overfitting, even fixed-yATM curves will be jagged.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import MultiModel
from dataset import DataProcessor
from utils import load_config, parse_list_config, parse_date, set_seed
from train import train_one_epoch, validate
from torch import optim

set_seed(42)
torch.set_default_dtype(torch.float64)

config = load_config(os.path.join(os.path.dirname(__file__), '..', 'src', 'config.ini'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

dp = DataProcessor(config)
dp()
train_start = parse_date(config['training']['train_start_date'])
train_end = parse_date(config['training']['train_end_date'])
batch_size = config['training'].getint('batch_size')
train_loader, val_loader, c6_loader = dp.Prepare_train_data(train_start, train_end, batch_size)

hidden_sizes = [int(x) for x in parse_list_config(config['model_sett']['hidden_sizes'], int)]
ensemble_num = config['model_sett'].getint('ensemble_num')
loss_weights = parse_list_config(config['model_sett']['loss_weights'])
model = MultiModel(hidden_sizes=hidden_sizes, ensemble_num=ensemble_num).to(device)
loss_function = __import__('model', fromlist=['WeightedSumLoss']).WeightedSumLoss(weights=loss_weights).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config['model_sett'].getfloat('learning_rate'))
gradient_clip = config['training'].getfloat('gradient_clip')

# Train 10 epochs
for epoch in range(10):
    train_loss = train_one_epoch(model, train_loader, c6_loader, loss_function, optimizer, device, gradient_clip)
    val_loss = validate(model, val_loader, c6_loader, loss_function, device)
    print(f"Epoch {epoch+1}/10 - Train: {train_loss:.4f}, Val: {val_loss:.4f}")

# ── Test 1: Fixed yATM, smooth logm grid ────────────────────────────
# This is the DEFINITIVE test: if model learned a proper surface,
# these curves MUST be smooth.
print("\n" + "=" * 65)
print("Test: Fixed yATM synthetic predictions")
print("=" * 65)

logm_grid = np.linspace(-0.4, 0.3, 200)
yatm_levels = [0.001, 0.003, 0.008, 0.020, 0.050]
tau_values = [0.04, 0.15, 0.30]  # ~10D, 38D, 76D

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

colors = ['steelblue', 'green', 'orange', 'crimson', 'purple']

for col, tau_val in enumerate(tau_values):
    ax = axes[col]
    days = int(round(tau_val * 252))

    for i, yatm_val in enumerate(yatm_levels):
        tau_t = torch.full((200, 1), tau_val, dtype=torch.float64).to(device)
        logm_t = torch.tensor(logm_grid.reshape(-1, 1), dtype=torch.float64).to(device)
        yatm_t = torch.full((200, 1), yatm_val, dtype=torch.float64).to(device)

        # Forward pass (needs grad for SmileModel)
        out, _, _, _ = model(tau_t, logm_t, yatm_t)
        tv_pred = out.detach().cpu().numpy().flatten()
        iv_pred = np.sqrt(np.maximum(tv_pred / tau_val, 0))

        ax.plot(logm_grid, iv_pred, color=colors[i], linewidth=2,
                label=f'yATM={yatm_val:.3f}', alpha=0.8)

    ax.set_title(f'tau = {tau_val:.2f} ({days}D)', fontsize=12)
    ax.set_xlabel('log-moneyness (k)')
    ax.set_ylabel('Implied Volatility')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

fig.suptitle('Fixed yATM → Smooth logm sweep (eSSVI surface test)\n'
             'If smooth: model is correct, zigzag was from mixed yATM in plots\n'
             'If jagged: SmileNN is overfitting',
             fontsize=12, fontweight='bold')
plt.tight_layout()

out1 = os.path.join(os.path.dirname(__file__), '..', 'iv_smooth_check.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.abspath(out1)}")

# ── Test 2: Real data but NARROW yATM band ──────────────────────────
# Show that within a tight yATM range, observed AND predicted are smooth
print("\nTest: Narrow yATM band from real validation data")

all_tau, all_logm, all_y_true, all_y_pred, all_yatm = [], [], [], [], []
for tau, logm, y, yATM in val_loader:
    tau_d, logm_d, yATM_d = tau.to(device), logm.to(device), yATM.to(device)
    out, _, _, _ = model(tau_d, logm_d, yATM_d)
    all_tau.append(tau.numpy().flatten())
    all_logm.append(logm.numpy().flatten())
    all_y_true.append(y.numpy().flatten())
    all_y_pred.append(out.detach().cpu().numpy().flatten())
    all_yatm.append(yATM.numpy().flatten())

tau_all = np.concatenate(all_tau)
logm_all = np.concatenate(all_logm)
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
yatm_all = np.concatenate(all_yatm)
iv_true = np.sqrt(np.maximum(y_true_all / tau_all, 0))
iv_pred = np.sqrt(np.maximum(y_pred_all / tau_all, 0))

# Focus on tau ≈ 0.15 (39D) where we had the clearest bands
target_tau = 0.1534
tau_mask = np.abs(tau_all - target_tau) < 0.015

# Split into narrow yATM bands
yatm_in_slice = yatm_all[tau_mask]
yatm_percentiles = np.percentile(yatm_in_slice, [10, 20, 30, 40, 50, 60, 70, 80, 90])

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
axes2 = axes2.flatten()

band_edges = list(zip(
    np.percentile(yatm_in_slice, [0, 15, 30, 50, 70, 85]),
    np.percentile(yatm_in_slice, [15, 30, 50, 70, 85, 100])
))

for idx, (yatm_lo, yatm_hi) in enumerate(band_edges):
    if idx >= 6:
        break
    ax = axes2[idx]

    mask = tau_mask & (yatm_all >= yatm_lo) & (yatm_all <= yatm_hi)
    if mask.sum() < 10:
        ax.set_title(f'yATM [{yatm_lo:.4f}, {yatm_hi:.4f}] (insufficient)')
        continue

    logm_s = logm_all[mask]
    sort_idx = np.argsort(logm_s)

    ax.scatter(logm_s[sort_idx], iv_true[mask][sort_idx],
              s=12, alpha=0.5, color='steelblue', label='Observed', zorder=2)
    ax.plot(logm_s[sort_idx], iv_pred[mask][sort_idx],
           color='crimson', linewidth=1.5, alpha=0.8, label='Predicted', zorder=3)

    mean_yatm = yatm_all[mask].mean()
    ax.set_title(f'yATM ∈ [{yatm_lo:.4f}, {yatm_hi:.4f}]\nmean={mean_yatm:.5f}, n={mask.sum()}',
                fontsize=10)
    ax.set_xlabel('log-moneyness (k)')
    ax.set_ylabel('Implied Volatility')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig2.suptitle(f'Narrow yATM bands at tau≈{target_tau:.4f} (39D)\n'
              'Each subplot = tight yATM range → predictions should be LESS jagged',
              fontsize=12, fontweight='bold')
plt.tight_layout()

out2 = os.path.join(os.path.dirname(__file__), '..', 'iv_narrow_band_check.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.abspath(out2)}")
