"""Diagnose: which loss component pushes rho positive/negative?

For each of the 6 loss components, compute the gradient direction w.r.t. rho
using real training data. This tells us exactly why rho drifts during training.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from utils import load_config, parse_date, parse_list_config, set_seed
from dataset import DataProcessor
from model import (MultiModel, WeightedSumLoss, RMSELoss, MAPELoss,
                   Loss_calendar, Loss_butterfly, Loss_linear, Loss_upperbound)

set_seed(42)
torch.set_default_dtype(torch.float64)

# ── Load real data ──────────────────────────────────────────────────
config = load_config(os.path.join(os.path.dirname(__file__), '..', 'src', 'config.ini'))
dp = DataProcessor(config)
dp()

train_start = parse_date(config['training']['train_start_date'])
train_end = parse_date(config['training']['train_end_date'])
train_loader, val_loader, c6_loader = dp.Prepare_train_data(train_start, train_end, batch_size=256)

# Get one batch of real data
tau_train, logm_train, y_train, yATM_train = next(iter(train_loader))
tau_syn, logm_syn, yATM_syn = next(iter(c6_loader))

print("=" * 70)
print("Data statistics")
print("=" * 70)
print(f"  y_train (total_var): mean={y_train.mean():.6f}, std={y_train.std():.6f}")
print(f"  yATM_train:          mean={yATM_train.mean():.6f}, std={yATM_train.std():.6f}")
print(f"  y_atm / total_var ratio: {(yATM_train.mean() / y_train.mean()).item():.4f}")
print(f"  logm_train:  range=[{logm_train.min():.4f}, {logm_train.max():.4f}]")
print(f"  logm_syn:    range=[{logm_syn.min():.4f}, {logm_syn.max():.4f}]")
print(f"  tau_train:   range=[{tau_train.min():.4f}, {tau_train.max():.4f}]")

# ── Method: finite-difference on rho ────────────────────────────────
# For each loss, compute loss(rho-eps) and loss(rho+eps)
# If loss decreases when rho increases (becomes less negative) -> loss pushes rho positive
# If loss increases when rho increases -> loss pushes rho more negative

loss_names = ['RMSE', 'MAPE', 'Calendar', 'Butterfly', 'Linear', 'UpperBound']
loss_fns = [RMSELoss(), MAPELoss(), Loss_calendar(), Loss_butterfly(), Loss_linear(), Loss_upperbound()]
weights = [1, 1, 10, 10, 10, 10]

# Test at multiple rho values
rho_test_points = [-0.7, -0.5, -0.3, -0.1]
eps = 0.01

print("\n" + "=" * 70)
print("Gradient of each loss w.r.t. rho (finite difference)")
print("Negative = loss wants rho MORE negative (correct)")
print("Positive = loss wants rho LESS negative / more positive (problematic)")
print("=" * 70)

for rho_center in rho_test_points:
    print(f"\n--- rho = {rho_center:.2f} ---")
    print(f"  {'Loss':>12} | {'Weight':>6} | {'d(loss)/d(rho)':>14} | {'Weighted':>14} | Direction")
    print(f"  {'-'*12}-+-{'-'*6}-+-{'-'*14}-+-{'-'*14}-+-{'-'*20}")

    total_weighted_grad = 0.0

    for name, loss_fn, w in zip(loss_names, loss_fns, weights):
        losses_at_rho = []

        for rho_val in [rho_center - eps, rho_center + eps]:
            # Create fresh model, set all SSVI rho to rho_val
            model = MultiModel(hidden_sizes=[64, 32, 16], ensemble_num=5)

            for single in model.ensemble_list:
                # Directly set raw_rho to match desired rho via -sigmoid
                # rho = -sigmoid(raw_rho) -> raw_rho = -log(-1/rho - 1) = log(-rho/(1+rho))
                # But simpler: just override rho in forward. Instead, use a hack:
                # Temporarily replace forward to use fixed rho
                pass

            # Actually, let's manually compute SSVI output with fixed rho
            # to avoid model complexity. Focus on the SSVI Prior only first.
            ssvi = model.ensemble_list[0].Prior
            rho_t = torch.tensor([rho_val], dtype=torch.float64)

            # SSVI forward with fixed rho (bounded: eta < 2, gamma < 1)
            eta = 2 * torch.sigmoid(ssvi.raw_eta)
            gamma_p = torch.sigmoid(ssvi.raw_gamma)
            phi = eta / (torch.pow(yATM_train, gamma_p) * torch.pow(1 + yATM_train, 1 - gamma_p))
            phi_k = phi * logm_train
            disc = torch.square(phi_k + rho_t) + (1 - torch.square(rho_t))
            sqrt_disc = torch.sqrt(disc)
            y_pred = yATM_train / 2 * (1 + rho_t * phi_k + sqrt_disc)
            grad_logm1 = yATM_train / 2 * (rho_t * phi + phi * (phi_k + rho_t) / sqrt_disc)
            grad_logm2 = yATM_train / 2 * torch.square(phi) * (1 - torch.square(rho_t)) / torch.pow(disc, 1.5)
            grad_tau = torch.zeros_like(y_pred)  # SSVI has no tau gradient

            # Same for c6
            phi_c6 = eta / (torch.pow(yATM_syn, gamma_p) * torch.pow(1 + yATM_syn, 1 - gamma_p))
            phi_k_c6 = phi_c6 * logm_syn
            disc_c6 = torch.square(phi_k_c6 + rho_t) + (1 - torch.square(rho_t))
            sqrt_disc_c6 = torch.sqrt(disc_c6)
            y_pred_c6 = yATM_syn / 2 * (1 + rho_t * phi_k_c6 + sqrt_disc_c6)
            grad_logm2_c6 = yATM_syn / 2 * torch.square(phi_c6) * (1 - torch.square(rho_t)) / torch.pow(disc_c6, 1.5)

            # Compute each loss
            if name == 'RMSE':
                l = loss_fn(y_pred, y_train).item()
            elif name == 'MAPE':
                l = loss_fn(y_pred, y_train).item()
            elif name == 'Calendar':
                l = loss_fn(y_pred, grad_tau).item()
            elif name == 'Butterfly':
                l = loss_fn(y_pred, logm_train, grad_logm1, grad_logm2).item()
            elif name == 'Linear':
                l = loss_fn(grad_logm2_c6).item()
            elif name == 'UpperBound':
                l = loss_fn(y_pred_c6, logm_syn).item()

            losses_at_rho.append(l)

        grad = (losses_at_rho[1] - losses_at_rho[0]) / (2 * eps)
        weighted_grad = grad * w
        total_weighted_grad += weighted_grad
        direction = "-> more negative" if grad > 0 else "-> LESS negative"
        print(f"  {name:>12} | {w:>6} | {grad:>+14.6f} | {weighted_grad:>+14.6f} | {direction}")

    print(f"  {'TOTAL':>12} | {'':>6} | {'':>14} | {total_weighted_grad:>+14.6f} | {'-> more negative' if total_weighted_grad > 0 else '-> LESS negative / POSITIVE'}")

print()
