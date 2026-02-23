"""Inspect SSVI parameters from the trained model to verify rho and other params."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from model import MultiModel

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'MultiModel.pt')
state_dict = torch.load(model_path, map_location='cpu', weights_only=False)

# Create model with same architecture
model = MultiModel(hidden_sizes=[64, 32, 16], ensemble_num=5)
model.load_state_dict(state_dict)
model.train(False)  # set to inference mode

print("=" * 65)
print("SSVI Parameter Inspection (Trained Model)")
print("=" * 65)

# Access all SingleModels in the ensemble
for i, single in enumerate(model.ensemble_list):
    ssvi = single.Prior
    print(f"\n--- SingleModel[{i}] / SSVIModel ---")

    # rho = -sigmoid(raw_rho) ∈ (-1, 0)
    raw_rho = ssvi.raw_rho.item()
    rho = -torch.sigmoid(ssvi.raw_rho).item()
    print(f"  raw_rho = {raw_rho:.6f}  ->  rho = -sigmoid(raw_rho) = {rho:.6f}")

    # phi function parameters
    if hasattr(ssvi, 'raw_lambda'):
        raw_lam = ssvi.raw_lambda.item()
        lam = torch.exp(ssvi.raw_lambda).item()
        print(f"  raw_lambda = {raw_lam:.6f}  ->  lambda = exp(raw_lambda) = {lam:.6f}")
        print(f"  phi_fun = heston_like: phi(theta) = lambda * theta")
    elif hasattr(ssvi, 'raw_eta'):
        raw_eta = ssvi.raw_eta.item()
        raw_gamma = ssvi.raw_gamma.item()
        eta = 2 * torch.sigmoid(ssvi.raw_eta).item()
        gamma = torch.sigmoid(ssvi.raw_gamma).item()
        print(f"  raw_eta = {raw_eta:.6f}  ->  eta = 2*sigmoid(raw_eta) = {eta:.6f}")
        print(f"  raw_gamma = {raw_gamma:.6f}  ->  gamma = sigmoid(raw_gamma) = {gamma:.6f}")
        print(f"  phi_fun = power_law: phi(theta) = eta / (theta^gamma * (1+theta)^(1-gamma))")
        print(f"  Gatheral-Jacquier check: eta*(1+|rho|) = {eta*(1+abs(rho)):.4f} {'< 2 OK' if eta*(1+abs(rho)) < 2 else '>= 2 VIOLATION!'}")

print("\n" + "=" * 65)
print("SSVI Parameter Summary")
print("=" * 65)

# Collect rho values
rhos = [-torch.sigmoid(m.Prior.raw_rho).item() for m in model.ensemble_list]
print(f"\nrho values across ensemble: {[f'{r:.4f}' for r in rhos]}")
print(f"  Mean rho = {np.mean(rhos):.4f}")
print(f"  All negative? {all(r < 0 for r in rhos)}")
print(f"  Initial rho = -sigmoid(0.0) = {-0.5:.4f}")

if all(r < 0 for r in rhos):
    print("\n  OK: rho is NEGATIVE across all ensemble members -> correct left-skew")
else:
    pos_count = sum(1 for r in rhos if r >= 0)
    print(f"\n  WARNING: {pos_count}/{len(rhos)} ensemble members have rho >= 0")

# Check the IV surface shape at a sample point
print("\n" + "=" * 65)
print("IV Surface Shape Check (left-skew verification)")
print("=" * 65)

with torch.no_grad():
    tau = torch.tensor([[0.5]], dtype=torch.float64)
    yATM = torch.tensor([[0.05]], dtype=torch.float64)

    logm_values = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    print(f"\n  tau=0.5, yATM=0.05")
    print(f"  {'logm':>8} | {'Total Var':>12} | {'Implied Vol':>12}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*12}")

    for lm in logm_values:
        logm_t = torch.tensor([[lm]], dtype=torch.float64)
        tv, _, _, _ = model(tau, logm_t, yATM)
        iv = np.sqrt(tv.item() / 0.5) if tv.item() > 0 else float('nan')
        print(f"  {lm:>8.1f} | {tv.item():>12.6f} | {iv:>12.4f}")

    # Check skew
    tv_put, _, _, _ = model(tau, torch.tensor([[-0.3]], dtype=torch.float64), yATM)
    tv_call, _, _, _ = model(tau, torch.tensor([[0.3]], dtype=torch.float64), yATM)
    tv_atm, _, _, _ = model(tau, torch.tensor([[0.0]], dtype=torch.float64), yATM)

    print(f"\n  OTM Put (logm=-0.3) TV = {tv_put.item():.6f}")
    print(f"  ATM     (logm= 0.0) TV = {tv_atm.item():.6f}")
    print(f"  OTM Call(logm=+0.3) TV = {tv_call.item():.6f}")

    if tv_put.item() > tv_call.item():
        print("  OK: Left skew confirmed: OTM put IV > OTM call IV (fear premium)")
    else:
        print("  WARNING: No left skew detected")

print()
