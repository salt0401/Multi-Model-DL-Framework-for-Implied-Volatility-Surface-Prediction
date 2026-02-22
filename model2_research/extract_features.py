import os
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser

from dupire_pinn import ICNNPriceNetwork, PriceNetwork, LocalVolNetwork
from module_d import GreekExtractor
from utils import load_config, setup_logging

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained DupireModel.pt")
    parser.add_argument("--use_icnn", action='store_true', default=True, help="Use ICNN (V2) instead of MLP (V1)")
    parser.add_argument("--out_file", type=str, default="data/module_d_features.csv")
    args = parser.parse_args()

    config = load_config('config.ini')
    log_dir = config.get('dupire', 'log_dir', fallback='logs')
    logger = setup_logging(log_dir, "module_d")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Extracting features on device: {device}")

    # Load Models
    hidden_dim = config['dupire'].getint('hidden_dim')
    n_layers = config['dupire'].getint('n_layers')
    
    if args.use_icnn:
        price_net = ICNNPriceNetwork(hidden_dim=hidden_dim, n_layers=n_layers).double().to(device)
    else:
        price_net = PriceNetwork(hidden_dim=hidden_dim, n_layers=n_layers).double().to(device)
        
    localvol_net = LocalVolNetwork(hidden_dim=hidden_dim, n_layers=n_layers).double().to(device)
    
    if not os.path.exists(args.model_path):
        logger.error(f"Checkpoint not found at {args.model_path}")
        return

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    
    # Check if this is a dictionary containing both state_dicts or just one
    if "price_net" in checkpoint and "localvol_net" in checkpoint:
        price_net.load_state_dict(checkpoint["price_net"])
        localvol_net.load_state_dict(checkpoint["localvol_net"])
    else:
        # Fallback if checkpoint only saved one dict (from older train_dupire)
        logger.warning("Checkpoint format old, attempting to load directly. This may fail.")
        price_net.load_state_dict(checkpoint) # will probably fail but kept for safety

    price_net.eval()
    localvol_net.eval()
    logger.info("Models loaded successfully.")

    # Generate standardized inquiry grid for Model 3
    # K_norm in [0.5, 1.5], tau in [0.01, 1.0]
    K_vals = np.linspace(0.8, 1.2, 21) # Typical moneyness range
    tau_vals = np.array([0.05, 0.1, 0.2, 0.5]) # Typical maturities (e.g., 2w, 1m, 2m, 6m)
    
    K_grid, tau_grid = np.meshgrid(K_vals, tau_vals)
    K_flat = K_grid.flatten()
    tau_flat = tau_grid.flatten()

    K_tensor = torch.tensor(K_flat, dtype=torch.float64, device=device).unsqueeze(1)
    tau_tensor = torch.tensor(tau_flat, dtype=torch.float64, device=device).unsqueeze(1)

    logger.info(f"Generated {len(K_flat)} query points for surface extraction.")

    # Extraction
    extractor = GreekExtractor(price_net, localvol_net, device=device)
    features = extractor.extract_features(K_tensor, tau_tensor)

    # Save to CSV for Model 3 consumption
    output_df = pd.DataFrame({
        'K_norm': K_flat,
        'tau': tau_flat,
        'local_vol': features['local_vol'].cpu().numpy().flatten(),
        'vanna_proxy': features['vanna'].cpu().numpy().flatten(),
        'volga_proxy': features['volga'].cpu().numpy().flatten(),
        'lv_gradient_K': features['lv_gradient_K'].cpu().numpy().flatten()
    })
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    output_df.to_csv(args.out_file, index=False)
    
    logger.info(f"Feature extraction complete. Saved {len(output_df)} rows to {args.out_file}")
    
    # Print some stats to verify sanity (no infs/nans)
    logger.info(f"Local Vol Mean:  {output_df['local_vol'].mean():.4f}")
    logger.info(f"Vanna Proxy Mean: {output_df['vanna_proxy'].mean():.4f}")
    logger.info(f"Volga Proxy Mean: {output_df['volga_proxy'].mean():.4f}")

if __name__ == "__main__":
    main()
