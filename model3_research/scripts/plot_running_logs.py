import os
import glob
import re
import json
import matplotlib.pyplot as plt

def parse_logs():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        print("Logs directory not found.")
        return
        
    # Find the most recently created log files for each variant
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    # We expect keywords 'tft_cpr', 'tft_adamw', 'baseline_cwd'
    target_prefixes = ['tft_cpr', 'tft_adamw', 'baseline_cwd']
    latest_logs = {p: None for p in target_prefixes}
    for lf in log_files:
        basename = os.path.basename(lf)
        for p in target_prefixes:
            if basename.startswith(p + "_adjustment"):
                if latest_logs[p] is None or os.path.getmtime(lf) > os.path.getmtime(latest_logs[p]):
                    latest_logs[p] = lf
                    
    # remove None values
    latest_logs = {k: v for k, v in latest_logs.items() if v is not None}
                
    if not latest_logs:
        print("No log files found.")
        return
        
    plt.figure(figsize=(14, 7))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    for idx, (prefix, log_path) in enumerate(latest_logs.items()):
        # Try to find corresponding metrics.json (which has epoch-by-epoch data)
        # prefix could be 'tft_cpr', 'tft_adamw', 'baseline_cwd'
        # The metrics file name format is usually {args.model}_{args.optimizer}_metrics.json
        # Wait, the filenames are like:
        # baseline_cwd_metrics.json
        # tft_adamw_fp32_metrics.json
        # tft_cpr_fp32_metrics.json
        metrics_candidates = glob.glob(os.path.join(log_dir, f"{prefix}*metrics.json"))
        
        epochs = []
        train_loss = []
        val_loss = []
        is_dense = False
        
        if metrics_candidates:
            # We found the completed metrics file - read dense epoch-by-epoch data!
            metrics_path = metrics_candidates[0]
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                train_loss = data.get('train_losses', [])
                val_loss = data.get('val_losses', [])
                epochs = list(range(1, len(train_loss) + 1))
            is_dense = True
            print(f"Loaded dense ({len(epochs)} epochs) metrics for {prefix} from JSON.")
        else:
            # Fallback for still-running models: read the console logs that only print every 10 epochs
            with open(log_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                match = re.search(r"Epoch\s+(\d+)/\d+.*?Train:\s+([\d.]+).*?Val:\s+([\d.]+)", line)
                if match:
                    epochs.append(int(match.group(1)))
                    train_loss.append(float(match.group(2)))
                    val_loss.append(float(match.group(3)))
            print(f"Parsed sparse ({len(epochs)} logs) metrics from {prefix}")
                
        if epochs:
            color = colors[idx % len(colors)]
            if is_dense:
                plt.plot(epochs, train_loss, linestyle='--', color=color, alpha=0.3, label=f'{prefix} Train')
                plt.plot(epochs, val_loss, linestyle='-', color=color, linewidth=2, label=f'{prefix} Val')
            else:
                plt.plot(epochs, train_loss, linestyle='--', marker='.', color=color, alpha=0.5, label=f'{prefix} Train (Live)')
                plt.plot(epochs, val_loss, linestyle='-', marker='o', color=color, linewidth=2, label=f'{prefix} Val (Live)')
        else:
            print(f"No epoch metrics found in {prefix}")
            
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curves - Dense Completed vs Sparse Live')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "live_loss_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    parse_logs()
