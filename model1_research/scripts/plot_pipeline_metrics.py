import sys, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
    metrics_file = os.path.join(log_dir, 'pipeline_stage1_metrics.json')
    
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found at {metrics_file}")
        sys.exit(1)
        
    with open(metrics_file, 'r') as f:
        data = json.load(f)
        
    train_losses = data['train_losses']
    val_losses = data['val_losses']
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4, alpha=0.8)
    plt.plot(epochs, val_losses, 'r-s', label='Val Loss', markersize=4, alpha=0.8)
    
    best_epoch = data.get('best_epoch', -1)
    if best_epoch >= 0:
        plt.axvline(x=best_epoch + 1, color='green', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch+1})')
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE + Arbitrage Penalties)')
    plt.title('Model 1 eSSVI+NN Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'm1_loss_curves.png')
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved loss curve to {out_path}")

if __name__ == '__main__':
    main()
