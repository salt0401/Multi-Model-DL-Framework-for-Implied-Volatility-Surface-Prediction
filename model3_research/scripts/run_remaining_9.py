"""
Run the 9 remaining model-optimizer combinations for the full 12-way comparison.
All outputs go to scripts/logs/ and scripts/models/ (will be moved to archived after).

Already completed:
  - TFT + CPR (fp32)
  - TFT + AdamW (fp32)
  - GRU + CWD (fp64)
"""
import subprocess
import sys
import os
import time
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, 'train_models.py')

# 9 remaining combinations ordered by architecture (to reuse data loading cache)
COMBINATIONS = [
    # Batch 1: TFT (fp32) — ~200 min each
    {'model': 'tft', 'optimizer': 'cwd', 'dtype': 'float32'},
    {'model': 'tft', 'optimizer': 'adam', 'dtype': 'float32'},
    # Batch 2: GRU (fp64) — ~45 min each
    {'model': 'baseline', 'optimizer': 'cpr', 'dtype': 'float64'},
    {'model': 'baseline', 'optimizer': 'adamw', 'dtype': 'float64'},
    {'model': 'baseline', 'optimizer': 'adam', 'dtype': 'float64'},
    # Batch 3: xLSTM (fp64) — ~300 min each
    {'model': 'xlstm', 'optimizer': 'cpr', 'dtype': 'float64'},
    {'model': 'xlstm', 'optimizer': 'adamw', 'dtype': 'float64'},
    {'model': 'xlstm', 'optimizer': 'cwd', 'dtype': 'float64'},
    {'model': 'xlstm', 'optimizer': 'adam', 'dtype': 'float64'},
]

def main():
    log_path = os.path.join(SCRIPT_DIR, 'logs', 'run_remaining_9_progress.log')
    
    with open(log_path, 'w') as progress:
        progress.write(f"=== 9-Combination Runner Started: {datetime.now().isoformat()} ===\n")
        progress.flush()
        
        for i, combo in enumerate(COMBINATIONS, 1):
            label = f"{combo['model']}+{combo['optimizer']} ({combo['dtype']})"
            progress.write(f"\n[{i}/9] Starting {label} at {datetime.now().isoformat()}\n")
            progress.flush()
            
            cmd = [
                sys.executable, TRAIN_SCRIPT,
                '--model', combo['model'],
                '--optimizer', combo['optimizer'],
                '--dtype', combo['dtype'],
            ]
            
            print(f"\n{'='*60}")
            print(f"[{i}/9] {label}")
            print(f"{'='*60}")
            
            start = time.time()
            result = subprocess.run(cmd, cwd=SCRIPT_DIR)
            elapsed = (time.time() - start) / 60
            
            status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            progress.write(f"[{i}/9] {label}: {status} in {elapsed:.1f} min\n")
            progress.flush()
            
            print(f"[{i}/9] {label}: {status} ({elapsed:.1f} min)")
            
            if result.returncode != 0:
                progress.write(f"  WARNING: {label} failed, continuing to next...\n")
                progress.flush()
        
        progress.write(f"\n=== All 9 combinations finished: {datetime.now().isoformat()} ===\n")
        progress.flush()
    
    print("\n=== ALL 9 COMBINATIONS COMPLETE ===")
    print(f"Progress log: {log_path}")

if __name__ == '__main__':
    main()
