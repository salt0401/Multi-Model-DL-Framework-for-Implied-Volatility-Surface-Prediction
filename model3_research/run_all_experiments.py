"""Run all 8 regularization experiments for Model 3.

Runs 4 optimizer configs × 2 models (TFT + GRU baseline).
TFT uses float32 for 3x speedup; GRU uses float64 (cuDNN optimized).

Expected total time: ~7 hours on RTX 4060 Laptop
  - TFT (float32): 4 × ~68 min = ~4.5 hr
  - GRU (float64): 4 × ~41 min = ~2.7 hr

Usage:
    python run_all_experiments.py
"""
import subprocess
import sys
import os
from datetime import datetime

EXPERIMENTS = [
    # === TFT experiments (float32 for speed) ===
    {
        'name': 'TFT × adam (baseline)',
        'args': ['--model', 'tft', '--optimizer', 'adam', '--dtype', 'float32'],
    },
    {
        'name': 'TFT × adamw',
        'args': ['--model', 'tft', '--optimizer', 'adamw', '--weight_decay', '0.01',
                 '--dtype', 'float32'],
    },
    {
        'name': 'TFT × cwd',
        'args': ['--model', 'tft', '--optimizer', 'cwd', '--weight_decay', '0.01',
                 '--dtype', 'float32'],
    },
    {
        'name': 'TFT × cpr',
        'args': ['--model', 'tft', '--optimizer', 'cpr', '--dtype', 'float32'],
    },

    # === GRU experiments (float64, cuDNN fast) ===
    {
        'name': 'GRU × adam (baseline)',
        'args': ['--model', 'baseline', '--optimizer', 'adam'],
    },
    {
        'name': 'GRU × adamw',
        'args': ['--model', 'baseline', '--optimizer', 'adamw', '--weight_decay', '0.01'],
    },
    {
        'name': 'GRU × cwd',
        'args': ['--model', 'baseline', '--optimizer', 'cwd', '--weight_decay', '0.01'],
    },
    {
        'name': 'GRU × cpr',
        'args': ['--model', 'baseline', '--optimizer', 'cpr'],
    },
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, 'train_models.py')

    total_start = datetime.now()
    results = []

    print(f'=== Model 3 Regularization Experiments ===')
    print(f'Total experiments: {len(EXPERIMENTS)}')
    print(f'Start time: {total_start.isoformat()}')
    print('=' * 60)

    for i, exp in enumerate(EXPERIMENTS):
        exp_start = datetime.now()
        print(f'\n[{i+1}/{len(EXPERIMENTS)}] {exp["name"]}')
        print(f'  Command: python train_models.py {" ".join(exp["args"])}')
        print(f'  Started: {exp_start.strftime("%H:%M:%S")}')

        try:
            result = subprocess.run(
                [sys.executable, train_script] + exp['args'],
                cwd=script_dir,
                capture_output=False,  # stream output in real time
                text=True,
            )
            status = 'OK' if result.returncode == 0 else f'FAILED (rc={result.returncode})'
        except Exception as e:
            status = f'ERROR: {e}'

        elapsed = (datetime.now() - exp_start).total_seconds() / 60
        print(f'  Finished: {status} ({elapsed:.1f} min)')
        results.append({
            'name': exp['name'],
            'status': status,
            'minutes': round(elapsed, 1),
        })

    total_elapsed = (datetime.now() - total_start).total_seconds() / 60
    print('\n' + '=' * 60)
    print(f'All experiments complete. Total time: {total_elapsed:.1f} min')
    print('\nSummary:')
    for r in results:
        icon = '✓' if r['status'] == 'OK' else '✗'
        print(f'  {icon} {r["name"]}: {r["status"]} ({r["minutes"]} min)')


if __name__ == '__main__':
    main()
