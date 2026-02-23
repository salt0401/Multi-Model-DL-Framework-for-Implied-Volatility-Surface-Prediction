"""Re-run TFT experiments only (after float32 fix).

Usage:
    python run_tft_experiments.py
"""
import subprocess
import sys
import os
from datetime import datetime

EXPERIMENTS = [
    {
        'name': 'TFT × adam (baseline, fp32)',
        'args': ['--model', 'tft', '--optimizer', 'adam', '--dtype', 'float32'],
    },
    {
        'name': 'TFT × adamw (fp32)',
        'args': ['--model', 'tft', '--optimizer', 'adamw', '--weight_decay', '0.01',
                 '--dtype', 'float32'],
    },
    {
        'name': 'TFT × cwd (fp32)',
        'args': ['--model', 'tft', '--optimizer', 'cwd', '--weight_decay', '0.01',
                 '--dtype', 'float32'],
    },
    {
        'name': 'TFT × cpr (fp32)',
        'args': ['--model', 'tft', '--optimizer', 'cpr', '--dtype', 'float32'],
    },
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, 'train_models.py')

    total_start = datetime.now()
    results = []

    print(f'=== TFT Regularization Experiments (float32 fix) ===')
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
                capture_output=False,
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
    print(f'All TFT experiments complete. Total time: {total_elapsed:.1f} min')
    print('\nSummary:')
    for r in results:
        icon = '✓' if r['status'] == 'OK' else '✗'
        print(f'  {icon} {r["name"]}: {r["status"]} ({r["minutes"]} min)')


if __name__ == '__main__':
    main()
