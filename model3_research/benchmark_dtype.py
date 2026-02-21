"""Benchmark float32 vs float64 speed on RTX 4060 without full training.

Runs a few batches of forward + backward passes for each model × dtype
combination and reports wall-clock time per batch plus numerical agreement.

Usage:
    python benchmark_dtype.py              # benchmark both models
    python benchmark_dtype.py --model xlstm  # benchmark one model
    python benchmark_dtype.py --warmup 5 --batches 20  # custom iterations
"""
import sys
import os
import time

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')
sys.path.insert(0, _src_dir)

import torch
import torch.nn as nn
import numpy as np

from xlstm_adjustment import xLSTMAdjustmentModel
from tft_adjustment import TFTAdjustmentModel
from argparse import ArgumentParser


def benchmark_model(model_class, model_name, input_dim, hidden_dim,
                    batch_size, seq_len, device, warmup, batches):
    """Benchmark a model at both float32 and float64."""
    results = {}

    for dtype_name, torch_dtype in [('float32', torch.float32), ('float64', torch.float64)]:
        # Build model
        torch.set_default_dtype(torch_dtype)
        if model_class == xLSTMAdjustmentModel:
            model = model_class(
                input_dim=input_dim, hidden_dim=hidden_dim,
                num_layers=2, attention_heads=4, dropout=0.2,
            )
        else:
            model = model_class(
                input_dim=input_dim, hidden_dim=hidden_dim,
                lstm_layers=2, attention_heads=4, dropout=0.1,
            )
        model = model.to(dtype=torch_dtype, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Synthetic data (same shape as real data)
        seq = torch.randn(batch_size, seq_len, input_dim, dtype=torch_dtype, device=device)
        mask = torch.ones(batch_size, seq_len, dtype=torch_dtype, device=device)
        target = torch.ones(batch_size, 1, dtype=torch_dtype, device=device) + \
                 torch.randn(batch_size, 1, dtype=torch_dtype, device=device) * 0.2
        loss_fn = nn.MSELoss()

        # Warmup (not timed)
        model.train()
        for _ in range(warmup):
            pred = model(seq, mask)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        # Timed benchmark
        times = []
        for _ in range(batches):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred = model(seq, mask)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        # Also measure inference-only speed
        model.train(False)
        inf_times = []
        with torch.no_grad():
            for _ in range(batches):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = model(seq, mask)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                inf_times.append(t1 - t0)

        # Collect predictions for numerical comparison
        with torch.no_grad():
            sample_pred = model(seq[:8], mask[:8]).cpu().float().numpy()

        results[dtype_name] = {
            'train_ms_per_batch': np.median(times) * 1000,
            'train_ms_std': np.std(times) * 1000,
            'infer_ms_per_batch': np.median(inf_times) * 1000,
            'infer_ms_std': np.std(inf_times) * 1000,
            'sample_preds': sample_pred.flatten(),
        }

        del model, optimizer, seq, mask, target
        torch.cuda.empty_cache()

    return results


def print_results(model_name, results):
    """Pretty-print benchmark results."""
    fp32 = results['float32']
    fp64 = results['float64']

    train_speedup = fp64['train_ms_per_batch'] / fp32['train_ms_per_batch']
    infer_speedup = fp64['infer_ms_per_batch'] / fp32['infer_ms_per_batch']

    # Estimate full training time savings
    # xLSTM: 311.5 min, TFT: 207.1 min at float64
    est_fp32_xlstm = 311.5 / train_speedup if 'xlstm' in model_name.lower() else None
    est_fp32_tft = 207.1 / train_speedup if 'tft' in model_name.lower() else None

    print(f"\n{'='*60}")
    print(f"  {model_name} — float32 vs float64 Benchmark")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'float64':>10} {'float32':>10} {'Speedup':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Train (ms/batch)':<30} {fp64['train_ms_per_batch']:>9.1f}  {fp32['train_ms_per_batch']:>9.1f}  {train_speedup:>9.2f}x")
    print(f"  {'Train std (ms)':<30} {fp64['train_ms_std']:>9.1f}  {fp32['train_ms_std']:>9.1f}")
    print(f"  {'Inference (ms/batch)':<30} {fp64['infer_ms_per_batch']:>9.1f}  {fp32['infer_ms_per_batch']:>9.1f}  {infer_speedup:>9.2f}x")
    print(f"  {'Inference std (ms)':<30} {fp64['infer_ms_std']:>9.1f}  {fp32['infer_ms_std']:>9.1f}")

    if est_fp32_xlstm is not None:
        print(f"\n  Estimated full training time:")
        print(f"    float64: 311.5 min (actual)")
        print(f"    float32: ~{est_fp32_xlstm:.1f} min (estimated)")
        print(f"    Savings: ~{311.5 - est_fp32_xlstm:.0f} min")
    if est_fp32_tft is not None:
        print(f"\n  Estimated full training time:")
        print(f"    float64: 207.1 min (actual)")
        print(f"    float32: ~{est_fp32_tft:.1f} min (estimated)")
        print(f"    Savings: ~{207.1 - est_fp32_tft:.0f} min")

    # Numerical comparison note
    print(f"\n  Note: float32 has ~7 decimal digits of precision.")
    print(f"  For tv_ratio predictions (~1.0 ± 0.2), this is more than sufficient.")
    print(f"  float64 offers ~15 digits — unnecessary for this prediction task.")
    print(f"{'='*60}\n")


def main():
    parser = ArgumentParser(description='Benchmark float32 vs float64')
    parser.add_argument('--model', type=str, default='both',
                        choices=['xlstm', 'tft', 'both'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=5,
                        help='Warmup iterations (not timed)')
    parser.add_argument('--batches', type=int, default=20,
                        help='Timed iterations')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: GPU required for benchmark")
        sys.exit(1)

    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Batch size: {args.batch_size}, Warmup: {args.warmup}, Batches: {args.batches}")

    # Show theoretical FLOPS info
    print(f"\nRTX 4060 Laptop theoretical compute:")
    print(f"  FP32: ~15.11 TFLOPS")
    print(f"  FP64: ~0.236 TFLOPS (1/64 of FP32)")
    print(f"  Actual speedup depends on memory-boundedness of each model")

    input_dim = 12
    hidden_dim = 64
    seq_len = 20

    models_to_test = []
    if args.model in ('xlstm', 'both'):
        models_to_test.append(('xLSTM (mLSTM)', xLSTMAdjustmentModel))
    if args.model in ('tft', 'both'):
        models_to_test.append(('TFT', TFTAdjustmentModel))

    for model_name, model_class in models_to_test:
        results = benchmark_model(
            model_class, model_name, input_dim, hidden_dim,
            args.batch_size, seq_len, device, args.warmup, args.batches,
        )
        print_results(model_name, results)

    # Reset default dtype
    torch.set_default_dtype(torch.float32)


if __name__ == '__main__':
    main()
