"""Plot training curves from the 20-epoch GPU training run (Fixes 1+3 applied)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Training data from logs/base_model_20260219_132415.log
epochs = list(range(1, 21))

train_loss = [
    31299.359085, 7961.363834, 53188.640271, 433811.175582,
    6688190.292269, 37259.138441, 2381.736935, 221825.287934,
    2.511381, 2.542114, 2.616456, 2.723123,
    417.134650, 540.646093, 651.945504, 6217.396799,
    2.395959, 2.419997, 2.630675, 5327.708291
]

val_loss = [
    574.927668, 1001.196184, 23934.688263, 67575.108233,
    389186.834514, 16727704.576099, 2.026525, 2.048517,
    2.042926, 2.085205, 2.162704, 2.237895,
    2.310071, 2.129016, 2.363707, 1.966668,
    1.939649, 2.019014, 2.097883, 2.022951
]

best_epochs = [1, 7, 16, 17]  # epochs where best model was saved

out_dir = os.path.join(os.path.dirname(__file__), '..', 'LossPlot')
os.makedirs(out_dir, exist_ok=True)

# ── Figure 1: Full training curve (log scale) ───────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.semilogy(epochs, train_loss, 'b-o', markersize=4, label='Train Loss', alpha=0.8)
ax1.semilogy(epochs, val_loss, 'r-s', markersize=4, label='Val Loss', alpha=0.8)
for be in best_epochs:
    ax1.axvline(x=be, color='green', linestyle='--', alpha=0.3)
ax1.axvspan(3, 6, alpha=0.1, color='red', label='Gradient Explosion Zone')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (log scale)', fontsize=12)
ax1.set_title('Full Training Curve (20 epochs, log scale)\nFixes: rho=-0.5, per-day y_atm', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs)

# ── Figure 2: Stable phase (epochs 7-20, linear scale) ─────────────
stable_epochs = epochs[6:]  # epoch 7 onward
stable_val = val_loss[6:]

ax2.plot(stable_epochs, stable_val, 'r-s', markersize=6, label='Val Loss', linewidth=2)
for be in [7, 16, 17]:
    idx = be - 7
    if 0 <= idx < len(stable_val):
        ax2.plot(be, stable_val[idx], '*', color='gold', markersize=15, zorder=5)
ax2.axhline(y=1.939649, color='green', linestyle='--', alpha=0.5, label=f'Best: 1.9396 (ep17)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Val Loss', fontsize=12)
ax2.set_title('Validation Loss - Stable Phase (epoch 7-20)\nBest model: epoch 17, val_loss=1.9396', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(stable_epochs)

plt.tight_layout()
fig.savefig(os.path.join(out_dir, 'fix1_fix3_training_20ep.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(out_dir, 'fix1_fix3_training_20ep.png')}")

# ── Figure 3: Summary table ─────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(12, 8))
ax3.axis('off')

# Summary table
col_labels = ['Epoch', 'Train Loss', 'Val Loss', 'Status']
table_data = []
for i, (e, tl, vl) in enumerate(zip(epochs, train_loss, val_loss)):
    status = ''
    if e in best_epochs:
        status = 'NEW BEST'
    elif 3 <= e <= 6:
        status = 'EXPLOSION'
    elif tl > 100 and e > 7:
        status = 'train spike'

    tl_str = f'{tl:,.2f}' if tl > 10 else f'{tl:.6f}'
    vl_str = f'{vl:,.2f}' if vl > 10 else f'{vl:.6f}'
    table_data.append([str(e), tl_str, vl_str, status])

table = ax3.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color code rows
for i, (e, tl, vl) in enumerate(zip(epochs, train_loss, val_loss)):
    row_idx = i + 1  # +1 for header
    if e in best_epochs:
        for j in range(4):
            table[row_idx, j].set_facecolor('#c8e6c9')  # green
    elif 3 <= e <= 6:
        for j in range(4):
            table[row_idx, j].set_facecolor('#ffcdd2')  # red

ax3.set_title('20-Epoch Training Summary\n(Green = New Best, Red = Gradient Explosion)',
              fontsize=14, fontweight='bold', pad=20)

fig2.savefig(os.path.join(out_dir, 'fix1_fix3_summary_table.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {os.path.join(out_dir, 'fix1_fix3_summary_table.png')}")

plt.close('all')
print("\nDone!")
