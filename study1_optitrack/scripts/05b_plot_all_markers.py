import sys


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE   = "data/opti_aligned.csv"
OUTPUT_DIR   = "outputs/marker_trajectories"

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(INPUT_FILE)

# ── Collect all moving markers ──
x_cols = [c for c in df.columns if c.startswith('X')]
markers = []
for xc in x_cols:
    suffix = xc[1:]
    yc = 'Y' + suffix
    zc = 'Z' + suffix
    if yc not in df.columns or zc not in df.columns:
        continue
    x = pd.to_numeric(df[xc], errors='coerce')
    y = pd.to_numeric(df[yc], errors='coerce')
    z = pd.to_numeric(df[zc], errors='coerce')
    if x.notna().sum() < 100:
        continue
    std = max(x.std(), y.std())
    if std > 50:
        markers.append({
            'id': xc, 'suffix': suffix,
            'x': x, 'y': y, 'z': z,
            'x_mean': x.mean(), 'y_mean': y.mean(), 'z_mean': z.mean(),
            'std': std
        })

# Sort by Y height
markers.sort(key=lambda m: m['y_mean'], reverse=True)
print(f"Found {len(markers)} moving markers")
print(f"Saving individual trajectory plots to: {OUTPUT_DIR}/")

# ── Plot 1: ALL markers overview (front + side view) ──
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
colors = plt.cm.tab20(np.linspace(0, 1, len(markers)))
t = pd.to_numeric(df['Time (Seconds)'], errors='coerce').values

ax = axes[0]
for i, m in enumerate(markers):
    ax.plot(m['x'].values, m['y'].values,
            color=colors[i], alpha=0.4, linewidth=0.5)
    ax.scatter(m['x_mean'], m['y_mean'],
               color=colors[i], s=60, zorder=5)
    ax.annotate(m['id'], (m['x_mean'], m['y_mean']),
                fontsize=7, color=colors[i],
                xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm) Height')
ax.set_title('Front View — All Marker Trajectories\n(dot = mean position)')
ax.grid(True, alpha=0.3)

ax = axes[1]
for i, m in enumerate(markers):
    ax.plot(m['z'].values, m['y'].values,
            color=colors[i], alpha=0.4, linewidth=0.5)
    ax.scatter(m['z_mean'], m['y_mean'],
               color=colors[i], s=60, zorder=5)
    ax.annotate(m['id'], (m['z_mean'], m['y_mean']),
                fontsize=7, color=colors[i],
                xytext=(3, 3), textcoords='offset points')
ax.set_xlabel('Z (mm) Depth')
ax.set_ylabel('Y (mm) Height')
ax.set_title('Side View — All Marker Trajectories\n(dot = mean position)')
ax.grid(True, alpha=0.3)

plt.suptitle('All Moving Markers — Identify Your Body Markers',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/00_all_markers_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: 00_all_markers_overview.png")

# ── Plot 2: Individual trajectory for each marker ──
# Group into pages of 12
n_cols = 4
n_rows = 3
per_page = n_cols * n_rows
n_pages = int(np.ceil(len(markers) / per_page))

for page in range(n_pages):
    batch = markers[page*per_page : (page+1)*per_page]
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
    axes = axes.flatten()

    for idx, m in enumerate(batch):
        ax = axes[idx]
        # X trajectory (left-right)
        ax.plot(t, m['x'].values, color='blue', linewidth=0.8, label='X', alpha=0.8)
        # Y trajectory (height)
        ax.plot(t, m['y'].values, color='red',  linewidth=0.8, label='Y', alpha=0.8)
        # Z trajectory (depth)
        ax.plot(t, m['z'].values, color='green',linewidth=0.8, label='Z', alpha=0.8)

        ax.set_title(f'{m["id"]}  (Y_mean={m["y_mean"]:.0f}mm)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('mm', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for idx in range(len(batch), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Individual Marker Trajectories — Page {page+1}/{n_pages}\n'
                 f'Blue=X(left/right)  Red=Y(height)  Green=Z(depth)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fname = f'{OUTPUT_DIR}/page{page+1:02d}_markers.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: page{page+1:02d}_markers.png")

# ── Plot 3: 3D trajectory for each marker individually ──
print("\nSaving 3D trajectory plots...")
for i, m in enumerate(markers):
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111, projection='3d')

    x = m['x'].dropna().values
    y = m['y'].dropna().values
    z = m['z'].dropna().values
    n = min(len(x), len(y), len(z))

    sc = ax.scatter(x[:n], z[:n], y[:n],
                    c=np.arange(n), cmap='viridis',
                    s=2, alpha=0.6)
    ax.plot(x[:n], z[:n], y[:n],
            color='gray', linewidth=0.3, alpha=0.4)

    # Mark start and end
    ax.scatter([x[0]], [z[0]], [y[0]],
               color='green', s=80, zorder=5, label='Start')
    ax.scatter([x[-1]], [z[-1]], [y[-1]],
               color='red', s=80, zorder=5, label='End')

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Z (mm)')
    ax.set_zlabel('Y/Height (mm)')
    ax.set_title(f'{m["id"]}  —  Y_mean={m["y_mean"]:.0f}mm  std={m["std"]:.0f}mm',
                 fontweight='bold')
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='Time progression', shrink=0.5)
    plt.tight_layout()

    rank = i + 1
    fname = f'{OUTPUT_DIR}/rank{rank:02d}_{m["id"].replace(".","")}_3D.png'
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close()

print(f"\nAll plots saved to: {OUTPUT_DIR}/")
print(f"\nFiles to check:")
print(f"  1. 00_all_markers_overview.png  — see all markers at once")
print(f"  2. page01_markers.png etc       — X/Y/Z time series per marker")
print(f"  3. rank01_*.png etc             — individual 3D trajectory per marker")
print(f"\nLook at the plots and tell me which markers are:")
print(f"  Head (2), RHand (3), LHand (3), RKnee (2), LKnee (2)")