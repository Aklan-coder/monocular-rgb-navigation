import sys
sys.path.insert(0, '/home/ataiwo/sandbox/packages')

import subprocess
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    'matplotlib', 'scipy', 'pandas', 'numpy',
    '--target', '/home/ataiwo/sandbox/packages', '--quiet'
], check=True)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
INPUT_FILE  = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/paired_dataset.csv"
OUTPUT_DIR  = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/evaluation"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading data...")
df = pd.read_csv(INPUT_FILE)
t      = df['time_sec'].values
gt_x   = df['Head_x_mm'].values
gt_y   = df['Head_y_mm'].values
gt_z   = df['Head_z_mm'].values
rgb_x  = df['body_center_x_px'].values
rgb_y  = df['body_center_y_px'].values
lk_y   = df['left_knee_y_px'].values
rk_y   = df['right_knee_y_px'].values

print(f"  Frames: {len(df)} | Time: {t.min():.1f}s - {t.max():.1f}s")

# Normalize + flip
gt_x_norm  = (gt_x - gt_x.min()) / (gt_x.max() - gt_x.min())
rgb_x_norm = (rgb_x - rgb_x.min()) / (rgb_x.max() - rgb_x.min())
corr_test, _ = pearsonr(gt_x_norm, rgb_x_norm)
if corr_test < 0:
    rgb_x_norm = 1 - rgb_x_norm
corr, _ = pearsonr(gt_x_norm, rgb_x_norm)

# ══════════════════════════════════════════════
# PLOT 1 — Depth Ambiguity
# ══════════════════════════════════════════════
print("\nGenerating Plot 1: Depth Ambiguity...")
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(t, gt_z, color='blue', linewidth=1.2,
        label='Real depth Z — OptiTrack GT', alpha=0.9)
ax.axhline(gt_z.mean(), color='red', linestyle='--', linewidth=2,
           label='RGB: no depth available')
ax.fill_between(t, gt_z, gt_z.mean(), alpha=0.2, color='orange',
                label=f'Unobservable depth zone ({gt_z.max()-gt_z.min():.0f} mm range)')

ax.set_title('Plot 1: Depth Ambiguity — What Monocular RGB Cannot See',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Depth Z (mm)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot1_depth_ambiguity.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: plot1_depth_ambiguity.png")
print(f"     Depth range: {gt_z.max()-gt_z.min():.1f} mm")

# ══════════════════════════════════════════════
# PLOT 2 — Lateral Correlation
# ══════════════════════════════════════════════
print("\nGenerating Plot 2: Lateral Correlation...")
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(t, gt_x_norm,  color='blue', linewidth=1.2,
        label='OptiTrack lateral (normalized)', alpha=0.9)
ax.plot(t, rgb_x_norm, color='red',  linewidth=1.2,
        label='RGB lateral (normalized)',       alpha=0.9)

ax.set_title(f'Plot 2: Lateral Motion Comparison\nPearson r = {corr:.4f}',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Normalized position [0-1]', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot2_lateral_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: plot2_lateral_correlation.png")
print(f"     Pearson r = {corr:.4f}")

# ══════════════════════════════════════════════
# PLOT 3 — Real Walking Path (OptiTrack X vs Z)
# ══════════════════════════════════════════════
print("\nGenerating Plot 3: Real Walking Path...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: OptiTrack floor path (X vs Z)
ax = axes[0]
sc = ax.scatter(gt_x, gt_z, c=t, cmap='viridis', s=2, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Time (s)')
# Mark start and end
ax.scatter(gt_x[0],  gt_z[0],  color='green', s=150, zorder=5,
           label='Start', marker='o')
ax.scatter(gt_x[-1], gt_z[-1], color='red',   s=150, zorder=5,
           label='End',   marker='s')
ax.set_title('OptiTrack: Real 3D Walking Path\n(floor plane X vs Z)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('X (mm) — lateral', fontsize=10)
ax.set_ylabel('Z (mm) — depth', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Right: RGB observed path (x vs y pixels)
ax = axes[1]
sc2 = ax.scatter(rgb_x, rgb_y, c=t, cmap='viridis', s=2, alpha=0.7)
plt.colorbar(sc2, ax=ax, label='Time (s)')
ax.scatter(rgb_x[0],  rgb_y[0],  color='green', s=150, zorder=5,
           label='Start', marker='o')
ax.scatter(rgb_x[-1], rgb_y[-1], color='red',   s=150, zorder=5,
           label='End',   marker='s')
ax.set_title('RGB Camera: Observed 2D Path\n(pixel coordinates)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('x (pixels) — lateral', fontsize=10)
ax.set_ylabel('y (pixels) — image height', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Plot 3: Real 3D Path (OptiTrack) vs Observed 2D Path (RGB)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot3_walking_path.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: plot3_walking_path.png")

# ══════════════════════════════════════════════
# PLOT 4 — Speed Comparison
# ══════════════════════════════════════════════
print("\nGenerating Plot 4: Speed Comparison...")

# OptiTrack speed: real 3D speed in mm/s (X and Z components)
dt = np.gradient(t)
dx_gt = np.gradient(gt_x, t)
dz_gt = np.gradient(gt_z, t)
speed_gt  = np.sqrt(dx_gt**2 + dz_gt**2)   # real 3D speed mm/s

# RGB speed: pixel/s (lateral only)
dx_rgb    = np.gradient(rgb_x, t)
speed_rgb = np.abs(dx_rgb)                  # pixel/s

# Smooth both for cleaner visualization
speed_gt_smooth  = uniform_filter1d(speed_gt,  size=30)
speed_rgb_smooth = uniform_filter1d(speed_rgb, size=30)

# Normalize RGB speed for comparison
speed_rgb_norm = (speed_rgb_smooth - speed_rgb_smooth.min()) / \
                 (speed_rgb_smooth.max() - speed_rgb_smooth.min())
speed_gt_norm  = (speed_gt_smooth - speed_gt_smooth.min()) / \
                 (speed_gt_smooth.max() - speed_gt_smooth.min())

speed_corr, _ = pearsonr(speed_gt_norm, speed_rgb_norm)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Top: raw speeds with dual axis
ax = axes[0]
ax2 = ax.twinx()
ax.plot(t,  speed_gt_smooth,  color='blue', linewidth=1.2,
        label='OptiTrack speed (mm/s)', alpha=0.9)
ax2.plot(t, speed_rgb_smooth, color='red',  linewidth=1.2,
         label='RGB speed (pixels/s)', alpha=0.9)
ax.set_ylabel('OptiTrack speed (mm/s)', color='blue', fontsize=10)
ax2.set_ylabel('RGB speed (pixels/s)',  color='red',  fontsize=10)
ax.set_xlabel('Time (seconds)', fontsize=10)
ax.set_title('Speed Over Time: OptiTrack (3D) vs RGB (lateral only)',
             fontsize=11, fontweight='bold')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# Bottom: normalized comparison
ax = axes[1]
ax.plot(t, speed_gt_norm,  color='blue', linewidth=1.2,
        label='OptiTrack speed (normalized)', alpha=0.9)
ax.plot(t, speed_rgb_norm, color='red',  linewidth=1.2,
        label='RGB speed (normalized)',       alpha=0.9)
ax.set_title(f'Normalized Speed Comparison — Pearson r = {speed_corr:.4f}',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Time (seconds)', fontsize=10)
ax.set_ylabel('Normalized speed [0-1]', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle('Plot 4: Walking Speed Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot4_speed_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: plot4_speed_comparison.png")
print(f"     Speed correlation r = {speed_corr:.4f}")
print(f"     OptiTrack mean speed: {speed_gt_smooth.mean():.1f} mm/s")
print(f"     OptiTrack max speed:  {speed_gt_smooth.max():.1f} mm/s")

# ══════════════════════════════════════════════
# PLOT 5 — Signal Noise Comparison
# ══════════════════════════════════════════════
print("\nGenerating Plot 5: Signal Noise Comparison...")

# Compute noise = raw signal minus smoothed signal
window = 15
gt_smooth  = uniform_filter1d(gt_x_norm,  size=window)
rgb_smooth = uniform_filter1d(rgb_x_norm, size=window)

gt_noise   = gt_x_norm  - gt_smooth
rgb_noise  = rgb_x_norm - rgb_smooth

gt_noise_std  = gt_noise.std()
rgb_noise_std = rgb_noise.std()

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Top: raw vs smoothed OptiTrack
ax = axes[0]
ax.plot(t, gt_x_norm,  color='lightblue', linewidth=0.8, alpha=0.8, label='Raw OptiTrack')
ax.plot(t, gt_smooth,  color='blue',      linewidth=1.5, label='Smoothed OptiTrack')
ax.set_title(f'OptiTrack Signal — Noise std = {gt_noise_std:.5f}',
             fontsize=11, fontweight='bold')
ax.set_ylabel('Normalized position', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Middle: raw vs smoothed RGB
ax = axes[1]
ax.plot(t, rgb_x_norm, color='lightsalmon', linewidth=0.8, alpha=0.8, label='Raw RGB')
ax.plot(t, rgb_smooth, color='red',         linewidth=1.5, label='Smoothed RGB')
ax.set_title(f'RGB Signal — Noise std = {rgb_noise_std:.5f}',
             fontsize=11, fontweight='bold')
ax.set_ylabel('Normalized position', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom: noise comparison
ax = axes[2]
ax.plot(t, gt_noise,  color='blue', linewidth=0.8,
        label=f'OptiTrack noise (std={gt_noise_std:.5f})', alpha=0.8)
ax.plot(t, rgb_noise, color='red',  linewidth=0.8,
        label=f'RGB noise (std={rgb_noise_std:.5f})', alpha=0.8)
ax.set_title(f'Noise Comparison — RGB is {rgb_noise_std/gt_noise_std:.1f}x noisier than OptiTrack',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Time (seconds)', fontsize=10)
ax.set_ylabel('Noise (residual)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.5)

plt.suptitle('Plot 5: Signal Noise Comparison — OptiTrack vs RGB',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/plot5_signal_noise.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ Saved: plot5_signal_noise.png")
print(f"     OptiTrack noise std : {gt_noise_std:.5f}")
print(f"     RGB noise std       : {rgb_noise_std:.5f}")
print(f"     RGB is {rgb_noise_std/gt_noise_std:.1f}x noisier than OptiTrack")

# ── Summary ───────────────────────────────────────────────────────────
print()
print("="*55)
print("  ALL PLOTS GENERATED")
print("="*55)
print(f"  plot1_depth_ambiguity.png")
print(f"  plot2_lateral_correlation.png")
print(f"  plot3_walking_path.png")
print(f"  plot4_speed_comparison.png")
print(f"  plot5_signal_noise.png")
print()
print("KEY NUMBERS FOR PAPER:")
print(f"  Lateral correlation  : r = {corr:.4f}")
print(f"  Speed correlation    : r = {speed_corr:.4f}")
print(f"  Depth range lost     : {gt_z.max()-gt_z.min():.1f} mm")
print(f"  RGB noise ratio      : {rgb_noise_std/gt_noise_std:.1f}x noisier than OptiTrack")
print(f"  OptiTrack mean speed : {speed_gt_smooth.mean():.1f} mm/s")