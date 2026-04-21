import sys
sys.path.insert(0, '/home/ataiwo/sandbox/packages')
"""
3D Plot of OptiTrack Head Trajectory
- Identifies which axis is height, lateral, depth
"""
import sys
sys.path.insert(0, '/home/ataiwo/sandbox/packages')

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ── Paths ─────────────────────────────────────────────────────
CSV_PATH   = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/paired_dataset.csv"
PLOT_3D    = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/optitrack_3d_trajectory.png"
PLOT_AXES  = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/optitrack_axes_analysis.png"

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

x = df['Head_x_mm'].values
y = df['Head_y_mm'].values
z = df['Head_z_mm'].values

# ── Print Stats ───────────────────────────────────────────────
print("=" * 55)
print("OPTITRACK HEAD MARKER - AXIS ANALYSIS")
print("=" * 55)
print(f"  X: min={x.min():.1f}, max={x.max():.1f}, "
      f"range={x.max()-x.min():.1f}, std={x.std():.1f} mm")
print(f"  Y: min={y.min():.1f}, max={y.max():.1f}, "
      f"range={y.max()-y.min():.1f}, std={y.std():.1f} mm")
print(f"  Z: min={z.min():.1f}, max={z.max():.1f}, "
      f"range={z.max()-z.min():.1f}, std={z.std():.1f} mm")

# Identify axes automatically
stds = {'X': x.std(), 'Y': y.std(), 'Z': z.std()}
height_axis = min(stds, key=stds.get)
print(f"\n  → Likely HEIGHT axis : {height_axis} "
      f"(std={stds[height_axis]:.1f}mm, least variation)")
print(f"  → Mean head height   : "
      f"{dict(X=x.mean(), Y=y.mean(), Z=z.mean())[height_axis]:.1f} mm")

# ── Figure 1: 3D Trajectory ───────────────────────────────────
fig = plt.figure(figsize=(16, 6))

# 3D view
ax1 = fig.add_subplot(131, projection='3d')
sc = ax1.scatter(x, y, z, c=df['time_sec'], cmap='viridis',
                 s=1, alpha=0.6)
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
ax1.set_title('3D Head Trajectory\n(color = time)')
plt.colorbar(sc, ax=ax1, label='time (s)', shrink=0.5)

# Top-down view (X-Z plane) → floor movement
ax2 = fig.add_subplot(132)
ax2.scatter(x, z, c=df['time_sec'], cmap='viridis', s=1, alpha=0.6)
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Z (mm)')
ax2.set_title('Top-Down View (X-Z)\nFloor movement')
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal')

# Side view (X-Y plane)
ax3 = fig.add_subplot(133)
ax3.scatter(x, y, c=df['time_sec'], cmap='viridis', s=1, alpha=0.6)
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.set_title('Side View (X-Y)')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_3D, dpi=150, bbox_inches='tight')
print(f"\n✅ 3D plot saved to: {PLOT_3D}")

# ── Figure 2: Each Axis Over Time ────────────────────────────
fig2, axes = plt.subplots(3, 1, figsize=(14, 10))
fig2.suptitle("OptiTrack Head - Each Axis Over Time", fontsize=13)

time = df['time_sec']

axes[0].plot(time, x, 'b-', linewidth=0.8)
axes[0].set_title(f'X axis  — std={x.std():.1f}mm')
axes[0].set_ylabel('mm')
axes[0].grid(True, alpha=0.3)

axes[1].plot(time, y, 'g-', linewidth=0.8)
axes[1].set_title(f'Y axis  — std={y.std():.1f}mm  '
                  f'{"← LIKELY HEIGHT (least variation)" if height_axis=="Y" else ""}')
axes[1].set_ylabel('mm')
axes[1].grid(True, alpha=0.3)

axes[2].plot(time, z, 'r-', linewidth=0.8)
axes[2].set_title(f'Z axis  — std={z.std():.1f}mm  '
                  f'{"← LIKELY HEIGHT (least variation)" if height_axis=="Z" else ""}')
axes[2].set_ylabel('mm')
axes[2].set_xlabel('time (s)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_AXES, dpi=150, bbox_inches='tight')
print(f"✅ Axes plot saved to: {PLOT_AXES}")
print("\nShare both images to confirm axis interpretation.")