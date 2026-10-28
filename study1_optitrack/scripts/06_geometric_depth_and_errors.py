"""
Full Pipeline v2 — Fixes applied:
  1. Lower confidence threshold (0.3 → 0.1) — keep more valid data
  2. Flip RGB nose X (mirror correction)
  3. Flip RGB nose Y (image Y is downward, OptiTrack Y is upward)
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────
INPUT_CSV    = "data/paired_dataset.csv"
FILTERED_CSV = "data/head_nose_filtered.csv"
DEPTH_CSV    = "data/depth_estimated.csv"
METRICS_CSV  = "outputs/metrics/error_metrics.csv"
TRAJ_PLOT    = "outputs/figures/trajectory_comparison.png"
DEPTH_PLOT   = "outputs/figures/depth_comparison.png"
ERROR_PLOT   = "outputs/figures/error_plot.png"

# ── Camera Parameters ─────────────────────────────────────────
W, H    = 1920, 1080
FOV_DEG = 65
fx      = W / (2 * np.tan(np.radians(FOV_DEG / 2)))

# ── Anthropometric Prior ───────────────────────────────────────
REAL_NOSE_TO_KNEE_MM = 900.0

# ── Filtering Thresholds (FIXED) ──────────────────────────────
CONF_THRESH = 0.1    # FIXED: was 0.3, too aggressive for YOLO
NOSE_Y_JUMP = 80
OPTI_Y_MIN  = 1650


# ═══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)
df = pd.read_csv(INPUT_CSV)
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: FILTERING")
print("=" * 60)
n_before = len(df)

# Filter 1: Low nose confidence (lowered threshold)
df = df[df['nose_conf'] > CONF_THRESH]
print(f"After confidence filter  : {len(df)} rows "
      f"(removed {n_before - len(df)})")

# Filter 2: OptiTrack occlusions
n = len(df)
df = df[df['Head_y_mm'] > OPTI_Y_MIN]
print(f"After OptiTrack filter   : {len(df)} rows "
      f"(removed {n - len(df)})")

# Filter 3: RGB nose Y spikes
n = len(df)
df = df.sort_values('time_sec').reset_index(drop=True)
nose_y_diff = df['nose_y_px'].diff().abs()
df = df[nose_y_diff < NOSE_Y_JUMP]
print(f"After nose spike filter  : {len(df)} rows "
      f"(removed {n - len(df)})")

# Filter 4: Drop NaN keypoints
n = len(df)
df = df.dropna(subset=['left_knee_y_px', 'right_knee_y_px',
                        'nose_y_px', 'nose_x_px'])
print(f"After NaN filter         : {len(df)} rows "
      f"(removed {n - len(df)})")

print(f"\nTotal removed : {n_before - len(df)} rows "
      f"({100*(n_before - len(df))/n_before:.1f}%)")
print(f"Remaining     : {len(df)} rows")

df = df.reset_index(drop=True)
df.to_csv(FILTERED_CSV, index=False)
print(f"✅ Saved filtered data : {FILTERED_CSV}")


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: AXIS CORRECTIONS (FIXED)")
print("=" * 60)

# FIX 1: Flip RGB nose X — camera mirror correction
df['nose_x_px_flipped'] = W - df['nose_x_px']
print(f"✅ RGB nose X flipped (mirror): W - nose_x_px")

# FIX 2: Flip RGB nose Y — image Y is downward, OptiTrack Y is upward
df['nose_y_px_flipped'] = H - df['nose_y_px']
print(f"✅ RGB nose Y flipped (vertical): H - nose_y_px")


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: GEOMETRIC DEPTH ESTIMATION")
print("=" * 60)

# Knee midpoint Y
df['knee_mid_y_px'] = (df['left_knee_y_px'] + df['right_knee_y_px']) / 2

# Pixel height
df['pixel_height'] = df['knee_mid_y_px'] - df['nose_y_px']

# Remove invalid pixel heights
n = len(df)
df = df[df['pixel_height'] > 20].reset_index(drop=True)
print(f"Removed invalid pixel heights : {n - len(df)} rows")

# Raw geometric depth
df['Z_rgb_raw'] = (fx * REAL_NOSE_TO_KNEE_MM) / df['pixel_height']

print(f"\nRaw Z_rgb stats (mm):")
print(f"  min  : {df['Z_rgb_raw'].min():.1f}")
print(f"  max  : {df['Z_rgb_raw'].max():.1f}")
print(f"  mean : {df['Z_rgb_raw'].mean():.1f}")

# Scale calibration using OptiTrack Z
opti_z         = df['Head_z_mm'].values
opti_z_shifted = opti_z - opti_z.min() + 100
rgb_z          = df['Z_rgb_raw'].values
scale          = np.median(opti_z_shifted / rgb_z)

df['Z_rgb_mm']       = df['Z_rgb_raw'] * scale
df['Z_opti_shifted'] = opti_z_shifted

print(f"\nScale factor applied : {scale:.4f}")
print(f"\nCalibrated Z_rgb stats (mm):")
print(f"  min  : {df['Z_rgb_mm'].min():.1f}")
print(f"  max  : {df['Z_rgb_mm'].max():.1f}")
print(f"  mean : {df['Z_rgb_mm'].mean():.1f}")

df.to_csv(DEPTH_CSV, index=False)
print(f"✅ Saved depth data : {DEPTH_CSV}")


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: ERROR METRICS")
print("=" * 60)

def compute_metrics(gt, pred, name):
    err  = pred - gt
    rmse = np.sqrt(np.mean(err**2))
    mae  = np.mean(np.abs(err))
    maxe = np.max(np.abs(err))
    corr = np.corrcoef(gt, pred)[0, 1]
    print(f"\n  [{name}]")
    print(f"    RMSE        : {rmse:.2f} mm")
    print(f"    MAE         : {mae:.2f} mm")
    print(f"    Max Error   : {maxe:.2f} mm")
    print(f"    Correlation : {corr:.4f}")
    return {'Axis': name, 'RMSE_mm': round(rmse, 2),
            'MAE_mm': round(mae, 2), 'MaxError_mm': round(maxe, 2),
            'Correlation': round(corr, 4)}

# Normalize flipped RGB pixels to mm space
def normalize_to_mm(rgb_series, opti_series):
    return ((rgb_series - rgb_series.mean()) *
            (opti_series.std() / rgb_series.std()) +
             opti_series.mean())

nose_x_mm = normalize_to_mm(df['nose_x_px_flipped'], df['Head_x_mm'])
nose_y_mm = normalize_to_mm(df['nose_y_px_flipped'], df['Head_y_mm'])

metrics = []
metrics.append(compute_metrics(df['Head_x_mm'].values,
                                nose_x_mm.values,          'Lateral (X)'))
metrics.append(compute_metrics(df['Head_y_mm'].values,
                                nose_y_mm.values,           'Height (Y)'))
metrics.append(compute_metrics(df['Z_opti_shifted'].values,
                                df['Z_rgb_mm'].values,      'Depth (Z)'))

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(METRICS_CSV, index=False)
print(f"\n✅ Saved metrics : {METRICS_CSV}")
print(f"\n{metrics_df.to_string(index=False)}")


# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: PLOTTING")
print("=" * 60)

time = df['time_sec']

# ── Plot 1: Side-by-side Trajectory Comparison ────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("OptiTrack vs RGB — Trajectory Comparison (v2)",
             fontsize=14, fontweight='bold')

# Lateral X
axes[0,0].plot(time, df['Head_x_mm'],         'b-', lw=0.8)
axes[0,0].set_title('Lateral X — OptiTrack (mm)')
axes[0,0].set_ylabel('mm'); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(time, df['nose_x_px_flipped'], 'r-', lw=0.8)
axes[0,1].set_title('Lateral X — RGB Nose flipped (pixels)')
axes[0,1].set_ylabel('pixels'); axes[0,1].grid(True, alpha=0.3)

# Height Y
axes[1,0].plot(time, df['Head_y_mm'],         'b-', lw=0.8)
axes[1,0].set_title('Height Y — OptiTrack (mm)')
axes[1,0].set_ylabel('mm'); axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(time, df['nose_y_px_flipped'], 'r-', lw=0.8)
axes[1,1].set_title('Height Y — RGB Nose flipped (pixels)')
axes[1,1].set_ylabel('pixels'); axes[1,1].grid(True, alpha=0.3)

# Depth Z
axes[2,0].plot(time, df['Z_opti_shifted'],    'b-', lw=0.8)
axes[2,0].set_title('Depth Z — OptiTrack shifted (mm)')
axes[2,0].set_ylabel('mm'); axes[2,0].set_xlabel('time (s)')
axes[2,0].grid(True, alpha=0.3)

axes[2,1].plot(time, df['Z_rgb_mm'],          'r-', lw=0.8)
axes[2,1].set_title('Depth Z — RGB Estimated (mm)')
axes[2,1].set_ylabel('mm'); axes[2,1].set_xlabel('time (s)')
axes[2,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(TRAJ_PLOT, dpi=150)
print(f"✅ Trajectory plot : {TRAJ_PLOT}")

# ── Plot 2: Depth Overlay ──────────────────────────────────────
fig2, ax = plt.subplots(figsize=(14, 5))
ax.plot(time, df['Z_opti_shifted'], 'b-', lw=0.8,
        label='OptiTrack Z (shifted)')
ax.plot(time, df['Z_rgb_mm'],       'r-', lw=0.8, alpha=0.8,
        label='RGB Estimated Z')
ax.set_title('Depth Comparison: OptiTrack vs RGB Estimate')
ax.set_xlabel('time (s)'); ax.set_ylabel('mm')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(DEPTH_PLOT, dpi=150)
print(f"✅ Depth plot      : {DEPTH_PLOT}")

# ── Plot 3: Error Over Time ────────────────────────────────────
fig3, axes3 = plt.subplots(3, 1, figsize=(14, 10))
fig3.suptitle("Perception Error Over Time (RGB - OptiTrack)",
              fontsize=13, fontweight='bold')

errors = [
    (nose_x_mm - df['Head_x_mm'],           'b',
     f'Lateral X Error — RMSE={metrics[0]["RMSE_mm"]:.1f}mm'),
    (nose_y_mm - df['Head_y_mm'],            'g',
     f'Height Y Error  — RMSE={metrics[1]["RMSE_mm"]:.1f}mm'),
    (df['Z_rgb_mm'] - df['Z_opti_shifted'],  'r',
     f'Depth Z Error   — RMSE={metrics[2]["RMSE_mm"]:.1f}mm'),
]
for i, (err, col, title) in enumerate(errors):
    axes3[i].plot(time, err, color=col, lw=0.6)
    axes3[i].axhline(0, color='k', lw=0.8, linestyle='--')
    axes3[i].set_title(title)
    axes3[i].set_ylabel('error (mm)')
    axes3[i].grid(True, alpha=0.3)
axes3[2].set_xlabel('time (s)')

plt.tight_layout()
plt.savefig(ERROR_PLOT, dpi=150)
print(f"✅ Error plot      : {ERROR_PLOT}")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE — ALL OUTPUTS SAVED")
print("=" * 60)