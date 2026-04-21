"""
Final Pipeline — Corrected Version
Evaluating Monocular RGB-Based Human Motion Estimation and Prediction
Against OptiTrack Ground Truth for Robot Navigation

Evaluation design:
  - KF applied to RGB only (to smooth noisy estimates)
  - OptiTrack = ground truth only (no KF applied)
  - KF-RGB predictions compared against OptiTrack ground truth
  - Horizons: 0.5s, 1.0s, 2.0s
"""
import sys
sys.path.insert(0, '/home/ataiwo/sandbox/packages')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
INPUT_CSV         = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/paired_dataset.csv"
OUT_DIR           = Path("/home/ataiwo/sandbox/NEWDATA ROBOTICS/results_final")
OUT_DIR.mkdir(exist_ok=True)

W, H              = 1920, 1080
FOV_DEG           = 65
fx                = W / (2 * np.tan(np.radians(FOV_DEG / 2)))
REAL_NOSE_KNEE_MM = 900.0
CONF_THRESH       = 0.1
NOSE_Y_JUMP       = 80
OPTI_Y_MIN        = 1650
HORIZONS          = [0.5, 1.0, 2.0]

COLORS = {
    'opti_gt' : '#1f77b4',  # blue  — OptiTrack ground truth
    'kf_rgb'  : '#d62728',  # red   — KF RGB prediction
}


# ═══════════════════════════════════════════════════════════════
# STEP 1: LOAD & FILTER
# ═══════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: LOADING & FILTERING")
print("=" * 65)
df = pd.read_csv(INPUT_CSV)
n0 = len(df)
print(f"Loaded: {n0} rows")

df = df[df['nose_conf'] > CONF_THRESH]
df = df[df['Head_y_mm'] > OPTI_Y_MIN]
df = df.sort_values('time_sec').reset_index(drop=True)
df = df[df['nose_y_px'].diff().abs() < NOSE_Y_JUMP]
df = df.dropna(subset=['left_knee_y_px', 'right_knee_y_px',
                        'nose_y_px', 'nose_x_px'])
df = df.reset_index(drop=True)
print(f"After filtering: {len(df)} rows ({100*len(df)/n0:.1f}% retained)")
df.to_csv(OUT_DIR / "01_filtered.csv", index=False)


# ═══════════════════════════════════════════════════════════════
# STEP 2: AXIS CORRECTIONS + DEPTH ESTIMATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: AXIS CORRECTIONS + DEPTH ESTIMATION")
print("=" * 65)

# Flip RGB axes to match OptiTrack orientation
df['nose_x_flip'] = W - df['nose_x_px']
df['nose_y_flip'] = H - df['nose_y_px']

# Geometric depth from nose-to-knee pixel span
df['knee_mid_y'] = (df['left_knee_y_px'] + df['right_knee_y_px']) / 2
df['pixel_h']    = df['knee_mid_y'] - df['nose_y_px']
df               = df[df['pixel_h'] > 20].reset_index(drop=True)
df['Z_rgb_raw']  = (fx * REAL_NOSE_KNEE_MM) / df['pixel_h']

# Scale to metric using OptiTrack Z
opti_z_shift     = df['Head_z_mm'] - df['Head_z_mm'].min() + 100
scale            = float(np.median(opti_z_shift / df['Z_rgb_raw']))
df['Z_rgb_mm']   = df['Z_rgb_raw'] * scale
df['Z_opti_mm']  = opti_z_shift.values

# Normalize RGB pixels to mm space
def norm_to_mm(rgb, opti):
    return ((rgb - rgb.mean()) *
            (opti.std() / rgb.std()) + opti.mean())

df['X_rgb_mm'] = norm_to_mm(df['nose_x_flip'], df['Head_x_mm'])
df['Y_rgb_mm'] = norm_to_mm(df['nose_y_flip'], df['Head_y_mm'])

print(f"Scale factor : {scale:.4f}")
df.to_csv(OUT_DIR / "02_depth_estimated.csv", index=False)
print("✅ Depth estimation complete")


# ═══════════════════════════════════════════════════════════════
# STEP 3: KALMAN FILTER ON RGB → PREDICT → COMPARE VS OPTITRACK
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3: KF ON RGB — PREDICTION vs OPTITRACK GROUND TRUTH")
print("=" * 65)
print("  KF applied to RGB only to smooth sensor noise")
print("  OptiTrack = ground truth only (no KF applied)")


class KalmanFilter1D:
    """1D Kalman Filter — state: [position, velocity]"""
    def __init__(self, dt, process_noise=1e4, meas_noise=1e4):
        self.dt = dt
        self.x  = None
        self.P  = np.eye(2) * 1e6
        self.F  = np.array([[1, dt], [0, 1]])
        self.H  = np.array([[1, 0]])
        self.Q  = np.eye(2) * process_noise
        self.R  = np.array([[meas_noise]])

    def update(self, z):
        if self.x is None:
            self.x = np.array([z, 0.0])
            return self.x.copy()
        x_p    = self.F @ self.x
        P_p    = self.F @ self.P @ self.F.T + self.Q
        S      = self.H @ P_p @ self.H.T + self.R
        K      = P_p @ self.H.T @ np.linalg.inv(S)
        self.x = x_p + K @ (np.array([z]) - self.H @ x_p)
        self.P = (np.eye(2) - K @ self.H) @ P_p
        return self.x.copy()

    def predict_ahead(self, horizon):
        steps = int(round(horizon / self.dt))
        F_h   = np.array([[1, self.dt * steps], [0, 1]])
        return float((F_h @ self.x)[0])


def run_kf_rgb(rgb_positions, opti_positions, times, horizon):
    """
    Apply KF to RGB positions.
    Predict future position at t + horizon.
    Compare predicted position against OptiTrack ground truth at t + horizon.
    """
    if len(times) < 2:
        return np.array([]), np.array([]), np.array([])

    dt_mean = float(np.median(np.diff(times)))
    kf      = KalmanFilter1D(dt=dt_mean)

    preds, gts, t_out = [], [], []
    for i, (rgb_z, t) in enumerate(zip(rgb_positions, times)):
        kf.update(rgb_z)                        # feed RGB into KF
        fut_idx = np.searchsorted(times, t + horizon)
        if fut_idx >= len(times):
            continue
        pred = kf.predict_ahead(horizon)        # predict future position
        gt   = opti_positions[fut_idx]          # compare vs OptiTrack GT
        preds.append(pred)
        gts.append(gt)
        t_out.append(t)

    return np.array(preds), np.array(gts), np.array(t_out)


# Axis configuration
axes_cfg = {
    'Horizontal Position': {
        'rgb' : df['X_rgb_mm'].values,
        'opti': df['Head_x_mm'].values,
        'time': df['time_sec'].values,
    },
    'Distance from Camera': {
        'rgb' : df['Z_rgb_mm'].values,
        'opti': df['Z_opti_mm'].values,
        'time': df['time_sec'].values,
    },
    'Vertical Position': {
        'rgb' : df['Y_rgb_mm'].values,
        'opti': df['Head_y_mm'].values,
        'time': df['time_sec'].values,
    },
}

results    = []
pred_store = {}

for ax_name, cfg in axes_cfg.items():
    pred_store[ax_name] = {}
    for h in HORIZONS:
        preds, gts, t_out = run_kf_rgb(
            cfg['rgb'], cfg['opti'], cfg['time'], h)

        pred_store[ax_name][h] = (preds, gts, t_out)

        if len(preds) == 0:
            continue
        err  = preds - gts
        rmse = float(np.sqrt(np.mean(err**2)))
        mae  = float(np.mean(np.abs(err)))
        results.append({
            'Axis'                  : ax_name,
            'Prediction Horizon (s)': h,
            'RMSE (mm)'             : round(rmse, 2),
            'MAE (mm)'              : round(mae,  2),
        })

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(OUT_DIR / "03_prediction_metrics.csv", index=False)
print("\n── Prediction Results ───────────────────────────────")
print(metrics_df.to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# STEP 4: ESTIMATION ERROR METRICS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: ESTIMATION ERROR METRICS")
print("=" * 65)

est_results = []
for ax_name, cfg in axes_cfg.items():
    err  = cfg['rgb'] - cfg['opti']
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    corr = float(np.corrcoef(cfg['opti'], cfg['rgb'])[0, 1])
    est_results.append({
        'Axis'       : ax_name,
        'RMSE (mm)'  : round(rmse, 2),
        'MAE (mm)'   : round(mae,  2),
        'Correlation': round(corr, 4),
    })
    print(f"  {ax_name}: RMSE={rmse:.1f}mm  "
          f"MAE={mae:.1f}mm  r={corr:.3f}")

est_df = pd.DataFrame(est_results)
est_df.to_csv(OUT_DIR / "04_estimation_metrics.csv", index=False)


# ═══════════════════════════════════════════════════════════════
# STEP 5: PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5: GENERATING PLOTS")
print("=" * 65)

time = df['time_sec']

# ── Plot 1: Estimation Comparison ────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Human Position Estimation: OptiTrack vs RGB Camera",
             fontsize=14, fontweight='bold')

plot_cfg = [
    ('Horizontal Position',  'Head_x_mm', 'X_rgb_mm'),
    ('Distance from Camera', 'Z_opti_mm', 'Z_rgb_mm'),
    ('Vertical Position',    'Head_y_mm', 'Y_rgb_mm'),
]
for i, (label, oc, rc) in enumerate(plot_cfg):
    axes[i,0].plot(time, df[oc], color=COLORS['opti_gt'], lw=0.8)
    axes[i,0].set_title(f'{label} — OptiTrack (Ground Truth)')
    axes[i,0].set_ylabel('Position (mm)')
    axes[i,0].grid(True, alpha=0.3)

    axes[i,1].plot(time, df[rc], color=COLORS['kf_rgb'], lw=0.8)
    axes[i,1].set_title(f'{label} — RGB Camera Estimate')
    axes[i,1].set_ylabel('Position (mm)')
    axes[i,1].grid(True, alpha=0.3)

for ax in axes[-1]:
    ax.set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_01_estimation_comparison.png", dpi=150)
print("✅ Plot 1: Estimation Comparison")

# ── Plot 2: Depth Overlay ─────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(time, df['Z_opti_mm'], color=COLORS['opti_gt'],
         lw=1.0, label='OptiTrack (Ground Truth)')
ax2.plot(time, df['Z_rgb_mm'],  color=COLORS['kf_rgb'],
         lw=0.8, alpha=0.8, label='RGB Camera Estimate')
ax2.set_title('Distance from Camera: OptiTrack vs RGB Estimate',
              fontsize=13)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Distance from Camera (mm)')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_02_depth_comparison.png", dpi=150)
print("✅ Plot 2: Depth Comparison")

# ── Plot 3: Estimation Error Over Time ────────────────────────
fig3, axes3 = plt.subplots(3, 1, figsize=(14, 10))
fig3.suptitle("Position Estimation Error Over Time\n"
              "(RGB Camera − OptiTrack Ground Truth)",
              fontsize=13, fontweight='bold')

err_cfg = [
    ('Horizontal Position',  df['X_rgb_mm'] - df['Head_x_mm'], 'b'),
    ('Vertical Position',    df['Y_rgb_mm'] - df['Head_y_mm'], 'g'),
    ('Distance from Camera', df['Z_rgb_mm'] - df['Z_opti_mm'], 'r'),
]
for i, (label, err, col) in enumerate(err_cfg):
    rmse = float(np.sqrt(np.mean(err**2)))
    axes3[i].plot(time, err, color=col, lw=0.6)
    axes3[i].axhline(0, color='k', lw=0.8, linestyle='--')
    axes3[i].fill_between(time, err, 0, alpha=0.15, color=col)
    axes3[i].set_title(f'{label} Error — RMSE = {rmse:.1f} mm')
    axes3[i].set_ylabel('Position Error (mm)')
    axes3[i].grid(True, alpha=0.3)
axes3[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_03_estimation_error.png", dpi=150)
print("✅ Plot 3: Estimation Error Over Time")

# ── Plot 4: Prediction RMSE vs Horizon ────────────────────────
fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))
fig4.suptitle("KF-RGB Prediction Error vs Time Horizon\n"
              "(compared against OptiTrack Ground Truth)",
              fontsize=13, fontweight='bold')

for i, ax_name in enumerate(axes_cfg.keys()):
    sub = metrics_df[metrics_df['Axis'] == ax_name]\
          .sort_values('Prediction Horizon (s)')
    axes4[i].plot(sub['Prediction Horizon (s)'],
                  sub['RMSE (mm)'],
                  color=COLORS['kf_rgb'],
                  marker='o', lw=2.0,
                  label='KF-RGB Prediction')
    axes4[i].set_title(ax_name)
    axes4[i].set_xlabel('Prediction Horizon (seconds)')
    axes4[i].set_ylabel('Prediction Error — RMSE (mm)')
    axes4[i].set_xticks(HORIZONS)
    axes4[i].legend(fontsize=9)
    axes4[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / "plot_04_prediction_vs_horizon.png", dpi=150)
print("✅ Plot 4: Prediction Error vs Horizon")

# ── Plot 5: KF Prediction vs Ground Truth (1.0s) ─────────────
fig5, axes5 = plt.subplots(3, 1, figsize=(14, 12))
fig5.suptitle("KF-RGB Trajectory Prediction vs OptiTrack Ground Truth\n"
              "Prediction Horizon: 1.0 second",
              fontsize=13, fontweight='bold')

for i, (ax_name, cfg) in enumerate(axes_cfg.items()):
    preds, gts, t_out = pred_store[ax_name][1.0]
    axes5[i].plot(t_out, gts,   color=COLORS['opti_gt'],
                  lw=1.0, label='Ground Truth (OptiTrack)')
    axes5[i].plot(t_out, preds, color=COLORS['kf_rgb'],
                  lw=0.8, linestyle='-',
                  label='KF Prediction (RGB input)')
    axes5[i].set_title(ax_name)
    axes5[i].set_ylabel('Position (mm)')
    axes5[i].legend(fontsize=9)
    axes5[i].grid(True, alpha=0.3)
axes5[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_05_kf_prediction_vs_gt.png", dpi=150)
print("✅ Plot 5: KF Prediction vs Ground Truth")

# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PIPELINE COMPLETE")
print("=" * 65)
print(f"\nAll outputs saved to: {OUT_DIR}")
print("\n── Estimation Results ───────────────────────────────")
print(est_df.to_string(index=False))
print("\n── Prediction Results (KF-RGB vs OptiTrack GT) ─────")
print(metrics_df.to_string(index=False))