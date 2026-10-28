import sys

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# Paths
# -----------------------------
csv_path = "data/paired_dataset.csv"
fig_dir = "outputs/figures"
metrics_dir = "outputs/metrics"
data_dir = "data"
for d in (fig_dir, metrics_dir, data_dir):
    os.makedirs(d, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(csv_path)

# -----------------------------
# Select comparable 2D trajectories
# Reference trajectory = motion-capture reference
# Estimated trajectory = RGB pose estimate
# -----------------------------
ref = df[["Head_x_mm", "Head_y_mm"]].copy()
est = df[["nose_x_px", "nose_y_px"]].copy()

# Keep only valid rows
mask = ref.notna().all(axis=1) & est.notna().all(axis=1) & df["time_sec"].notna()
ref = ref[mask].to_numpy(dtype=float)
est = est[mask].to_numpy(dtype=float)
time_sec = df.loc[mask, "time_sec"].to_numpy(dtype=float)

if len(ref) < 3:
    raise ValueError("Not enough valid rows to compute trajectories and derivatives.")

# -----------------------------
# Affine alignment: reference -> estimated pixel space
# -----------------------------
model = LinearRegression()
model.fit(ref, est)
ref_aligned = model.predict(ref)

# -----------------------------
# Normalization helper
# -----------------------------
def normalize_trajectory(traj: np.ndarray) -> np.ndarray:
    traj = np.asarray(traj, dtype=float)
    traj0 = traj - traj[0]
    step_lengths = np.linalg.norm(np.diff(traj0, axis=0), axis=1)
    path_length = np.sum(step_lengths)
    if path_length == 0 or np.isnan(path_length):
        path_length = 1.0
    return traj0 / path_length

ref_norm = normalize_trajectory(ref_aligned)
est_norm = normalize_trajectory(est)

# -----------------------------
# Basic metrics
# -----------------------------
pos_errors = np.linalg.norm(ref_aligned - est, axis=1)
ADE = float(np.mean(pos_errors))
FDE = float(pos_errors[-1])

dt = np.diff(time_sec)
dt[dt == 0] = np.nan

ref_vel = np.diff(ref_aligned, axis=0) / dt[:, None]
est_vel = np.diff(est, axis=0) / dt[:, None]

vel_errors = np.linalg.norm(ref_vel - est_vel, axis=1)
velocity_ADE = float(np.nanmean(vel_errors))

# -----------------------------
# Normalized metrics
# -----------------------------
norm_errors = np.linalg.norm(ref_norm - est_norm, axis=1)
norm_ADE = float(np.mean(norm_errors))
norm_FDE = float(norm_errors[-1])

# -----------------------------
# Strong analysis metrics
# -----------------------------
# Velocity consistency: cosine similarity between velocity vectors
ref_vel_norm = np.linalg.norm(ref_vel, axis=1)
est_vel_norm = np.linalg.norm(est_vel, axis=1)
denom = ref_vel_norm * est_vel_norm
denom[denom == 0] = np.nan
cos_sim = np.sum(ref_vel * est_vel, axis=1) / denom
mean_cos_sim = float(np.nanmean(cos_sim))

# Direction similarity on step vectors
ref_step = np.diff(ref_aligned, axis=0)
est_step = np.diff(est, axis=0)
ref_step_norm = np.linalg.norm(ref_step, axis=1)
est_step_norm = np.linalg.norm(est_step, axis=1)
denom_dir = ref_step_norm * est_step_norm
denom_dir[denom_dir == 0] = np.nan
dir_sim = np.sum(ref_step * est_step, axis=1) / denom_dir
mean_dir_sim = float(np.nanmean(dir_sim))

# Jitter / noise: second derivative magnitude of estimated trajectory
est_acc = np.diff(est, n=2, axis=0)
jitter = np.linalg.norm(est_acc, axis=1)

# -----------------------------
# Application-level insight
# Choose a robot position in image space.
# Replace this with your actual robot-center pixel coordinate if available.
# -----------------------------
robot_pos = np.array([640.0, 360.0])  # example image center
safety_margin_px = 100.0              # example safety buffer

dist_ref = np.linalg.norm(ref_aligned - robot_pos, axis=1)
dist_est = np.linalg.norm(est - robot_pos, axis=1)
distance_error = np.abs(dist_ref - dist_est)

# Safety margin implications
ref_within = dist_ref <= safety_margin_px
est_within = dist_est <= safety_margin_px

false_safe = np.sum((~ref_within) & est_within)   # estimate says safe, reference says unsafe
false_unsafe = np.sum(ref_within & (~est_within)) # estimate says unsafe, reference says safe
total_frames = len(dist_ref)

false_safe_rate = float(false_safe / total_frames)
false_unsafe_rate = float(false_unsafe / total_frames)

# -----------------------------
# Save metrics summary
# -----------------------------
metrics = pd.DataFrame([{
    "ADE_px": ADE,
    "FDE_px": FDE,
    "Velocity_ADE_px_per_s": velocity_ADE,
    "Normalized_ADE": norm_ADE,
    "Normalized_FDE": norm_FDE,
    "Mean_Velocity_Cosine_Similarity": mean_cos_sim,
    "Mean_Direction_Similarity": mean_dir_sim,
    "Mean_Distance_Error_px": float(np.mean(distance_error)),
    "False_Safe_Rate": false_safe_rate,
    "False_Unsafe_Rate": false_unsafe_rate
}])

metrics_path = os.path.join(metrics_dir, "trajectory_metrics_summary.csv")
metrics.to_csv(metrics_path, index=False)

# -----------------------------
# Save aligned trajectory data
# -----------------------------
aligned_out = df.loc[mask, ["frame", "time_sec"]].copy()
aligned_out["ref_x_mm"] = ref[:, 0]
aligned_out["ref_y_mm"] = ref[:, 1]
aligned_out["est_x_px"] = est[:, 0]
aligned_out["est_y_px"] = est[:, 1]
aligned_out["ref_x_px_aligned"] = ref_aligned[:, 0]
aligned_out["ref_y_px_aligned"] = ref_aligned[:, 1]
aligned_out["pos_error_px"] = pos_errors
aligned_out["dist_ref_px"] = dist_ref
aligned_out["dist_est_px"] = dist_est
aligned_out["distance_error_px"] = distance_error

aligned_path = os.path.join(data_dir, "aligned_motion_data.csv")
aligned_out.to_csv(aligned_path, index=False)

# -----------------------------
# Print summary
# -----------------------------
print("Basic metrics")
print(f"ADE (px): {ADE:.4f}")
print(f"FDE (px): {FDE:.4f}")
print(f"Velocity ADE (px/s): {velocity_ADE:.4f}")
print()
print("Normalized comparison")
print(f"Normalized ADE: {norm_ADE:.6f}")
print(f"Normalized FDE: {norm_FDE:.6f}")
print()
print("Strong analysis")
print(f"Mean velocity cosine similarity: {mean_cos_sim:.6f}")
print(f"Mean direction similarity: {mean_dir_sim:.6f}")
print(f"Mean jitter magnitude: {float(np.mean(jitter)):.6f}")
print()
print("Application-level insight")
print(f"Mean distance error (px): {float(np.mean(distance_error)):.4f}")
print(f"False safe rate: {false_safe_rate:.4f}")
print(f"False unsafe rate: {false_unsafe_rate:.4f}")
print()
print("Saved files:")
print(metrics_path)
print(aligned_path)

# -----------------------------
# Plot 1: Aligned trajectories
# -----------------------------
plt.figure(figsize=(8, 7))
plt.plot(ref_aligned[:, 0], ref_aligned[:, 1], label="Reference (aligned)")
plt.plot(est[:, 0], est[:, 1], label="Estimated trajectory")
plt.title("Aligned trajectories in pixel space")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.legend()
plt.axis("equal")
plt.grid(True)

path1 = os.path.join(fig_dir, "01_aligned_trajectories.png")
plt.tight_layout()
plt.savefig(path1, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot 2: Normalized trajectories
# -----------------------------
plt.figure(figsize=(8, 7))
plt.plot(ref_norm[:, 0], ref_norm[:, 1], label="Reference (normalized)")
plt.plot(est_norm[:, 0], est_norm[:, 1], label="Estimated (normalized)")
plt.title("Normalized trajectories")
plt.xlabel("X (normalized)")
plt.ylabel("Y (normalized)")
plt.legend()
plt.axis("equal")
plt.grid(True)

path2 = os.path.join(fig_dir, "02_normalized_trajectories.png")
plt.tight_layout()
plt.savefig(path2, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot 3: Temporal error profile
# -----------------------------
plt.figure(figsize=(9, 4))
plt.plot(time_sec, pos_errors)
plt.title("Temporal error profile")
plt.xlabel("Time (s)")
plt.ylabel("Position error (px)")
plt.grid(True)

path3 = os.path.join(fig_dir, "03_temporal_error_profile.png")
plt.tight_layout()
plt.savefig(path3, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot 4: Velocity comparison
# -----------------------------
plt.figure(figsize=(9, 4))
plt.plot(time_sec[1:], np.linalg.norm(ref_vel, axis=1), label="Reference velocity")
plt.plot(time_sec[1:], np.linalg.norm(est_vel, axis=1), label="Estimated velocity")
plt.title("Velocity comparison")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (px/s)")
plt.legend()
plt.grid(True)

path4 = os.path.join(fig_dir, "04_velocity_comparison.png")
plt.tight_layout()
plt.savefig(path4, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot 5: Velocity consistency
# -----------------------------
plt.figure(figsize=(9, 4))
plt.plot(time_sec[1:], cos_sim)
plt.title("Velocity consistency")
plt.xlabel("Time (s)")
plt.ylabel("Cosine similarity")
plt.ylim(-1.05, 1.05)
plt.grid(True)

path5 = os.path.join(fig_dir, "05_velocity_consistency.png")
plt.tight_layout()
plt.savefig(path5, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot 6: Direction similarity
# -----------------------------
plt.figure(figsize=(9, 4))
plt.plot(time_sec[1:], dir_sim)
plt.title("Direction similarity")
plt.xlabel("Time (s)")
plt.ylabel("Cosine similarity")
plt.ylim(-1.05, 1.05)
plt.grid(True)

path6 = os.path.join(fig_dir, "06_direction_similarity.png")
plt.tight_layout()
plt.savefig(path6, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot 7: Jitter / noise
# -----------------------------
plt.figure(figsize=(9, 4))
plt.plot(time_sec[2:], jitter)
plt.title("Noise / jitter in estimated trajectory")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration magnitude")
plt.grid(True)

path7 = os.path.join(fig_dir, "07_jitter.png")
plt.tight_layout()
plt.savefig(path7, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot 8: Distance estimation error
# -----------------------------
plt.figure(figsize=(9, 4))
plt.plot(time_sec, distance_error)
plt.title("Distance estimation error")
plt.xlabel("Time (s)")
plt.ylabel("Absolute distance error (px)")
plt.grid(True)

path8 = os.path.join(fig_dir, "08_distance_error.png")
plt.tight_layout()
plt.savefig(path8, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot 9: Safety margin view
# -----------------------------
plt.figure(figsize=(9, 4))
plt.plot(time_sec, dist_ref, label="Reference distance to robot")
plt.plot(time_sec, dist_est, label="Estimated distance to robot")
plt.axhline(safety_margin_px, linestyle="--", label="Safety margin")
plt.title("Safety margin implication")
plt.xlabel("Time (s)")
plt.ylabel("Distance (px)")
plt.legend()
plt.grid(True)

path9 = os.path.join(fig_dir, "09_safety_margin.png")
plt.tight_layout()
plt.savefig(path9, dpi=300, bbox_inches="tight")
plt.close()

# -----------------------------
# Plot 10: Error heatmap on image plane
# -----------------------------
plt.figure(figsize=(7, 6))
sc = plt.scatter(est[:, 0], est[:, 1], c=pos_errors, s=8)
plt.colorbar(sc, label="Position error (px)")
plt.title("Spatial error distribution")
plt.xlabel("X (px)")
plt.ylabel("Y (px)")
plt.grid(True)

path10 = os.path.join(fig_dir, "10_error_heatmap.png")
plt.tight_layout()
plt.savefig(path10, dpi=300, bbox_inches="tight")
plt.close()

print()
print("Saved plots:")
print(path1)
print(path2)
print(path3)
print(path4)
print(path5)
print(path6)
print(path7)
print(path8)
print(path9)
print(path10)