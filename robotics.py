import pandas as pd

# Paths
rgb_path = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/rgb_clean.csv"
opti_path = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/opti_clean.csv"

rgb_out = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/rgb_aligned.csv"
opti_out = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/opti_aligned.csv"

# Load data
rgb = pd.read_csv(rgb_path)
opti = pd.read_csv(opti_path)

# Time columns
rgb_time_col = "time_sec"
opti_time_col = "Time (Seconds)"

# RGB pose columns that must be present
rgb_pose_cols = [
    "nose_x_px", "nose_y_px",
    "left_wrist_x_px", "left_wrist_y_px",
    "right_wrist_x_px", "right_wrist_y_px"
]

# Convert RGB numeric columns
for col in [rgb_time_col] + rgb_pose_cols:
    rgb[col] = pd.to_numeric(rgb[col], errors="coerce")

# Convert OptiTrack time to numeric
opti[opti_time_col] = pd.to_numeric(opti[opti_time_col], errors="coerce")

# Drop rows with missing RGB pose data or missing time
rgb = rgb.dropna(subset=[rgb_time_col] + rgb_pose_cols).copy()

# Drop rows with missing OptiTrack time
opti = opti.dropna(subset=[opti_time_col]).copy()

# Round timestamps to reduce float mismatch
rgb[rgb_time_col] = rgb[rgb_time_col].round(3)
opti[opti_time_col] = opti[opti_time_col].round(3)

# Sort by time
rgb = rgb.sort_values(rgb_time_col).reset_index(drop=True)
opti = opti.sort_values(opti_time_col).reset_index(drop=True)

# Align OptiTrack to RGB timestamps
aligned = pd.merge_asof(
    rgb,
    opti,
    left_on=rgb_time_col,
    right_on=opti_time_col,
    direction="nearest",
    tolerance=0.01
)

# Drop RGB rows that did not find a valid OptiTrack match
aligned = aligned.dropna(subset=[opti_time_col]).copy()

# Split back into aligned files
rgb_aligned = aligned[rgb.columns].copy()
opti_aligned = aligned[opti.columns].copy()

# Save
rgb_aligned.to_csv(rgb_out, index=False)
opti_aligned.to_csv(opti_out, index=False)

print("Saved:", rgb_out)
print("Saved:", opti_out)
print("RGB aligned rows:", len(rgb_aligned))
print("OptiTrack aligned rows:", len(opti_aligned))