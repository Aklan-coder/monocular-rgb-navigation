import sys

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OPTI_FILE   = "data/opti_aligned.csv"
RGB_FILE    = "data/rgb_pose_extracted.csv"
OUTPUT_FILE = "data/paired_dataset.csv"

# Your confirmed head markers
HEAD_MARKERS = ['X.5', 'X.6', 'X.83']
# ─────────────────────────────────────────────

print("=" * 55)
print("  STEP 1: Extract OptiTrack Head Markers")
print("=" * 55)

opti = pd.read_csv(OPTI_FILE)
print(f"  Loaded: {len(opti)} frames | "
      f"{opti['Time (Seconds)'].min():.2f}s - {opti['Time (Seconds)'].max():.2f}s")

# Extract X, Y, Z for each head marker
head_data = pd.DataFrame()
head_data['time_sec'] = pd.to_numeric(opti['Time (Seconds)'], errors='coerce')

for xc in HEAD_MARKERS:
    suffix = xc[1:]     # '', '.5' etc — get the suffix after X
    # Find the actual column
    # X.5 → column name is 'X.5', Y is 'Y.5', Z is 'Z.5'
    if xc == 'X.5':
        yc, zc = 'Y.5', 'Z.5'
    elif xc == 'X.6':
        yc, zc = 'Y.6', 'Z.6'
    elif xc == 'X.83':
        yc, zc = 'Y.83', 'Z.83'
    else:
        yc = 'Y' + suffix
        zc = 'Z' + suffix

    head_data[f'{xc}_x'] = pd.to_numeric(opti[xc],  errors='coerce')
    head_data[f'{xc}_y'] = pd.to_numeric(opti[yc],  errors='coerce')
    head_data[f'{xc}_z'] = pd.to_numeric(opti[zc],  errors='coerce')
    nan_pct = head_data[f'{xc}_x'].isna().mean() * 100
    print(f"  {xc}: NaN={nan_pct:.1f}%")

# Average the 3 head markers → 1 clean head position
x_cols = [f'{m}_x' for m in HEAD_MARKERS]
y_cols = [f'{m}_y' for m in HEAD_MARKERS]
z_cols = [f'{m}_z' for m in HEAD_MARKERS]

head_data['Head_x_mm'] = head_data[x_cols].mean(axis=1)
head_data['Head_y_mm'] = head_data[y_cols].mean(axis=1)
head_data['Head_z_mm'] = head_data[z_cols].mean(axis=1)

# Keep only needed columns
opti_clean = head_data[['time_sec', 'Head_x_mm', 'Head_y_mm', 'Head_z_mm']].copy()
nan_pct = opti_clean['Head_x_mm'].isna().mean() * 100
print(f"\n  Head trajectory: {len(opti_clean)} frames | NaN={nan_pct:.1f}%")
print(f"  Head_x range: {opti_clean['Head_x_mm'].min():.1f} - {opti_clean['Head_x_mm'].max():.1f} mm")
print(f"  Head_z range: {opti_clean['Head_z_mm'].min():.1f} - {opti_clean['Head_z_mm'].max():.1f} mm")

print()
print("=" * 55)
print("  STEP 2: Compute RGB Body Center")
print("=" * 55)

rgb = pd.read_csv(RGB_FILE)
print(f"  Loaded: {len(rgb)} frames | "
      f"{rgb['time_sec'].min():.2f}s - {rgb['time_sec'].max():.2f}s")

# Compute body center from shoulders + hips
rgb['body_center_x_px'] = rgb[[
    'left_shoulder_x_px', 'right_shoulder_x_px',
    'left_hip_x_px',      'right_hip_x_px'
]].mean(axis=1)

rgb['body_center_y_px'] = rgb[[
    'left_shoulder_y_px', 'right_shoulder_y_px',
    'left_hip_y_px',      'right_hip_y_px'
]].mean(axis=1)

# Also keep nose for reference
rgb_clean = rgb[['time_sec', 'frame',
                  'body_center_x_px', 'body_center_y_px',
                  'nose_x_px', 'nose_y_px', 'nose_conf',
                  'left_knee_x_px',  'left_knee_y_px',
                  'right_knee_x_px', 'right_knee_y_px']].copy()

print(f"  body_center_x range: {rgb_clean['body_center_x_px'].min():.1f} - "
      f"{rgb_clean['body_center_x_px'].max():.1f} px")
print(f"  body_center_y range: {rgb_clean['body_center_y_px'].min():.1f} - "
      f"{rgb_clean['body_center_y_px'].max():.1f} px")

print()
print("=" * 55)
print("  STEP 3: Align on Timestamp (nearest match)")
print("=" * 55)

# Sort both by time
opti_clean = opti_clean.sort_values('time_sec').reset_index(drop=True)
rgb_clean  = rgb_clean.sort_values('time_sec').reset_index(drop=True)

# Merge using nearest timestamp
# For each RGB frame, find closest OptiTrack frame
from pandas import merge_asof

paired = merge_asof(
    rgb_clean.rename(columns={'time_sec': 'time_sec'}),
    opti_clean.rename(columns={'time_sec': 'time_sec_opti',
                                'Head_x_mm': 'Head_x_mm',
                                'Head_y_mm': 'Head_y_mm',
                                'Head_z_mm': 'Head_z_mm'}),
    left_on='time_sec',
    right_on='time_sec_opti',
    direction='nearest',
    tolerance=0.05   # max 50ms difference allowed
)

# Check alignment quality
time_diff = (paired['time_sec'] - paired['time_sec_opti']).abs()
print(f"  Aligned frames    : {len(paired)}")
print(f"  Max time gap      : {time_diff.max()*1000:.1f} ms")
print(f"  Mean time gap     : {time_diff.mean()*1000:.2f} ms")
print(f"  Unmatched frames  : {paired['Head_x_mm'].isna().sum()}")

# Final clean columns
final = paired[[
    'frame', 'time_sec',
    'Head_x_mm', 'Head_y_mm', 'Head_z_mm',
    'body_center_x_px', 'body_center_y_px',
    'nose_x_px', 'nose_y_px', 'nose_conf',
    'left_knee_x_px',  'left_knee_y_px',
    'right_knee_x_px', 'right_knee_y_px',
]].copy()

# Drop rows where either source has NaN
final = final.dropna(subset=['Head_x_mm', 'body_center_x_px'])

final.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 55)
print("  FINAL DATASET")
print("=" * 55)
print(f"  Saved to  : {OUTPUT_FILE}")
print(f"  Shape     : {final.shape}")
print(f"  Duration  : {final['time_sec'].min():.2f}s - {final['time_sec'].max():.2f}s")
print(f"  Columns   :")
for c in final.columns:
    print(f"    {c}")
print()
print("  Sample (first 3 rows):")
print(final.head(3).to_string())
print()
print("✅ Paired dataset ready for evaluation!")