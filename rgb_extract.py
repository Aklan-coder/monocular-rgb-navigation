import pandas as pd

input_path = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/Take 2026-04-10 01.50.05 PM DATA1.csv"
output_path = "/home/ataiwo/sandbox/NEWDATA ROBOTICS/opti_clean.csv"

# Your OptiTrack file has several metadata rows before the real table header.
# In the file you showed, the actual data header starts on the row:
# Frame, Time (Seconds), X, Y, Z, ...
df = pd.read_csv(input_path, header=5, low_memory=False)

# Time column in your OptiTrack CSV
time_col = "Time (Seconds)"

# Keep the same clean segments you used for RGB
segments = [
    (5, 11),
    (17, 65),
    (70, 85)
]

mask = False
for start, end in segments:
    mask = mask | ((df[time_col] >= start) & (df[time_col] <= end))

opti_clean = df[mask].copy()
opti_clean = opti_clean.sort_values(by=time_col).reset_index(drop=True)

opti_clean.to_csv(output_path, index=False)

print("Saved:", output_path)
print("Original rows:", len(df))
print("Clean rows:", len(opti_clean))