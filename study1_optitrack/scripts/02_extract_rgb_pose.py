"""
RGB Pose Extraction Script
- Uses YOLOv8-Pose
- CENTER CROP to isolate you from other people
- Extracts 7 keypoints needed for motion prediction:
    nose, left_shoulder, right_shoulder,
    left_hip, right_hip, left_knee, right_knee
- Saves annotated screenshots every N frames
- Keypoints remapped to full-frame pixel coordinates

Requirements:
    pip install ultralytics opencv-python pandas numpy
"""
import sys

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
VIDEO_PATH     = r"data/Take 2026-04-10 01.50.05 PM-Camera 17 (C11559).avi"  # raw RGB video (not in repo; see README)
OUTPUT_CSV     = "data/rgb_pose_extracted.csv"
SCREENSHOT_DIR = "outputs/screenshots"
SCREENSHOT_EVERY = 30          # 1 screenshot per second at 30fps

TIME_RANGES = [
    (5,  11),
    (17, 65),
    (70, 85),
]

MODEL_SIZE  = "yolov8m-pose.pt"
CONF_THRESH = 0.5

# Center crop — removes people on left/right edges
CROP_X1 = 460
CROP_X2 = 1680
CROP_Y1 = 0
CROP_Y2 = None
# ─────────────────────────────────────────────

# COCO 17 keypoints (full list — we extract all, save only needed)
ALL_KEYPOINTS = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]

# Only these 7 keypoints will be saved to CSV
SAVE_KEYPOINTS = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
]

# Skeleton connections for drawing
SKELETON = [
    (0,5),(0,6),           # nose to shoulders
    (5,6),                 # shoulders
    (5,7),(7,9),           # left arm
    (6,8),(8,10),          # right arm
    (5,11),(6,12),         # shoulders to hips
    (11,12),               # hips
    (11,13),(13,15),       # left leg
    (12,14),(14,16),       # right leg
]

# Colors
COLOR_BOX  = (0,255,0)
COLOR_KP   = (0,255,255)
COLOR_BONE = (255,128,0)
COLOR_SAVE = (0,0,255)    # red dot = saved keypoints


def is_in_range(t, ranges):
    return any(s <= t <= e for s, e in ranges)


def crop_frame(frame):
    h, w = frame.shape[:2]
    x1 = CROP_X1
    x2 = CROP_X2 if CROP_X2 else w
    y1 = CROP_Y1
    y2 = CROP_Y2 if CROP_Y2 else h
    return frame[y1:y2, x1:x2], x1, y1


def draw_frame(frame, boxes, kps_all, target_idx, t, frame_idx):
    vis = frame.copy()
    for i, box in enumerate(boxes):
        x1,y1,x2,y2 = map(int, box)
        is_target = (i == target_idx)
        color = COLOR_BOX if is_target else (0,0,255)
        cv2.rectangle(vis, (x1,y1),(x2,y2), color, 3 if is_target else 1)
        cv2.putText(vis, "YOU" if is_target else f"other",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if is_target:
            kps = kps_all[i]
            for a,b in SKELETON:
                xa,ya,ca = kps[a]; xb,yb,cb = kps[b]
                if ca>0.3 and cb>0.3:
                    cv2.line(vis,(int(xa),int(ya)),(int(xb),int(yb)),COLOR_BONE,2)
            for j,(x,y,c) in enumerate(kps):
                if c>0.3:
                    kp_name = ALL_KEYPOINTS[j]
                    color_kp = COLOR_SAVE if kp_name in SAVE_KEYPOINTS else COLOR_KP
                    cv2.circle(vis,(int(x),int(y)),5,color_kp,-1)
    # Info overlay
    for li, line in enumerate([f"Frame:{frame_idx}", f"t={t:.2f}s",
                                 f"People:{len(boxes)}", "RED=saved KPs"]):
        cv2.putText(vis, line, (10, 25+li*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    return vis


def extract():
    print(f"Loading model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {VIDEO_PATH}")

    fps     = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_w  = (CROP_X2 or frame_w) - CROP_X1

    ss_path = Path(SCREENSHOT_DIR)
    ss_path.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nVideo  : {Path(VIDEO_PATH).name}")
    print(f"  Full frame : {frame_w}x{frame_h} @ {fps:.1f}fps")
    print(f"  Crop       : x=[{CROP_X1}:{CROP_X2}] → {crop_w}px wide")
    print(f"  Time ranges: {TIME_RANGES}")
    print(f"  Saving keypoints: {SAVE_KEYPOINTS}")

    # Build CSV columns
    cols = ["frame", "time_sec"]
    for kp in SAVE_KEYPOINTS:
        cols += [f"{kp}_x_px", f"{kp}_y_px", f"{kp}_conf"]

    rows      = []
    frame_idx = 0
    extracted = 0
    no_det    = 0
    screenshots = 0

    print("\nExtracting...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        if not is_in_range(t, TIME_RANGES):
            frame_idx += 1
            continue

        # Crop
        cropped, x_off, y_off = crop_frame(frame)

        # Run YOLO
        results = model(cropped, conf=CONF_THRESH, verbose=False)
        row = {"frame": frame_idx, "time_sec": round(t, 4)}

        if results and results[0].keypoints is not None:
            kps_all = results[0].keypoints.data.cpu().numpy()
            boxes   = results[0].boxes.xyxy.cpu().numpy()

            if len(boxes) > 0:
                # Pick largest box
                areas      = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
                target_idx = int(np.argmax(areas))
                kps_target = kps_all[target_idx]

                # Save only needed keypoints
                for kp_name in SAVE_KEYPOINTS:
                    idx = ALL_KEYPOINTS.index(kp_name)
                    x, y, c = kps_target[idx]
                    row[f"{kp_name}_x_px"] = round(float(x) + x_off, 4)
                    row[f"{kp_name}_y_px"] = round(float(y) + y_off, 4)
                    row[f"{kp_name}_conf"] = round(float(c), 4)

                # Screenshot
                if extracted % SCREENSHOT_EVERY == 0:
                    ann = draw_frame(cropped, boxes, kps_all, target_idx, t, frame_idx)
                    cv2.imwrite(str(ss_path/f"frame_{frame_idx:06d}_t{t:.2f}s.jpg"), ann)
                    screenshots += 1
            else:
                no_det += 1
                for kp_name in SAVE_KEYPOINTS:
                    row[f"{kp_name}_x_px"] = np.nan
                    row[f"{kp_name}_y_px"] = np.nan
                    row[f"{kp_name}_conf"] = np.nan
        else:
            no_det += 1
            for kp_name in SAVE_KEYPOINTS:
                row[f"{kp_name}_x_px"] = np.nan
                row[f"{kp_name}_y_px"] = np.nan
                row[f"{kp_name}_conf"] = np.nan

        rows.append(row)
        extracted += 1

        if extracted % 200 == 0:
            print(f"  {extracted} frames | t={t:.1f}s | no_det={no_det}")

        frame_idx += 1

    cap.release()

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Done!")
    print(f"  Extracted : {extracted} frames")
    print(f"  No detect : {no_det} ({100*no_det/max(extracted,1):.1f}%)")
    print(f"  Screenshots: {screenshots}")
    print(f"  Saved to  : {OUTPUT_CSV}")
    print(f"  Shape     : {df.shape}")
    print(f"\nColumns saved:")
    for c in df.columns:
        print(f"  {c}")
    print(f"\nSample:")
    print(df.head(3).to_string())
    return df


if __name__ == "__main__":
    extract()