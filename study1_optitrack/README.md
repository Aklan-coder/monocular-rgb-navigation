# Study 1 — Monocular RGB vs. OptiTrack Ground Truth

Paper: A. Taiwo and C. C. Ekechi, *Decision-Level Safety Evaluation of Monocular RGB Human Motion Tracking for Robot Navigation*, [SSRN 7439432](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7439432).

A single participant walked, turned and paused for 80 s under a ceiling-mounted RGB camera (120 Hz) while an OptiTrack system (125 Hz, 13 markers) recorded ground truth. A YOLOv8-Pose → geometric depth → Kalman filter pipeline was compared against OptiTrack, and every frame was classified against a 500 mm safety threshold with a lateral-only distance rule.

## Results (reproduced by `scripts/08` and `scripts/07`)

| Axis | RMSE (mm) | MAE (mm) | Correlation r |
|---|---|---|---|
| Horizontal (lateral) | 118.4 | 84.7 | 0.962 |
| Depth (distance from camera) | 711.5 | 473.4 | 0.767 |
| Vertical | 19.1 | 14.4 | 0.323 |

| Prediction horizon | Lateral RMSE (mm) | Depth RMSE (mm) |
|---|---|---|
| 0.5 s | 246 | 810 |
| 1.0 s | 389 | 1,005 |
| 2.0 s | 719 | 1,529 |

**Safety decisions (500 mm threshold, lateral-only rule):** 0.0% false negatives, 65.0% false positives. The system predicted unsafe conditions in 70.1% of frames, while only 5.1% were truly unsafe.

**Takeaway:** a missed-danger rate of zero can hide a large false-stop burden, so false negatives and false positives must be reported together. Study 2 explains where this asymmetry comes from.

Sources: `outputs/results_final/04_estimation_metrics.csv`, `outputs/results_final/03_prediction_metrics.csv`, `outputs/evaluation/evaluation_summary.csv`.

## Pipeline

Run every script from this folder (`study1_optitrack/`), e.g. `python scripts/08_final_kalman_pipeline.py`.

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `01_clean_optitrack.py` | raw OptiTrack export | `data/opti_clean.csv` |
| 2 | `02_extract_rgb_pose.py` | raw RGB video | `data/rgb_pose_extracted.csv` |
| 3 | `03_align_streams.py` | `rgb_clean.csv`, `opti_clean.csv` | `data/rgb_aligned.csv`, `data/opti_aligned.csv` |
| 4 | `04_build_paired_dataset.py` | aligned streams | `data/paired_dataset.csv` |
| 5a | `05a_inspect_head_axes.py` | paired dataset | axis-identification figures |
| 5b | `05b_plot_all_markers.py` | `data/opti_aligned.csv` | per-marker trajectory plots |
| 6 | `06_geometric_depth_and_errors.py` | paired dataset | `outputs/metrics/error_metrics.csv`, depth figures |
| 7 | `07_safety_evaluation.py` | paired dataset | `outputs/evaluation/` (FN/FP, safety plots) |
| 8 | `08_final_kalman_pipeline.py` | paired dataset | `outputs/results_final/` — **the paper's numbers** |
| 9 | `09_trajectory_similarity_metrics.py` | paired dataset | `outputs/metrics/trajectory_metrics_summary.csv`, figures 01–10 |

Steps 5–9 run from the committed `data/paired_dataset.csv`; rerunning steps 6–8 reproduces the committed metrics exactly.

```
pip install -r requirements.txt
```

## Known gaps

- The raw RGB video (`Take 2026-04-10 … Camera 17.avi`) is not in the repository because of its size, so steps 1–4 cannot be rerun from scratch. Their outputs are committed in `data/`.
- Step 3 expects `rgb_clean.csv` (RGB keypoints trimmed to the same time segments as step 1). The script that produced it is not in the repository.
- `outputs/results_earlier_run/` is an earlier run kept for reference; `outputs/results_final/` supersedes it.
- `models/pose_landmarker_lite.task` (MediaPipe) is not used by the final pipeline, which uses YOLOv8-Pose.
- Single participant, single session: results are a proof of concept, not population-level evidence.
