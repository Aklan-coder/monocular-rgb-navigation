# Evaluating Vision-Based Human Motion Prediction for Safe Robot Navigation

A multi-layer evaluation study quantifying how aggregate component metrics can mask deployment-level failure in learned perception systems.

## What this project is

A quantitative case study in evaluation methodology, using monocular RGB-based human motion prediction for safe robot navigation as the test domain. A YOLOv8-Pose + Kalman-filter pipeline is evaluated against synchronized OptiTrack motion-capture ground truth across 8,242 frames, with performance decomposed into three layers:

1. **Component metrics**: per-axis position error (RMSE, MAE, Pearson r) and multi-horizon trajectory prediction error (0.5 s, 1.0 s, 2.0 s).
2. **Task-derived metrics**: estimated human-robot distance vs. ground-truth distance.
3. **Deployment-level metrics**: collision-risk classification (true/false positives and negatives against a 500 mm safety threshold).

The central methodological finding: strong component-level performance on some axes (horizontal RMSE = 118 mm, r = 0.962) coexists with severe deployment-level failure (13.7× overestimate of collision risk, 65% false-positive rate, 0% false-negative rate). The headline tracking metric alone gives no indication of the deployment failure.

## Why this matters beyond robotics

This is a worked instance of a recurring pattern in learned-system evaluation: aggregate component metrics can systematically overstate deployment reliability when failure modes are correlated with task-critical conditions. The same structural pattern is a central concern in LLM and agent evaluation, where high aggregate benchmark scores can coexist with deployment-critical failure modes invisible to aggregate scoring. This project is a small but quantitatively explicit demonstration of why multi-layer evaluation is necessary.

## Methods

- **Data**: 80 s of synchronized monocular RGB video (120 Hz) and OptiTrack motion capture (125 Hz) of a single participant performing natural locomotion in a controlled lab.
- **Vision pipeline**: YOLOv8-Pose 2D keypoint detection, geometric 3D lifting using a nose-to-knee anthropometric prior, per-axis Kalman filtering for smoothing and short-horizon prediction.
- **Ground truth**: OptiTrack 3D head-marker average, synchronized within a 50 ms tolerance window.
- **Evaluation**: per-axis RMSE / MAE / Pearson r; multi-horizon prediction error; human-robot distance error; collision-risk classification against a 500 mm safety threshold.

Full methodological detail is in the accompanying paper.

## Key results

| Layer | Metric | Value | Interpretation |
| --- | --- | --- | --- |
| Component | Horizontal RMSE | 118.4 mm | Strong lateral tracking |
| Component | Horizontal r | 0.962 | High shape agreement |
| Component | Depth RMSE | 711.5 mm | Severe depth-axis failure |
| Task-derived | Distance MAE | 1,198 mm | Exceeds 500 mm safety threshold |
| Deployment | False-positive rate | 65.0% | Robot halts in 2 of every 3 frames |
| Deployment | False-negative rate | 0.0% | Consequence of systematic underestimation, not safety |
| Deployment | Risk overestimate | 13.7× | Aggregate vs. ground-truth unsafe-frame rate |

## Author and contribution

**Abdulahi Taiwo**. Sole author. PhD student, Computer Engineering (AI / Robotics / Computer Vision), University of South Carolina. Project completed for CSCE574 (Robotics), Spring 2026.

Designed the evaluation pipeline, implemented all code, conducted the data collection at the lab's OptiTrack facility, performed all analysis, and wrote the paper.

## Limitations and next steps

This is a single-subject case study in a controlled environment. The methodology generalizes; the specific magnitudes do not. The natural extension is replication on multi-subject benchmarks such as Human3.6M or 3DPW to test whether the same component-vs-deployment metric divergences appear across subject populations.

Separately, I am currently exploring how this multi-layer evaluation pattern applies to LLM and agent benchmarks (e.g., SWE-bench Verified, τ-bench), where similar gaps between aggregate scores and deployment-critical failure modes are emerging.
