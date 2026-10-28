# Study 2 — Decision-Level Error Attribution on ETH/UCY

Manuscript (in preparation): A. Taiwo and C. C. Ekechi, *Decision-Level Error Attribution and Depth-Aware Safety Evaluation of Monocular RGB Human Tracking for Autonomous Robot Navigation.*

> **Code status:** the evaluation code for this study is being consolidated and will be released here with the journal submission. This page summarises the method and the results reported in the manuscript.

## Why this study

[Study 1](../study1_optitrack/) found 0% missed dangers but 65% unnecessary stops for one participant, without being able to explain why. This study asks: **when a monocular RGB pipeline drives a stop/proceed rule, how much of the decision error comes from the rule's geometry, how much from localisation error, and how much from missed detections?**

## Method

- **Proposition 1:** a lateral-only distance is a lower bound on planar distance (Cauchy–Schwarz). On accurate input the lateral rule therefore cannot miss a danger, and its false stops follow from geometry rather than from the sensor.
- **Two-factor Shapley attribution** splits decision errors (FNR, FPR) into a rule share and a perception share. The perception share is split further into missed detections and localisation error.
- **Data:** ETH (seq_eth, seq_hotel) and UCY (zara01) via OpenTraj, 898 annotated pedestrians in total. Robots are simulated as stationary discs on a 1 m grid.
- **Pipeline:** YOLOv8m-Pose → three ground-plane localisers (M0 geometric, M1 homography ground plane, M2 Depth Anything V2 + calibration) → constant-velocity Kalman filter → lateral, planar, and planar + kσ uncertainty-margin rules.
- **Leakage control:** calibration and evaluation halves never share a pedestrian, and every parameter is fitted on the calibration half only. Intervals use a time-block bootstrap.

## Main results

| Localiser | Planar RMSE, Zara01 | Planar RMSE, Hotel |
|---|---|---|
| M0 geometric (Study 1 method) | 1.72 m | 1.39 m |
| M2 Depth Anything V2 + calibration | 0.81 m | 0.68 m |
| **M1 ground plane** | **0.59 m** | **0.51 m** |

- On annotated positions, the lateral rule had **zero missed dangers in all 12 scene–threshold cells**, with stop precision of only 5.6–31.9%, closely matching a closed-form occupancy model.
- Detector recall on these elevated views was 0.43 (Zara01) and 0.26 (Hotel), and **missed detections were the largest source of missed dangers**.
- With the M1 localiser, a **2σ uncertainty margin** on planar clearance missed no more danger than the lateral rule in **8/8** cells and produced fewer unnecessary stops in **7/8**. With the geometric baseline, the same margin multiplied unnecessary stops by 2.4–3.3×.

## Limitations

ETH/UCY ground truth is manual annotation at 2.5 Hz, not motion capture. Only two scenes were usable with the full RGB pipeline, both viewed from above rather than from a robot. Robots are simulated, so results are offline decision classifications, not evidence of collision avoidance.
