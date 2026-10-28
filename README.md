# Decision-Level Safety Evaluation of Monocular RGB Human Tracking for Robot Navigation

**Abdulahi Taiwo** · University of South Carolina · with Chijioke C. Ekechi

Accuracy metrics such as RMSE do not say whether a perception error changes a robot's stop/proceed decision. This repository evaluates monocular RGB human tracking by the decisions it drives, in two studies.

| | Study 1 (published) | Study 2 (extension, in preparation) |
|---|---|---|
| Question | Is a monocular RGB pipeline safe enough to drive stop/proceed decisions? | *Why* does it fail, and how much is the rule vs. perception? |
| Data | 1 participant, OptiTrack (125 Hz) + overhead RGB (120 Hz), 80 s | ETH/UCY pedestrian video, 898 pedestrians, 3 scenes |
| Pipeline | YOLOv8-Pose → geometric depth → Kalman filter | Same chain + ground-plane and Depth Anything V2 localisers |
| Key result | 0.0% missed dangers, but 65.0% unnecessary stops | Proof that the lateral rule lower-bounds planar distance; missed detections drive most missed dangers |
| Paper | [SSRN 7439432](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7439432) | Manuscript under preparation |
| Code | [`study1_optitrack/`](study1_optitrack/) — complete, reproduces the paper's numbers | [`study2_eth_ucy/`](study2_eth_ucy/) — results summary; code being consolidated |

Study 2 grew directly out of Study 1's limitations: Study 1 found the asymmetry (no misses, many false stops) but could not explain it from a single participant.

## Layout

```
study1_optitrack/
  scripts/   numbered pipeline, 01 → 09
  data/      raw OptiTrack export and intermediate CSVs
  outputs/   results_final/ (paper numbers), evaluation/, metrics/, figures/
  paper/     course report for the original study
study2_eth_ucy/
  README.md  method, results and code status
```
