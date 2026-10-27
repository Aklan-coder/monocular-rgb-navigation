
---

## Research Extension: Pedestrian Perception and Safety (Summer 2026)

**Research Assistant | University of South Carolina | Advisor: Dr. Lewis**

This research extends the original vision-based human motion prediction study by investigating pedestrian localization, uncertainty-aware safety decisions, and the relationship between perception errors and robot navigation behavior.

Unlike the original OptiTrack-based experiment, this extension evaluates pedestrian motion using the ETH/UCY benchmark.

### Research Objectives

1. Evaluate pedestrian localization accuracy against geometric and learned-depth baselines.
2. Investigate how localization uncertainty affects human–robot safety decisions.
3. Develop and analyze a lateral-only distance rule for conservative safety assessment.
4. Determine whether uncertainty margins can reduce unnecessary robot stops without increasing missed-danger cases.

### Experimental Setup

- **Dataset:** ETH/UCY pedestrian trajectories.
- **Evaluation population:** 898 pedestrians.
- **Method:** YOLOv8-Pose and Kalman-filter-based perception pipeline.
- **Evaluation protocol:** Leakage-free dataset splits.
- **Baselines:** Geometric localization and Depth Anything V2.
- **Metrics:** Planar localization RMSE, missed-danger cases, and unnecessary robot stops.

### Localization Results

| Method | Planar Localization RMSE |
|---|---:|
| Geometric localization | 1.39–1.72 m |
| Depth Anything V2 | 0.68–0.81 m |
| Evaluated perception approach | **0.51–0.59 m** |

The evaluated approach achieved lower planar localization error than the reported geometric and Depth Anything V2 baselines.

### Lateral-Only Safety Rule

Established a lateral-only distance rule that provides a mathematical lower bound on planar human–robot distance.

The research also used Shapley attribution to analyze the contribution of different error sources to downstream safety decisions.

### Uncertainty-Aware Safety Evaluation

A 2σ uncertainty margin was evaluated across eight cases.

**Observed results:**

- Matched the lateral-only rule's missed-danger rate in 8 out of 8 evaluated cases.
- Reduced unnecessary robot stops in 7 out of 8 evaluated cases.

These findings suggest that incorporating uncertainty into safety decisions can reduce unnecessary stopping while preserving the evaluated rule's missed-danger performance.

### Relationship to the Original Study

The original CSCE 574 study demonstrated that relatively accurate lateral tracking could coexist with poor depth estimates and unreliable downstream safety decisions.

This extension investigates more accurate localization and conservative safety-decision methods.

The experiments use different datasets and evaluation protocols and should not be interpreted as a direct comparison between the original OptiTrack study and the ETH/UCY extension.

### Research Limitations

- The safety results cover eight evaluated cases and do not establish universal safety guarantees.
- The localization comparisons depend on the reported dataset and evaluation protocol.
- The extension does not establish that the method will generalize to every robot navigation environment.

---

## Future Research

A proposed next direction is decision-aware selective depth estimation for human-aware robot navigation.

The proposed system would invoke additional depth estimation when uncertainty could change a downstream robot safety decision.

Potential evaluation directions include:

- RGB-only versus always-depth versus selectively activated depth.
- Safety quality versus computational cost.
- Depth invocation rate and processing latency.
- Generalization to additional robot-perspective datasets.

These are proposed future investigations, not completed experimental results.
