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




EXTENSION

Research Extension Plan
From CSCE 574 Pilot Study to a Stronger Research Project
Proposed Research Title
When Is Depth Necessary?
Decision-Aware Selective Depth for Human-Aware Robot Navigation

Alternative, more technical title:

Decision-Aware Selective Monocular Depth Estimation for Safe Human–Robot Navigation
1. Starting Point: What I Already Did

The original project investigated whether a monocular RGB camera could estimate and predict human motion for robot navigation.

The experimental pipeline was approximately:

Monocular RGB Video
        ↓
YOLOv8-Pose
        ↓
2D Human Keypoints
        ↓
Geometric 3D / Depth Estimation
        ↓
Kalman Filtering
        ↓
Future Human Position
        ↓
Human–Robot Distance
        ↓
Collision-Risk Decision

At the same time:

OptiTrack
    ↓
Ground-Truth 3D Human Position

The RGB-based results were compared against synchronized OptiTrack measurements.

The experiment contained approximately 80 seconds and 8,242 synchronized frames. The repository describes RGB acquisition at 120 Hz and OptiTrack at 125 Hz.

2. Important Finding From the Pilot Study

The interesting finding was not simply whether the RGB system worked or failed.

Performance was highly dependent on the spatial dimension being measured.

Horizontal/lateral tracking
$$ RMSE = 118.4\ mm $$ $$ r = 0.962 $$

This suggests that lateral human motion was represented relatively well.

Depth estimation
$$ RMSE = 711.5\ mm $$

This was substantially worse.

Downstream distance estimation
$$ MAE = 1198\ mm $$
Safety decision

Using the original 500-mm safety threshold:

$$ False\ Positive\ Rate = 65\% $$ $$ False\ Negative\ Rate = 0\% $$

The repository therefore demonstrates an important phenomenon:

Good performance on one perception metric does not necessarily produce a reliable downstream robot decision.

The README itself identifies this component-to-deployment gap as the central methodological finding.

3. The Main Limitation of the Original Work

There are several.

A. Weak depth-estimation assumption

The original system estimated depth geometrically from human body proportions.

Therefore, the poor depth result does not necessarily prove that monocular RGB itself is unsuitable.

It may instead demonstrate that:

The particular geometric depth estimator was inadequate.

This distinction needs to be investigated.

B. One participant

The original experiment cannot establish population-level generalization.

C. Controlled environment

Lighting, background, motion and occlusion complexity were limited.

D. Stationary robot assumption

The experiment did not fully represent simultaneous robot and human motion.

E. One safety threshold

The 500-mm threshold itself may contribute to the false-positive behavior.

These limitations are already acknowledged in the reframed study.

4. First New Research Question

Before developing an adaptive system, answer this:

Was the original safety failure caused primarily by monocular vision itself, or by the geometric depth-estimation method?

This should be Study 1 of the extension.

5. Study 1 — Modern Depth Re-Evaluation

Use the existing RGB data.

Instead of only:

RGB
 ↓
Geometric Depth

compare:

                    RGB
                     │
        ┌────────────┴────────────┐
        ↓                         ↓
 Geometric Depth          Learned Monocular
 Original Method              Depth Model
        │                         │
        └────────────┬────────────┘
                     ↓
               OptiTrack GT
                     ↓
                  Compare

Possible modern pretrained monocular depth models can be evaluated rather than trained from scratch.

Compare

A. Original geometric depth

B. Modern learned monocular depth

C. OptiTrack ground truth

6. Study 1 Metrics

Don't evaluate only depth RMSE.

Evaluate the complete chain.

Perception Level
Depth RMSE
Depth MAE
Position error
Correlation
Motion Level
Trajectory prediction error
Velocity error
Direction error
Decision Level
Human–robot distance error
False positive rate
False negative rate
Collision-risk accuracy
Unnecessary stop rate

This answers:

Does improving depth estimation actually improve the robot's downstream safety decision?

That is more meaningful than simply showing that one depth model has lower RMSE.

7. What Happens If Learned Depth Works?

Suppose learned depth substantially reduces:

$$ 711.5\text{ mm depth RMSE} $$

and reduces the:

$$ 65\%\text{ false-positive rate}. $$

Then you have established:

Better depth information is valuable for the navigation decision.

But that creates another question:

Does the robot need to pay the computational cost of obtaining that better depth on every frame?

And that becomes the main new research contribution.

8. Main Research Question
When is additional depth information actually necessary for a safe human–robot navigation decision?

More specifically:

Can uncertainty in RGB-based human tracking and safety estimation determine when a stronger monocular depth model should be invoked?

9. Proposed New System
Decision-Aware Selective Depth
                   RGB VIDEO
                       ↓
                Human Detection
                       ↓
                  2D Tracking
                       ↓
               Motion Prediction
                       ↓
             RGB Safety Estimate
                       ↓
              ┌────────────────┐
              │ Decision /     │
              │ Uncertainty    │
              │ Monitor        │
              └───────┬────────┘
                      ↓
               Is RGB enough?
                 /          \
               YES           NO
                ↓             ↓
          Skip Depth      Run Learned
                │            Depth
                │             ↓
                │       Better Distance
                │          Estimate
                │             │
                └──────┬──────┘
                       ↓
               Collision Risk
                       ↓
                STOP / PROCEED
10. What Would Be New?

The contribution should not be:

❌ “I used YOLO.”

or:

❌ “I used Depth Anything.”

or:

❌ “I used a Kalman filter.”

Those are existing tools.

The contribution becomes:

A decision-aware mechanism for selectively invoking stronger depth estimation when uncertainty in the human–robot safety decision indicates that RGB-only perception may be insufficient.

In simple words:

The system decides when depth is worth computing based on whether the robot actually needs that information to make a reliable safety decision.

11. What Could Trigger Depth?

Your repository already contains analyses of:

trajectory error;
velocity consistency;
direction similarity;
jitter;
distance error;
safety margin.

These give candidate signals for an uncertainty monitor.

Possible signals include:

Pose confidence

Is human detection/keypoint confidence deteriorating?

Motion instability

Is the estimated trajectory suddenly unstable?

Prediction uncertainty

Is future human position uncertain?

Distance-to-safety threshold

Suppose:

$$ Safety\ Threshold=500mm $$

and RGB estimates:

$$ Distance=3m. $$

You probably don't need expensive depth.

But if RGB estimates:

$$ Distance=550mm $$

then the decision is close to the boundary.

Better depth may be much more valuable.

Direction

Is the person approaching the robot or moving away?

12. A Particularly Strong Trigger

I would investigate decision-boundary uncertainty seriously.

Imagine:

$$ d_{safe}=500mm $$
Case 1

RGB predicts:

$$ d=4000mm $$

Clearly safe.

Depth may be unnecessary.

Case 2

RGB predicts:

$$ d=520mm $$

Now the estimate is extremely close to:

$$ 500mm. $$

This is precisely where an estimation error could flip:

$$ PROCEED $$

into:

$$ STOP $$

or vice versa.

So your controller could ask:

Is the predicted distance sufficiently close to the safety boundary that depth uncertainty could change the decision?

That is a strong decision-aware trigger.

13. Study 2 — Compare Four Depth Policies

Once the system works, compare:

Policy	Depth usage
RGB Only	Never
Always Depth	Every frame
Fixed-Rate Depth	Every \(N\) frames
Decision-Aware Depth — Ours	When the safety decision is uncertain

Keep everything else as constant as possible.

14. Evaluation Question

You're now evaluating:

$$ \boxed{\text{Safety Quality}} $$

against:

$$ \boxed{\text{Computational Cost}} $$
Safety metrics
False positives ↓
False negatives ↓
Unnecessary stops ↓
Missed-danger events ↓
Collision-risk accuracy ↑
Distance error ↓
Systems metrics
Latency ↓
FPS ↑
Memory ↓
Depth invocation rate ↓
15. Main Experiment

This would be especially strong:

Give:

Fixed-Rate

and:

Decision-Aware

approximately the same depth-computation budget.

For example:

$$ 20\%\text{ depth invocation} $$

for both.

Then ask:

At equal computational cost, does invoking depth near safety-critical/uncertain moments produce better navigation decisions than invoking it periodically?

That gets beyond:

“Using less depth is faster.”

Instead you're testing:

Does it matter when the computation is spent?

16. Study 3 — Safety Threshold Sensitivity

Your original work used:

$$ 500mm. $$

Don't assume that's the only meaningful threshold.

Test:

$$ 300mm,\quad500mm,\quad750mm,\quad1000mm $$

Then measure how the threshold changes:

false positives;
false negatives;
unnecessary stops;
depth invocation frequency.

Your existing paper already identifies threshold sensitivity as an important follow-up.

17. Study 4 — Public Dataset Validation

Don't immediately depend on collecting more participants.

Use your existing dataset for development, then add public data.

TUM RGB-D

Useful for:

$$ RGB + measured\ depth $$

This can help validate monocular depth estimates.

JRDB

Potentially more important for the final robotics story because it provides realistic robot-perspective scenes with pedestrians and 3D/LiDAR information.

Conceptually:

YOUR DATA
RGB + OptiTrack
      ↓
Proof of concept
      ↓
TUM RGB-D
      ↓
Depth validation
      ↓
JRDB
      ↓
Robot-perspective validation
18. Study 5 — New Human Experiment

Only after the method works would I collect another controlled OptiTrack dataset.

Instead of one participant simply walking naturally, design scenarios deliberately.

For each participant:

Lateral crossing
Direct approach
Moving away
Diagonal approach
Stop near safety threshold
Sudden direction change
Partial occlusion
Different walking speeds

Then recruit multiple participants.

This directly addresses the single-subject limitation in the original study.

19. Strong Research Questions

I would document these:

RQ1

How much does modern learned monocular depth improve human–robot distance and safety decisions compared with geometric monocular depth?

RQ2

Under what conditions is RGB-only perception sufficient for the downstream navigation decision?

RQ3

Can decision uncertainty reliably determine when additional depth computation is necessary?

RQ4

Can selective depth retain most of the safety benefit of always-on depth while substantially reducing depth computation?

RQ5

At equal depth-computation budgets, does decision-aware depth outperform fixed-rate depth?

RQ5 may ultimately become your strongest systems question.

20. Possible Paper Contributions

If the experiments support them, a future paper could claim contributions along the lines of:

1. Empirical characterization

Demonstrating how monocular depth errors propagate from perception into human–robot safety decisions.

2. Decision-aware depth controller

A mechanism that selectively invokes stronger depth estimation based on downstream safety uncertainty.

3. Multi-level evaluation

Evaluating:

$$ Perception \rightarrow Motion \rightarrow Safety \rightarrow Systems\ Cost. $$

4. Controlled policy comparison

RGB-only vs always-depth vs fixed-rate vs decision-aware depth.

5. Cross-dataset validation

Existing OptiTrack experiment + public robot/RGB-D data.

21. The Complete Research Story

This is how I would document the progression:

ORIGINAL CSCE 574 STUDY
        ↓
Monocular RGB human tracking
        ↓
Strong lateral tracking
        ↓
BUT
        ↓
Poor depth estimation
        ↓
Large human-robot distance error
        ↓
65% false-positive safety decisions
        ↓
        ↓
NEW QUESTION
        ↓
Is monocular vision fundamentally inadequate,
or was the geometric depth method the problem?
        ↓
        ↓
STUDY 1
        ↓
Geometric Depth
vs
Learned Monocular Depth
vs
Ground Truth
        ↓
        ↓
If better depth improves safety:
        ↓
        ↓
NEW SYSTEMS QUESTION
        ↓
Do we need that expensive depth
on EVERY frame?
        ↓
        ↓
PROPOSED METHOD
        ↓
Decision-Aware Selective Depth
        ↓
        ↓
RGB ONLY ──────────────┐
ALWAYS DEPTH ──────────┤
FIXED-RATE ────────────┼──→ Compare
DECISION-AWARE ────────┘
        ↓
        ↓
Safety Quality
vs
Computational Cost
My recommended final direction

Don't try to turn the old project into a publication merely by adding more participants.

Don't simply replace one depth estimator with another and stop there.

Use the original experiment as the pilot observation that motivates a new problem:

Accurate perception is not equally valuable at every moment; what matters is whether uncertainty can change the downstream robot decision.

Then investigate:

When Is Depth Necessary?
Decision-Aware Selective Depth for Human-Aware Robot Navigation

That gives the old CSCE 574 work a clear path from class project → pilot study → new method → controlled evaluation → broader validation → potential research paper.
