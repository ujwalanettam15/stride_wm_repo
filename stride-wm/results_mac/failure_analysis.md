# Failure and error analysis (computed from saved results; nothing re-run)

## 1. Cells where a model is worse than copying the last frame (cube position error, cm)
Copy-last never moves the cube, so it is the right answer while the cube is at rest (Reach). A model that loses here is adding drift to static content.

| depth | phase | model | model_cm | copy_last_cm | excess_cm |
|---|---|---|---|---|---|
| 8 | reach | chunk k=1 | 3.54 | 3.04 | 0.50 |
| 8 | reach | chunk k=2 | 3.06 | 3.04 | 0.02 |
| 40 | reach | chunk k=1 | 5.85 | 5.35 | 0.50 |

## 2. Error per recursive call, and seed divergence
mse at the first call vs at the deepest shared depth; gap = max - min over seeds of mean latent MSE. Equal gaps at the first call that widen with depth are compounding, not seed luck.

|  | first_call_depth | mse_first | mse_at_40 | seed_gap_first | seed_gap_40 |
|---|---|---|---|---|---|
| DINO-WM k=5 | 5 | 0.14 | 0.26 | 0.00 | 0.04 |
| chunk k=1 | 1 | 0.11 | 0.50 | 0.00 | 0.13 |
| chunk k=2 | 2 | 0.12 | 0.36 | 0.00 | 0.05 |
| chunk k=4 | 4 | 0.13 | 0.27 | 0.00 | 0.03 |
| chunk k=5 | 5 | 0.14 | 0.26 | 0.00 | 0.04 |
| chunk k=8 | 8 | 0.14 | 0.23 | 0.00 | 0.06 |

## 3. Planning: success and progress by how far the cube starts from the goal
Success needs the cube within 4 cm. A start under 8 cm needs a nudge of at most 4 cm, so successes there are weak evidence of planning. Progress = start distance minus final distance (cm).

| model | start | pairs | success | ci | progress_cm |
|---|---|---|---|---|---|
| DINO-WM k=5 | 8-15 cm | 2 | 0/2 | 0-66% | 5.30 |
| DINO-WM k=5 | >=15 cm | 8 | 1/8 | 2-47% | 4.10 |
| chunk k=1 | 8-15 cm | 2 | 0/2 | 0-66% | 4.30 |
| chunk k=1 | >=15 cm | 8 | 0/8 | 0-32% | 3.60 |
| chunk k=2 | 8-15 cm | 2 | 0/2 | 0-66% | 0.70 |
| chunk k=2 | >=15 cm | 8 | 0/8 | 0-32% | 3.50 |
| chunk k=4 | 8-15 cm | 2 | 1/2 | 9-91% | 8.40 |
| chunk k=4 | >=15 cm | 8 | 1/8 | 2-47% | 4.30 |
| chunk k=5 | 8-15 cm | 2 | 1/2 | 9-91% | 7.30 |
| chunk k=5 | >=15 cm | 8 | 0/8 | 0-32% | 3.20 |
| chunk k=8 | 8-15 cm | 2 | 0/2 | 0-66% | 1.40 |
| chunk k=8 | >=15 cm | 8 | 0/8 | 0-32% | 3.70 |
| expert replay | 8-15 cm | 2 | 2/2 | 34-100% | 9.80 |
| expert replay | >=15 cm | 8 | 8/8 | 68-100% | 27.00 |
| random actions | 8-15 cm | 2 | 0/2 | 0-66% | 3.60 |
| random actions | >=15 cm | 8 | 0/8 | 0-32% | 3.50 |

Successes from planners with start < 8 cm: 0 of 4.
Random-action null: 0/10 succeed. A planner is only distinguishable from chance when its interval excludes this rate.

## 4. Is the planning cost informative? (true latents, no model)
Spearman rank correlation, per goal frame, between latent MSE to the goal and the true distance to the goal. 1 = the cost is a perfect proxy. Goals are from the last 30% of held-out episodes.

| distance | mean rho | goals |
|---|---|---|
| cube to goal cube | 0.28 | 250 |
| end effector to goal end effector | 0.83 | 250 |
| cube to goal cube, arm already within 3 cm of goal arm | 0.42 | 5 |

If the cube row is clearly below the end-effector row, the cost mostly rewards moving the arm.
