# Failure and error analysis (computed from saved results; nothing re-run)

## 1. Cells where a model is worse than copying the last frame (cube position error, cm)
Copy-last never moves the cube, so it is the right answer while the cube is at rest (Reach). A model that loses here is adding drift to static content.

| depth | phase | model | model_cm | copy_last_cm | excess_cm |
|---|---|---|---|---|---|
| 8 | reach | chunk k=1 | 3.75 | 2.67 | 1.07 |
| 8 | reach | chunk k=2 | 2.99 | 2.67 | 0.31 |
| 8 | reach | chunk k=4 | 2.92 | 2.67 | 0.24 |
| 8 | reach | chunk k=8 | 2.68 | 2.67 | 0.00 |
| 40 | reach | chunk k=1 | 6.49 | 4.42 | 2.08 |
| 40 | reach | chunk k=2 | 4.52 | 4.42 | 0.10 |

## 2. Error per recursive call, and seed divergence
mse at the first call vs at the deepest shared depth; gap = max - min over seeds of mean latent MSE. Equal gaps at the first call that widen with depth are compounding, not seed luck.

|  | first_call_depth | mse_first | mse_at_40 | seed_gap_first | seed_gap_40 |
|---|---|---|---|---|---|
| DINO-WM k=5 | 5 | 0.14 | 0.32 | 0.00 | 0.12 |
| chunk k=1 | 1 | 0.12 | 0.49 | 0.00 | 0.14 |
| chunk k=2 | 2 | 0.13 | 0.36 | 0.00 | 0.05 |
| chunk k=4 | 4 | 0.14 | 0.32 | 0.00 | 0.08 |
| chunk k=5 | 5 | 0.14 | 0.32 | 0.00 | 0.09 |
| chunk k=8 | 8 | 0.15 | 0.24 | 0.00 | 0.05 |

## 3. Planning: success and progress by how far the cube starts from the goal
Success needs the cube within 4 cm. A start under 8 cm needs a nudge of at most 4 cm, so successes there are weak evidence of planning. Progress = start distance minus final distance (cm).

| model | start | pairs | success | ci | progress_cm |
|---|---|---|---|---|---|
| DINO-WM k=5 | 8-15 cm | 9 | 1/9 | 2-44% | 0.20 |
| DINO-WM k=5 | >=15 cm | 21 | 3/21 | 5-35% | 5.40 |
| DINO-WM k=5 · probe_cube cost | 8-15 cm | 9 | 1/9 | 2-44% | 0.90 |
| DINO-WM k=5 · probe_cube cost | >=15 cm | 21 | 1/21 | 1-23% | 5.60 |
| chunk k=1 | 8-15 cm | 30 | 3/30 | 3-26% | 1.60 |
| chunk k=1 | >=15 cm | 60 | 6/60 | 5-20% | 4.00 |
| chunk k=1 · probe_cube cost | 8-15 cm | 30 | 6/30 | 10-37% | 1.80 |
| chunk k=1 · probe_cube cost | >=15 cm | 60 | 3/60 | 2-14% | 3.80 |
| chunk k=2 | 8-15 cm | 30 | 3/30 | 3-26% | 2.30 |
| chunk k=2 | >=15 cm | 60 | 6/60 | 5-20% | 3.70 |
| chunk k=2 · probe_cube cost | 8-15 cm | 30 | 2/30 | 2-21% | 1.60 |
| chunk k=2 · probe_cube cost | >=15 cm | 60 | 5/60 | 4-18% | 4.00 |
| chunk k=4 | 8-15 cm | 30 | 4/30 | 5-30% | 1.70 |
| chunk k=4 | >=15 cm | 60 | 7/60 | 6-22% | 4.60 |
| chunk k=4 · probe_cube cost | 8-15 cm | 30 | 4/30 | 5-30% | 2.00 |
| chunk k=4 · probe_cube cost | >=15 cm | 60 | 5/60 | 4-18% | 4.40 |
| chunk k=5 | 8-15 cm | 9 | 0/9 | 0-30% | 0.50 |
| chunk k=5 | >=15 cm | 21 | 1/21 | 1-23% | 3.80 |
| chunk k=5 · probe_cube cost | 8-15 cm | 9 | 0/9 | 0-30% | 0.60 |
| chunk k=5 · probe_cube cost | >=15 cm | 21 | 2/21 | 3-29% | 5.30 |
| chunk k=8 | 8-15 cm | 30 | 3/30 | 3-26% | 0.90 |
| chunk k=8 | >=15 cm | 60 | 7/60 | 6-22% | 4.90 |
| chunk k=8 · 300x30 search | 8-15 cm | 9 | 0/9 | 0-30% | 0.40 |
| chunk k=8 · 300x30 search | >=15 cm | 21 | 2/21 | 3-29% | 5.00 |
| chunk k=8 · probe_cube cost | 8-15 cm | 30 | 3/30 | 3-26% | 2.30 |
| chunk k=8 · probe_cube cost | >=15 cm | 60 | 7/60 | 6-22% | 4.60 |
| chunk k=8 · probe_cube cost · 300x30 search | 8-15 cm | 9 | 0/9 | 0-30% | 0.00 |
| chunk k=8 · probe_cube cost · 300x30 search | >=15 cm | 21 | 4/21 | 8-40% | 5.60 |
| expert replay | 8-15 cm | 30 | 26/30 | 70-95% | 8.20 |
| expert replay | >=15 cm | 60 | 57/60 | 86-98% | 20.70 |
| random actions | 8-15 cm | 30 | 3/30 | 3-26% | 0.70 |
| random actions | >=15 cm | 60 | 0/60 | 0-6% | 1.10 |
| simulator as model · latent cost | 8-15 cm | 5 | 1/5 | 4-62% | 2.10 |
| simulator as model · latent cost | >=15 cm | 5 | 1/5 | 4-62% | 7.60 |
| simulator as model · probe_cube cost | 8-15 cm | 5 | 1/5 | 4-62% | 2.20 |
| simulator as model · probe_cube cost | >=15 cm | 5 | 1/5 | 4-62% | 15.00 |
| simulator as model · true_cube cost | 8-15 cm | 30 | 17/30 | 39-73% | 6.00 |
| simulator as model · true_cube cost | >=15 cm | 60 | 29/60 | 36-61% | 15.20 |

Successes from planners with start < 8 cm: 0 of 139.
Random-action null: 3/90 succeed. A planner is only distinguishable from chance when its interval excludes this rate.

## 4. Is the planning cost informative? (true latents, no model)
Spearman rank correlation, per goal frame, between latent MSE to the goal and the true distance to the goal. 1 = the cost is a perfect proxy. Goals are from the last 30% of held-out episodes.

| distance | mean rho | goals |
|---|---|---|
| cube to goal cube | 0.31 | 250 |
| end effector to goal end effector | 0.82 | 250 |
| cube to goal cube, arm already within 3 cm of goal arm | — | 0 |

If the cube row is clearly below the end-effector row, the cost mostly rewards moving the arm.
