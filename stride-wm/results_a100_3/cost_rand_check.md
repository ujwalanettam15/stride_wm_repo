# Refit cube probe: check on held-out episodes (no model, no planner)

Off-expert frames from scripts/collect_rand.py (data/cache_rand): random, held-random (blocks) and noisy-expert actions, 8-40 steps from expert states. Held out: 10050 expert frames, 3000 off-expert frames (2616 with the arm > 10 cm from the cube), 600 starts for the rank correlation.

| cost | cube error, expert frames (cm) | cube error, off-expert (cm) | off-expert, arm > 10 cm from cube (cm) | cube rank correlation over planner candidates |
|---|---|---|---|---|
| latent MSE | — | — | — | 0.27 |
| probe_cube (expert frames only) | 2.37 | 3.70 | 3.72 | 0.64 |
| probe_cube_rand, linear, grid 4 | 2.39 | 2.83 | 2.79 | 0.65 |
| probe_cube_rand, linear, grid 8 | 2.05 | 2.42 | 2.36 | 0.65 |
| probe_cube_rand, MLP, grid 8 | 1.54 | 1.82 | 1.77 | 0.70 |

Kept (lowest off-expert cube error, rule fixed before planning): **probe_cube_rand, MLP, grid 8** -> data/cache/cost_rand.pt (`--cost probe_cube_rand`)
