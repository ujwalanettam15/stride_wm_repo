# Planning-cost check (true latents, held-out episodes; no model, no simulator)

Cache: data/cache (64 patches). Spearman rank correlation, per goal frame, between each cost and the true distance to the goal; 1 = a perfect proxy. Fit on training episodes only.

| cost | cube rho | end-effector rho | goals |
|---|---|---|---|
| latent | 0.30 | 0.81 | 250 |
| patchw | 0.38 | 0.68 | 250 |
| probe_cube | 0.57 | 0.24 | 250 |
| probe_cube_ee | 0.47 | 0.91 | 250 |

Chosen (highest cube rho, rule fixed before planning): **probe_cube** -> data/cache/cost.pt
