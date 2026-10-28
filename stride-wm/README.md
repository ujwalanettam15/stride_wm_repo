# Prediction Stride in Latent World Models

Does the stride k of a latent world model (predict z<sub>t+k</sub> from z<sub>t</sub> and the k actions
a<sub>t</sub>..a<sub>t+k-1</sub>) change prediction accuracy and planning success on single-cube
pick-and-place? Built on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
(pinned to `21446f1`): its DINO-WM predictor, single-cube OGBench env, expert policy and CEM solver
are imported unchanged. Everything this repo adds is the stride / action-chunk logic and the
evaluation around it.

## Layout

| file | what it does |
|---|---|
| `stride/model.py` | `StrideWM`: swm's DINO-WM `CausalPredictor` + action conditioning. `arch=chunk` (proposed): 2-layer transformer over the k actions → one 64-d embedding concatenated to every patch. `arch=dinowm` (original): k actions flattened → linear 10-d embedding (swm `Embedder`) |
| `stride/phases.py` | per-frame Reach / Grasp / Transport / Place / Other labels; thresholds fit once on training episodes by fixed rules |
| `stride/data.py` | latent cache, linear probe (latent → cube pose, end-effector pose, gripper), physical errors in cm / degrees |
| `stride/encoder.py`, `stride/env.py` | frozen DINOv2 ViT-S/14 (224 px, ImageNet norm, patch tokens); the swm cube env |
| `scripts/collect.py` | episodes from swm's `OGBCube-v0` + `ExpertPolicy` (noisy Markov oracle) in swm's HDF5 schema |
| `scripts/encode.py` | DINOv2 latents for every frame + state + phase labels → `data/cache/` |
| `scripts/train.py` | one model: single k-step transitions, MSE to the frozen latent at t+k, no rollout loss |
| `scripts/eval_openloop.py` | **E1/E4** open-loop latent MSE, cosine distance, probe error by stride × depth × phase; copy-last reference and probe floor |
| `scripts/eval_plan.py` | **E2/E3** MPC with swm's `CEMSolver`: success by goal offset, model calls and seconds per plan |
| `scripts/report.py` | markdown tables + figures → `results/` |
| `tests/test_model.py` | shuffling the actions inside a chunk must change the prediction (both archs, k = 2, 4, 5, 8); rollout = repeated k-steps |

## Setup

```bash
pip install -r requirements.txt          # Python 3.11/3.12; see the note in requirements.txt for 3.13
export MUJOCO_GL=egl                     # headless rendering
```

## Run

Apple-silicon prelim (what produced `results/`, ~1 h): `bash run_prelim.sh`.
CPU pilot (`results_cpu_pilot/`): `bash run_pilot.sh`.

Full scale (GPU): same scripts, with their defaults (DINO-WM-sized predictor, 16×16 patch grid,
swm's CEM budget 300 samples × 30 iterations × top 30):

```bash
python scripts/collect.py --episodes 1000 --steps 200          # or pass the LeWM cube .h5 to encode.py
python scripts/encode.py --h5 data/cube_single.h5 --out data/cache --n-eval 100
for s in 0 1 2; do
  python scripts/train.py --arch dinowm --k 5 --seed $s --steps 50000
  for k in 1 2 4 5 8; do python scripts/train.py --arch chunk --k $k --seed $s --steps 50000; done
done
python scripts/eval_openloop.py --ckpts runs/*.pt --horizon 80 --every 1
for H in 40 80; do
  python scripts/eval_plan.py --ckpts runs/*.pt --horizon $H --offsets 25,50,100 --pairs 50 --out results/plan_H$H.jsonl
done
python scripts/report.py
```

### On an Apple-silicon Mac

`--device` defaults to CUDA, then the Mac GPU (MPS), then CPU, so nothing needs changing. MuJoCo
renders natively on macOS (no `MUJOCO_GL`, no `--fast-render`). A laptop GPU is far slower than a
data-center GPU, so `run_mac.sh` runs a mid-scale version: 8×8 patch grid (`--pool 2`),
DINO-WM-sized predictor, 3 seeds, reduced CEM budget for planning. Time one model before
committing to the grid:

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/collect.py --episodes 50                      # small first, to time things
python scripts/encode.py --pool 2 --n-eval 10
python scripts/train.py --k 1 --steps 300 --eval-every 100   # 'sec' in the log / 300 = seconds per step
```

Seconds per step × 20,000 ≈ time per model; the grid is 18 models. Then `bash run_mac.sh`
(delete `data/` first so it regenerates the full dataset).

## Fairness rules built into the code

- All proposed models share architecture, data, steps, batch, optimizer (AdamW 5e-4, clip 1.0, as DINO-WM); only k differs.
- The planner always optimizes all H raw actions; stride k only groups them into H/k chunks. Same CEM budget, replanning interval (8 steps), start/goal pairs and seeds for every condition.
- Strides are compared only at shared depths: multiples of 8 for k ∈ {1, 2, 4, 8}, multiples of 5 for proposed k = 5 vs the DINO-WM baseline, multiples of 40 for all.
- Phase thresholds are fit on training episodes before any model is trained (`data/cache/phase_thresholds.json`); E1 groups errors by the phase of the predicted frame and reports frame counts.
- `train.py` logs validation loss next to a copy-last reference, and a chunk-diversity statistic (RMS nearest-neighbour distance per action dim among training chunks) for each k.

## Decisions to know about

- **History length 1.** A stride-k model sees one frame, matching the proposal's definition. Stock DINO-WM uses 3 history frames; the baseline here also uses 1, so E4 isolates the action conditioning.
- **No proprioception in the latent.** Stock DINO-WM concatenates a 10-d proprio embedding; here the latent is DINOv2 patches only, so the planning cost is purely visual and the probe's end-effector error is not trivial.
- **Contact phases are events, not windows.** The expert's gripper noise makes the gripper twitch while carrying and after release, so Grasp also requires the cube on the table and Place requires the hand at the cube. Fixed after looking at label counts, before any model was trained. At 20 Hz a grasp or release lasts 1-2 frames (pilot: 239 Grasp / 173 Place frames out of 24k), so per-phase results carry their frame counts.
- **Data.** Generated with the same swm env + expert policy family as swm's `cube_single_expert`; `encode.py` also reads swm-format `.h5` files whose columns use `_` instead of `/`.
- **Pilot-only reductions** (2 CPU cores, no GPU): faster software rendering (`--fast-render`: 1024 shadow map, no MSAA), 16×16 patch grid average-pooled to 4×4, predictor depth 3 / 6 heads / MLP 1024 (4.9M params vs 22M), 5k steps, 1-2 seeds, CEM 64 × 5, near goals only. Pilot numbers show the pipeline works end to end; they are not the study's conclusions.

## Prelim results (8×8 patches; full tables in `results/summary.md`)

`bash run_prelim.sh` (~1 h on an M-series laptop). Same data scale, predictor and step count as the
CPU pilot below; the one thing changed is the patch grid, 4×4 → 8×8 (`--pool 2`), to test whether
the pilot's pooling was hiding contact detail. 120 episodes (100 train / 20 held out, 24,120
frames), both seeds for all six conditions. `±` is the s.d. over the two training seeds.

**Latent MSE is not comparable to the pilot's.** The targets are 8×8 pooled DINOv2 patches rather
than 4×4, so the same models score higher here by construction; read the ordering and the in-run
`copy_last` / probe-floor references, not the absolute values.

- **E1:** coarser stride gives lower open-loop latent error, monotone from k=1 to k=4, then flat.
  At 40 steps: k=1 0.591 ± 0.169, k=2 0.438 ± 0.044, k=4 0.405 ± 0.002, k=5 0.345 ± 0.013,
  k=8 0.364 ± 0.070; copy-last 1.458. k = 4, 5, 8 are within seed noise of each other, and k=4 and
  k=8 are identical at depth 8 (0.229 vs 0.230). Compounding dominates, and the benefit saturates
  near k=4.
- **H3 still not supported, and 4×4 pooling was not the reason.** At depth 8, k=8 beats k=1 in both
  contact phases: Grasp 0.221 vs 0.284, Place 0.198 vs 0.278. The null survives the finer grid.
  Frame counts remain the limit (16 Grasp / 13 Place at depth 8), so this is consistent with the
  pilot rather than a settled result.
- **E4:** chunk encoder vs DINO-WM conditioning at k=5 is indistinguishable at every depth —
  0.209 vs 0.213 (5), 0.239 vs 0.240 (10), 0.281 vs 0.281 (20), 0.345 vs 0.355 (40), against seed
  s.d. of ±0.013 and ±0.011. Same answer as the pilot, now with both seeds on all conditions.
- **Probe:** positions survive, orientation does not. End-effector 4.3-7.5 cm at 40 steps (floor
  1.10 cm), cube 5.1-10.6 cm (floor 2.54 cm), yaw ≈ 43° against a 6.1° floor, i.e. chance.
- **E2/E3:** with CEM 100 × 10 at offset 25, success over 5 start/goal pairs is 2/5 for the DINO-WM
  baseline, 1/5 for k = 1, 2, 4, 8 and 0/5 for k=5; expert replay reaches 4/5, so env resets and the
  success check work. **Five pairs cannot separate these models** — the spread is what 5 Bernoulli
  draws look like. Cost per plan scales cleanly with H/k: 400 model calls / 9.34 s for k=1 down to
  50 calls / 1.15 s for k=8.

## Pilot results (CPU scale; full tables in `results_cpu_pilot/`)

120 episodes (100 train / 20 held out, 24,120 frames), seeds 0 and 1 for k = 1, 5, 8 and the DINO-WM baseline, seed 0 for k = 2, 4.

- **E1:** at every shared depth, coarser stride gives lower open-loop latent error. At 40 steps: k=1 0.35 / 0.46 (seed 0 / 1), k=2 0.28, k=4 0.24, k=5 0.24 / 0.21, k=8 0.20 / 0.23; copying the last frame gives 0.94. Per model call k=1 is the most accurate, so compounding dominates. k = 4, 5, 8 are within seed noise of each other.
- **H3 not supported yet:** in Grasp and Place frames, k=8 is no worse than k=1 at matched depth (8 steps: Grasp latent MSE 0.129 vs 0.178, Place 0.130 vs 0.177). Few contact frames (39 Grasp / 46 Place at depth 8) and 4×4 pooling may hide contact detail; recheck at 16×16.
- **Probe:** predicted latents keep positions (end-effector 4-11 cm at 40 steps; floor 1.1 cm) but lose orientation (yaw error ≈ chance), as LeWM reports for long predictions.
- **E4:** chunk encoder vs DINO-WM conditioning at k=5 is within seed noise (40 steps: 0.236 vs 0.275 on seed 0, 0.213 vs 0.212 on seed 1).
- **E2/E3:** with the CPU budget (CEM 64 × 5), planning reaches 1 of 8 non-trivial near goals for k=8 and 0 of 8 for every other model, so E2 needs the GPU budget. Replaying the logged actions from the same start states reaches 7 of 8 (the miss is a drop-and-regrasp that does not reproduce exactly from qpos/qvel), so env resets and the success check work. Cost per plan scales with H/k: 200 model calls for k=1 vs 25 for k=8 (8.9 s vs 1.1 s on CPU).

## Training · final k-step MSE (train = mean of last 500 steps; val = held-out episodes)
| model | seed | train | val | copy-last on same val pairs | chunk diversity (RMS NN dist / dim) |
|---|---|---|---|---|---|
| chunk_k1 | 0 | 0.1634 | 0.1487 | 0.2560 | 0.125 |
| chunk_k1 | 1 | 0.1624 | 0.1498 | 0.2560 | 0.126 |
| chunk_k2 | 0 | 0.1925 | 0.1716 | 0.4266 | 0.212 |
| chunk_k2 | 1 | 0.1911 | 0.1712 | 0.4266 | 0.216 |
| chunk_k4 | 0 | 0.2257 | 0.2011 | 0.7391 | 0.306 |
| chunk_k4 | 1 | 0.2256 | 0.2009 | 0.7391 | 0.310 |
| chunk_k5 | 0 | 0.2365 | 0.2115 | 0.8507 | 0.342 |
| chunk_k5 | 1 | 0.2365 | 0.2099 | 0.8507 | 0.344 |
| chunk_k8 | 0 | 0.2564 | 0.2282 | 1.0737 | 0.418 |
| chunk_k8 | 1 | 0.2582 | 0.2328 | 1.0737 | 0.421 |
| dinowm_k5 | 0 | 0.2468 | 0.2141 | 0.8507 | 0.342 |
| dinowm_k5 | 1 | 0.2442 | 0.2147 | 0.8507 | 0.344 |

## E1 · open-loop latent MSE by rollout depth (env steps), all phases
± = s.e. over prediction samples (one seed); † = s.d. over training seeds

| label | 8 | 16 | 24 | 32 | 40 |
| --- | --- | --- | --- | --- | --- |
| chunk k=1 | 0.3015 ± 0.0134† | 0.4006 ± 0.0572† | 0.4775 ± 0.1002† | 0.5404 ± 0.1383† | 0.5911 ± 0.1685† |
| chunk k=2 | 0.2471 ± 0.0051† | 0.3103 ± 0.0176† | 0.3586 ± 0.0302† | 0.4000 ± 0.0388† | 0.4377 ± 0.0443† |
| chunk k=4 | 0.2292 ± 0.0004† | 0.2778 ± 0.0046† | 0.3215 ± 0.0048† | 0.3640 ± 0.0024† | 0.4047 ± 0.0016† |
| chunk k=5 | — | — | — | — | 0.3450 ± 0.0134† |
| chunk k=8 | 0.2295 ± 0.0029† | 0.2620 ± 0.0150† | 0.2945 ± 0.0320† | 0.3297 ± 0.0505† | 0.3642 ± 0.0699† |
| DINO-WM k=5 | — | — | — | — | 0.3550 ± 0.0105† |
| copy_last | 1.0516 ± 0.0108 | 1.3080 ± 0.0132 | 1.4281 ± 0.0134 | 1.4885 ± 0.0129 | 1.4579 ± 0.0134 |

## E1 · probe-decoded error at depth 40 (cube / end-effector position, cm)
| model | calls | latent MSE | cos dist | cube cm | cube deg | EE cm | EE deg | grip |
|---|---|---|---|---|---|---|---|---|
| chunk k=1 | 40 | 0.5911 ± 0.1685† | 0.0690 ± 0.0198† | 10.64 ± 1.02† | 22.4 ± 0.5† | 7.50 ± 1.68† | 42.9 ± 2.2† | 0.105 ± 0.003† |
| chunk k=2 | 20 | 0.4377 ± 0.0443† | 0.0516 ± 0.0055† | 7.25 ± 1.21† | 22.2 ± 0.3† | 6.20 ± 1.01† | 43.1 ± 4.0† | 0.084 ± 0.015† |
| chunk k=4 | 10 | 0.4047 ± 0.0016† | 0.0479 ± 0.0001† | 6.71 ± 0.66† | 20.1 ± 0.5† | 5.70 ± 1.38† | 43.7 ± 1.0† | 0.067 ± 0.002† |
| chunk k=5 | 8 | 0.3450 ± 0.0134† | 0.0409 ± 0.0017† | 6.00 ± 0.72† | 19.7 ± 2.0† | 4.31 ± 0.36† | 43.8 ± 0.3† | 0.078 ± 0.010† |
| chunk k=8 | 5 | 0.3642 ± 0.0699† | 0.0430 ± 0.0082† | 5.05 ± 0.38† | 19.0 ± 0.7† | 5.39 ± 1.17† | 43.3 ± 0.9† | 0.070 ± 0.004† |
| DINO-WM k=5 | 8 | 0.3550 ± 0.0105† | 0.0419 ± 0.0012† | 6.57 ± 1.60† | 20.8 ± 0.0† | 4.77 ± 1.11† | 45.1 ± 0.7† | 0.087 ± 0.023† |
| copy_last | 0 | 1.4579 ± 0.0134 | 0.1645 ± 0.0015 | 8.06 ± 0.25 | 9.7 ± 0.3 | 24.11 ± 0.29 | 40.9 ± 0.7 | 0.160 ± 0.005 |
| probe floor (true latents) | – | – | – | 2.54 | 6.9 | 1.10 | 6.1 | 0.064 |

## E1 · cube position error (cm) by phase of the predicted frame, depth 8
| label | reach | grasp | transport | place |
| --- | --- | --- | --- | --- |
| chunk k=1 | 5.68 ± 0.03† | 3.83 ± 1.00† | 6.32 ± 0.64† | 5.27 ± 2.14† |
| chunk k=2 | 4.52 ± 0.24† | 3.08 ± 0.12† | 5.65 ± 0.60† | 4.41 ± 1.26† |
| chunk k=4 | 3.99 ± 0.01† | 2.69 ± 0.48† | 5.06 ± 0.50† | 4.08 ± 0.67† |
| chunk k=8 | 3.50 ± 0.03† | 2.36 ± 0.06† | 4.47 ± 0.05† | 3.27 ± 0.57† |
| copy_last | 3.11 ± 0.13 | 4.32 ± 1.01 | 11.33 ± 0.29 | 9.44 ± 1.53 |
| probe_floor | 2.50 ± 0.03 | 2.07 ± 0.16 | 2.66 ± 0.06 | 2.18 ± 0.20 |

## E1 · latent MSE by phase, depth 8  (frames: reach n=575, grasp n=16, transport n=283, place n=13)
| label | reach | grasp | transport | place |
| --- | --- | --- | --- | --- |
| chunk k=1 | 0.3036 ± 0.0058† | 0.2841 ± 0.0065† | 0.2740 ± 0.0132† | 0.2777 ± 0.0223† |
| chunk k=2 | 0.2532 ± 0.0014† | 0.2318 ± 0.0035† | 0.2243 ± 0.0103† | 0.2231 ± 0.0014† |
| chunk k=4 | 0.2381 ± 0.0019† | 0.2227 ± 0.0064† | 0.2060 ± 0.0022† | 0.2045 ± 0.0003† |
| chunk k=8 | 0.2345 ± 0.0013† | 0.2211 ± 0.0058† | 0.2074 ± 0.0004† | 0.1977 ± 0.0035† |
| copy_last | 1.0831 ± 0.0181 | 0.8601 ± 0.0675 | 0.8739 ± 0.0237 | 0.6621 ± 0.0805 |

## E1 · cube position error (cm) by phase of the predicted frame, depth 40
| label | reach | grasp | transport | place |
| --- | --- | --- | --- | --- |
| chunk k=1 | 10.98 ± 1.26† | 8.36 ± 1.09† | 8.62 ± 1.16† | 9.44 ± 1.24† |
| chunk k=2 | 7.38 ± 1.61† | 5.12 ± 1.13† | 6.50 ± 2.02† | 5.88 ± 0.86† |
| chunk k=4 | 6.57 ± 0.23† | 5.79 ± 2.34† | 6.47 ± 1.35† | 6.08 ± 1.49† |
| chunk k=5 | 6.04 ± 0.49† | 5.49 ± 2.58† | 5.91 ± 1.41† | 5.62 ± 2.72† |
| chunk k=8 | 5.04 ± 0.48† | 4.36 ± 0.40† | 5.64 ± 0.51† | 4.67 ± 0.53† |
| copy_last | 4.45 ± 0.23 | 18.36 ± 4.65 | 22.99 ± 1.21 | 27.84 ± 3.66 |
| probe_floor | 2.50 ± 0.03 | 2.07 ± 0.16 | 2.66 ± 0.06 | 2.18 ± 0.20 |

## E1 · latent MSE by phase, depth 40  (frames: reach n=643, grasp n=3, transport n=105, place n=10)
| label | reach | grasp | transport | place |
| --- | --- | --- | --- | --- |
| chunk k=1 | 0.5828 ± 0.1340† | 0.4825 ± 0.1229† | 0.4615 ± 0.1501† | 0.5554 ± 0.1429† |
| chunk k=2 | 0.4352 ± 0.0217† | 0.3096 ± 0.0279† | 0.4030 ± 0.1077† | 0.3143 ± 0.0329† |
| chunk k=4 | 0.4016 ± 0.0036† | 0.3277 ± 0.0180† | 0.3730 ± 0.0355† | 0.3078 ± 0.0185† |
| chunk k=5 | 0.3417 ± 0.0239† | 0.3136 ± 0.0594† | 0.3176 ± 0.0018† | 0.3301 ± 0.1319† |
| chunk k=8 | 0.3727 ± 0.0723† | 0.2466 ± 0.0028† | 0.2822 ± 0.0050† | 0.2544 ± 0.0107† |
| copy_last | 1.4337 ± 0.0201 | 1.4709 ± 0.3456 | 1.4622 ± 0.0682 | 1.9750 ± 0.2199 |

## E4 · DINO-WM conditioning vs chunk encoder, k=5 (latent MSE)
| label | 5 | 10 | 20 | 40 |
| --- | --- | --- | --- | --- |
| chunk k=5 | 0.2092 ± 0.0008† | 0.2392 ± 0.0027† | 0.2814 ± 0.0012† | 0.3450 ± 0.0134† |
| DINO-WM k=5 | 0.2134 ± 0.0006† | 0.2399 ± 0.0022† | 0.2811 ± 0.0028† | 0.3550 ± 0.0105† |

## E2 · planning success rate by goal offset (env steps)
| model | d=25 |
|---|---|
| DINO-WM k=5 | 40% (5) |
| chunk k=1 | 20% (5) |
| chunk k=2 | 20% (5) |
| chunk k=4 | 20% (5) |
| chunk k=5 | 0% (5) |
| chunk k=8 | 20% (5) |
| expert replay (pipeline check) | 80% (5) |

## E3 · cost per plan (same CEM budget, horizon and replanning for all)
| model | model calls / plan | sec / plan |
|---|---|---|
| chunk k=1 | 400 | 9.34 |
| chunk k=2 | 200 | 4.68 |
| chunk k=4 | 100 | 2.34 |
| chunk k=5 | 80 | 1.98 |
| chunk k=8 | 50 | 1.15 |
| DINO-WM k=5 | 80 | 1.86 |



## Next (GPU)

1. Encode at full resolution (`--pool 1`) on the LeWM cube data or ~1000 collected episodes; train with the DINO-WM-sized defaults for longer (validation loss was still falling at 5k steps), 3 seeds each.
2. E1 at H = 80; E2 with swm's CEM budget (300 × 30) at H = 40 and 80, near / mid / far = 25 / 50 / 100 steps. k=1 planning dominates the cost (1,200 model calls per replan at H = 40), so size the pair count to the GPU budget.
3. Open checks: does the H3 null hold at 16×16 patches; E4 with 3 seeds; action-embedding bottleneck (`--emb-dim` 32 / 64 / 128 at k=8); then E5 (adaptive stride) only after E1-E4.
