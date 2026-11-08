# Prediction Stride in Latent World Models

Does the stride k of a latent world model (predict z<sub>t+k</sub> from z<sub>t</sub> and the k actions
a<sub>t</sub>..a<sub>t+k-1</sub>) change prediction accuracy and planning success on single-cube
pick-and-place? Built on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel)
(pinned to `21446f1`): its DINO-WM predictor, single-cube OGBench env, expert policy and CEM solver
are imported unchanged. Everything this repo adds is the stride / action-chunk logic and the
evaluation around it.

## New changes 

Full results in "Failure and error analysis" and
"Changes to the original design"

- **Planning evaluation (`eval_plan.py`) was fixed.** The goal offset now equals the planning horizon,
  CEM candidates are clamped to the env's action bounds before they are scored, and start/goal pairs must
  start at least 8 cm apart. A random-action planner (`--ckpts random`) is the new null baseline.
- **Statistics were tightened.** Open-loop `±` is now the s.e. over held-out *episodes*; planning success
  gets Wilson intervals and a progress-toward-goal table.
- **Three optional model/training switches were added, all off by default** (the headline models are
  unchanged): `--residual`, `--unroll`, `--dinowm-emb`. They exist for the ablations.
- **New tools:** `scripts/analyze_failures.py`, `run_ablations.sh`, `stride/cost.py`, `stride/stats.py`,
  `tests/test_utils.py`; `eval_openloop.py` gained `--shuffle-actions` and `--probe-grid`.

## Getting started (step by step)


### Step 0 — What you need

| | |
|---|---|
| OS | macOS (Apple silicon or Intel) or Linux. Windows is untested. |
| Python | **3.11 or 3.12** (3.13 needs the workaround noted in `requirements.txt`) |
| Tools | `git`, `python3 -m venv` (ships with Python) |
| Disk | roughly 6-10 GB free for the 120-episode run (≈3.5 GB of episodes, ≈1 GB of latents, checkpoints) |
| Network | needed on first use: `pip` packages, the pinned `stable-worldmodel` from GitHub, and the DINOv2 weights (`facebook/dinov2-small`, ≈90 MB, from Hugging Face) |
| GPU | optional. The scripts pick CUDA, then the Mac GPU (MPS), then CPU automatically. |

### Step 1 — Get the code

```bash
git clone https://github.com/ujwalanettam15/stride_wm_repo.git
cd stride_wm_repo/stride-wm          # the project is the stride-wm/ subfolder
# if the changes described above are on another branch, check it out: git checkout <branch>
```

### Step 2 — Install and check that it works

```bash
python3.12 -m venv .venv             # or python3.11
source .venv/bin/activate            # run this in every new terminal before anything else
pip install --upgrade pip
pip install -r requirements.txt      # several minutes: torch, mujoco, dm_control, stable-worldmodel @ 21446f1
python -c "import torch, stable_worldmodel, mujoco; print('imports ok')"
python -m pytest -q tests            # small CPU tests, no data needed; all should pass
```

If a test fails, stop and fix that first: the later steps depend on the code the tests cover.

### Step 3 — Look at the existing results (no data, no GPU, a minute)

The repo ships the outputs of the earlier runs, so you can regenerate the report and the failure analysis
without training anything. **Write the report to a scratch folder** (`--out`), because without checkpoints in
`runs/` it cannot rebuild the "Training" table and would overwrite the committed `results/summary.md`.

```bash
python scripts/analyze_failures.py                       # writes results/failure_analysis.md (sections 1-3)
python scripts/report.py --out /tmp/stride_report        # prelim tables + figures, written to /tmp
python scripts/report.py --openloop results_cpu_pilot/openloop.csv --plan results_cpu_pilot/plan.jsonl \
  --out /tmp/stride_report_pilot                         # same for the CPU pilot
```

Open `/tmp/stride_report/summary.md` and `results/failure_analysis.md`. The failure analysis is what
supports the "Failure and error analysis" section below; section 4 of it (is the planning cost informative?)
needs the latent cache from Step 5.

### Step 4 — Back up the committed results

Steps 5-6 overwrite files in `results/`. Keep the originals so you can compare:

```bash
cp -r results results_committed
rm results/plan.jsonl                # eval_plan.py APPENDS to its output file; start the new log empty
```

### Step 5 — Reproduce the prelim: data, latents, 12 models, E1/E4 (≈1 h on an M-series Mac)

```bash
bash run_prelim.sh
```

What it does, in order (you can also run the lines by hand; see the script):

1. `scripts/collect.py` — 120 expert episodes of the single-cube task → `data/cube_single.h5` (≈6 min, ≈3.5 GB)
2. `scripts/encode.py --pool 2 --n-eval 20` — frozen DINOv2 latents (8×8 patch grid), physical state and
   phase labels → `data/cache/`. The last 20 episodes are held out. Downloads the DINOv2 weights the first time.
3. `pytest` — the unit tests again.
4. `scripts/train.py` × 12 — the DINO-WM baseline and chunk models for k = 1, 2, 4, 5, 8, seeds 0 and 1,
   5,000 steps each (≈3 min each) → `runs/*.pt`. Training logs validation loss next to a copy-last reference.
5. `scripts/eval_openloop.py` — E1/E4: open-loop error by stride × depth × phase → `results/openloop.csv`
6. `scripts/report.py` — tables and figures → `results/summary.md`, `results/fig_*.png`

The script skips collection and encoding if `data/` already holds them. To start from scratch, delete `data/`.
Seeds are fixed, but GPU kernels are not bit-deterministic, so expect numbers close to, not identical with,
the committed ones.

### Step 6 — Planning (E2/E3, the slow part)

`run_prelim.sh` deliberately does not run planning. Two versions, depending on what you want:

**6a. Reproduce the committed E2/E3 table** (old planner: goal 25 steps away, 4 cm minimum start distance;
≈25-35 min for 5 pairs). Candidate clipping cannot be switched off, so even this will not match exactly:

```bash
python scripts/eval_plan.py --ckpts replay runs/*_s0.pt --offsets 25 --pairs 5 \
  --samples 100 --iters 10 --topk 10 --horizon 40 --replan 8 --allow-misaligned --min-start-cm 4
python scripts/report.py
```

**6b. Run the fixed planner** (goal = horizon, clipped candidates, ≥ 8 cm starts, random-action null).
This is the version to quote from now on. Use a new output file, since the script appends:

```bash
python scripts/eval_plan.py --ckpts replay random runs/*_s0.pt --horizon 40 --pairs 10 \
  --samples 100 --iters 10 --topk 10 --out results/plan_aligned.jsonl
python scripts/report.py --plan results/plan_aligned.jsonl
```

`replay` re-executes the logged expert actions (checks that env resets and the success test work);
`random` is the null planner. Read a model's success rate only next to the random row and the Wilson
interval. Cost scales with H/k, so k=1 is ≈8× slower than k=8 (9.3 s vs 1.2 s per plan in the prelim). Planning must render
frames the way the data was collected: pass `--fast-render` only if you also collected with it (`run_pilot.sh` does both).

### Step 7 — Full failure analysis, including the cost check

```bash
python scripts/analyze_failures.py --cache data/cache --plan results/plan_aligned.jsonl
```

Adds section 4: the rank correlation between the latent-MSE planning cost and the true cube / end-effector
distance to the goal. If the cube correlation is much lower than the end-effector one, the cost mostly
rewards moving the arm, which would explain planning failing for every stride.

### Step 8 — Ablations (written, not yet run)

Needs Step 5 first (the script checks and refuses to start otherwise).

```bash
bash run_ablations.sh                                    # full set: several hours (estimate, not measured)
# a trimmed first pass:
SEEDS="0" SPAN=8 PAIRS=5 PLAN_CKPTS="runs/chunk_k1_s0.pt runs/chunk_k8_s0.pt runs/dinowm_k5_s0.pt" \
  bash run_ablations.sh
```

| ablation | what it trains / runs | question |
|---|---|---|
| A1 | each stride with `--unroll SPAN/k` (chained-step loss; a stride whose unroll would be 1 is skipped, it is the default model) | is k=1's deficit just compounding? |
| A2 | DINO-WM with a 64-d embedding; chunk with 10-d; chunk width 32 / 128 | is the E4 null hiding a width effect? |
| A3 | `--residual` for k = 1, 5, 8 and DINO-WM k=5 | does it stop the Reach-phase drift? |
| A4 | random-action null planner, `--shuffle-actions`, `--probe-grid 8` | do planners beat chance; do the models use their actions? |

Variant checkpoints carry a suffix (`chunk_k1-u16_s0.pt`, `chunk_k5-res_s0.pt`, …) and are kept out of the
headline tables; `summary.md` gets separate "Ablations" and "Action sensitivity" tables. The longest part is
A1 at k=1 (`SPAN` steps of backprop per training step); lower `SPAN`, `SEEDS` or `PLAN_CKPTS` to save time.
Settings are variables at the top of the script (`SEEDS`, `SPAN`, `PAIRS`, `PLAN_CKPTS`, `SMALL`).

### Step 9 — Where each result comes from

| result | file | produced by |
|---|---|---|
| E1 / E4 tables, probe errors, by-phase tables | `results/summary.md` ← `results/openloop.csv` | `eval_openloop.py` → `report.py` |
| Training loss table | `results/summary.md` ← `runs/*.pt` | `train.py` → `report.py` |
| E2 success + progress, E3 cost per plan | `results/summary.md` ← `results/plan*.jsonl` | `eval_plan.py` → `report.py` |
| Failure analysis (sections 1-4) | `results/failure_analysis.md` | `analyze_failures.py` |
| Ablation and action-sensitivity tables | `results/summary.md` | `run_ablations.sh` |
| Figures | `results/fig_e1_e4_depth.png`, `results/fig_e1_phase.png` | `report.py` |

### Step 10 — Full scale (GPU)

Same scripts at their defaults (22M-parameter predictor, 16×16 patches, 50k steps, 3 seeds, swm's CEM
budget). The commands are in "Other run configurations" below.

### If something goes wrong

| symptom | fix |
|---|---|
| `ModuleNotFoundError: stable_worldmodel` (or torch) | the venv is not active: `source .venv/bin/activate` |
| `pip` fails building `labmaze` | you are on Python 3.13; use 3.11/3.12 or follow `requirements.txt` |
| MuJoCo / OpenGL errors on Linux | `export MUJOCO_GL=egl` (or `osmesa`); install the system EGL/OSMesa libraries |
| "operator not implemented for MPS" | `export PYTORCH_ENABLE_MPS_FALLBACK=1` (the run scripts already set it) |
| out of memory in `eval_openloop.py` | lower `--batch` (default 256) |
| duplicated planning rows | `eval_plan.py` appends; delete or rename the `.jsonl` before re-running |
| `run_ablations.sh` says data or runs are missing | run `bash run_prelim.sh` first (Step 5) |
| numbers differ slightly from the committed results | expected: GPU nondeterminism, and the planner fixes (Step 6) |

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
| `scripts/report.py` | markdown tables + figures → `results/` (headline tables use default-setting models only; ablations, controls and planning intervals have their own tables) |
| `scripts/analyze_failures.py` | failure analysis from saved outputs (no model, no simulator): where models lose to copy-last, error per call, seed divergence, planning success by start distance vs a random-action null, and whether the planning cost tracks the cube (`--cache`) |
| `stride/cost.py`, `stride/stats.py` | CEM goal cost (candidate clipping, goal-depth indexing); episode-clustered s.e. and Wilson intervals |
| `run_ablations.sh` | the ablation set (A1-A4), written but **not yet run** |
| `tests/test_model.py` | shuffling the actions inside a chunk must change the prediction (both archs, k = 2, 4, 5, 8); rollout = repeated k-steps |

## Other run configurations (CPU pilot, full-scale GPU, Mac timing)

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
for H in 40 80; do   # goal offset = horizon by default (see "Changes to the original design")
  python scripts/eval_plan.py --ckpts replay random runs/*.pt --horizon $H --pairs 50 --out results/plan_H$H.jsonl
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

## Status (what is done, at what scale)

| item | status | scale / caveat |
|---|---|---|
| E1 open-loop accuracy by stride × depth × phase | **done** (pilot and prelim) | 120 episodes (20 held out), 5k steps, small predictor (5M params), 2 seeds; GPU scale not run |
| E4 chunk encoder vs DINO-WM conditioning (k=5) | **done**, null result | confounded by embedding width (64 vs 10); control written (A2), not run |
| E2 planning success | **run, inconclusive** | 5 pairs, seed 0, CEM 100×10; pre-fix planner (see below); all successes on the two easiest pairs |
| E3 cost per plan | **done** | scales as H/k, as designed |
| Ablations A1-A4 | **code written, not run** | `run_ablations.sh` |
| E5 adaptive stride | not started | only after E1-E4, per the proposal |
| Already-completed ablations to report | 4×4 vs 8×8 patch grid; chunk vs DINO-WM conditioning; stride sweep k ∈ {1,2,4,5,8} | |

## Failure and error analysis (from the existing prelim / pilot outputs)

`python scripts/analyze_failures.py [--cache data/cache]` regenerates these from saved files.

1. **Planning fails for every model, not for particular strides.** All 6 prelim successes are on the two
   pairs that start 5.6 and 7.4 cm from the goal (a 1.6-3.4 cm nudge inside a 4 cm success radius); there are
   none at ≥ 14 cm. Mean progress toward the goal is ≈ 0 cm for every model (start 14.7 cm → final 11-16 cm)
   against 10.9 cm for expert replay. Pilot: 1 success in 48 planner runs. The 0-40% rates are noise.
2. **Models add drift to a static cube.** In the Reach phase (cube at rest) copy-last has 3.11 cm cube error at
   depth 8 and every model is worse: k=8 3.50, k=1 5.68 (same at depth 40: 4.45 vs 5.04 for the best). A
   model that cannot even preserve what does not move cannot be steered to move it by 4 cm.
3. **k=1's deficit looks like compounding.** The two k=1 seeds agree at depth 1 (MSE 0.147 / 0.148) and diverge
   to 0.710 / 0.472 at depth 40 — the instability grows with the number of recursive calls, in a model trained
   only on ground-truth inputs. This confounds "coarser stride is more accurate" with "fewer chained calls".
4. **The planner had three specification problems** (read from the code, not measured): the cost compared the
   prediction at t+H with a goal 25 steps away; CEM candidates were scored unclipped but executed clipped; and
   pairs starting inside 8 cm were included. All three are fixed (below), so the prelim E2 numbers are not
   comparable to new runs.
5. **Probe resolution does not follow the patch grid.** `probe_features` always pooled to 4×4, so the probe-based
   cube-cm numbers cannot show whether 8×8 patches add contact detail; the latent-MSE evidence for H3 can.
   `--probe-grid 8` removes the limit.
6. **Uncertainty was understated.** `±` was the s.e. over overlapping rollouts from 20 episodes; it is now over
   episodes. Grasp / Place have 3-16 frames per cell, so H3 is untestable at this data size; with 2 seeds an s.d.
   is a rough guide only.

## Changes to the original design

The research question (does stride k change accuracy and planning success), the frozen DINOv2 latents, the
DINO-WM predictor, swm's CEM and the fairness rules are unchanged. Changed:

| change | where | why (evidence above) | default? |
|---|---|---|---|
| Goal offset = planning horizon; cost reads the prediction at the goal's depth | `eval_plan.py`, `stride/cost.py` | cost compared frames up to 15 steps apart (#4) | yes (`--allow-misaligned` restores old) |
| CEM candidates clamped to env bounds before scoring | `stride/cost.py` | scored actions differed from executed actions (#4) | yes |
| Pairs start ≥ 8 cm from goal; random-action null planner; `min_dist_cm` logged | `eval_plan.py` | successes on 5-7 cm starts are not planning (#1) | yes (`--min-start-cm`, `--ckpts random`) |
| Episode-clustered s.e.; Wilson intervals; progress table | `stride/stats.py`, `eval_openloop.py`, `report.py` | #6 | yes |
| Probe grid is an argument | `eval_openloop.py`, `data.py` | #5 | default 4 (unchanged) |
| Residual prediction `z_t + predictor(...)` | `model.py` `--residual` | #2 | **off**; ablation A3, to be adopted for the headline models only if it helps and applied to both architectures |
| Chained-step training loss `--unroll` | `train.py` | #3 | **off**; ablation A1 |
| `--dinowm-emb` | `model.py`, `train.py` | E4 confound (64-d vs 10-d) | **off**; ablation A2 |

## Ablations (A1-A4; code written, not run — `run_ablations.sh`)

- **A1 stride vs compounding:** train each stride on the same SPAN-step chain (`--unroll SPAN/k`). If k=1 catches up, the E1 stride effect is mostly fewer recursive calls.
- **A2 conditioning:** DINO-WM with a 64-d embedding, chunk with a 10-d one, chunk width 32/128 at k=8. Decides whether the E4 null is real or hides a width effect.
- **A3 residual:** with / without, k ∈ {1, 5, 8}, both architectures. Prediction: Reach-phase cube error moves from worse-than-copy-last to at most equal.
- **A4 planning controls:** random-action null, expert replay, aligned goal. Plus `--shuffle-actions` in open loop (do the actions matter?) and `--probe-grid 8`.

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
- **E2/E3:** (pre-fix planner: goal 25 steps away, H = 40, unclipped candidates, easy pairs included) with CEM 100 × 10 at offset 25, success over 5 start/goal pairs is 2/5 for the DINO-WM
  baseline, 1/5 for k = 1, 2, 4, 8 and 0/5 for k=5; expert replay reaches 4/5, so env resets and the
  success check work. **Five pairs cannot separate these models** — the spread is what 5 Bernoulli
  draws look like, and every success is on a pair that starts within 8 cm (see Failure analysis). Cost per plan scales cleanly with H/k: 400 model calls / 9.34 s for k=1 down to
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

0. Before any GPU time: run `bash run_ablations.sh` at prelim scale (A1-A4), then `python scripts/analyze_failures.py --cache data/cache`. If A3 fixes the Reach-phase drift and the cost-validity check shows the cost tracks the cube, adopt residual prediction for all headline models and rerun E1/E2; otherwise keep the default and report the ablation as a negative result.
1. Encode at full resolution (`--pool 1`) on the LeWM cube data or ~1000 collected episodes; train with the DINO-WM-sized defaults for longer (validation loss was still falling at 5k steps), 3 seeds each.
2. E1 at H = 80; E2 with swm's CEM budget (300 × 30) at H = 40 and 80, near / mid / far = 25 / 50 / 100 steps. k=1 planning dominates the cost (1,200 model calls per replan at H = 40), so size the pair count to the GPU budget.
3. Open checks: does the H3 null hold at 16×16 patches; E4 with 3 seeds; action-embedding bottleneck (`--emb-dim` 32 / 64 / 128 at k=8); then E5 (adaptive stride) only after E1-E4.
