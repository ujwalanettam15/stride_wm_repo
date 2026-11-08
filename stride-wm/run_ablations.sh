#!/usr/bin/env bash
# Ablations and controls added after the prelim. NOT yet run: written so the whole set is one command
# once the GPU / time budget exists. Same data, predictor size, steps, batch and seeds as run_prelim.sh;
# each variant differs from the default in exactly one setting, and gets its own checkpoint suffix
# (-res, -uN, -eN, -embN) so report.py keeps it out of the headline tables.
#
# Prerequisite: run_prelim.sh has been run (it creates data/cache and the 12 headline checkpoints in
# runs/ that the variants are compared against). Planning goes to results/plan_aligned.jsonl, separate
# from the prelim's results/plan.jsonl (the old, misaligned planner), because eval_plan.py APPENDS.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
[ -f data/cache/z.npy ] || { echo "data/cache missing: run 'bash run_prelim.sh' first (README, step 5)"; exit 1; }
[ -f runs/chunk_k1_s0.pt ] || { echo "runs/ has no headline checkpoints: run 'bash run_prelim.sh' first"; exit 1; }
[ ! -f results/plan_aligned.jsonl ] || { echo "results/plan_aligned.jsonl exists (eval_plan.py appends): move it first"; exit 1; }
SEEDS=${SEEDS:-"0 1"}
PAIRS=${PAIRS:-10}              # planning pairs per model; planning is the slowest step
PLAN_CKPTS=${PLAN_CKPTS:-"runs/*_s0.pt"}   # restrict to save time, e.g. "runs/chunk_k1_s0.pt runs/chunk_k8_s0.pt runs/dinowm_k5_s0.pt"
SPAN=${SPAN:-16}               # env steps of chained prediction in the A1 training loss
SMALL=${SMALL:-"--depth 3 --heads 6 --mlp-dim 1024 --steps 5000 --batch 32"}   # match run_prelim.sh

python -m pytest -q tests

for s in $SEEDS; do
  # A1  stride vs compounding: every stride trained on the same SPAN-step stretch of chained predictions
  #     (k=8 at SPAN=8 is the default setup, a built-in check). If k=1 closes its gap, the stride
  #     advantage in E1 is fewer recursive calls on a model trained only on ground-truth inputs, not a
  #     better k-step predictor. Cost per step grows with SPAN/k (k=1 at SPAN=16: ~16x, order of an hour
  #     per model at the prelim size): lower SPAN if time is short. k=5 is skipped (5 does not divide 16).
  for k in 1 2 4 8; do
    u=$((SPAN / k))
    [ $((u * k)) -eq "$SPAN" ] || { echo "SPAN=$SPAN is not a multiple of k=$k"; exit 1; }
    [ "$u" -gt 1 ] || continue   # unroll 1 is the default model (same name): already trained, do not overwrite it
    python scripts/train.py --arch chunk --k "$k" --unroll "$u" --seed "$s" $SMALL
  done
  # A2  E4 confound: DINO-WM's 10-d action embedding vs the chunk encoder's 64-d. dinowm-emb 64 and
  #     chunk emb-dim 10 each remove the width difference from one side; the encoder sweep is at k=8.
  python scripts/train.py --arch dinowm --k 5 --dinowm-emb 64 --seed "$s" $SMALL
  python scripts/train.py --arch chunk  --k 5 --emb-dim 10    --seed "$s" $SMALL
  for e in 32 128; do python scripts/train.py --arch chunk --k 8 --emb-dim "$e" --seed "$s" $SMALL; done
  # A3  residual prediction (z_t + predictor), both architectures, for the k that bracket the range.
  for k in 1 5 8; do python scripts/train.py --arch chunk --k "$k" --residual --seed "$s" $SMALL; done
  python scripts/train.py --arch dinowm --k 5 --residual --seed "$s" $SMALL
done

# Open loop for every checkpoint (headline + variants), with the action-sensitivity control and a
# probe that sees the 8x8 grid (the default probe pools to 4x4 whatever --pool was).
python scripts/eval_openloop.py --ckpts runs/*.pt --horizon 40 --every 2 --out results/openloop.csv
python scripts/eval_openloop.py --ckpts runs/*_s0.pt --horizon 40 --every 2 --shuffle-actions \
  --out results/openloop_shuf.csv
python scripts/eval_openloop.py --ckpts runs/chunk_k1_s0.pt runs/chunk_k8_s0.pt --horizon 40 --every 2 \
  --probe-grid 8 --out results/openloop_probe8.csv
# report.py reads one csv: append the control rows to the main one before reporting.
python - <<'PY'
import pandas as pd
a = pd.read_csv('results/openloop.csv')
b = pd.read_csv('results/openloop_shuf.csv')
b = b[b.arch.str.endswith('-shuf')]
pd.concat([a, b]).to_csv('results/openloop.csv', index=False)
PY

# A4  planning controls (this is the slow part; seed-0 checkpoints; aligned goal = horizon).
#     'random' is the null planner; 'replay' the pipeline check. Run H=40 first, 80 if time allows.
python scripts/eval_plan.py --ckpts replay random $PLAN_CKPTS --horizon 40 --pairs "$PAIRS" \
  --samples 100 --iters 10 --topk 10 --out results/plan_aligned.jsonl

python scripts/report.py --plan results/plan_aligned.jsonl
python scripts/analyze_failures.py --cache data/cache --plan results/plan_aligned.jsonl
