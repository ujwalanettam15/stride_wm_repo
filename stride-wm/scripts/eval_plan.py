"""E2 + E3: MPC in the frozen latent space with stable-worldmodel's CEMSolver.

Start = simulator state of a held-out episode at t0; goal = that episode's frame at t0 + d.
Pairs where the cube is already within --min-start-cm of its goal, or that holding still would solve
(a dropped cube landing on the goal), are skipped.
The planner always optimizes H raw actions; stride k only groups them into H/k chunks, i.e. H/k
model calls per candidate. Cost = MSE between the latent predicted d steps ahead and the goal latent
(stride.cost.LatentGoalCost). Every condition shares the CEM budget, replanning interval, pairs and seeds.
Success = cube within 4 cm of the goal cube position at any step of a 2d-step budget.

Changes after the prelim (see README, "Changes to the original design"):
  * goal offset = horizon. The prelim used H = 40 with goals 25 steps away and scored the prediction at
    t+40, so the cost compared frames 15 steps apart. Offsets default to [H]; run --horizon 40 and
    --horizon 80 separately. --allow-misaligned restores the old behaviour for comparison.
  * candidates are clamped to the env bounds before scoring (they were scored unclipped, executed clipped).
  * --min-start-cm 8: the prelim's only successes were pairs starting 5.6 and 7.4 cm from the goal, i.e.
    a 1.6-3.4 cm nudge inside a 4 cm success radius. Success on those says almost nothing.
  * 'random' pseudo-checkpoint: executes actions drawn from the training action distribution. It is the
    null planner every success rate must beat. ('replay' remains the pipeline check.)
  * logs min_dist_cm (closest approach) so progress can be reported next to success.
  * --cost picks how the predicted latent is compared with the goal (stride/cost.py): the original
    latent MSE, or a cost fit by scripts/fit_cost.py ('best' = the one it chose). Each row records it.
  * reruns skip (model, cost, pair, budget) rows already in --out, so a job killed at its time limit
    resumes where it stopped instead of duplicating rows.
  * pair selection, row keys and resuming live in stride/planning.py, shared with scripts/eval_oracle.py
    (the same planner with the simulator as its model), so both plan on exactly the same pairs.
  * a model whose stride does not divide the horizon is skipped with a message (k=5 at H=16 or 24)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.env import make_env  # first: sets MUJOCO_GL=egl

import gymnasium as gym
from stable_worldmodel.planning.solver.cem import CEMSolver
from stable_worldmodel.policy import PlanConfig

from stride.cost import COSTS, LatentGoalCost, load_score
from stride.data import Cache
from stride.encoder import encode, load_encoder
from stride.model import default_device, enable_tf32, load, variant_tag
from stride.planning import load_done, row_key, select_pairs

p = argparse.ArgumentParser()
p.add_argument('--cache', default='data/cache')
p.add_argument('--ckpts', nargs='+', required=True,
               help="checkpoints; 'replay' executes the logged actions (pipeline check), "
                    "'random' executes random actions (null planner)")
p.add_argument('--horizon', type=int, default=40)
p.add_argument('--replan', type=int, default=8)
p.add_argument('--offsets', default=None, help='goal offsets in env steps; default = the horizon')
p.add_argument('--allow-misaligned', action='store_true', help='allow offsets != horizon (prelim behaviour)')
p.add_argument('--min-start-cm', type=float, default=8.0, help='skip pairs whose cube starts closer to the goal')
p.add_argument('--pairs', type=int, default=20, help='start/goal pairs per offset')
p.add_argument('--samples', type=int, default=300)  # swm cem.yaml defaults
p.add_argument('--iters', type=int, default=30)
p.add_argument('--topk', type=int, default=30)
p.add_argument('--fast-render', action='store_true')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--cost', default='latent', choices=COSTS + ('best',),
               help="planning cost (stride/cost.py); 'best' = the one scripts/fit_cost.py chose")
p.add_argument('--cost-file', default=None, help='fit_cost.py output; default <cache>/cost.pt')
p.add_argument('--tf32', action='store_true', help='TF32 matmuls on CUDA (A100: ~2x faster)')
p.add_argument('--out', default='results/plan.jsonl')
p.add_argument('--device', default=default_device())
args = p.parse_args()
enable_tf32(args.tf32)

H, R = args.horizon, args.replan
offsets = [int(x) for x in args.offsets.split(',')] if args.offsets else [H]
if not args.allow_misaligned and any(d != H for d in offsets):
    sys.exit(f'goal offsets {offsets} != horizon {H}: the cost compares the prediction at the goal depth '
             f'to the goal frame, so run one horizon per offset (or pass --allow-misaligned)')
c = Cache(args.cache, mmap=True)
m = c.m
mu_a, sd_a = c.action_stats()
enc_cfg = json.loads((Path(args.cache) / 'encoder.json').read_text())  # encode frames like the cache
enc = load_encoder(enc_cfg['encoder']).to(args.device)
env = make_env(args.fast_render)
score = load_score(args.cost, args.cost_file or Path(args.cache) / 'cost.pt', args.device)
print(f'planning cost: {score.kind}', flush=True)

pairs = select_pairs(env, c, offsets, args.pairs, args.min_start_cm, args.seed)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
done = load_done(args.out)
for path in args.ckpts:
    mode = path if path in ('replay', 'random') else 'model'
    if mode == 'replay':
        info_row = dict(model='expert_replay', k=0, arch='replay', variant='', seed=0, cost='')
    elif mode == 'random':
        info_row = dict(model='random_actions', k=0, arch='random', variant='', seed=args.seed, cost='')
    else:
        model, ck = load(path, args.device)
        if ck['model_kw']['num_patches'] != c.z.shape[1]:
            sys.exit(f"{path} was trained on {ck['model_kw']['num_patches']} patches but {args.cache} has "
                     f'{c.z.shape[1]}: pass the cache the model was trained on')
        k, A = model.k, model.action_dim
        if H % k:
            print(f'skip {path}: stride k={k} does not divide the horizon H={H}', flush=True)
            continue
        mu, sd = ck['act_mean'], ck['act_std']
        info_row = dict(model=Path(path).stem, k=k, arch=ck['model_kw']['arch'],
                        variant=variant_tag(ck['cfg']), seed=ck['cfg']['seed'], cost=score.kind)
    for d, t0 in pairs:
        if row_key(dict(**info_row, offset=d, t0=t0, horizon=H, samples=args.samples, iters=args.iters)) in done:
            continue
        act_rng = np.random.default_rng([args.seed, t0])  # random null: per pair, so a resumed run matches
        if mode == 'model':  # the cost scores the prediction at the goal's depth (d = H unless misaligned)
            solver = CEMSolver(LatentGoalCost(model, mu, sd, depth=min(d, H), score=score),
                               num_samples=args.samples, n_steps=args.iters, topk=args.topk, var_scale=1.0,
                               device=args.device, seed=args.seed)
            solver.configure(action_space=gym.spaces.Box(-1, 1, (1, A)), n_envs=1,
                             config=PlanConfig(horizon=H // k, receding_horizon=R, action_block=k))
        env.reset(options={'state': np.r_[m['qpos'][t0], m['qvel'][t0]]})
        goal = torch.from_numpy(c.z[t0 + d].astype(np.float32))[None]
        goal_cube = m['cube_pos'][t0 + d]
        plan, solve_s, success, dist, best = None, [], False, None, float('inf')
        for step in range(d if mode == 'replay' else 2 * d):
            if mode == 'replay':
                a = m['action'][t0 + step]
            elif mode == 'random':
                a = np.clip(act_rng.standard_normal(5) * sd_a + mu_a, -1, 1)
            else:
                if step % R == 0:
                    z = encode(enc, env.unwrapped.render()[None], enc_cfg['pool'])
                    init = None
                    if plan is not None:  # warm start: shift the last plan by R raw actions
                        init = torch.cat([plan[R:], torch.zeros(R, A)]).reshape(1, H // k, k * A)
                    t = time.time()
                    plan = solver.solve({'z': z, 'goal': goal}, init_action=init)['actions'][0].reshape(H, A)
                    solve_s.append(time.time() - t)
                a = np.clip(plan[step % R].numpy() * sd + mu, -1, 1)
            _, _, _, _, info = env.step(a)
            dist = float(np.linalg.norm(info['privileged/block_0_pos'] - goal_cube))
            best = min(best, dist)
            if dist <= 0.04:
                success = True
                break
        row = dict(**info_row, offset=d, t0=t0, success=success, steps=step + 1, final_dist_cm=100 * dist,
                   min_dist_cm=100 * best,
                   start_dist_cm=float(100 * np.linalg.norm(m['cube_pos'][t0] - goal_cube)),
                   solves=len(solve_s), solve_sec=float(np.mean(solve_s)) if solve_s else 0.0,
                   calls_per_solve=args.iters * H // k if mode == 'model' else 0, horizon=H,
                   samples=args.samples, iters=args.iters, aligned=(d == H))
        with open(args.out, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
