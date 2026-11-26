"""E2 diagnosis: the same CEM planner as scripts/eval_plan.py, with the simulator itself as the model.

Why (first A100 run, results_a100/): no learned model planned better than chance, a model with half the
prediction error (k=8 vs k=1) planned no better, and a cost that tracks the cube twice as well did not help.
Giving the planner perfect predictions separates the world model from everything else:
  * if even this fails, the bottleneck is the planner/task setup (search budget, 40-step horizon, a cost
    that is flat until the gripper has the cube), and no world model could have passed this test;
  * if it succeeds, the learned models are the bottleneck.

Identical to eval_plan.py in everything else: the same start/goal pairs (stride.planning.select_pairs), CEM
budget, horizon, replanning interval, action normalization and clipping, success rule and row format.
Planning acts on raw actions (no stride grouping).
  --cost true_cube   cube distance read from the simulator at the goal depth: perfect predictions and a
                     perfect cost.
  --cost latent | probe_cube | probe_cube_ee | patchw | best
                     the frame rendered at the goal depth, encoded like the cache, scored with that learned
                     cost against the goal latent: perfect predictions, our cost. Separates cost from model.
Each candidate costs `depth` simulator steps (plus one render for the learned costs), so episodes run in
parallel worker processes (--workers); results do not depend on the number of workers.
Rows: model 'oracle_sim', arch 'oracle', k 1; calls_per_solve = simulator steps per candidate path."""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.env import make_env  # first: sets MUJOCO_GL=egl

ORACLE = dict(model='oracle_sim', k=1, arch='oracle', variant='')
A = 5  # action dim
G = {}  # per-process state, filled by init()


def init(cfg):
    """Build this process's simulator (and encoder/cost for the learned costs)."""
    torch.set_num_threads(1)
    from stride.data import Cache
    G['cfg'], G['c'], G['env'] = cfg, Cache(cfg['cache'], mmap=True), make_env()
    G['mu'], G['sd'] = G['c'].action_stats()
    if cfg['cost'] != 'true_cube':
        from stride.cost import load_score
        from stride.encoder import load_encoder
        enc_cfg = json.loads((Path(cfg['cache']) / 'encoder.json').read_text())
        G['enc'], G['pool'] = load_encoder(enc_cfg['encoder']).to(cfg['device']), enc_cfg['pool']
        G['score'] = load_score(cfg['cost'], cfg['cost_file'], cfg['device'])


def plan_episode(pair):
    """Plan and execute one start/goal pair; returns its result row."""
    import gymnasium as gym
    from stable_worldmodel.planning.solver.cem import CEMSolver
    from stable_worldmodel.policy import PlanConfig

    from stride.planning import SUCCESS_M, SimCost
    d, t0 = pair
    cfg, c, env, mu, sd = G['cfg'], G['c'], G['env'], G['mu'], G['sd']
    m, H, R = c.m, cfg['horizon'], cfg['replan']
    goal_cube = m['cube_pos'][t0 + d]
    kw = {}
    if cfg['cost'] != 'true_cube':
        kw = dict(goal_z=torch.from_numpy(c.z[t0 + d].astype(np.float32)).to(cfg['device']),
                  score=G['score'], enc=G['enc'], pool=G['pool'])
    cost = SimCost(env, mu, sd, depth=min(d, H), kind=cfg['cost'], goal_cube=goal_cube, **kw)
    solver = CEMSolver(cost, num_samples=cfg['samples'], n_steps=cfg['iters'], topk=cfg['topk'], var_scale=1.0,
                       device='cpu', seed=cfg['seed'])
    solver.configure(action_space=gym.spaces.Box(-1, 1, (1, A)), n_envs=1,
                     config=PlanConfig(horizon=H, receding_horizon=R, action_block=1))
    info_in = {'goal': torch.zeros(1, 1)}  # unused by SimCost; CEMSolver reads the batch size from it
    env.reset(options={'state': np.r_[m['qpos'][t0], m['qvel'][t0]]})
    plan, solve_s, success, dist, best = None, [], False, None, math.inf
    for step in range(2 * d):
        if step % R == 0:
            cost.snapshot()  # candidates start from the current state
            init_a = None if plan is None else torch.cat([plan[R:], torch.zeros(R, A)])[None]
            t = time.time()
            plan = solver.solve(info_in, init_action=init_a)['actions'][0].reshape(H, A)
            solve_s.append(time.time() - t)
        info = env.step(np.clip(plan[step % R].numpy() * sd + mu, -1, 1))[-1]
        dist = float(np.linalg.norm(info['privileged/block_0_pos'] - goal_cube))
        best = min(best, dist)
        if dist <= SUCCESS_M:
            success = True
            break
    return dict(**ORACLE, seed=cfg['seed'], cost=cfg['cost'], offset=d, t0=t0, success=success, steps=step + 1,
                final_dist_cm=100 * dist, min_dist_cm=100 * best,
                start_dist_cm=float(100 * np.linalg.norm(m['cube_pos'][t0] - goal_cube)),
                solves=len(solve_s), solve_sec=float(np.mean(solve_s)), calls_per_solve=cfg['iters'] * H,
                horizon=H, samples=cfg['samples'], iters=cfg['iters'], aligned=(d == H))


def default_workers(kind):
    n = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    return n if kind == 'true_cube' else min(n, 4)  # rendered costs also share one GPU for DINOv2


def main():
    from stride.cost import COSTS
    from stride.data import Cache
    from stride.model import default_device
    from stride.planning import load_done, row_key, select_pairs

    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='data/cache')
    p.add_argument('--horizon', type=int, default=40)
    p.add_argument('--replan', type=int, default=8)
    p.add_argument('--offsets', default=None, help='goal offsets in env steps; default = the horizon')
    p.add_argument('--min-start-cm', type=float, default=8.0)
    p.add_argument('--pairs', type=int, default=20, help='start/goal pairs per offset')
    p.add_argument('--samples', type=int, default=300)
    p.add_argument('--iters', type=int, default=30)
    p.add_argument('--topk', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cost', default='true_cube', choices=('true_cube',) + COSTS + ('best',))
    p.add_argument('--cost-file', default=None, help='fit_cost.py output; default <cache>/cost.pt')
    p.add_argument('--workers', type=int, default=None, help='parallel episodes; default = available CPUs '
                   '(at most 4 for the learned costs)')
    p.add_argument('--out', default='results/plan.jsonl')
    p.add_argument('--device', default=default_device())
    args = p.parse_args()

    cost_file = args.cost_file or str(Path(args.cache) / 'cost.pt')
    kind = args.cost
    if kind == 'best':  # record the cost that fit_cost.py chose, as eval_plan.py does
        kind = torch.load(cost_file, map_location='cpu', weights_only=False)['chosen']
    H = args.horizon
    offsets = [int(x) for x in args.offsets.split(',')] if args.offsets else [H]
    if any(d != H for d in offsets):
        sys.exit(f'goal offsets {offsets} != horizon {H}: run one horizon per offset, as eval_plan.py does')

    c = Cache(args.cache, mmap=True)
    pairs = select_pairs(make_env(), c, offsets, args.pairs, args.min_start_cm, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = load_done(args.out)
    todo = [(d, t) for d, t in pairs if row_key(dict(**ORACLE, seed=args.seed, cost=kind, offset=d, t0=t, horizon=H,
                                                     samples=args.samples, iters=args.iters)) not in done]
    workers = max(1, min(args.workers or default_workers(kind), len(todo) or 1))
    print(f'simulator planner | cost {kind} | {len(todo)} of {len(pairs)} pairs to plan | {workers} workers '
          f'| budget {args.samples} x {args.iters}', flush=True)
    cfg = dict(cache=args.cache, horizon=H, replan=args.replan, samples=args.samples, iters=args.iters,
               topk=args.topk, seed=args.seed, cost=kind, cost_file=cost_file, device=args.device)
    pool = None
    if workers == 1:
        init(cfg)
        results = map(plan_episode, todo)
    else:
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ.setdefault(v, '1')  # one thread per worker process
        # maxtasksperchild=1: a fresh process per pair, so memory cannot build up across pairs (the env
        # recompiles its MuJoCo model on every reset); start-up takes seconds, a pair takes minutes
        pool = mp.get_context('spawn').Pool(workers, initializer=init, initargs=(cfg,), maxtasksperchild=1)
        results = pool.imap_unordered(plan_episode, todo)
    for row in results:
        with open(args.out, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    if pool is not None:
        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
