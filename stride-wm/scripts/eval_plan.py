"""E2 + E3: MPC in the frozen latent space with stable-worldmodel's CEMSolver.

Start = simulator state of a held-out episode at t0; goal = that episode's frame at t0 + d, with
d in near/mid/far offsets. Pairs where the cube is already within 4 cm of its goal, or that holding
still would solve (a dropped cube landing on the goal), are skipped.
The planner always optimizes H raw actions; stride k only groups them into H/k chunks, i.e. H/k
model calls per candidate. Cost = MSE between the final predicted latent and the goal latent.
Every condition shares the CEM budget, replanning interval, pairs and seeds.
Success = cube within 4 cm of the goal cube position at any step of a 2d-step budget."""
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

from stride.data import Cache
from stride.encoder import encode, load_encoder
from stride.model import default_device, load


class LatentGoalCost:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def get_cost(self, info, cand):  # cand: (B, N, H/k, k*A) normalized actions
        B, N = cand.shape[:2]
        z0 = info['z'].reshape(B * N, *info['z'].shape[2:])
        zH = self.model.rollout(z0, cand.reshape(B * N, -1, self.model.action_dim))[:, -1]
        return ((zH - info['goal'].reshape(zH.shape)) ** 2).mean((1, 2)).view(B, N)


p = argparse.ArgumentParser()
p.add_argument('--cache', default='data/cache')
p.add_argument('--ckpts', nargs='+', required=True,
               help="checkpoints; 'replay' executes the logged actions instead (pipeline check)")
p.add_argument('--horizon', type=int, default=40)
p.add_argument('--replan', type=int, default=8)
p.add_argument('--offsets', default='25,50,100', help='near,mid,far goal offsets (env steps)')
p.add_argument('--pairs', type=int, default=20, help='start/goal pairs per offset')
p.add_argument('--samples', type=int, default=300)  # swm cem.yaml defaults
p.add_argument('--iters', type=int, default=30)
p.add_argument('--topk', type=int, default=30)
p.add_argument('--fast-render', action='store_true')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default='results/plan.jsonl')
p.add_argument('--device', default=default_device())
args = p.parse_args()

H, R = args.horizon, args.replan
c = Cache(args.cache, mmap=True)
m = c.m
enc_cfg = json.loads((Path(args.cache) / 'encoder.json').read_text())  # encode frames like the cache
enc = load_encoder(enc_cfg['encoder']).to(args.device)
env = make_env(args.fast_render)

def solved_without_planning(t0, d):
    """True if holding still (zero action) already brings the cube to the goal, e.g. a dropped
    cube landing on it. Such pairs test the dynamics, not the planner, so they are skipped."""
    env.reset(options={'state': np.r_[m['qpos'][t0], m['qvel'][t0]]})
    for _ in range(2 * d):
        info = env.step(np.zeros(5))[-1]
        if np.linalg.norm(info['privileged/block_0_pos'] - m['cube_pos'][t0 + d]) <= 0.04:
            return True
    return False


rng = np.random.default_rng(args.seed)
pairs = []
for d in map(int, args.offsets.split(',')):
    t0 = c.starts('eval', d)
    t0 = t0[np.linalg.norm(m['cube_pos'][t0 + d] - m['cube_pos'][t0], axis=1) > 0.04]
    chosen = []
    for t in rng.permutation(t0):
        if len(chosen) == args.pairs:
            break
        if not solved_without_planning(int(t), d):
            chosen.append(int(t))
    pairs += [(d, t) for t in chosen]

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
for path in args.ckpts:
    replay = path == 'replay'
    if replay:
        info_row = dict(model='expert_replay', k=0, arch='replay', seed=0)
    else:
        model, ck = load(path, args.device)
        k, A = model.k, model.action_dim
        mu, sd = ck['act_mean'], ck['act_std']
        info_row = dict(model=Path(path).stem, k=k, arch=ck['model_kw']['arch'], seed=ck['cfg']['seed'])
        solver = CEMSolver(LatentGoalCost(model), num_samples=args.samples, n_steps=args.iters,
                           topk=args.topk, var_scale=1.0, device=args.device, seed=args.seed)
        solver.configure(action_space=gym.spaces.Box(-1, 1, (1, A)), n_envs=1,
                         config=PlanConfig(horizon=H // k, receding_horizon=R, action_block=k))
    for d, t0 in pairs:
        env.reset(options={'state': np.r_[m['qpos'][t0], m['qvel'][t0]]})
        goal = torch.from_numpy(c.z[t0 + d].astype(np.float32))[None]
        goal_cube = m['cube_pos'][t0 + d]
        plan, solve_s, success, dist = None, [], False, None
        for step in range(d if replay else 2 * d):
            if replay:
                a = m['action'][t0 + step]
            elif step % R == 0:
                z = encode(enc, env.unwrapped.render()[None], enc_cfg['pool'])
                init = None
                if plan is not None:  # warm start: shift the last plan by R raw actions
                    init = torch.cat([plan[R:], torch.zeros(R, A)]).reshape(1, H // k, k * A)
                t = time.time()
                plan = solver.solve({'z': z, 'goal': goal}, init_action=init)['actions'][0].reshape(H, A)
                solve_s.append(time.time() - t)
            if not replay:
                a = np.clip(plan[step % R].numpy() * sd + mu, -1, 1)
            _, _, _, _, info = env.step(a)
            dist = float(np.linalg.norm(info['privileged/block_0_pos'] - goal_cube))
            if dist <= 0.04:
                success = True
                break
        row = dict(**info_row, offset=d, t0=t0, success=success, steps=step + 1, final_dist_cm=100 * dist,
                   start_dist_cm=float(100 * np.linalg.norm(m['cube_pos'][t0] - goal_cube)),
                   solves=len(solve_s), solve_sec=float(np.mean(solve_s)) if solve_s else 0.0,
                   calls_per_solve=0 if replay else args.iters * H // k, horizon=H,
                   samples=args.samples, iters=args.iters)
        with open(args.out, 'a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
