"""Pieces shared by the planning scripts (eval_plan.py: learned models; eval_oracle.py: the simulator).

select_pairs  start/goal pairs, identical for every planner given the same cache, offsets and seed
row_key       what identifies a result row, so reruns skip finished rows
load_done     row keys already in a results file (drops a half-written last row first)
SimCost       CEM cost with the simulator itself as the model: perfect predictions
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

SUCCESS_M = 0.04  # success = cube within 4 cm of its goal position


def solved_without_planning(env, m, t0, d):
    """True if holding still (zero action) already brings the cube to the goal, e.g. a dropped
    cube landing on it. Such pairs test the dynamics, not the planner, so they are skipped."""
    env.reset(options={'state': np.r_[m['qpos'][t0], m['qvel'][t0]]})
    for _ in range(2 * d):
        info = env.step(np.zeros(5))[-1]
        if np.linalg.norm(info['privileged/block_0_pos'] - m['cube_pos'][t0 + d]) <= SUCCESS_M:
            return True
    return False


def select_pairs(env, c, offsets, n_pairs, min_start_cm, seed):
    """[(d, t0), ...]: held-out starts whose cube must move more than min_start_cm to reach the frame d steps
    later, and that holding still would not solve. Same inputs -> same pairs, whichever planner runs."""
    m = c.m
    rng = np.random.default_rng(seed)
    pairs = []
    for d in offsets:
        t0 = c.starts('eval', d)
        t0 = t0[np.linalg.norm(m['cube_pos'][t0 + d] - m['cube_pos'][t0], axis=1) > min_start_cm / 100]
        chosen = []
        for t in rng.permutation(t0):
            if len(chosen) == n_pairs:
                break
            if not solved_without_planning(env, m, int(t), d):
                chosen.append(int(t))
        if len(chosen) < n_pairs:
            print(f'warning: only {len(chosen)} pairs at offset {d} (asked for {n_pairs})', flush=True)
        pairs += [(d, t) for t in chosen]
    return pairs


def row_key(r):  # rows written before --cost existed planned with the latent cost
    return (r['model'], r.get('cost', 'latent' if r['k'] else ''), r['offset'], r['t0'], r.get('horizon'),
            r.get('samples'), r.get('iters'), r['seed'])


def load_done(out):
    """Keys of the rows already in `out`. A job killed mid-write leaves a last line without its newline:
    drop it, or the next row would be appended onto it and both would be unreadable."""
    out = Path(out)
    if not out.exists():
        return set()
    raw = out.read_bytes()
    if raw and not raw.endswith(b'\n'):
        out.write_bytes(raw[: raw.rfind(b'\n') + 1])
        print(f'dropped a half-written last row in {out}', flush=True)
    done = {row_key(json.loads(x)) for x in open(out) if x.strip()}
    print(f'{len(done)} rows already in {out}; matching runs are skipped', flush=True)
    return done


class SimCost:
    """CEM cost that uses the simulator as the model: every candidate is executed from a copy of the current
    simulator state (exactly what env.step would do, without touching the real episode) and scored at the
    goal depth. Perfect predictions, so a planner that still fails is limited by search or task, not by a model.

    kind='true_cube'  squared distance between the cube and its goal position: perfect cost too.
    any GoalScore     the frame rendered at the goal depth is encoded like the cache and scored against the
                      goal latent with that score (latent, probe_cube, ...): perfect predictions, learned cost.
    Call snapshot() before each solve; get_cost leaves the real simulator state as it found it.
    """

    def __init__(self, env, mu, sd, depth, kind='true_cube', goal_cube=None, goal_z=None, score=None,
                 enc=None, pool=1, batch=100):
        self.u = env.unwrapped
        self.mu, self.sd = np.asarray(mu, np.float64), np.asarray(sd, np.float64)
        self.depth, self.kind, self.score = depth, kind, score
        self.goal_cube, self.goal_z = goal_cube, goal_z
        self.enc, self.pool, self.batch = enc, pool, batch
        self.base, self._base_model = None, None
        if kind != 'true_cube' and (score is None or enc is None or goal_z is None):
            raise ValueError(f'cost {kind!r} on rendered frames needs score, enc and goal_z')

    def snapshot(self):
        """Remember the current simulator state; candidates start from it."""
        u = self.u
        if self._base_model is not u._model:  # the env recompiles its model on reset
            self.base, self._base_model = mujoco.MjData(u._model), u._model
        mujoco.mj_copyData(self.base, u._model, u._data)

    def _restore(self):
        mujoco.mj_copyData(self.u._data, self.u._model, self.base)

    def execute(self, actions):
        """Run raw env actions (n, A) from the snapshot, as env.step does (set_control + n_steps physics steps)."""
        u = self.u
        self._restore()
        for a in actions:
            u.set_control(a)
            mujoco.mj_step(u._model, u._data, nstep=u._n_steps)

    def cube_pos(self):
        return self.u._data.joint('object_joint_0').qpos[:3].copy()

    @torch.no_grad()
    def get_cost(self, info, cand):  # cand: (B, N, H, A) normalized actions (action_block = 1)
        B, N, H, A = cand.shape
        acts = np.clip(cand.reshape(B * N, H, A).double().cpu().numpy() * self.sd + self.mu, -1, 1)
        acts = acts[:, : self.depth]
        costs, frames = [], []
        try:
            for seq in acts:
                self.execute(seq)
                if self.kind == 'true_cube':
                    costs.append(float(((self.cube_pos() - self.goal_cube) ** 2).sum()))
                else:
                    frames.append(self.u.render())
        finally:
            self._restore()  # leave the real episode exactly where it was
        if self.kind == 'true_cube':
            return torch.tensor(costs, dtype=torch.float32).view(B, N)
        from stride.encoder import encode
        z = torch.cat([encode(self.enc, np.stack(frames[i: i + self.batch]), self.pool)
                       for i in range(0, len(frames), self.batch)])
        dev = self.goal_z.device
        g = self.goal_z.reshape(1, *z.shape[1:]).expand(len(z), -1, -1)
        return self.score(z.to(dev), g).float().cpu().view(B, N)
