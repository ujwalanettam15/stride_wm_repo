"""Fit the candidate planning costs and check them on true latents. Runs no world model and no simulator.

Fitting uses training episodes only:
  patchw  per-patch weights (non-negative least squares) so that the weighted per-patch squared error
          between a frame and a goal frame tracks the cube's distance to its goal position
  probe   the same frozen linear probe eval_openloop.py fits (latent -> cube / end-effector pose)
Checking uses held-out episodes, as in analyze_failures.py section 4: for goal frames from the last 30%
of each held-out episode, the Spearman rank correlation between each cost (every frame of that episode
vs the goal) and the true cube and end-effector distances to the goal.

The choice rule is fixed before any planning run: the candidate with the highest mean cube rho is
written to <cache>/cost.pt as 'chosen', and `eval_plan.py --cost best` plans with it."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.cost import COSTS, GoalScore, nnls
from stride.data import Cache, Probe, probe_features, probe_targets

p = argparse.ArgumentParser()
p.add_argument('--cache', default='data/cache')
p.add_argument('--goals', type=int, default=5, help='goal frames per episode (fit and check)')
p.add_argument('--fit-episodes', type=int, default=200, help='training episodes used to fit patch weights')
p.add_argument('--probe-grid', type=int, default=4)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--out', default=None, help='markdown table; default <cache>/cost_check.md')
args = p.parse_args()
c = Cache(args.cache, mmap=True)
m = c.m
P = c.z.shape[1]
rng = np.random.default_rng(args.seed)


def episode(e):
    o, n = c.ep_offset[e], c.ep_len[e]
    return o, n, torch.from_numpy(c.z[o: o + n].astype(np.float32))


def goals_of(n):
    return rng.choice(np.arange(int(0.7 * n), n), min(args.goals, n - int(0.7 * n)), replace=False)


def feats(idx):
    return torch.cat([probe_features(c.z[idx[i: i + 4096]], args.probe_grid) for i in range(0, len(idx), 4096)])


# ---- probe (identical recipe to eval_openloop.py: lambda picked on the last 10% of training frames)
tr = c.frames('train')
cut = tr[int(0.9 * len(tr))]
probe = Probe().fit(feats(tr[tr < cut]), torch.from_numpy(probe_targets(m, tr[tr < cut])),
                    feats(tr[tr >= cut]), torch.from_numpy(probe_targets(m, tr[tr >= cut])))
print(f'probe fit, lambda {probe.lam}', flush=True)

# ---- patch weights: per-patch squared error to a goal frame -> cube distance to that goal
train_eps = list(c.episodes('train'))
train_eps = [train_eps[i] for i in np.sort(rng.choice(len(train_eps), min(args.fit_episodes, len(train_eps)),
                                                      replace=False))]
F, y = [], []
for e in train_eps:
    o, n, z = episode(e)
    for g in goals_of(n):
        F.append(((z - z[g]) ** 2).mean(2))
        y.append(torch.from_numpy(np.linalg.norm(m['cube_pos'][o: o + n] - m['cube_pos'][o + g], axis=1)))
w = nnls(torch.cat(F), torch.cat(y).double())
w = w / w.sum() if w.sum() > 0 else torch.full((P,), 1.0 / P, dtype=torch.float64)
print(f'patch weights fit on {len(train_eps)} episodes: {int((w > 0).sum())}/{P} patches non-zero', flush=True)
params = dict(patch_w=w.float(), probe=dict(mx=probe.mx, my=probe.my, W=probe.W, grid=args.probe_grid,
                                            lam=probe.lam), num_patches=P, cache=str(args.cache))

# ---- check on held-out episodes
rank = lambda a: pd.Series(np.asarray(a)).rank().to_numpy()
rho = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
scores = {k: GoalScore(k, params) for k in COSTS}
out = {k: {'cube': [], 'ee': []} for k in COSTS}
with torch.no_grad():
    for e in c.episodes('eval'):
        o, n, z = episode(e)
        cube, ee = m['cube_pos'][o: o + n], m['ee_pos'][o: o + n]
        for g in goals_of(n):
            dc, de = np.linalg.norm(cube - cube[g], axis=1), np.linalg.norm(ee - ee[g], axis=1)
            for k, s in scores.items():
                cost = s(z, z[g].expand_as(z)).numpy()
                out[k]['cube'].append(rho(cost, dc))
                out[k]['ee'].append(rho(cost, de))

rows = [dict(cost=k, cube_rho=np.nanmean(v['cube']), ee_rho=np.nanmean(v['ee']), goals=len(v['cube']))
        for k, v in out.items()]
best = max(rows, key=lambda r: r['cube_rho'])['cost']
params.update(chosen=best, rho={r['cost']: (r['cube_rho'], r['ee_rho']) for r in rows})
torch.save(params, Path(args.cache) / 'cost.pt')

md = ['# Planning-cost check (true latents, held-out episodes; no model, no simulator)\n\n',
      f'Cache: {args.cache} ({P} patches). Spearman rank correlation, per goal frame, between each cost and the '
      'true distance to the goal; 1 = a perfect proxy. Fit on training episodes only.\n\n',
      '| cost | cube rho | end-effector rho | goals |\n|---|---|---|---|\n']
md += [f"| {r['cost']} | {r['cube_rho']:.2f} | {r['ee_rho']:.2f} | {r['goals']} |\n" for r in rows]
md.append(f'\nChosen (highest cube rho, rule fixed before planning): **{best}** -> {Path(args.cache) / "cost.pt"}\n')
out_md = Path(args.out) if args.out else Path(args.cache) / 'cost_check.md'
out_md.parent.mkdir(parents=True, exist_ok=True)
out_md.write_text(''.join(md))
print(''.join(md))
