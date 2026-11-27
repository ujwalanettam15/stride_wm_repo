"""Refit the cube probe with off-expert frames (scripts/collect_rand.py) and check it before any planning run.

Fit: the expert frames of the training episodes (as fit_cost.py) plus the off-expert frames that start in training
episodes. Ridge regression, lambda picked on the last 10% of the training episodes (both kinds of frames).
Candidates: linear probes on two feature grids, 4 (the original probe: the 8x8 latent average-pooled to 4x4) and 8 (no
pooling), and, when a CUDA GPU is available, a small MLP on the 8x8 grid. The one with the lowest cube error on held-out
off-expert frames is kept; the rule is fixed before planning.

Check, on held-out episodes only, for the original probe (<cache>/cost.pt) and the refit one:
  * cube position error (cm) on expert frames, on off-expert frames, and on off-expert frames with the arm far from
    the cube (> 10 cm), where a probe that reads the arm to place the cube goes wrong
  * cube rank correlation as in fit_cost.py, but over the candidates a planner compares: for each held-out start,
    the expert frames 0, 8, ..., 32 steps on plus that start's five off-expert frames, against the expert frame
    40 steps on as the goal. 1 = the cost orders candidates exactly by the cube's true distance to its goal.
Writes <cache>/cost_rand.pt (eval_plan.py / eval_oracle.py --cost probe_cube_rand) and a markdown table."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.cost import GoalScore
from stride.data import Cache, probe_features, probe_targets
from stride.model import default_device

p = argparse.ArgumentParser()
p.add_argument('--cache', default='data/cache')
p.add_argument('--rand', default='data/cache_rand', help='scripts/collect_rand.py output')
p.add_argument('--grids', default='4,8', help='feature grids to fit; the best on held-out off-expert frames is kept')
p.add_argument('--mlp', choices=('auto', 'on', 'off'), default='auto',
               help='also fit a small MLP probe (auto = when a CUDA GPU is available)')
p.add_argument('--mlp-grid', type=int, default=8)
p.add_argument('--mlp-epochs', type=int, default=30)
p.add_argument('--chunk', type=int, default=2048)
p.add_argument('--device', default=default_device())
p.add_argument('--out', default=None, help='markdown table; default <cache>/cost_rand_check.md')
args = p.parse_args()
dev = torch.device(args.device if args.device != 'mps' else 'cpu')  # float64 solves: not on MPS
c, r = Cache(args.cache, mmap=True), Cache.__new__(Cache)
r.z = np.load(Path(args.rand) / 'z.npy', mmap_mode='r')
r.m = dict(np.load(Path(args.rand) / 'meta.npz'))
if r.z.shape[1:] != c.z.shape[1:]:
    sys.exit(f'{args.rand} latents {r.z.shape[1:]} do not match {args.cache} {c.z.shape[1:]}')
grids = [int(g) for g in args.grids.split(',')]

# ---- frames: (source, index) for fit / validation / held-out check
tr = c.frames('train')
cut = tr[int(0.9 * len(tr))]  # validation = the last 10% of training episodes (frames from cut on)
rs = r.m['start']
SETS = {
    'fit': [(c, tr[tr < cut]), (r, np.where((r.m['split'] == 0) & (rs < cut))[0])],
    'val': [(c, tr[tr >= cut]), (r, np.where((r.m['split'] == 0) & (rs >= cut))[0])],
}
ev_expert, ev_rand = c.frames('eval'), np.where(r.m['split'] == 1)[0]
print(f"fit: {len(SETS['fit'][0][1])} expert + {len(SETS['fit'][1][1])} off-expert frames | validation: "
      f"{len(SETS['val'][0][1])} + {len(SETS['val'][1][1])} | held out: {len(ev_expert)} + {len(ev_rand)}", flush=True)


def batches(sets, grid):
    for src, idx in sets:
        for i in range(0, len(idx), args.chunk):
            j = idx[i: i + args.chunk]
            yield probe_features(src.z[j], grid), torch.from_numpy(probe_targets(src.m, j))


def fit(grid):
    """Ridge regression from running sums (the 8x8 grid has 24,576 features: no copy of the data in memory)."""
    n, sx, sy, sxx, sxy = 0, 0, 0, 0, 0
    for X, Y in batches(SETS['fit'], grid):
        X, Y = X.to(dev, torch.float64), Y.to(dev, torch.float64)
        n, sx, sy = n + len(X), sx + X.sum(0), sy + Y.sum(0)
        sxx, sxy = sxx + X.T @ X, sxy + X.T @ Y
    mx, my = sx / n, sy / n
    G, XY = sxx.sub_(n * torch.outer(mx, mx)), sxy - n * torch.outer(mx, my)  # in place: G is 4.8 GB at grid 8
    scale, Ws = G.diagonal().mean(), {}
    for lam in (1e-3, 1e-2, 1e-1, 1.0):
        A = G.clone()
        A.diagonal().add_(lam * scale)
        Ws[lam] = torch.linalg.solve(A, XY)
        del A
    err = {lam: 0.0 for lam in Ws}
    for X, Y in batches(SETS['val'], grid):
        X, Y = X.to(dev, torch.float64) - mx, Y.to(dev, torch.float64)
        for lam, W in Ws.items():
            err[lam] += ((X @ W + my - Y) ** 2).sum().item()
    lam = min(err, key=err.get)
    return dict(mx=mx.float().cpu(), my=my.float().cpu(), W=Ws[lam].float().cpu(), grid=grid, lam=lam)


def fit_mlp(grid, hidden=512, wd=1e-4):
    """Two hidden layers, standardized inputs and targets, cube columns weighted 4x; the epoch with the lowest
    validation cube error is kept (early stopping)."""
    def load(sets):
        X, Y = zip(*batches(sets, grid))
        return torch.cat(X).to(dev, torch.float16), torch.cat(Y).to(dev)
    X, Y = load(SETS['fit'])
    Xv, Yv = load(SETS['val'])
    mx, sx = X.float().mean(0), X.float().std(0).clamp_min(1e-3)
    my, sy = Y.mean(0), Y.std(0).clamp_min(1e-3)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, Y.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=wd)
    w = torch.ones(Y.shape[1], device=dev)
    w[:3] = 4.0
    best = (float('inf'), None, -1)
    for ep in range(args.mlp_epochs):
        net.train()
        for b in torch.randperm(len(X), device=dev).split(256):
            loss = ((((net((X[b].float() - mx) / sx) - (Y[b] - my) / sy)) ** 2) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            yv = torch.cat([net((Xv[i: i + 4096].float() - mx) / sx) for i in range(0, len(Xv), 4096)]) * sy + my
            err = 100 * (yv[:, :3] - Yv[:, :3]).norm(dim=1).mean().item()
        if err < best[0]:
            best = (err, [(m.weight.T.detach().cpu().clone(), m.bias.detach().cpu().clone())
                          for m in net if isinstance(m, torch.nn.Linear)], ep)
        print(f'  MLP epoch {ep + 1}: validation cube error {err:.2f} cm', flush=True)
    print(f'MLP grid {grid}: kept epoch {best[2] + 1} ({best[0]:.2f} cm)', flush=True)
    return dict(mx=mx.cpu(), sx=sx.cpu(), my=my.cpu(), sy=sy.cpu(), layers=best[1], grid=grid, epoch=best[2] + 1)


def cube_cm(score, src, idx):
    out = []
    for i in range(0, len(idx), args.chunk):
        j = idx[i: i + args.chunk]
        y = score.probe(torch.from_numpy(src.z[j].astype(np.float32)))[:, :3].numpy()
        out.append(100 * np.linalg.norm(y - src.m['cube_pos'][j], axis=1))
    return np.concatenate(out)


old = torch.load(Path(args.cache) / 'cost.pt', map_location='cpu', weights_only=False)
probes = {'probe_cube (expert frames only)': GoalScore('probe_cube', old)}
fits = {}
for g in grids:
    name = f'probe_cube_rand, linear, grid {g}'
    fits[name] = fit(g)
    probes[name] = GoalScore('probe_cube_rand', dict(probe=fits[name]))
    print(f'{name}: fit, lambda {fits[name]["lam"]}', flush=True)
if args.mlp == 'on' or (args.mlp == 'auto' and dev.type == 'cuda'):
    name = f'probe_cube_rand, MLP, grid {args.mlp_grid}'
    fits[name] = fit_mlp(args.mlp_grid)
    probes[name] = GoalScore('probe_cube_rand', dict(probe=fits[name]))

far = ev_rand[100 * np.linalg.norm(r.m['ee_pos'][ev_rand] - r.m['cube_pos'][ev_rand], axis=1) > 10]
rank = lambda a: pd.Series(np.asarray(a)).rank().to_numpy()
rho = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
starts = np.unique(rs[ev_rand])
rows = []
for name, s in [('latent MSE', GoalScore('latent'))] + list(probes.items()):
    row = dict(cost=name)
    if name != 'latent MSE':
        row.update(expert=cube_cm(s, c, ev_expert).mean(), off_expert=cube_cm(s, r, ev_rand).mean(),
                   arm_far=cube_cm(s, r, far).mean())
    rhos = []
    with torch.no_grad():
        for t in starts:
            ri = np.where(rs == t)[0]
            ci = t + np.arange(0, 40, 8)
            zc = torch.from_numpy(np.concatenate([c.z[ci], r.z[ri]]).astype(np.float32))
            goal = torch.from_numpy(c.z[t + 40].astype(np.float32))[None].expand_as(zc)
            dist = np.linalg.norm(np.concatenate([c.m['cube_pos'][ci], r.m['cube_pos'][ri]]) - c.m['cube_pos'][t + 40], axis=1)
            if dist.std() > 1e-4:
                rhos.append(rho(s(zc, goal).numpy(), dist))
    row['rho'] = float(np.nanmean(rhos))
    rows.append(row)

best = min(fits, key=lambda n: next(x['off_expert'] for x in rows if x['cost'] == n))
params = dict(probe=fits[best], chosen='probe_cube_rand', robust=True, num_patches=c.z.shape[1], cache=args.cache,
              rand=args.rand, check=rows, kept=best)
torch.save(params, Path(args.cache) / 'cost_rand.pt')

f = lambda v: f'{v:.2f}' if isinstance(v, float) else '—'
md = ['# Refit cube probe: check on held-out episodes (no model, no planner)\n\n',
      f'Off-expert frames from scripts/collect_rand.py ({args.rand}): random, held-random (blocks) and noisy-expert '
      f'actions, 8-40 steps from expert states. Held out: {len(ev_expert)} expert frames, {len(ev_rand)} off-expert '
      f'frames ({len(far)} with the arm > 10 cm from the cube), {len(starts)} starts for the rank correlation.\n\n',
      '| cost | cube error, expert frames (cm) | cube error, off-expert (cm) | off-expert, arm > 10 cm from cube (cm) '
      '| cube rank correlation over planner candidates |\n|---|---|---|---|---|\n']
md += [f"| {x['cost']} | {f(x.get('expert'))} | {f(x.get('off_expert'))} | {f(x.get('arm_far'))} | {f(x['rho'])} |\n"
       for x in rows]
md.append(f'\nKept (lowest off-expert cube error, rule fixed before planning): **{best}** -> '
          f'{Path(args.cache) / "cost_rand.pt"} (`--cost probe_cube_rand`)\n')
out_md = Path(args.out) if args.out else Path(args.cache) / 'cost_rand_check.md'
out_md.parent.mkdir(parents=True, exist_ok=True)
out_md.write_text(''.join(md))
print(''.join(md))
