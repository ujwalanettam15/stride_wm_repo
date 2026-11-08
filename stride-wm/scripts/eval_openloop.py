"""E1 + E4: open-loop rollouts with the logged actions from held-out states.

For every model and every predicted frame: latent MSE, cosine distance (mean over patches), and
probe-decoded physical error, grouped by rollout depth (env steps) and the phase of the predicted
frame. Also writes a copy-last reference (predict z_{t+d} = z_t) and the probe floor (probe on
the true eval latents). Compare strides only at depths they share (see report.py).

Standard errors (`*_se`) treat the held-out EPISODE as the unit of replication (stride.stats), because
rollouts that start a few frames apart in one episode overlap and are not independent.

--shuffle-actions is a control, not a model: each rollout is given another sample's action sequence
(same episode split, fixed permutation). If the error barely moves, the model is not using its
actions, which would explain a null E4 and planners that cannot steer. Rows are tagged
arch='<arch>-shuf' so the main tables ignore them."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride import phases
from stride.data import Cache, Probe, physical_errors, probe_features, probe_targets
from stride.model import default_device, load, variant_tag
from stride.stats import cluster_mean_se

p = argparse.ArgumentParser()
p.add_argument('--cache', default='data/cache')
p.add_argument('--ckpts', nargs='+', required=True)
p.add_argument('--horizon', type=int, default=40)
p.add_argument('--every', type=int, default=2, help='start-frame spacing on eval episodes')
p.add_argument('--batch', type=int, default=256)
p.add_argument('--shuffle-actions', action='store_true', help='action-sensitivity control (see docstring)')
p.add_argument('--probe-grid', type=int, default=4, help='probe sees the latent pooled to at most this grid')
p.add_argument('--out', default='results/openloop.csv')
p.add_argument('--device', default=default_device())
args = p.parse_args()
H = args.horizon
c = Cache(args.cache, mmap=True)
m, ph = c.m, c.m['phase']


def feats(idx):
    return torch.cat([probe_features(c.z[idx[i: i + 4096]], args.probe_grid) for i in range(0, len(idx), 4096)])


# ---- probe: fit on true latents of training episodes (lambda chosen on their last 10% of frames)
tr = c.frames('train')
cut = tr[int(0.9 * len(tr))]
probe = Probe().fit(feats(tr[tr < cut]), torch.from_numpy(probe_targets(m, tr[tr < cut])),
                    feats(tr[tr >= cut]), torch.from_numpy(probe_targets(m, tr[tr >= cut])))
Y = torch.from_numpy(probe_targets(m, np.arange(len(ph))))

stats = {}  # (model, k, arch, variant, seed, depth, phase) -> list of (per-sample metric dict, episode ids)


def add(key, metrics, tgt):
    ep = c.episode_of(tgt)
    for name in ['all'] + [phases.NAMES[i] for i in range(5)]:
        sel = np.ones(len(tgt), bool) if name == 'all' else ph[tgt] == phases.NAMES.index(name)
        if sel.any():
            stats.setdefault(key + (name,), []).append(
                ({n: v[torch.from_numpy(sel)] for n, v in metrics.items()}, ep[sel]))


def score(pred, tgt):
    true = torch.from_numpy(c.z[tgt].astype(np.float32))
    out = dict(mse=((pred - true) ** 2).mean((1, 2)),
               cos=(1 - F.cosine_similarity(pred, true, dim=-1)).mean(1))
    out.update(physical_errors(probe(probe_features(pred, args.probe_grid)), Y[tgt]))
    return out


starts = c.starts('eval', H)[:: args.every]
ev = c.frames('eval')
for i in range(0, len(ev), 4096):  # probe floor: probe applied to the true latents
    idx = ev[i: i + 4096]
    add(('probe_floor', 0, '-', '', 0, 0), physical_errors(probe(feats(idx)), Y[idx]), idx)
for i in range(0, len(starts), args.batch):  # copy-last reference
    b = starts[i: i + args.batch]
    z0 = torch.from_numpy(c.z[b].astype(np.float32))
    for d in range(1, H + 1):
        add(('copy_last', 0, '-', '', 0, d), score(z0, b + d), b + d)

with torch.no_grad():
    for path in args.ckpts:
        model, ck = load(path, args.device)
        k, name = model.k, Path(path).stem
        arch, variant = ck['model_kw']['arch'], variant_tag(ck['cfg'])
        if args.shuffle_actions:
            name, arch = name + '_shuf', arch + '-shuf'
        A = torch.from_numpy((m['action'] - ck['act_mean']) / ck['act_std']).float()
        gen = torch.Generator().manual_seed(0)
        for i in range(0, len(starts), args.batch):
            b = starts[i: i + args.batch]
            z0 = torch.from_numpy(c.z[b].astype(np.float32))
            a = A[b[:, None] + np.arange(H)]
            if args.shuffle_actions:
                a = a[torch.randperm(len(b), generator=gen)]
            preds = model.rollout(z0.to(args.device), a.to(args.device)).cpu()
            for j in range(H // k):
                d = (j + 1) * k
                add((name, k, arch, variant, ck['cfg']['seed'], d), score(preds[:, j], b + d), b + d)
        print('done', name, flush=True)

rows = []
for (name, k, arch, variant, seed, d, phase), parts in stats.items():
    row = dict(model=name, k=k, arch=arch, variant=variant, seed=seed, depth=d,
               calls=d // k if k else 0, phase=phase)
    ep = np.concatenate([q[1] for q in parts])
    for metric in parts[0][0]:
        v = torch.cat([q[0][metric] for q in parts]).double().numpy()
        row[metric], row[metric + '_se'], row['n_eps'] = cluster_mean_se(v, ep)
        row['n'] = len(v)
    rows.append(row)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(args.out, index=False)
print('probe lambda', probe.lam, '->', args.out)
