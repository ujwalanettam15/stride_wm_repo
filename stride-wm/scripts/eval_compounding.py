"""Does a coarse stride win because it learned better dynamics, or because it chains fewer calls?

Coarser strides have lower open-loop error at a given depth, but they also make fewer recursive calls to
get there, so the two explanations are confounded in E1. This script separates them in two ways.

1. Decomposition (needs no new training). For every model call j of an open-loop rollout
   (stride.compound.decompose):
     total       error of the open-loop prediction
     fresh       error of the same call when it is given the TRUE previous latent (the call's own error)
     propagated  difference between the two predictions (error carried in through the input)
     gain        propagated_j / total_{j-1}: the share of the incoming error that survives one call.
                 gain ** (1/k) is the same quantity per env step, comparable across strides.
   Reading it: if strides have about the same per-step gain, their models carry errors forward equally
   well and coarse strides win only by injecting fresh error less often (accumulation). If the per-step
   gain falls with stride, the coarse models also damp incoming error better (a property of the learned
   dynamics, not of the call count).

2. Equal-span ablation (A1; needs the `-uN` checkpoints from run_a100.sh, read from --openloop). Every
   stride is trained on the same chained span, so each model has been trained on its own outputs over
   the same number of env steps and only the stride differs. If the spread across strides collapses,
   the E1 ordering came from accumulation on models never trained on their own outputs. If it stays,
   stride itself matters.

Writes <out>.csv (one row per model and call) and <out>.md (summary tables)."""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.compound import decompose
from stride.data import Cache
from stride.model import default_device, enable_tf32, load, variant_tag

p = argparse.ArgumentParser()
p.add_argument('--cache', default='data/cache')
p.add_argument('--ckpts', nargs='+', required=True)
p.add_argument('--horizon', type=int, default=40)
p.add_argument('--every', type=int, default=8, help='start-frame spacing on eval episodes')
p.add_argument('--batch', type=int, default=64)
p.add_argument('--openloop', default=None, help='openloop.csv holding the -uN variants (equal-span table)')
p.add_argument('--out', default='results/compounding.csv')
p.add_argument('--tf32', action='store_true')
p.add_argument('--device', default=default_device())
args = p.parse_args()
enable_tf32(args.tf32)
H = args.horizon
c = Cache(args.cache, mmap=True)
starts = c.starts('eval', H)[:: args.every]

rows = []
for path in args.ckpts:
    if not Path(path).exists():  # an unmatched shell glob arrives as a literal pattern
        print(f'skip {path}: no such checkpoint', flush=True)
        continue
    model, ck = load(path, args.device)
    k = model.k
    if ck['model_kw']['num_patches'] != c.z.shape[1] or H % k:
        print(f'skip {path}: trained on another patch grid, or k={k} does not divide H={H}', flush=True)
        continue
    n = H // k
    A = torch.from_numpy((c.m['action'] - ck['act_mean']) / ck['act_std']).float()
    tot, fr, pr = (torch.zeros(n, dtype=torch.float64) for _ in range(3))
    for i in range(0, len(starts), args.batch):
        b = starts[i: i + args.batch]
        idx = b[:, None] + k * np.arange(n + 1)
        Z = torch.from_numpy(c.z[idx.ravel()].astype(np.float32)).reshape(len(b), n + 1, *c.z.shape[1:])
        t, f, q = decompose(model, Z.to(args.device), A[b[:, None] + np.arange(H)].to(args.device))
        t, f, q = (x.sum(0).cpu().double() for x in (t, f, q))  # to CPU first: MPS has no float64
        tot, fr, pr = tot + t, fr + f, pr + q
    tot, fr, pr = (x / len(starts) for x in (tot, fr, pr))
    for j in range(n):
        inc = tot[j - 1].item() if j else 0.0
        rows.append(dict(model=Path(path).stem, k=k, arch=ck['model_kw']['arch'], variant=variant_tag(ck['cfg']),
                         seed=ck['cfg']['seed'], depth=(j + 1) * k, calls=j + 1, n=len(starts),
                         total=tot[j].item(), fresh=fr[j].item(), propagated=pr[j].item(), incoming=inc,
                         gain=pr[j].item() / inc if j else float('nan')))
    print('done', Path(path).stem, flush=True)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
d = pd.DataFrame(rows)
d.to_csv(out, index=False)
d['label'] = d.apply(lambda r: f"{'DINO-WM' if r.arch == 'dinowm' else 'chunk'} k={r.k}{r.variant}", axis=1)


def ms(v, nd=3):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return '—'
    return f'{v.mean():.{nd}f} ± {v.std(ddof=1):.{nd}f}' if len(v) > 1 else f'{v.mean():.{nd}f}'


md = ['# Stride vs error accumulation\n\n',
      f'Open-loop rollouts from {len(starts)} held-out start frames ({args.cache}). ± = s.d. over training seeds.\n',
      '`fresh` = error of the last call when given the true previous latent; `carried in` = share of the total '
      'that is not fresh error; `gain` = share of the incoming error that survives one call, over calls 2..n; '
      '`gain per step` = gain ** (1/k), comparable across strides.\n']
for depth in (8, 40):
    sub = d[d.depth <= depth]
    labels = [lab for lab, g in sub.groupby('label', sort=False) if (g.depth == depth).any()]
    if not labels:
        continue
    md.append(f'\n## Decomposition at depth {depth} env steps\n')
    md.append('| model | calls | total MSE | fresh (last call) | carried in | gain per call | gain per step |\n'
              '|---|---|---|---|---|---|---|\n')
    for lab in labels:
        g = sub[sub.label == lab]
        k = int(g.k.iloc[0])
        per = []
        for _, s in g.groupby('seed'):
            last, later = s[s.depth == depth].iloc[0], s[s.calls > 1]
            gain = later.propagated.sum() / later.incoming.sum() if len(later) else float('nan')
            per.append((last.total, last.fresh, 1 - last.fresh / last.total, gain, gain ** (1 / k)))
        per = np.array(per)
        md.append(f'| {lab} | {depth // k} | {ms(per[:, 0])} | {ms(per[:, 1])} | {ms(100 * per[:, 2], 0)}% | '
                  f'{ms(per[:, 3], 2)} | {ms(per[:, 4], 3)} |\n')

if args.openloop and Path(args.openloop).exists():
    o = pd.read_csv(args.openloop)
    o['variant'] = o['variant'].fillna('') if 'variant' in o else ''
    o = o[(o.phase == 'all') & (o.arch == 'chunk')]
    unr = o[o.variant.map(lambda v: bool(re.fullmatch(r'-u\d+', v)))]
    if len(unr):
        ks = sorted(unr.k.unique())
        span = int(unr.k.iloc[0]) * int(unr.variant.iloc[0][2:])
        base = o[(o.variant == '') & o.k.isin(ks)]
        md.append(f'\n## Equal-span ablation (A1): every stride trained on the same {span}-step chained loss\n')
        md.append('Latent MSE; `default` = single k-step loss, `equal span` = chained loss over the span. '
                  'A model whose unroll would be 1 is its own default.\n\n')
        depths = [x for x in (8, 16, 24, 40) if all((o[o.k == k].depth == x).any() for k in ks)]
        md.append('| stride | ' + ' | '.join(f'default @{x} | equal span @{x}' for x in depths) + ' |\n'
                  + '|---' * (1 + 2 * len(depths)) + '|\n')
        means = {x: ([], []) for x in depths}
        for k in ks:
            cells = []
            for x in depths:
                b = base[(base.k == k) & (base.depth == x)].groupby('seed').mse.mean()
                u = unr[(unr.k == k) & (unr.depth == x)].groupby('seed').mse.mean()
                u = u if len(u) else b
                cells += [ms(b.values), ms(u.values)]
                means[x][0].append(b.mean())
                means[x][1].append(u.mean())
            md.append(f'| k={k} | ' + ' | '.join(cells) + ' |\n')
        md.append('| **spread across strides (max − min)** | '
                  + ' | '.join(f'{np.ptp(means[x][0]):.3f} | {np.ptp(means[x][1]):.3f}' for x in depths) + ' |\n')
        md.append('\nIf the spread shrinks toward zero under equal span, the E1 stride ordering was error '
                  'accumulation; if it stays, stride itself changes what the model learns.\n')
out.with_suffix('.md').write_text(''.join(md))
print(''.join(md))
