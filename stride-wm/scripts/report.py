"""Tables (markdown) + figures from results/openloop.csv (E1, E4) and results/plan.jsonl (E2, E3).
Strides are compared only at depths they share: multiples of 8 for k in {1,2,4,8}, multiples
of 5 for k=5 vs the DINO-WM baseline, multiples of 40 for everything."""
import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('--openloop', default='results/openloop.csv')
p.add_argument('--plan', default='results/plan.jsonl')
p.add_argument('--runs', default='runs')
p.add_argument('--out', default='results')
args = p.parse_args()
out = Path(args.out)
PHASES = ['reach', 'grasp', 'transport', 'place']
RAMP = {1: '#86b6ef', 2: '#5598e7', 4: '#2a78d6', 5: '#1c5cab', 8: '#0d366b'}  # ordinal blue
BASE, REF, INK, MUTED, GRID = '#eb6834', '#898781', '#0b0b0b', '#52514e', '#e1e0d9'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'figure.facecolor': '#fcfcfb',
                     'axes.facecolor': '#fcfcfb', 'legend.frameon': False})


def fmt(v, se=None, nd=3):
    if pd.isna(v):
        return '—'
    return f'{v:.{nd}f}' if se is None or pd.isna(se) else f'{v:.{nd}f} ± {se:.{nd}f}'


def cell(col, nd):
    """One seed: mean ± s.e. over prediction samples. Several seeds: mean ± s.d. over seeds (†)."""
    def f(g):
        if not len(g) or g[col].isna().all():
            return '—'
        if g.seed.nunique() > 1:
            per_seed = g.groupby('seed')[col].mean()
            return f'{per_seed.mean():.{nd}f} ± {per_seed.std():.{nd}f}†'
        return fmt(g[col].mean(), g[col + '_se'].mean(), nd)
    return f


def table(df, index, columns, cell):
    rows = [index + [str(c) for c in columns]]
    rows.append(['---'] * len(rows[0]))
    for key, g in df.groupby(index, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rows.append([str(x) for x in key] + [cell(g[g[columns.name] == c]) for c in columns])
    return '\n'.join('| ' + ' | '.join(r) + ' |' for r in rows) + '\n'


md = ['## Training · final k-step MSE (train = mean of last 500 steps; val = held-out episodes)\n',
      '| model | seed | train | val | copy-last on same val pairs | chunk diversity (RMS NN dist / dim) |\n|---|---|---|---|---|---|\n']
for f in sorted(Path(args.runs).glob('*.pt')):
    ck = torch.load(f, map_location='cpu', weights_only=False)
    last = ck['log'][-1]
    md.append(f"| {f.stem.rsplit('_s', 1)[0]} | {ck['cfg']['seed']} | {last['train']:.4f} | {last['val']:.4f} | "
              f"{last['val_copy']:.4f} | {ck['chunk_nn']:.3f} |\n")
md.append('\n')
d = pd.read_csv(args.openloop)
d['label'] = d.apply(lambda r: f"{'DINO-WM' if r.arch == 'dinowm' else 'chunk'} k={r.k}" if r.k else r.model, axis=1)
prop, base = d[d.arch == 'chunk'], d[d.arch == 'dinowm']
ref = d[d.model == 'copy_last']
order = sorted(prop.k.unique())

# ---- E1 overall: latent MSE vs depth
alld = pd.concat([prop, base, ref])
alld = alld[alld.phase == 'all']
depths = pd.Index([8, 16, 24, 32, 40], name='depth')
md.append('## E1 · open-loop latent MSE by rollout depth (env steps), all phases\n')
md.append('± = s.e. over prediction samples (one seed); † = s.d. over training seeds\n\n')
md.append(table(alld, ['label'], depths, cell('mse', 4)))
md.append('\n## E1 · probe-decoded error at depth 40 (cube / end-effector position, cm)\n')
d40 = alld[alld.depth == 40]
md.append('| model | calls | latent MSE | cos dist | cube cm | cube deg | EE cm | EE deg | grip |\n|---|---|---|---|---|---|---|---|---|\n')
for lab, g in d40.groupby('label', sort=False):
    md.append(f'| {lab} | {g.calls.iloc[0]} | ' + ' | '.join(cell(c, n)(g) for c, n in
              [('mse', 4), ('cos', 4), ('cube_cm', 2), ('cube_deg', 1), ('ee_cm', 2), ('ee_deg', 1), ('grip', 3)]) + ' |\n')
fl = d[(d.model == 'probe_floor') & (d.phase == 'all')]
if len(fl):
    r = fl.iloc[0]
    md.append(f'| probe floor (true latents) | – | – | – | {fmt(r.cube_cm, nd=2)} | {fmt(r.cube_deg, nd=1)} | '
              f'{fmt(r.ee_cm, nd=2)} | {fmt(r.ee_deg, nd=1)} | {fmt(r.grip)} |\n')

# ---- E1 by phase (H3): shared depths 8 (k in 1,2,4,8) and 40 (all)
for depth in [8, 40]:
    sub = pd.concat([prop, ref, d[d.model == 'probe_floor']])
    sub = sub[((sub.depth == depth) | (sub.model == 'probe_floor')) & sub.phase.isin(PHASES)]
    md.append(f'\n## E1 · cube position error (cm) by phase of the predicted frame, depth {depth}\n')
    md.append(table(sub, ['label'], pd.Index(PHASES, name='phase'),
                    cell('cube_cm', 2)))
    md.append(f'\n## E1 · latent MSE by phase, depth {depth}  (frames: '
              + ', '.join(f'{ph} n={int(sub[(sub.phase == ph) & (sub.model == "copy_last")].n.sum())}' for ph in PHASES)
              + ')\n')
    md.append(table(sub[sub.model != 'probe_floor'], ['label'], pd.Index(PHASES, name='phase'),
                    cell('mse', 4)))

# ---- E4: original DINO-WM conditioning vs proposed chunk encoder at k=5
e4 = pd.concat([prop[prop.k == 5], base])
e4 = e4[e4.phase == 'all']
md.append('\n## E4 · DINO-WM conditioning vs chunk encoder, k=5 (latent MSE)\n')
md.append(table(e4, ['label'], pd.Index([5, 10, 20, 40], name='depth'),
                cell('mse', 4)))

# ---- figures
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
def curve(a, df, x, color, label):  # thin line per seed, thick line for the mean
    df = df[df.phase == 'all']
    for _, g in df.groupby('seed'):
        if df.seed.nunique() > 1:
            a.plot(g[x], g.mse, color=color, lw=0.8, alpha=0.45)
    g = df.groupby(x).mse.mean()
    a.plot(g.index, g.values, color=color, lw=2, marker='o', ms=4, label=label)


for k in order:
    for a, x in [(ax[0], 'depth'), (ax[1], 'calls')]:
        curve(a, prop[prop.k == k], x, RAMP.get(k, INK), f'k={k}')
copy40 = ref[(ref.phase == 'all') & (ref.depth == ref.depth.max())].mse.mean()
ax[0].set(xlabel='rollout depth (env steps)', ylabel='latent MSE',
          title=f'E1 · error vs depth (copy-last: {copy40:.2f} at {ref.depth.max()})')
ax[1].set(xlabel='recursive model calls', ylabel='latent MSE', title='E1 · same rollouts vs model calls')
ax[1].set_xscale('log', base=2)
ax[0].legend(fontsize=8)
for lab, g, col in [('chunk k=5 (proposed)', prop[prop.k == 5], RAMP[5]), ('DINO-WM k=5 (original)', base, BASE)]:
    curve(ax[2], g, 'depth', col, lab)
ax[2].set(xlabel='rollout depth (env steps)', ylabel='latent MSE',
          title='E4 · action conditioning at k=5 (thin = per seed)')
ax[2].legend(fontsize=8)
for a in ax:
    a.title.set_color(INK)
fig.savefig(out / 'fig_e1_e4_depth.png', dpi=160)

sub = prop[(prop.depth == 40) & prop.phase.isin(PHASES)]
fig, ax = plt.subplots(figsize=(6.2, 3.2), constrained_layout=True)
w = 0.8 / max(len(order), 1)
for i, k in enumerate(order):
    seeds = sorted(sub[sub.k == k].seed.unique())
    for si, seed in enumerate(seeds):  # one dot per training seed, 95% CI over prediction samples
        g = sub[(sub.k == k) & (sub.seed == seed)].set_index('phase').reindex(PHASES)
        x = [j + (i - (len(order) - 1) / 2) * w + (si - (len(seeds) - 1) / 2) * w / 3 for j in range(len(PHASES))]
        ax.errorbar(x, g.cube_cm, yerr=1.96 * g.cube_cm_se, fmt='o', ms=4, color=RAMP.get(k, INK), lw=1.1,
                    label=f'k={k}' if si == 0 else None)
fl = d[(d.model == 'probe_floor')].set_index('phase').reindex(PHASES)
ax.scatter(range(len(PHASES)), fl.cube_cm, marker='_', s=300, color=REF, label='probe floor', zorder=3)
ax.set_xticks(range(len(PHASES)), PHASES)
ax.set(ylabel='cube position error (cm)', title='E1 · cube error at depth 40 by phase (one dot per seed)')
ax.title.set_color(INK)
ax.grid(axis='x', visible=False)
ax.legend(fontsize=8, ncol=len(order) + 1, loc='upper center', bbox_to_anchor=(0.5, -0.1))
fig.savefig(out / 'fig_e1_phase.png', dpi=160)

# ---- E2/E3 planning
if Path(args.plan).exists():
    pl = pd.DataFrame([json.loads(x) for x in open(args.plan)])
    pl['label'] = pl.apply(lambda r: 'expert replay (pipeline check)' if r.arch == 'replay' else
                           f"{'DINO-WM' if r.arch == 'dinowm' else 'chunk'} k={r.k}", axis=1)
    md.append('\n## E2 · planning success rate by goal offset (env steps)\n')
    g = pl.groupby(['label', 'offset']).success.agg(['mean', 'count']).reset_index()
    offs = sorted(pl.offset.unique())
    md.append('| model | ' + ' | '.join(f'd={o}' for o in offs) + ' |\n' + '|---' * (len(offs) + 1) + '|\n')
    for lab, gg in g.groupby('label', sort=False):
        cells = [f"{100 * r['mean']:.0f}% ({int(r['count'])})" for _, r in gg.set_index('offset').reindex(offs).iterrows()]
        md.append(f'| {lab} | ' + ' | '.join(cells) + ' |\n')
    md.append('\n## E3 · cost per plan (same CEM budget, horizon and replanning for all)\n')
    md.append('| model | model calls / plan | sec / plan |\n|---|---|---|\n')
    for lab, gg in pl[pl.arch != 'replay'].groupby('label', sort=False):
        md.append(f'| {lab} | {int(gg.calls_per_solve.iloc[0])} | {gg.solve_sec.mean():.2f} |\n')

(out / 'summary.md').write_text(''.join(md))
print(''.join(md))
