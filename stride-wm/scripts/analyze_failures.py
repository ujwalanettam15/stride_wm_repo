"""Failure / error analysis from outputs that already exist. Runs no model and no simulator.

  results/openloop.csv  -> where models are worse than copy-last, how errors grow per call, seed divergence
  results/plan.jsonl    -> success and progress by start distance, against the random-action null
  data/cache (optional) -> is the planning cost informative? Rank correlation between the latent MSE to a
                           goal frame and the true cube / end-effector distance to that goal, over frames
                           of the held-out episodes. If the cost tracks the arm and not the cube, a
                           planner minimizing it will not move the cube, whatever the stride.

Writes results/failure_analysis.md."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.stats import wilson

p = argparse.ArgumentParser()
p.add_argument('--openloop', default='results/openloop.csv')
p.add_argument('--plan', default='results/plan.jsonl')
p.add_argument('--cache', default=None, help='latent cache dir; enables the cost-validity section')
p.add_argument('--goals', type=int, default=5, help='goal frames per held-out episode (cost validity)')
p.add_argument('--out', default='results/failure_analysis.md')
args = p.parse_args()
PH = ['reach', 'grasp', 'transport', 'place']
md = ['# Failure and error analysis (computed from saved results; nothing re-run)\n']


def label(r):
    return f"{'DINO-WM' if r.arch == 'dinowm' else 'chunk'} k={r.k}"


def md_table(df):
    cols = list(df.columns)
    rows = ['| ' + ' | '.join(map(str, [df.index.name or ''] + cols)) + ' |',
            '|' + '---|' * (len(cols) + 1)]
    for idx, r in df.iterrows():
        rows.append('| ' + ' | '.join([str(idx)] + [str(int(v)) if isinstance(v, float) and v.is_integer() and 'depth' in str(c)
                                                      else f'{v:.2f}' if isinstance(v, float) else str(v)
                                                      for c, v in r.items()]) + ' |')
    return '\n'.join(rows) + '\n'


# ------------------------------------------------------------------ open loop
if Path(args.openloop).exists():
    d = pd.read_csv(args.openloop)
    if 'variant' not in d:
        d['variant'] = ''
    d['variant'] = d['variant'].fillna('')
    d = d[(d.variant == '') & ~d.arch.str.endswith('-shuf')]
    mdl = d[d.arch.isin(['chunk', 'dinowm'])].assign(label=lambda x: x.apply(label, axis=1))
    ref = d[d.model == 'copy_last']

    md.append('\n## 1. Cells where a model is worse than copying the last frame (cube position error, cm)\n')
    md.append('Copy-last never moves the cube, so it is the right answer while the cube is at rest (Reach). '
              'A model that loses here is adding drift to static content.\n\n')
    rows = []
    for depth in (8, 40):
        for ph in PH:
            r = ref[(ref.depth == depth) & (ref.phase == ph)]
            if r.empty:
                continue
            for lab, g in mdl[(mdl.depth == depth) & (mdl.phase == ph)].groupby('label'):
                v, base = g.groupby('seed').cube_cm.mean().mean(), r.cube_cm.mean()
                if v > base:
                    rows.append(dict(depth=depth, phase=ph, model=lab, model_cm=round(v, 2),
                                     copy_last_cm=round(base, 2), excess_cm=round(v - base, 2)))
    md.append(md_table(pd.DataFrame(rows).set_index('depth')) if rows else 'None: every model beats copy-last.\n')

    md.append('\n## 2. Error per recursive call, and seed divergence\n')
    md.append('mse at the first call vs at the deepest shared depth; gap = max - min over seeds of mean latent MSE. '
              'Equal gaps at the first call that widen with depth are compounding, not seed luck.\n\n')
    allp = mdl[mdl.phase == 'all']
    rows = []
    for lab, g in allp.groupby('label'):
        first, last = g.depth.min(), 40
        piv = g.pivot_table(index='depth', columns='seed', values='mse')
        row = dict(first_call_depth=int(first), mse_first=round(piv.loc[first].mean(), 3))
        if last in piv.index:
            row['mse_at_40'] = round(piv.loc[last].mean(), 3)
        if piv.shape[1] > 1:
            row['seed_gap_first'] = round(float(np.ptp(piv.loc[first].to_numpy())), 3)
            if last in piv.index:
                row['seed_gap_40'] = round(float(np.ptp(piv.loc[last].to_numpy())), 3)
        rows.append(pd.Series(row, name=lab))
    md.append(md_table(pd.DataFrame(rows)))

# ------------------------------------------------------------------ planning
if Path(args.plan).exists():
    pl = pd.DataFrame([json.loads(x) for x in open(args.plan)])
    for col, default in (('variant', ''), ('min_dist_cm', np.nan)):
        if col not in pl:
            pl[col] = default
    pl['variant'] = pl['variant'].fillna('')
    pl['model_label'] = pl.apply(lambda r: {'replay': 'expert replay', 'random': 'random actions'}.get(r.arch)
                                 or f"{label(r)}{r.variant}", axis=1)
    pl['progress'] = pl.start_dist_cm - pl.final_dist_cm
    pl['bin'] = pd.cut(pl.start_dist_cm, [0, 8, 15, 1e9], labels=['<8 cm', '8-15 cm', '>=15 cm'])

    md.append('\n## 3. Planning: success and progress by how far the cube starts from the goal\n')
    md.append('Success needs the cube within 4 cm. A start under 8 cm needs a nudge of at most 4 cm, so successes '
              'there are weak evidence of planning. Progress = start distance minus final distance (cm).\n\n')
    rows = []
    for (lab, b), g in pl.groupby(['model_label', 'bin'], observed=True):
        lo, hi = wilson(int(g.success.sum()), len(g))
        rows.append(dict(model=lab, start=str(b), pairs=len(g), success=f'{int(g.success.sum())}/{len(g)}',
                         ci=f'{100 * lo:.0f}-{100 * hi:.0f}%', progress_cm=round(g.progress.mean(), 1)))
    md.append(md_table(pd.DataFrame(rows).set_index('model')))
    n_easy = int(pl[(pl.start_dist_cm < 8) & ~pl.arch.isin(['replay', 'random'])].success.sum())
    n_all = int(pl[~pl.arch.isin(['replay', 'random'])].success.sum())
    md.append(f'\nSuccesses from planners with start < 8 cm: {n_easy} of {n_all}.\n')
    if (pl.arch == 'random').any():
        rnd = pl[pl.arch == 'random']
        md.append(f'Random-action null: {int(rnd.success.sum())}/{len(rnd)} succeed. A planner is only '
                  f'distinguishable from chance when its interval excludes this rate.\n')
    else:
        md.append('No random-action null in this log yet (eval_plan.py --ckpts random); until there is, '
                  'no success rate here can be called better than chance.\n')
    if 'aligned' in pl and not pl.aligned.all():
        md.append('\nNote: some rows used goal offsets different from the horizon (the cost compared frames '
                  'that were not the same time apart).\n')
    elif 'aligned' not in pl:
        md.append('\nNote: this log predates goal/horizon alignment (H = 40 with 25-step goals), so the cost '
                  'compared frames up to 15 steps apart.\n')

# ------------------------------------------------------------------ cost validity
if args.cache:
    from stride.data import Cache

    c = Cache(args.cache, mmap=True)
    m, rng = c.m, np.random.default_rng(0)
    rank = lambda a: pd.Series(a).rank().to_numpy()
    rho = lambda a, b: float(np.corrcoef(rank(a), rank(b))[0, 1])
    out = {k: [] for k in ('cube', 'ee', 'cube_arm_matched')}
    for e in c.episodes('eval'):
        o, n = c.ep_offset[e], c.ep_len[e]
        z = c.z[o: o + n].astype(np.float32).reshape(n, -1)
        cube, ee = m['cube_pos'][o: o + n], m['ee_pos'][o: o + n]
        for g in rng.choice(np.arange(int(0.7 * n), n), args.goals, replace=False):
            cost = ((z - z[g]) ** 2).mean(1)
            dc, de = np.linalg.norm(cube - cube[g], axis=1), np.linalg.norm(ee - ee[g], axis=1)
            out['cube'].append(rho(cost, dc))
            out['ee'].append(rho(cost, de))
            near = de < 0.03  # frames where the arm is already (nearly) at the goal arm pose
            if near.sum() > 10 and dc[near].std() > 0:
                out['cube_arm_matched'].append(rho(cost[near], dc[near]))
    md.append('\n## 4. Is the planning cost informative? (true latents, no model)\n')
    md.append('Spearman rank correlation, per goal frame, between latent MSE to the goal and the true distance to '
              'the goal. 1 = the cost is a perfect proxy. Goals are from the last 30% of held-out episodes.\n\n')
    md.append('| distance | mean rho | goals |\n|---|---|---|\n')
    for key, name in (('cube', 'cube to goal cube'), ('ee', 'end effector to goal end effector'),
                      ('cube_arm_matched', 'cube to goal cube, arm already within 3 cm of goal arm')):
        v = np.array(out[key])
        md.append(f'| {name} | {np.nanmean(v):.2f} | {len(v)} |\n' if len(v) else f'| {name} | — | 0 |\n')
    md.append('\nIf the cube row is clearly below the end-effector row, the cost mostly rewards moving the arm.\n')

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(''.join(x if x.endswith('\n') else x + '\n' for x in md))
print(Path(args.out).read_text())
