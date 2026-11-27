"""Frames OFF the expert's path, with their true cube and arm state, for refitting the cube probe (third A100 run).

Why: with perfect predictions (the simulator as the model, results_a100_2) the planner solved 7/10 pairs with the true
cube distance but 2/10 with the probe_cube cost. That probe was fit on expert frames only, where the cube rides in the
gripper during transport, so cube and arm positions move together and a linear probe can read the arm to place the
cube. The planner's candidates break that link (the arm moves, the cube stays) and can lower the cost by moving the arm
alone. This script records frames from such actions; scripts/fit_probe_rand.py refits the probe with them.

For each sampled start frame (from training episodes for fitting, held-out episodes for the check): reset the
simulator to that frame's state, run 40 steps of one action kind, render after every 8 steps (the depths CEM scores at
k = 1, 2, 4, 8) and encode the frames like the cache. Action kinds, in turn:
  random   i.i.d. actions from the training action distribution: what CEM samples in its first iteration
  blocks   one such random action held for 8 steps, then the next: wider arm sweeps that leave the cube behind
  noisy    the episode's own expert actions plus noise (0.5 in normalized units): near-expert frames in which the
           cube is often dropped or nudged
Writes <out>/z.npy (N, P, D) float16 and <out>/meta.npz: the probe-target columns (cube_pos, cube_yaw, ee_pos,
ee_yaw, grip, as in the cache, so stride.data.probe_targets reads it), start (cache frame index), episode, depth,
kind (0 random, 1 blocks, 2 noisy), split (0 train, 1 eval). meta.npz is written last: if it exists, the data is
complete and a rerun does nothing."""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.env import make_env  # first: sets MUJOCO_GL=egl

KINDS = ('random', 'blocks', 'noisy')
DEPTHS = (8, 16, 24, 32, 40)
SPAN = DEPTHS[-1]
NOISE = 0.5  # noisy kind: standard deviation in normalized action units
BLOCK = 8    # blocks kind: steps each random action is held
COLS = dict(cube_pos='privileged/block_0_pos', cube_yaw='privileged/block_0_yaw', ee_pos='proprio/effector_pos',
            ee_yaw='proprio/effector_yaw', grip='proprio/gripper_opening')
G = {}  # per-process state


def init(cache):
    import torch
    torch.set_num_threads(1)
    from stride.data import Cache
    c = Cache(cache, mmap=True)
    G['m'], G['env'] = c.m, make_env()
    G['mu'], G['sd'] = c.action_stats()


def rollout(task):
    """One start: (index, frames (len(DEPTHS), H, W, 3) uint8, {column: values per depth})."""
    i, t, kind, seed = task
    m, env, mu, sd = G['m'], G['env'], G['mu'], G['sd']
    rng = np.random.default_rng(seed)
    if kind == 0:
        a_norm = rng.standard_normal((SPAN, len(mu)))
    elif kind == 1:
        a_norm = np.repeat(rng.standard_normal((SPAN // BLOCK, len(mu))), BLOCK, axis=0)
    else:
        a_norm = (m['action'][t: t + SPAN] - mu) / sd + NOISE * rng.standard_normal((SPAN, len(mu)))
    acts = np.clip(a_norm * sd + mu, -1, 1)
    env.reset(options={'state': np.r_[m['qpos'][t], m['qvel'][t]]})
    frames, labels = [], {k: [] for k in COLS}
    for step in range(1, SPAN + 1):
        info = env.step(acts[step - 1])[-1]
        if step in DEPTHS:
            frames.append(env.unwrapped.render().copy())
            for k, col in COLS.items():  # copy: the env reuses its info arrays between steps
                v = np.array(info[col], dtype=np.float64).reshape(-1)
                labels[k].append(v if len(v) > 1 else v[0])
    return i, np.stack(frames), {k: np.array(v) for k, v in labels.items()}


def main():
    import torch
    from stride.data import Cache
    from stride.encoder import encode, load_encoder
    from stride.model import default_device

    p = argparse.ArgumentParser()
    p.add_argument('--cache', default='data/cache')
    p.add_argument('--out', default='data/cache_rand')
    p.add_argument('--starts', type=int, default=3000, help='start frames from training episodes (fit)')
    p.add_argument('--eval-starts', type=int, default=600, help='start frames from held-out episodes (check)')
    p.add_argument('--workers', type=int, default=None, help='simulator processes; default = available CPUs - 2')
    p.add_argument('--batch', type=int, default=256, help='frames per DINOv2 batch')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=default_device())
    args = p.parse_args()
    out = Path(args.out)
    if (out / 'meta.npz').exists():
        print(f'{out}/meta.npz exists: off-expert frames already collected', flush=True)
        return

    c = Cache(args.cache, mmap=True)
    rng = np.random.default_rng(args.seed)
    tasks, split = [], []
    for s, (name, n) in enumerate((('train', args.starts), ('eval', args.eval_starts))):
        t0 = c.starts(name, SPAN)  # the noisy kind replays the next SPAN expert actions of the same episode
        t0 = np.sort(rng.choice(t0, min(n, len(t0)), replace=False))
        tasks += [(len(tasks) + j, int(t), (len(tasks) + j) % len(KINDS), int(rng.integers(2 ** 31)))
                  for j, t in enumerate(t0)]
        split += [s] * len(t0)
    n_frames = len(tasks) * len(DEPTHS)
    enc_cfg = json.loads((Path(args.cache) / 'encoder.json').read_text())
    enc = load_encoder(enc_cfg['encoder']).to(args.device)
    n_cpu = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    workers = max(1, min(args.workers or n_cpu - 2, len(tasks)))
    print(f'{len(tasks)} starts ({split.count(0)} train, {split.count(1)} held out) x {len(DEPTHS)} depths = '
          f'{n_frames} frames | {workers} simulator processes | encoder on {args.device}', flush=True)

    out.mkdir(parents=True, exist_ok=True)
    z = None
    labels = {k: [None] * len(tasks) for k in COLS}
    buf, buf_at, done, t_start = [], 0, 0, time.time()

    def flush():
        nonlocal z, buf, buf_at
        if not buf:
            return
        lat = torch.cat([encode(enc, np.stack(buf[i: i + args.batch]), enc_cfg['pool'])
                         for i in range(0, len(buf), args.batch)]).half().numpy()
        if z is None:
            z = np.lib.format.open_memmap(out / 'z.npy', 'w+', np.float16, (n_frames, *lat.shape[1:]))
        z[buf_at: buf_at + len(lat)] = lat
        buf_at, buf = buf_at + len(lat), []

    if workers == 1:
        init(args.cache)
        results, pool = map(rollout, tasks), None
    else:
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[v] = '1'
        # a fresh process every 50 starts: the env recompiles its MuJoCo model on every reset
        pool = mp.get_context('spawn').Pool(workers, initializer=init, initargs=(args.cache,), maxtasksperchild=50)
        results = pool.imap(rollout, tasks, chunksize=4)  # in order, so frames land in task order
    for i, frames, lab in results:
        assert i == done, 'results out of order'
        buf += list(frames)
        for k in COLS:
            labels[k][i] = lab[k]
        done += 1
        if len(buf) >= args.batch:
            flush()
        if done % 200 == 0 or done == len(tasks):
            el = time.time() - t_start
            print(f'{done}/{len(tasks)} starts | {el / 60:.1f} min | ~{el / done * (len(tasks) - done) / 60:.1f} min left',
                  flush=True)
    flush()
    if pool is not None:
        pool.close()
        pool.join()
    z.flush()

    per = len(DEPTHS)
    meta = {k: np.concatenate(v) for k, v in labels.items()}
    meta.update(start=np.repeat([t for _, t, _, _ in tasks], per), episode=np.repeat(c.episode_of(np.array(
        [t for _, t, _, _ in tasks])), per), depth=np.tile(DEPTHS, len(tasks)),
        kind=np.repeat([k for _, _, k, _ in tasks], per), split=np.repeat(split, per),
        n_starts=np.array(len(tasks)))
    np.savez(out / 'meta.npz', **meta)  # last: marks the data as complete
    arm_cube = 100 * np.linalg.norm(meta['ee_pos'] - meta['cube_pos'], axis=1)
    moved = 100 * np.linalg.norm(meta['cube_pos'] - c.m['cube_pos'][meta['start']], axis=1)
    for k, name in enumerate(KINDS):
        s = meta['kind'] == k
        print(f'{name:7} {s.sum():6d} frames | arm-cube distance median {np.median(arm_cube[s]):.1f} cm | '
              f'cube moved > 2 cm in {np.mean(moved[s] > 2):.0%}', flush=True)
    print(f'wrote {out}/z.npy and {out}/meta.npz in {(time.time() - t_start) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
