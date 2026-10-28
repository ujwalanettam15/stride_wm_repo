"""Encode every frame with the frozen DINOv2 ViT-S/14 and cache patch latents + physical state +
phase labels. Phase thresholds are fit here on the training episodes, before any model exists."""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride import phases
from stride.encoder import encode, load_encoder
from stride.model import default_device

# cache key -> HDF5 column (the swm writer keeps '/'; some swm datasets use '_' instead)
COLS = dict(cube_pos='privileged/block_0_pos', cube_yaw='privileged/block_0_yaw',
            ee_pos='proprio/effector_pos', ee_yaw='proprio/effector_yaw',
            grip='proprio/gripper_opening', contact='proprio/gripper_contact',
            grip_vel='proprio/gripper_vel', target_pos='privileged/target_block_pos',
            qpos='qpos', qvel='qvel')


def col(f, name):
    x = f[name][:] if name in f else f[name.replace('/', '_')][:]
    return x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x


p = argparse.ArgumentParser()
p.add_argument('--h5', default='data/cube_single.h5')
p.add_argument('--out', default='data/cache')
p.add_argument('--encoder', default='facebook/dinov2-small', help='HF id or local dir')
p.add_argument('--pool', type=int, default=1, help='avg-pool the 16x16 patch grid (4 -> 4x4)')
p.add_argument('--n-eval', type=int, default=20, help='last N episodes are held out')
p.add_argument('--max-episodes', type=int)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--device', default=default_device())
args = p.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

f = h5py.File(args.h5, 'r')
ep_len, ep_off = f['ep_len'][: args.max_episodes], f['ep_offset'][: args.max_episodes]
rows = np.concatenate([o + np.arange(n) for o, n in zip(ep_off, ep_len)])
assert (np.diff(rows) == 1).all(), 'expects contiguous episodes'
meta = {k: col(f, c)[rows] for k, c in COLS.items()}
meta.update(action=f['action'][:][rows].astype(np.float32), ep_len=ep_len,
            ep_offset=np.r_[0, np.cumsum(ep_len)[:-1]], n_eval=np.array(args.n_eval))

last = np.zeros(len(rows), bool)
last[meta['ep_offset'] + ep_len - 1] = True
train = np.arange(meta['ep_offset'][len(ep_len) - args.n_eval])
th = phases.fit_thresholds({k: meta[k][train] for k in COLS})
meta['phase'] = phases.label(meta, th, last)
(out / 'phase_thresholds.json').write_text(json.dumps(th, indent=1))
np.savez(out / 'meta.npz', **meta)
print('thresholds', th, '\nphase counts',
      dict(zip(phases.NAMES, np.bincount(meta['phase'], minlength=5).tolist())), flush=True)

enc, z = load_encoder(args.encoder).to(args.device), None
for i in range(0, len(rows), args.batch):
    r = rows[i: i + args.batch]
    lat = encode(enc, f['pixels'][r[0]: r[-1] + 1], args.pool).half().numpy()
    if z is None:
        z = np.lib.format.open_memmap(out / 'z.npy', 'w+', np.float16, (len(rows), *lat.shape[1:]))
    z[i: i + len(r)] = lat
    if i // args.batch % 50 == 0:
        print(f'encoded {i + len(r)}/{len(rows)}', flush=True)
z.flush()
(out / 'encoder.json').write_text(json.dumps(dict(encoder=args.encoder, pool=args.pool)))
