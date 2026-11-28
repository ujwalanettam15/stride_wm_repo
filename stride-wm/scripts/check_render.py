"""Do frames rendered on this machine match the frames in the dataset? Runs no world model.

Planning resets the simulator to a dataset state, renders and encodes the current frame here, and compares
the prediction with a goal latent taken from the cache (rendered wherever the dataset was collected). If
this machine renders differently (another MuJoCo version, EGL instead of macOS OpenGL), every current frame
is slightly off the training distribution and the planning cost compares mismatched images.

For held-out frames: reset to the stored simulator state, render, encode exactly as the cache was encoded,
and compare with the cached latent. Reported as a fraction of the copy-last MSE at k=1 (how much the latent
changes in one env step), so 0 = identical and 1 = as different as two consecutive frames.

Reference: 0.09 on the M1 Pro that collected the dataset. Even there, resetting to a stored state does not
reproduce the arm exactly (faint differences along the arm and gripper edges, 0.23/255 per pixel; frame
alignment is right: the stored frame at the same step matches ~10x better than its neighbours), so every
planning run, including the laptop's, starts from frames ~9% of a step away from the dataset's. What matters
on another machine is whether it is clearly worse than that. Warns above --warn; never fails, because only
planning (not training or open-loop evaluation) depends on it."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.env import make_env  # first: sets MUJOCO_GL=egl on Linux

from stride.data import Cache
from stride.encoder import encode, load_encoder
from stride.model import default_device

p = argparse.ArgumentParser()
p.add_argument('--cache', default='data/cache')
p.add_argument('--frames', type=int, default=32)
p.add_argument('--warn', type=float, default=0.20, help='warn above this ratio (laptop reference: 0.09)')
p.add_argument('--fast-render', action='store_true', help='only if the dataset was collected with it')
p.add_argument('--device', default=default_device())
args = p.parse_args()

c = Cache(args.cache, mmap=True)
m = c.m
cfg = json.loads((Path(args.cache) / 'encoder.json').read_text())
enc = load_encoder(cfg['encoder']).to(args.device)
env = make_env(args.fast_render)
idx = np.sort(np.random.default_rng(0).choice(c.starts('eval', 1), args.frames, replace=False))

diff = []
for t in idx:
    env.reset(options={'state': np.r_[m['qpos'][t], m['qvel'][t]]})
    z = encode(enc, env.unwrapped.render()[None], cfg['pool'])[0].numpy()
    diff.append(float(((z - c.z[t].astype(np.float32)) ** 2).mean()))
copy = float(((c.z[idx + 1].astype(np.float32) - c.z[idx].astype(np.float32)) ** 2).mean())
ratio = float(np.mean(diff)) / copy
print(f'render vs cache on {len(idx)} held-out frames: MSE {np.mean(diff):.4f} (max {np.max(diff):.4f}); '
      f'one-step change (copy-last) {copy:.4f}; ratio {ratio:.3f} (laptop reference 0.09)')
if ratio > args.warn:
    print(f'WARNING: frames rendered here differ from the dataset by {ratio:.0%} of a one-step change, clearly more '
          f'than on the laptop that collected it (9%). Planning starts every episode further off the training '
          f'distribution than the laptop run did; report this next to plan8 results. Training and open-loop '
          f'results are unaffected.')
else:
    print('ok: rendering here matches the dataset about as well as on the laptop that collected it')
