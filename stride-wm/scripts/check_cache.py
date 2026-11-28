"""Sanity checks on latent caches before any GPU hours go into training. Runs no model.

1. Copy-last MSE at k=1 on held-out pairs, the number that exposed the MPS encoder bug (corrupted
   latents gave 0.93 at 8x8; correct ones give ~0.26). Fails above --max-copy for the 8x8 cache.
2. With both caches: average-pooling the 16x16 latents 2x2 must reproduce the 8x8 latents of the same
   frames (both come from one DINOv2 forward pass; only float16 rounding differs). This checks the GPU
   16x16 encoding against the 8x8 cache, e.g. one encoded on the laptop's CPU. Fails if they disagree."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.data import Cache

p = argparse.ArgumentParser()
p.add_argument('--cache8', default='data/cache')
p.add_argument('--cache16', default=None)
p.add_argument('--frames', type=int, default=2048)
p.add_argument('--max-copy', type=float, default=0.5, help='fail if 8x8 copy-last MSE (k=1) exceeds this')
p.add_argument('--max-rel-diff', type=float, default=0.01, help='fail if pooled 16x16 differs from 8x8 by more')
args = p.parse_args()
ok = True


def copy_last(c):
    rng = np.random.default_rng(0)
    s = np.sort(rng.choice(c.starts('eval', 1), min(args.frames, len(c.starts('eval', 1))), replace=False))
    return float(((c.z[s].astype(np.float32) - c.z[s + 1].astype(np.float32)) ** 2).mean())


c8 = Cache(args.cache8, mmap=True)
v = copy_last(c8)
print(f'{args.cache8}: {c8.z.shape[0]} frames, {c8.z.shape[1]} patches, copy-last MSE (k=1, held out) {v:.3f}')
if c8.z.shape[1] == 64 and v > args.max_copy:
    print(f'FAIL: above {args.max_copy}; the latents look corrupted (laptop reference 0.26)')
    ok = False

if args.cache16:
    c16 = Cache(args.cache16, mmap=True)
    print(f'{args.cache16}: {c16.z.shape[0]} frames, {c16.z.shape[1]} patches, '
          f'copy-last MSE (k=1, held out) {copy_last(c16):.3f}')
    if c16.z.shape[0] != c8.z.shape[0] or c16.z.shape[1] != 4 * c8.z.shape[1]:
        print('FAIL: the caches do not hold the same frames at 16x16 and 8x8')
        ok = False
    else:
        idx = np.sort(np.random.default_rng(1).choice(c8.z.shape[0], min(args.frames, c8.z.shape[0]), replace=False))
        z16 = torch.from_numpy(c16.z[idx].astype(np.float32))
        n, p16, d = z16.shape
        g = int(round(p16 ** 0.5))
        pooled = F.avg_pool2d(z16.transpose(1, 2).reshape(n, d, g, g), 2).flatten(2).transpose(1, 2)
        z8 = torch.from_numpy(c8.z[idx].astype(np.float32))
        rel = ((pooled - z8).abs().mean() / z8.abs().mean()).item()
        print(f'pooled 16x16 vs 8x8 on {n} frames: mean |diff| / mean |z| = {rel:.2e} (float16 rounding ~1e-3)')
        if rel > args.max_rel_diff:
            print(f'FAIL: above {args.max_rel_diff}; the two encodings disagree (device bug or different frames)')
            ok = False
sys.exit(0 if ok else 1)
