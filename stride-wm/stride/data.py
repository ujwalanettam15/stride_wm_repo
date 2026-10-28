"""Latent cache (written by scripts/encode.py) and the frozen linear probe.

cache/z.npy        (N, P, 384) float16  frozen DINOv2 patch latents, one row per env step
cache/meta.npz     action (N, 5) [row t = action taken at t; NaN on each episode's last row],
                   ep_offset, ep_len, n_eval (last n_eval episodes are held out), phase (N,),
                   physical state: cube_pos, cube_yaw, ee_pos, ee_yaw, grip, contact, grip_vel,
                   target_pos, qpos, qvel
cache/phase_thresholds.json
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class Cache:
    def __init__(self, root, mmap=False):
        root = Path(root)
        self.z = np.load(root / 'z.npy', mmap_mode='r' if mmap else None)
        meta = np.load(root / 'meta.npz')
        self.m = {k: meta[k] for k in meta.files}
        self.ep_offset, self.ep_len = self.m['ep_offset'], self.m['ep_len']
        self.n_eval = int(self.m['n_eval'])
        self.thresholds = json.loads((root / 'phase_thresholds.json').read_text())

    def episodes(self, split):
        cut = len(self.ep_len) - self.n_eval
        return range(cut) if split == 'train' else range(cut, len(self.ep_len))

    def starts(self, split, span):
        """Global frame indices t whose frame t+span is in the same episode."""
        return np.concatenate([self.ep_offset[e] + np.arange(max(self.ep_len[e] - span, 0))
                               for e in self.episodes(split)])

    def frames(self, split):
        return self.starts(split, 0)

    def action_stats(self):
        a = self.m['action'][self.frames('train')]
        a = a[~np.isnan(a).any(1)]
        return a.mean(0).astype(np.float32), a.std(0).astype(np.float32)


# ---- frozen linear probe: latent -> cube pose, end-effector pose, gripper ----------------
def probe_targets(m, idx):
    cy, ey = m['cube_yaw'][idx], m['ee_yaw'][idx]
    return np.concatenate([m['cube_pos'][idx], np.sin(4 * cy)[:, None], np.cos(4 * cy)[:, None],
                           m['ee_pos'][idx], np.sin(2 * ey)[:, None], np.cos(2 * ey)[:, None],
                           m['grip'][idx][:, None]], 1).astype(np.float32)


def probe_features(z):
    """(N, P, D) -> (N, 16 * D): average-pool the patch grid to 4x4 (a linear map)."""
    z = torch.as_tensor(z, dtype=torch.float32)
    n, p, d = z.shape
    g = int(round(p ** 0.5))
    if g > 4:
        z = F.adaptive_avg_pool2d(z.transpose(1, 2).reshape(n, d, g, g), 4).flatten(2).transpose(1, 2)
    return z.reshape(n, -1)


class Probe:
    """Ridge regression; lambda picked on held-out training episodes, then frozen."""

    def fit(self, X, Y, X_val, Y_val):
        self.mx, self.my = X.mean(0), Y.mean(0)
        Xc, Yc = X - self.mx, Y - self.my
        G, XY = Xc.T @ Xc, Xc.T @ Yc
        scale = G.diagonal().mean()
        best = None
        for lam in [1e-3, 1e-2, 1e-1, 1.0]:
            W = torch.linalg.solve(G + lam * scale * torch.eye(len(G)), XY)
            err = (((X_val - self.mx) @ W + self.my - Y_val) ** 2).mean().item()
            if best is None or err < best[0]:
                best = (err, W, lam)
        _, self.W, self.lam = best
        return self

    def __call__(self, X):
        return (X - self.mx) @ self.W + self.my


def physical_errors(Yh, Y):
    """Per-sample errors: positions in cm, yaw in degrees (cube mod 90, gripper mod 180)."""
    def ang(i, sym):
        a = torch.atan2(Yh[:, i], Yh[:, i + 1]) - torch.atan2(Y[:, i], Y[:, i + 1])
        return torch.rad2deg(torch.atan2(torch.sin(a), torch.cos(a)).abs() / sym)
    return dict(cube_cm=100 * (Yh[:, :3] - Y[:, :3]).norm(dim=1), cube_deg=ang(3, 4),
                ee_cm=100 * (Yh[:, 5:8] - Y[:, 5:8]).norm(dim=1), ee_deg=ang(8, 2),
                grip=(Yh[:, 10] - Y[:, 10]).abs())
