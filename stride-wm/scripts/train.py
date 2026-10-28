"""Train one stride-k predictor on single k-step transitions: MSE between the prediction and the
frozen latent at t+k, no rollout loss. Every stride gets the same steps, batch and optimizer."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.data import Cache
from stride.model import StrideWM, default_device

p = argparse.ArgumentParser()
p.add_argument('--cache', default='data/cache')
p.add_argument('--k', type=int, required=True)
p.add_argument('--arch', default='chunk', choices=['chunk', 'dinowm'])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--steps', type=int, default=20000)
p.add_argument('--batch', type=int, default=32)
p.add_argument('--lr', type=float, default=5e-4)
# predictor defaults = DINO-WM's (swm scripts/train/config/prejepa.yaml); shrink only for CPU pilots
p.add_argument('--depth', type=int, default=6)
p.add_argument('--heads', type=int, default=16)
p.add_argument('--dim-head', type=int, default=64)
p.add_argument('--mlp-dim', type=int, default=2048)
p.add_argument('--dropout', type=float, default=0.1)
p.add_argument('--emb-dim', type=int, default=64, help='chunk-encoder output size (shared by all k)')
p.add_argument('--eval-every', type=int, default=500)
p.add_argument('--out', default='runs')
p.add_argument('--device', default=default_device())
args = p.parse_args()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
c = Cache(args.cache, mmap=True)
N, P, D = c.z.shape
mu, sd = c.action_stats()
A = torch.from_numpy((c.m['action'] - mu) / sd).float()  # NaN rows are never sampled
k, dev = args.k, args.device
tr = c.starts('train', k)
va = c.starts('eval', k)
va = np.random.default_rng(0).choice(va, min(2048, len(va)), replace=False)  # same rows every run


def batch(idx):
    idx = np.sort(idx)
    z0 = torch.from_numpy(c.z[idx].astype(np.float32))
    z1 = torch.from_numpy(c.z[idx + k].astype(np.float32))
    return z0.to(dev), A[idx[:, None] + np.arange(k)].to(dev), z1.to(dev)


model_kw = dict(k=k, arch=args.arch, num_patches=P, latent_dim=D, action_dim=A.shape[1],
                emb_dim=args.emb_dim, depth=args.depth, heads=args.heads, dim_head=args.dim_head,
                mlp_dim=args.mlp_dim, dropout=args.dropout)
model = StrideWM(**model_kw).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)


@torch.no_grad()
def val_loss():
    model.eval()
    tot = copy = 0.0
    for i in range(0, len(va), 256):
        z0, a, z1 = batch(va[i: i + 256])
        tot += F.mse_loss(model(z0, a), z1, reduction='sum').item()
        copy += F.mse_loss(z0, z1, reduction='sum').item()
    model.train()
    return tot / (len(va) * P * D), copy / (len(va) * P * D)


# chunk diversity: RMS nearest-neighbour distance per action dim among sampled training chunks
ch = A[rng.choice(tr, min(2000, len(tr)), replace=False)[:, None] + np.arange(k)].flatten(1)
dist = torch.cdist(ch, ch).fill_diagonal_(float('inf'))
chunk_nn = (dist.min(1).values / (ch.shape[1] ** 0.5)).mean().item()

name = f'{args.arch}_k{k}_s{args.seed}'
print(name, f'{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params,',
      f'{len(tr)} train pairs, chunk_nn={chunk_nn:.3f}', flush=True)
log, run, n_run, t0 = [], 0.0, 0, time.time()
for step in range(1, args.steps + 1):
    z0, a, z1 = batch(rng.choice(tr, args.batch))
    loss = F.mse_loss(model(z0, a), z1)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    run, n_run = run + loss.item(), n_run + 1
    if step % args.eval_every == 0 or step == args.steps:
        v, cp = val_loss()
        log.append(dict(step=step, train=run / n_run, val=v, val_copy=cp, sec=round(time.time() - t0)))
        print(json.dumps(log[-1]), flush=True)
        run, n_run = 0.0, 0

Path(args.out).mkdir(parents=True, exist_ok=True)
torch.save(dict(model_kw=model_kw, state=model.state_dict(), act_mean=mu, act_std=sd, cfg=vars(args),
                log=log, chunk_nn=chunk_nn), Path(args.out) / f'{name}.pt')
