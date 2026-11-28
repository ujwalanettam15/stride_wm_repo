"""Port check: seed-0 models trained on this machine at the laptop settings (8x8, 20k steps) against the
same models from the Apple-silicon run (results_mac/summary.md). Runs no model; reads checkpoint logs.

val_copy (copy-last loss on the same validation pairs) depends only on the data. If it matches, the data
are the same and the validation losses should match within seed-level noise; a large gap then means
training behaves differently here (the MPS encoder bug was this kind of silent difference), and the
script fails so no GPU hours go into the 16x16 run. If val_copy differs, the data differ (for example
episodes re-collected on this machine), the losses are not directly comparable, and it only warns."""
import argparse
import sys
from pathlib import Path

import torch

# results_mac/summary.md, seed 0: (val, val_copy) after 20k steps
LAPTOP = {'chunk_k1': (0.1143, 0.2585), 'chunk_k2': (0.1231, 0.4303), 'chunk_k4': (0.1333, 0.7269),
          'chunk_k5': (0.1359, 0.8451), 'chunk_k8': (0.1449, 1.0866), 'dinowm_k5': (0.1377, 0.8451)}

p = argparse.ArgumentParser()
p.add_argument('--runs', default='runs8')
p.add_argument('--max-val-gap', type=float, default=0.10, help='fail above this relative val gap (same data)')
args = p.parse_args()
fail = False
print('| model | val here | val laptop | gap | val_copy here | val_copy laptop | same data |')
print('|---|---|---|---|---|---|---|')
for name, (val_ref, copy_ref) in LAPTOP.items():
    f = Path(args.runs) / f'{name}_s0.pt'
    if not f.exists():
        print(f'| {name} | missing | {val_ref:.4f} | | | {copy_ref:.4f} | |')
        continue
    last = torch.load(f, map_location='cpu', weights_only=False)['log'][-1]
    gap = last['val'] / val_ref - 1
    same = abs(last['val_copy'] / copy_ref - 1) < 0.02
    print(f"| {name} | {last['val']:.4f} | {val_ref:.4f} | {100 * gap:+.1f}% | {last['val_copy']:.4f} | "
          f"{copy_ref:.4f} | {'yes' if same else 'no'} |")
    if same and abs(gap) > args.max_val_gap:
        fail = True
if fail:
    print(f'FAIL: same data but validation loss differs by more than {100 * args.max_val_gap:.0f}% for a model')
sys.exit(1 if fail else 0)
