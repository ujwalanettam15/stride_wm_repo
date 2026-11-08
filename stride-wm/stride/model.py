"""Stride-k latent world models on frozen DINOv2 patch tokens.

The predictor is DINO-WM's, imported unchanged from stable-worldmodel. The only thing that
varies is how the k actions a_t..a_{t+k-1} become the embedding concatenated to every patch:
  arch='chunk'  (proposed): 2-layer transformer with learned positions -> one embedding
  arch='dinowm' (original): actions flattened and linearly embedded to 10 dims
                            (swm Embedder with in_chans = k * action_dim, as DINO-WM does
                            for frameskip k)
Optional, off by default (the headline comparison uses neither):
  residual=True   predict z_{t+k} = z_t + predictor(...). Same network and parameters; the output is
                  a change to the current latent, so static scene content is carried through by
                  construction. Diagnosed from the prelim: copy-last beats every model on cube
                  position in the Reach phase (cube at rest), i.e. models add drift to a static cube.
  dinowm_emb=N    width of the DINO-WM action embedding (10 in DINO-WM). The chunk encoder emits 64,
                  so E4 as first run also changes the embedding width; dinowm_emb=64 removes that.
"""
import torch
from stable_worldmodel.wm.prejepa.module import CausalPredictor, Embedder
from torch import nn


class ChunkEncoder(nn.Module):
    def __init__(self, action_dim, emb_dim=64, max_k=8, width=64, depth=2, heads=4):
        super().__init__()
        self.emb_dim = emb_dim
        self.inp = nn.Linear(action_dim, width)
        self.pos = nn.Parameter(0.02 * torch.randn(1, max_k, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Linear(width, emb_dim)

    def forward(self, a):  # (B, k, A) -> (B, emb_dim)
        x = self.inp(a) + self.pos[:, : a.shape[1]]
        return self.out(self.tf(x).mean(1))


class StrideWM(nn.Module):
    def __init__(self, k, arch='chunk', num_patches=256, latent_dim=384, action_dim=5,
                 emb_dim=64, depth=6, heads=16, dim_head=64, mlp_dim=2048, dropout=0.1,
                 residual=False, dinowm_emb=10):
        super().__init__()
        self.k, self.arch, self.action_dim, self.residual = k, arch, action_dim, residual
        if arch == 'chunk':
            self.act_enc = ChunkEncoder(action_dim, emb_dim, max_k=max(k, 8))
        elif arch == 'dinowm':
            self.act_enc = Embedder(in_chans=k * action_dim, emb_dim=dinowm_emb)
        else:
            raise ValueError(arch)
        self.predictor = CausalPredictor(
            num_patches=num_patches, num_frames=1, dim=latent_dim + self.act_enc.emb_dim,
            depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)

    def embed_actions(self, a):  # (B, k, A) -> (B, e)
        if self.arch == 'chunk':
            return self.act_enc(a)
        return self.act_enc(a.flatten(1)[:, None])[:, 0]

    def forward(self, z, a):
        """z: (B, P, D) latent at t; a: (B, k, A) normalized actions -> predicted latent at t+k."""
        c = self.embed_actions(a)[:, None].expand(-1, z.shape[1], -1)
        out = self.predictor(torch.cat([z, c], -1))[..., : z.shape[-1]]
        return z + out if self.residual else out

    def rollout(self, z, actions):
        """Open loop. actions: (B, H, A) with H % k == 0 -> latents at t+k, ..., t+H: (B, H//k, P, D)."""
        B, H, A = actions.shape
        out = []
        for a in actions.reshape(B, H // self.k, self.k, A).unbind(1):
            z = self(z, a)
            out.append(z)
        return torch.stack(out, 1)


def variant_tag(cfg):
    """Suffix naming a non-default variant ('' for the headline models), from a checkpoint's cfg.
    Used in checkpoint names and the `variant` column of results/openloop.csv, so ablation runs never
    mix into the main tables."""
    tag = '-res' if cfg.get('residual') else ''
    if cfg.get('unroll', 1) > 1:
        tag += f"-u{cfg['unroll']}"
    if cfg['arch'] == 'dinowm' and cfg.get('dinowm_emb', 10) != 10:
        tag += f"-e{cfg['dinowm_emb']}"
    if cfg['arch'] == 'chunk' and cfg.get('emb_dim', 64) != 64:
        tag += f"-emb{cfg['emb_dim']}"
    return tag


def default_device():
    """CUDA if present, else the Apple-silicon GPU (MPS), else CPU."""
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


def load(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = StrideWM(**ck['model_kw']).to(device)
    model.load_state_dict(ck['state'])
    return model.eval().requires_grad_(False), ck
