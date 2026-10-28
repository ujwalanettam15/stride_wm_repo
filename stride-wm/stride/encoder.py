"""Frozen DINOv2 ViT-S/14, used as in DINO-WM: 224 px, ImageNet normalization, patch tokens
(CLS dropped). Optional average pooling of the 16x16 patch grid for CPU-scale runs."""
import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_encoder(name='facebook/dinov2-small'):
    from transformers import AutoModel, Dinov2Config, Dinov2Model
    if name == 'random-init':  # pipeline smoke tests only -- never report results from this
        torch.manual_seed(0)
        return Dinov2Model(Dinov2Config(hidden_size=384, num_hidden_layers=12, num_attention_heads=6,
                                        intermediate_size=1536, patch_size=14, image_size=518)).eval()
    return AutoModel.from_pretrained(name).eval()


@torch.no_grad()
def encode(enc, pixels, pool=1):
    """(B, H, W, 3) uint8 -> (B, (16 // pool) ** 2, 384) float32 patch latents, on CPU."""
    x = torch.as_tensor(np.asarray(pixels)).permute(0, 3, 1, 2).float() / 255
    if x.shape[-1] != 224:
        x = F.interpolate(x, size=(224, 224), mode='bilinear', antialias=True)
    x = ((x - MEAN) / STD).to(next(enc.parameters()).device)
    t = enc(pixel_values=x).last_hidden_state[:, 1:]
    if pool > 1:
        n, p, d = t.shape
        g = int(p ** 0.5)
        t = F.avg_pool2d(t.transpose(1, 2).reshape(n, d, g, g), pool).flatten(2).transpose(1, 2)
    return t.float().cpu()
