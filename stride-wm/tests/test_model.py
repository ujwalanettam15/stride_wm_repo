"""Action-conditioning checks (proposal sec. 7): a k-step prediction must depend on every action
in its chunk, so shuffling them has to change the output, for both conditioning types."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.model import StrideWM

SMALL = dict(num_patches=16, latent_dim=32, action_dim=5, emb_dim=16, depth=1, heads=2,
             dim_head=8, mlp_dim=32, dropout=0.0)


@pytest.mark.parametrize('arch', ['chunk', 'dinowm'])
@pytest.mark.parametrize('k', [2, 4, 5, 8])
def test_shuffling_actions_changes_prediction(arch, k):
    torch.manual_seed(0)
    m = StrideWM(k, arch, **SMALL).eval()
    z, a = torch.randn(4, 16, 32), torch.randn(4, k, 5)
    shuffled = a[:, torch.roll(torch.arange(k), 1)]
    assert not torch.allclose(m(z, a), m(z, shuffled), atol=1e-5)
    last_only = a.clone()
    last_only[:, -1] += 1.0  # the last action of the chunk must matter too
    assert not torch.allclose(m(z, a), m(z, last_only), atol=1e-5)


@pytest.mark.parametrize('k', [1, 2, 4, 5, 8])
def test_rollout_is_repeated_k_step_prediction(k):
    torch.manual_seed(0)
    m = StrideWM(k, 'chunk', **SMALL).eval()
    z, a = torch.randn(3, 16, 32), torch.randn(3, 40, 5)
    out = m.rollout(z, a)
    assert out.shape == (3, 40 // k, 16, 32)
    ref = z
    for j in range(40 // k):
        ref = m(ref, a[:, j * k:(j + 1) * k])
    assert torch.allclose(out[:, -1], ref, atol=1e-5)
