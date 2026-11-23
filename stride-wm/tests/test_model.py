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


@pytest.mark.parametrize('arch', ['chunk', 'dinowm'])
def test_residual_adds_the_input_latent(arch):
    """residual=True is the same network with z_t added to the output (same seed -> same weights)."""
    z, a = torch.randn(3, 16, 32), torch.randn(3, 4, 5)
    torch.manual_seed(0)
    plain = StrideWM(4, arch, **SMALL).eval()
    torch.manual_seed(0)
    res = StrideWM(4, arch, residual=True, **SMALL).eval()
    assert torch.allclose(res(z, a), plain(z, a) + z, atol=1e-5)


def test_residual_rollout_is_repeated_k_step_prediction():
    torch.manual_seed(0)
    m = StrideWM(4, 'chunk', residual=True, **SMALL).eval()
    z, a = torch.randn(2, 16, 32), torch.randn(2, 16, 5)
    ref = z
    for j in range(4):
        ref = m(ref, a[:, j * 4:(j + 1) * 4])
    assert torch.allclose(m.rollout(z, a)[:, -1], ref, atol=1e-5)


def test_dinowm_embedding_width_is_configurable():
    m = StrideWM(5, 'dinowm', dinowm_emb=64, **SMALL)
    assert m.embed_actions(torch.randn(2, 5, 5)).shape == (2, 64)
    assert StrideWM(5, 'dinowm', **SMALL).embed_actions(torch.randn(2, 5, 5)).shape == (2, 10)


def test_variant_tag_is_empty_for_headline_models():
    from stride.model import variant_tag
    assert variant_tag(dict(arch='chunk', emb_dim=64, unroll=1, residual=False)) == ''
    assert variant_tag(dict(arch='dinowm', dinowm_emb=10)) == ''
    assert variant_tag(dict(arch='chunk', residual=True, unroll=5)) == '-res-u5'
    assert variant_tag(dict(arch='dinowm', dinowm_emb=64)) == '-e64'


@pytest.mark.parametrize('k', [1, 4])
def test_error_decomposition_separates_fresh_and_carried_in_error(k):
    from stride.compound import decompose
    torch.manual_seed(0)
    m = StrideWM(k, 'chunk', **SMALL).eval()
    n = 3
    Z, a = torch.randn(2, n + 1, 16, 32), torch.randn(2, n * k, 5)
    total, fresh, prop = decompose(m, Z, a, chunk=4)
    assert total.shape == fresh.shape == prop.shape == (2, n)
    assert torch.allclose(prop[:, 0], torch.zeros(2), atol=1e-10)   # first call starts from the true latent
    assert torch.allclose(total[:, 0], fresh[:, 0], atol=1e-6)
    assert (prop[:, 1:] > 0).all()                                    # later calls start from predictions
    zs = m.rollout(Z[:, 0], a)
    assert torch.allclose(total, ((zs - Z[:, 1:]) ** 2).mean((2, 3)), atol=1e-6)
    one = m(Z[:, 1], a[:, k: 2 * k])                                  # call 2 given the true latent at depth k
    assert torch.allclose(fresh[:, 1], ((one - Z[:, 2]) ** 2).mean((1, 2)), atol=1e-6)
