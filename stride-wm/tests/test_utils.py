"""Planning cost (clipping, goal depth) and statistics helpers. No simulator, no checkpoints."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.cost import GoalScore, LatentGoalCost, load_score, nnls
from stride.stats import cluster_mean_se, wilson


class Spy(nn.Module):
    """Stand-in model: records the actions it is given; the 'latent' at call j is the constant j + 1."""

    def __init__(self, k=2, A=5):
        super().__init__()
        self.k, self.action_dim, self.w = k, A, nn.Parameter(torch.zeros(1))
        self.seen = None

    def rollout(self, z, a):
        self.seen = a
        n = a.shape[1] // self.k
        return torch.stack([torch.full_like(z, j + 1.0) for j in range(n)], 1)


def test_cost_clamps_candidates_to_env_bounds():
    mu, sd = np.full(5, 0.2, np.float32), np.full(5, 0.5, np.float32)
    model = Spy(k=2)
    cost = LatentGoalCost(model, mu, sd)
    info = {'z': torch.zeros(1, 3, 4, 3), 'goal': torch.zeros(1, 3, 4, 3)}  # CEM expands info to (B, N, ...)
    cand = torch.full((1, 3, 2, 10), 100.0)
    cost.get_cost(info, cand)
    assert torch.allclose(model.seen, torch.full_like(model.seen, (1 - 0.2) / 0.5))  # upper bound = 1.6
    cand = torch.full((1, 3, 2, 10), -100.0)
    cost.get_cost(info, cand)
    assert torch.allclose(model.seen, torch.full_like(model.seen, (-1 - 0.2) / 0.5))  # lower bound = -2.4


@pytest.mark.parametrize('depth,expected_latent', [(None, 3.0), (2, 1.0), (4, 2.0), (6, 3.0)])
def test_cost_scores_the_prediction_at_the_goal_depth(depth, expected_latent):
    model = Spy(k=2)  # H = 6 -> 3 calls -> predictions at depths 2, 4, 6
    cost = LatentGoalCost(model, np.zeros(5, np.float32), np.ones(5, np.float32), depth=depth)
    info = {'z': torch.zeros(1, 2, 4, 3), 'goal': torch.zeros(1, 2, 4, 3)}
    c = cost.get_cost(info, torch.zeros(1, 2, 3, 10))
    assert torch.allclose(c, torch.full((1, 2), expected_latent ** 2))


def test_cost_rejects_goal_depth_that_is_not_a_multiple_of_k():
    cost = LatentGoalCost(Spy(k=4), np.zeros(5, np.float32), np.ones(5, np.float32), depth=25)
    with pytest.raises(ValueError):
        cost.call_index(10)


def test_cluster_se_counts_episodes_not_samples():
    # 2 episodes, 50 near-identical samples each: s.e. must reflect 2 episodes, not 100 samples
    v = np.r_[np.zeros(50), np.ones(50)] + 1e-3 * np.random.default_rng(0).standard_normal(100)
    g = np.r_[np.zeros(50), np.ones(50)]
    mean, se, n = cluster_mean_se(v, g)
    assert n == 2 and mean == pytest.approx(0.5, abs=1e-2)
    assert se == pytest.approx(0.5, abs=1e-2)           # std of group means (0, 1) / sqrt(2)
    assert se > 5 * v.std(ddof=1) / np.sqrt(len(v))  # the naive s.e. over 100 samples is ~10x smaller
    assert np.isnan(cluster_mean_se(v, np.zeros(100))[1])


def test_wilson_interval_is_sane_at_small_n():
    lo, hi = wilson(1, 5)
    assert 0.0 < lo < 0.2 < hi < 0.7
    assert wilson(0, 5)[0] == pytest.approx(0.0, abs=1e-9) and wilson(5, 5)[1] == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(wilson(0, 0)[0])


# ---- planning-cost candidates (stride/cost.py, fit by scripts/fit_cost.py)
def test_latent_score_is_the_original_mse():
    z, g = torch.randn(3, 4, 6), torch.randn(3, 4, 6)
    assert torch.allclose(GoalScore('latent')(z, g), ((z - g) ** 2).mean((1, 2)))


def test_patch_weighted_score_only_sees_weighted_patches():
    w = torch.tensor([1.0, 0.0, 0.0, 0.0])
    s = GoalScore('patchw', {'patch_w': w})
    z, g = torch.zeros(2, 4, 6), torch.zeros(2, 4, 6)
    g[:, 1:] = 5.0  # differences on unweighted patches only
    assert torch.allclose(s(z, g), torch.zeros(2))
    g[:, 0] = 2.0
    assert torch.allclose(s(z, g), torch.full((2,), 4.0))
    with pytest.raises(ValueError):  # weights fit on another patch grid
        s(torch.zeros(2, 16, 6), torch.zeros(2, 16, 6))


def _probe_params(D=6, grid=2):
    # 2x2 grid of D-dim patches -> 4*D features; the 'probe' reads cube xyz from features 0-2 and EE xyz from 3-5
    W = torch.zeros(grid * grid * D, 11)
    W[0, 0] = W[1, 1] = W[2, 2] = 1.0
    W[3, 5] = W[4, 6] = W[5, 7] = 1.0
    return {'probe': {'mx': torch.zeros(grid * grid * D), 'my': torch.zeros(11), 'W': W, 'grid': grid}}


def test_probe_scores_compare_decoded_cube_and_end_effector_positions():
    z, g = torch.zeros(1, 4, 6), torch.zeros(1, 4, 6)
    g[0, 0, :3] = torch.tensor([3.0, 0.0, 0.0])  # decoded cube 3 away
    g[0, 0, 3:6] = torch.tensor([0.0, 4.0, 0.0])  # decoded end effector 4 away
    params = _probe_params()
    assert torch.allclose(GoalScore('probe_cube', params)(z, g), torch.tensor([9.0]))
    assert torch.allclose(GoalScore('probe_cube_ee', params)(z, g), torch.tensor([25.0]))


def test_cost_passes_its_score_through(tmp_path):
    params = dict(patch_w=torch.tensor([0.0, 0.0, 0.0, 1.0]), chosen='patchw', **_probe_params(D=3))
    torch.save(params, tmp_path / 'cost.pt')
    score = load_score('best', tmp_path / 'cost.pt')
    assert score.kind == 'patchw'
    model = Spy(k=2)
    cost = LatentGoalCost(model, np.zeros(5, np.float32), np.ones(5, np.float32), score=score)
    info = {'z': torch.zeros(1, 2, 4, 3), 'goal': torch.zeros(1, 2, 4, 3)}
    c = cost.get_cost(info, torch.zeros(1, 2, 3, 10))  # Spy's last latent is 3.0 everywhere
    assert torch.allclose(c, torch.full((1, 2), 9.0))


def test_nnls_recovers_non_negative_weights():
    g = torch.Generator().manual_seed(0)
    F = torch.rand(2000, 8, generator=g, dtype=torch.float64)
    w_true = torch.tensor([0.0, 2.0, 0.0, 1.0, 0.5, 0.0, 0.0, 3.0], dtype=torch.float64)
    y = F @ w_true + 7.0  # the intercept is free
    w = nnls(F, y)
    assert (w >= 0).all()
    assert torch.allclose(w, w_true, atol=1e-3)
    w = nnls(F, -F[:, 0])  # best unconstrained weight is negative: clamped to zero
    assert w[0] == 0 and (w >= 0).all()
