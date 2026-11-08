"""Planning cost (clipping, goal depth) and statistics helpers. No simulator, no checkpoints."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.cost import LatentGoalCost
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
