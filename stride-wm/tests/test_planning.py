"""Simulator-as-model planning cost (stride/planning.py). Builds the real simulator; no data, no checkpoints."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.planning import SimCost


@pytest.fixture(scope='module')
def env():
    try:
        from stride.env import make_env
        e = make_env()
    except ImportError as err:  # e.g. no headless OpenGL on this machine
        pytest.skip(f'simulator unavailable: {err}')
    e.reset(seed=0)
    return e


def test_sim_cost_matches_env_step_and_leaves_the_episode_untouched(env):
    u = env.unwrapped
    mu, sd = np.zeros(5), np.full(5, 0.5)
    goal = u._data.joint('object_joint_0').qpos[:3] + np.array([0.05, 0.0, 0.0])
    cost = SimCost(env, mu, sd, depth=6, goal_cube=goal)
    cost.snapshot()
    q0, v0 = u._data.qpos.copy(), u._data.qvel.copy()
    cand = torch.from_numpy(np.random.default_rng(0).standard_normal((1, 3, 8, 5)).astype(np.float32))
    c = cost.get_cost({}, cand)
    assert c.shape == (1, 3)
    assert np.array_equal(u._data.qpos, q0) and np.array_equal(u._data.qvel, v0)
    # candidate 1 executed with env.step from the same state ends where the cost said (first `depth` actions)
    for a in np.clip(cand[0, 1].double().numpy() * sd + mu, -1, 1)[:6]:
        info = env.step(a)[-1]
    assert c[0, 1].item() == pytest.approx(((info['privileged/block_0_pos'] - goal) ** 2).sum(), abs=1e-7)


def test_sim_cost_on_rendered_frames_needs_an_encoder_and_a_goal_latent(env):
    with pytest.raises(ValueError):
        SimCost(env, np.zeros(5), np.ones(5), depth=4, kind='latent')
