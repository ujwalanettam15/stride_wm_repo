"""Single-cube OGBench env from stable-worldmodel (the env LeWM's cube dataset comes from)."""
import os
import sys

if sys.platform.startswith('linux'):
    os.environ.setdefault('MUJOCO_GL', 'egl')  # headless Linux; macOS renders natively
import gymnasium as gym
import stable_worldmodel  # noqa: F401  (registers swm/OGBCube-v0)

ENV_ID = 'swm/OGBCube-v0'
# ob_type='states' + rendered pixels and no goal marker, as in swm's cube planning config.
ENV_KW = dict(env_type='single', ob_type='states', mode='data_collection',
              visualize_info=False, terminate_at_goal=False, width=224, height=224)


class FastRender(gym.Wrapper):
    """CPU-only pilot: shadow map 4096->1024 and no MSAA (~2x faster software rendering).
    Leave off on GPU machines so frames match the reference dataset."""

    def reset(self, **kw):  # swm recompiles the model on reset, so re-apply each time
        out = self.env.reset(**kw)
        q = self.env.unwrapped._model.vis.quality
        q.shadowsize, q.offsamples = 1024, 0
        return out


def make_env(fast_render=False):
    env = gym.make(ENV_ID, **ENV_KW)
    return FastRender(env) if fast_render else env
