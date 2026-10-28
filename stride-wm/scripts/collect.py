"""Collect single-cube pick-and-place episodes with stable-worldmodel's OGBench env + expert
(noisy Markov oracle) policy, written in swm's HDF5 schema. Use this when the LeWM cube .h5
can't be downloaded; encode.py reads either file the same way."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stride.env import ENV_ID, ENV_KW, FastRender  # first: sets MUJOCO_GL=egl

import stable_worldmodel as swm
from stable_worldmodel.envs.ogbench.expert_policy import ExpertPolicy

p = argparse.ArgumentParser()
p.add_argument('--out', default='data/cube_single.h5')
p.add_argument('--episodes', type=int, default=100)
p.add_argument('--steps', type=int, default=200)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--fast-render', action='store_true')
args = p.parse_args()

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
world = swm.World(ENV_ID, num_envs=1, image_shape=(224, 224), max_episode_steps=args.steps,
                  pre_wrappers=[FastRender] if args.fast_render else None, **ENV_KW)
world.set_policy(ExpertPolicy(seed=args.seed))
world.collect(args.out, episodes=args.episodes, seed=args.seed, format='hdf5')
