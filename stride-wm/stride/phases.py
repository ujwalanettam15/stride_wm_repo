"""Per-frame task phases from robot + cube state (proposal 4.1).

Thresholds are fit once on the training episodes with the fixed rules in fit_thresholds(),
saved next to the latent cache, and never tuned afterwards. Grasp and Place win overlaps.
"""
import numpy as np

NAMES = ['reach', 'grasp', 'transport', 'place', 'other']
REACH, GRASP, TRANSPORT, PLACE, OTHER = range(5)
GOAL_RADIUS = 0.04  # OGBench cube success radius (task definition, not fit)


def _mid(x):  # split point of a bimodal signal: midpoint of its 5th/95th percentiles
    lo, hi = np.percentile(x, [5, 95])
    return float((lo + hi) / 2)


def fit_thresholds(s):
    d = np.linalg.norm(s['ee_pos'] - s['cube_pos'], axis=1)
    contact_thr = _mid(s['contact'])
    held = s['contact'] > contact_thr
    eps = float(np.percentile(d[held], 95))           # EE-cube distance while the cube is in hand
    free = ~held & (d > eps)                           # untouched cube
    z = s['cube_pos'][:, 2]
    z_table = float(np.median(z[free]))
    static = free & (np.abs(np.gradient(z)) < 1e-4)    # resting, not falling
    delta = float(np.percentile(np.abs(z[static] - z_table), 99))  # resting-height noise floor
    return dict(contact_thr=contact_thr, closed_thr=_mid(s['grip']), eps=eps, z_table=z_table,
                delta=delta, vel_thr=float(0.1 * np.percentile(np.abs(s['grip_vel']), 95)),
                goal_radius=GOAL_RADIUS)


def label(s, th, ep_last):
    """s: dict of per-frame arrays; ep_last: bool mask of each episode's final frame."""
    d = np.linalg.norm(s['ee_pos'] - s['cube_pos'], axis=1)
    dd = np.r_[d[1:] - d[:-1], 0.0]
    idx = np.flatnonzero(ep_last)
    dd[idx] = dd[np.maximum(idx - 1, 0)]               # backward difference at episode ends
    held = s['contact'] > th['contact_thr']
    lifted = s['cube_pos'][:, 2] > th['z_table'] + th['delta']
    closing, opening = s['grip_vel'] > th['vel_thr'], s['grip_vel'] < -th['vel_thr']
    near_goal = np.linalg.norm(s['cube_pos'] - s['target_pos'], axis=1) < th['goal_radius']
    at_cube = d < th['eps']

    lab = np.full(len(d), OTHER, np.int8)
    lab[(s['grip'] < th['closed_thr']) & ~held & (dd < 0)] = REACH
    lab[held & lifted] = TRANSPORT
    # contact events only: the expert's noisy gripper also twitches while carrying the cube and
    # after it has let go, so Grasp needs the cube still on the table and Place the hand at the cube
    lab[opening & near_goal & at_cube] = PLACE
    lab[closing & at_cube & ~lifted] = GRASP
    return lab
