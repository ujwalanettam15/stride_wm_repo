"""Planning cost for CEM, kept out of scripts/eval_plan.py so it can be unit-tested (no simulator)."""
import torch


class LatentGoalCost:
    """MSE between the rolled-out latent and the goal latent.

    Two fixes over the first version (both found by reading the prelim planning logs, see README):
      * clipping: CEM samples unbounded normalized actions but the env executes clip(a*sd+mu, -1, 1).
        The model is now scored on the actions that are executed, i.e. candidates are clamped to the
        env bounds in normalized units before the rollout.
      * goal depth: the goal is the frame `depth` env steps after the start, so the prediction at that
        depth is compared to it (depth = H when the goal offset equals the horizon; the old code always
        used the last prediction even when the goal was 25 steps away and H was 40).
    """

    def __init__(self, model, mu, sd, depth=None):
        self.model = model
        dev = next(model.parameters()).device
        mu, sd = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)
        self.lo, self.hi = ((-1 - mu) / sd).to(dev), ((1 - mu) / sd).to(dev)
        self.depth = depth  # env steps; None = the end of the rollout

    def call_index(self, n_calls):
        """Index into the rollout (one entry per model call) holding the prediction at `depth`."""
        if self.depth is None:
            return n_calls - 1
        k = self.model.k
        if self.depth % k or not 0 < self.depth // k <= n_calls:
            raise ValueError(f'goal depth {self.depth} is not a multiple of k={k} within the horizon')
        return self.depth // k - 1

    @torch.no_grad()
    def get_cost(self, info, cand):  # cand: (B, N, H/k, k*A) normalized actions
        B, N = cand.shape[:2]
        z0 = info['z'].reshape(B * N, *info['z'].shape[2:])
        a = cand.reshape(B * N, -1, self.model.action_dim)
        a = torch.minimum(torch.maximum(a, self.lo), self.hi)
        zs = self.model.rollout(z0, a)
        zH = zs[:, self.call_index(zs.shape[1])]
        return ((zH - info['goal'].reshape(zH.shape)) ** 2).mean((1, 2)).view(B, N)
