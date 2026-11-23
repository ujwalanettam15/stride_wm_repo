"""Planning costs for CEM, kept out of scripts/eval_plan.py so they can be unit-tested (no simulator).

Every cost compares the latent predicted at the goal's depth with the goal latent. They differ only in
how two (P, D) latents become one number (GoalScore):
  latent         mean squared error over every patch and channel (the original cost)
  patchw         per-patch squared error, weighted by non-negative patch weights fit on training
                 episodes so the cost tracks the cube's distance to its goal (scripts/fit_cost.py)
  probe_cube     squared distance between the cube positions a frozen linear probe reads off the two
                 latents
  probe_cube_ee  probe_cube plus the same term for the end-effector position
patchw and the probe are fit with state labels on training episodes only. At planning time they see
nothing but latents, like the original cost. Why these exist: on true latents the original cost ranks
frames by arm distance (Spearman rho 0.83) far better than by cube distance (0.28), so a planner
minimizing it is rewarded for matching the arm pose whether or not the cube moved.
"""
import torch

from stride.data import probe_features

COSTS = ('latent', 'patchw', 'probe_cube', 'probe_cube_ee')
CUBE, EE = slice(0, 3), slice(5, 8)  # columns of stride.data.probe_targets


class GoalScore:
    """Maps a predicted latent and a goal latent, both (n, P, D), to a cost per sample, (n,)."""

    def __init__(self, kind='latent', params=None, device='cpu'):
        if kind not in COSTS:
            raise ValueError(f'unknown cost {kind!r}; choose from {COSTS}')
        self.kind = kind
        if kind == 'patchw':
            self.w = params['patch_w'].float().to(device)
        elif kind.startswith('probe'):
            pr = params['probe']
            self.grid = pr['grid']
            self.mx, self.my, self.W = (pr[n].float().to(device) for n in ('mx', 'my', 'W'))

    def probe(self, z):
        return (probe_features(z, self.grid) - self.mx) @ self.W + self.my

    def __call__(self, z, g):
        if self.kind == 'latent':
            return ((z - g) ** 2).mean((1, 2))
        if self.kind == 'patchw':
            if len(self.w) != z.shape[1]:
                raise ValueError(f'patch weights cover {len(self.w)} patches, latents have {z.shape[1]}: '
                                 f'fit_cost.py was run on a cache with a different patch grid')
            return (((z - g) ** 2).mean(2) * self.w).sum(1)
        y, yg = self.probe(z), self.probe(g)
        d = ((y[:, CUBE] - yg[:, CUBE]) ** 2).sum(1)
        if self.kind == 'probe_cube_ee':
            d = d + ((y[:, EE] - yg[:, EE]) ** 2).sum(1)
        return d


def load_score(kind, path, device='cpu'):
    """GoalScore for `kind`; 'best' reads the cost fit_cost.py chose. Only 'latent' needs no file."""
    if kind == 'latent':
        return GoalScore('latent')
    params = torch.load(path, map_location='cpu', weights_only=False)
    return GoalScore(params['chosen'] if kind == 'best' else kind, params, device)


def nnls(F, y, iters=3000):
    """Non-negative least squares with a free intercept: min ||(F - mean F) w - (y - mean y)||^2, w >= 0.

    Accelerated projected gradient (FISTA) on the Gram matrix, so no scipy dependency. F: (n, P)."""
    F, y = F.double(), y.double()
    Fc, yc = F - F.mean(0), y - y.mean()
    G, b = Fc.T @ Fc, Fc.T @ yc
    step = 1.0 / torch.linalg.eigvalsh(G).max().clamp_min(1e-12)
    w = z = torch.zeros(F.shape[1], dtype=torch.float64)
    t = 1.0
    for _ in range(iters):
        w_next = (z - step * (G @ z - b)).clamp_min(0)
        t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
        z = w_next + ((t - 1) / t_next) * (w_next - w)
        w, t = w_next, t_next
    return w


class LatentGoalCost:
    """CEM cost: rolls the candidates out with the model and scores the prediction at the goal's depth.

    Two fixes over the first version (both found by reading the prelim planning logs, see README):
      * clipping: CEM samples unbounded normalized actions but the env executes clip(a*sd+mu, -1, 1).
        The model is now scored on the actions that are executed, i.e. candidates are clamped to the
        env bounds in normalized units before the rollout.
      * goal depth: the goal is the frame `depth` env steps after the start, so the prediction at that
        depth is compared to it (depth = H when the goal offset equals the horizon; the old code always
        used the last prediction even when the goal was 25 steps away and H was 40).
    `score` picks how prediction and goal are compared (GoalScore); the default is the original latent MSE.
    """

    def __init__(self, model, mu, sd, depth=None, score=None):
        self.model = model
        dev = next(model.parameters()).device
        mu, sd = torch.as_tensor(mu, dtype=torch.float32), torch.as_tensor(sd, dtype=torch.float32)
        self.lo, self.hi = ((-1 - mu) / sd).to(dev), ((1 - mu) / sd).to(dev)
        self.depth = depth  # env steps; None = the end of the rollout
        self.score = score or GoalScore('latent')

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
        return self.score(zH, info['goal'].reshape(zH.shape)).view(B, N)
