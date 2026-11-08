"""Small statistics helpers shared by the evaluation scripts and the report."""
import numpy as np


def cluster_mean_se(values, groups):
    """Mean over samples and its standard error with episodes as the unit of replication.

    Rollouts that start a few frames apart in the same episode overlap and are strongly correlated, so
    a plain s.e. over samples is too small. Here each group (episode) contributes its own mean, and the
    s.e. is the spread of those group means. Returns (mean, se, n_groups); se is nan for < 2 groups.
    """
    v, g = np.asarray(values, np.float64), np.asarray(groups)
    mean = float(v.mean())
    ids, inv = np.unique(g, return_inverse=True)
    if len(ids) < 2:
        return mean, float('nan'), len(ids)
    gm = np.bincount(inv, weights=v) / np.bincount(inv)
    return mean, float(gm.std(ddof=1) / np.sqrt(len(ids))), len(ids)


def wilson(successes, n, z=1.96):
    """Wilson score interval for a binomial rate (valid for the tiny n used in planning)."""
    if n == 0:
        return float('nan'), float('nan')
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(c - h, 0.0)), float(min(c + h, 1.0))
