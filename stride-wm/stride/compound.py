"""Split a model's open-loop error into each call's own error and the error carried in from earlier calls."""
import torch


@torch.no_grad()
def decompose(model, Z, a, chunk=512):
    """Z: (B, n+1, P, D) true latents at depths 0, k, ..., n*k. a: (B, n*k, A) normalized actions.

    Returns three (B, n) tensors, one column per model call j = 1..n (depth j*k):
      total       MSE(zhat_j, z_j)              zhat_j = f(zhat_{j-1}, a_j), the open-loop prediction
      fresh       MSE(f(z_{j-1}, a_j), z_j)     the same call given the TRUE previous latent: its own error
      propagated  MSE(zhat_j, f(z_{j-1}, a_j))  same model and actions, only the input differs: error carried in
    total = fresh + propagated + a cross term. At the first call the input is true, so propagated = 0."""
    B, n = Z.shape[0], Z.shape[1] - 1
    zs = model.rollout(Z[:, 0], a)
    prev = Z[:, :-1].reshape(B * n, *Z.shape[2:])
    ak = a.reshape(B * n, model.k, a.shape[-1])
    tf = torch.cat([model(prev[i: i + chunk], ak[i: i + chunk]) for i in range(0, B * n, chunk)]).reshape(zs.shape)
    true = Z[:, 1:]
    mse = lambda x: (x ** 2).mean((2, 3))
    return mse(zs - true), mse(tf - true), mse(zs - tf)
