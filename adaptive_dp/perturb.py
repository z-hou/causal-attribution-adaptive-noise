"""Input perturbation (§5.3): generalised randomised response, Laplace, label randomised response.

Every record is perturbed once. Attribute j is ε_j-LDP, the label ε_y-LDP, so a record is
(Σ_j ε_j + ε_y)-LDP; training on the perturbed data is post-processing.
"""
import numpy as np

EPS_MEAN = 3.0      # ε̄: average attribute budget
EPS_Y = 3.0         # label budget


def grr_keep_prob(eps, k):
    """Probability that generalised randomised response keeps the true category."""
    return np.exp(eps) / (np.exp(eps) + k - 1)


def grr(codes, eps, k, rng):
    """Generalised randomised response on category indices in {0, …, k−1}.

    Keep with probability e^ε / (e^ε + k − 1); otherwise pick one of the other k − 1 categories
    uniformly at random."""
    codes = np.asarray(codes)
    keep = rng.random(len(codes)) < grr_keep_prob(eps, k)
    other = rng.integers(0, k - 1, size=len(codes))
    other = other + (other >= codes)                 # skip the true category
    return np.where(keep, codes, other)


def laplace_noise(n, eps, rng):
    """Lap(0, 1/ε) noise (sensitivity 1 on [0, 1])."""
    return rng.laplace(0.0, 1.0 / eps, size=n)


def laplace_mech(x, eps, rng):
    """Add Lap(0, 1/ε) and clip back to [0, 1]."""
    return np.clip(np.asarray(x, dtype=np.float64) + laplace_noise(len(x), eps, rng), 0.0, 1.0)


def label_rr(y, eps, rng):
    """Binary randomised response: keep with probability e^ε / (1 + e^ε), otherwise flip."""
    y = np.asarray(y)
    keep = rng.random(len(y)) < np.exp(eps) / (1 + np.exp(eps))
    return np.where(keep, y, 1 - y)


def perturb(X, y, params, eps_vec, eps_y, seed):
    """perturb(D, ε_vec, ε_y, seed): perturb every attribute of every record once, then the label.

    X is the encoded matrix (numpy, float32), y the 0/1 labels. Attributes are processed in the
    order j = 1…12, then the label, all with one numpy generator seeded by `seed`."""
    rng = np.random.default_rng(seed)
    Xn = np.array(X, dtype=np.float32, copy=True)
    for a, eps in zip(params["attributes"], eps_vec):
        cols = a["columns"]
        if a["type"] == "categorical":
            k = len(cols)
            codes = Xn[:, cols].argmax(1)
            new = grr(codes, eps, k, rng)
            block = np.zeros((len(Xn), k), dtype=np.float32)
            block[np.arange(len(Xn)), new] = 1.0
            Xn[:, cols] = block
        else:                                        # continuous: Lap(0, 1/ε), clipped to [0, 1]
            for c in cols:
                Xn[:, c] = laplace_mech(Xn[:, c], eps, rng)
    yn = label_rr(y, eps_y, rng)                     # binary randomised response (§5.3)
    return Xn, yn.astype(np.int64)
