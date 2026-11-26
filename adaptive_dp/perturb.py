"""Input perturbation of the attributes: generalised randomised response and Laplace.

Every record is perturbed once; attribute j is ε_j-LDP, so the attributes of a record are (Σ_j ε_j)-LDP and
training on the perturbed data is post-processing. The label is not sensitive and is not perturbed.
"""
import numpy as np


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


def perturb(X, params, eps_vec, seed):
    """Perturb every attribute of every record once and return the perturbed matrix.

    X is the encoded matrix (numpy, float32). Attributes are processed in the order j = 1, 2, …, all with
    one numpy generator seeded by `seed`: categorical attributes by generalised randomised response
    (the output stays a valid one-hot code), continuous ones by Laplace noise clipped to [0, 1]."""
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
        else:
            for c in cols:
                Xn[:, c] = laplace_mech(Xn[:, c], eps, rng)
    return Xn
