"""§10.4: input perturbation mechanisms (attributes only; labels are not perturbed)."""
import numpy as np

from adaptive_dp.data import prepare
from adaptive_dp.perturb import grr, grr_keep_prob, laplace_mech, laplace_noise, perturb

N = 200_000


def test_grr_frequencies_and_one_hot():
    rng = np.random.default_rng(0)
    k, eps = 5, 1.0
    out = grr(np.zeros(N, dtype=np.int64), eps, k, rng)
    p_keep = grr_keep_prob(eps, k)
    assert abs(np.mean(out == 0) - p_keep) < 0.005
    for c in range(1, k):
        assert abs(np.mean(out == c) - (1 - p_keep) / (k - 1)) < 0.005
    # the same mechanism inside perturb() always returns a valid one-hot code
    params = {"attributes": [{"j": 1, "name": "c", "type": "categorical", "categories": list("abcde"),
                              "columns": [0, 1, 2, 3, 4]}]}
    X = np.zeros((N, 5), dtype=np.float32)
    X[np.arange(N), rng.integers(0, 5, N)] = 1
    Xn = perturb(X, params, [eps], seed=1)
    assert np.isin(Xn, (0, 1)).all() and (Xn.sum(1) == 1).all()


def test_laplace_std_and_range():
    rng = np.random.default_rng(1)
    eps = 2.0
    noise = laplace_noise(N, eps, rng)
    assert abs(noise.std() - np.sqrt(2) / eps) / (np.sqrt(2) / eps) < 0.02
    out = laplace_mech(rng.random(N), eps, rng)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_perturb_on_adult_keeps_valid_encoding_and_is_seeded():
    X, y, splits, params, _ = prepare(scale=0.2)
    Xt, yt = X[splits["train"]], y[splits["train"]]
    a = perturb(Xt, params, np.full(12, 3.0), seed=0)
    b = perturb(Xt, params, np.full(12, 3.0), seed=0)
    assert np.array_equal(a, b)
    for att in params["attributes"]:
        block = a[:, att["columns"]]
        if att["type"] == "categorical":
            assert (block.sum(1) == 1).all()
        else:
            assert block.min() >= 0 and block.max() <= 1
    assert not np.array_equal(a, Xt)              # something was perturbed
