"""§10.4: input perturbation mechanisms and budgets."""
import numpy as np

from adaptive_dp.allocate import allocate, normalise
from adaptive_dp.data import prepare
from adaptive_dp.perturb import (EPS_MEAN, grr, grr_keep_prob, label_rr, laplace_mech, laplace_noise,
                                 perturb)

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
    Xn, _ = perturb(X, np.zeros(N, dtype=np.int64), params, [eps], 1.0, seed=1)
    assert np.isin(Xn, (0, 1)).all() and (Xn.sum(1) == 1).all()


def test_laplace_std_and_range():
    rng = np.random.default_rng(1)
    eps = 2.0
    noise = laplace_noise(N, eps, rng)
    assert abs(noise.std() - np.sqrt(2) / eps) / (np.sqrt(2) / eps) < 0.02
    out = laplace_mech(rng.random(N), eps, rng)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_label_rr_frequency():
    rng = np.random.default_rng(2)
    for eps in (1.0, 3.0):
        y = rng.integers(0, 2, N)
        out = label_rr(y, eps, rng)
        assert abs(np.mean(out == y) - np.exp(eps) / (1 + np.exp(eps))) < 0.005
        assert set(np.unique(out)) <= {0, 1}


def test_budget_sums_are_36():
    assert abs(np.full(12, EPS_MEAN).sum() - 36) / 36 < 1e-6
    rng = np.random.default_rng(3)
    for _ in range(20):
        V, U = normalise(rng.normal(-0.002, 0.004, 12), rng.normal(-0.01, 0.02, 12))
        w, _ = allocate(U, V)
        assert abs((EPS_MEAN * w).sum() - 36) / 36 < 1e-6


def test_perturb_on_adult_keeps_valid_encoding_and_is_seeded():
    X, y, splits, params, _ = prepare(scale=0.2)
    Xt, yt = X[splits["train"]], y[splits["train"]]
    a = perturb(Xt, yt, params, np.full(12, 3.0), 3.0, seed=0)
    b = perturb(Xt, yt, params, np.full(12, 3.0), 3.0, seed=0)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    for att in params["attributes"]:
        block = a[0][:, att["columns"]]
        if att["type"] == "categorical":
            assert (block.sum(1) == 1).all()
        else:
            assert block.min() >= 0 and block.max() <= 1
    assert not np.array_equal(a[0], Xt)              # something was perturbed
