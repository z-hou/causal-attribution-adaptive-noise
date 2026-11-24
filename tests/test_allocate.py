"""§10.5 (AGENT.md §10.5): allocation of the budget shares w_j."""
import numpy as np

from adaptive_dp.allocate import allocate, normalise


def _scores(seed):
    rng = np.random.default_rng(seed)
    dMIA = rng.normal(0.002, 0.004, 12)
    dAcc = rng.normal(0.01, 0.02, 12)
    return normalise(dMIA, dAcc)


def test_sum_and_range():
    for seed in range(20):
        V, U = _scores(seed)
        eps, _ = allocate(U, V)
        assert abs(eps.sum() - 12) / 12 < 1e-6
        assert eps.min() >= 0.1 and eps.max() <= 10
    # an extreme, very unequal input
    U = np.r_[12.0, np.zeros(11)]
    V = np.r_[0.0, np.full(11, 12 / 11)]
    eps, _ = allocate(U, V)
    assert abs(eps.sum() - 12) / 12 < 1e-6 and eps.min() >= 0.1 and eps.max() <= 10


def test_equal_scores_give_uniform_budget():
    for seed in range(5):
        V, _ = _scores(seed)
        eps, _ = allocate(V.copy(), V)
        assert np.allclose(eps, 1.0, atol=1e-6)


def test_raising_one_risk_score_does_not_raise_its_budget():
    for seed in range(10):
        V, U = _scores(seed)
        base, _ = allocate(U, V)
        for j in range(12):
            for bump in (0.1, 1.0, 10.0):
                V2 = V.copy()
                V2[j] += bump
                eps, _ = allocate(U, V2)
                assert eps[j] <= base[j] + 1e-9
