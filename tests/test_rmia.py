"""§10.2: fast RMIA score equals brute force; AUC behaves on synthetic data."""
import numpy as np

from adaptive_dp.rmia import auc, offline_score, rmia_score, rmia_score_bruteforce


def test_fast_score_equals_bruteforce():
    rng = np.random.default_rng(0)
    rz = rng.lognormal(0.0, 1.0, 2000)
    rz[:50] = rz[50:100]                         # ties among ratio(z)
    rx = rng.lognormal(0.0, 1.2, 3000)
    rx[:200] = 2.0 * rz[:200]                    # exactly on the γ = 2 boundary
    rx[200:300] = np.nextafter(2.0 * rz[200:300], 0)      # just below it
    rx[300:400] = np.nextafter(2.0 * rz[300:400], np.inf)  # just above it
    rx[400:410] = rz.min() / 10                  # score 0
    rx[410:420] = rz.max() * 10                  # score 1
    fast = rmia_score(rx, rz)
    brute = rmia_score_bruteforce(rx, rz)
    assert np.array_equal(fast, brute)
    for gamma in (1.0, 1.5, 4.0):
        assert np.array_equal(rmia_score(rx, rz, gamma), rmia_score_bruteforce(rx, rz, gamma))


def _synthetic(boost, n=20_000, n_z=2_000, n_ref=3, seed=1):
    """Per-record difficulty d; every model gives sigmoid(d + noise). Members of the target get +boost."""
    rng = np.random.default_rng(seed)

    def probs(d, extra=0.0):
        return 1 / (1 + np.exp(-(d + extra + rng.normal(0, 0.5, d.shape))))

    d_in, d_out, d_z = rng.normal(1, 1, n), rng.normal(1, 1, n), rng.normal(1, 1, n_z)
    pt_in, pt_out, pt_z = probs(d_in, boost), probs(d_out), probs(d_z)
    pr_in = np.stack([probs(d_in) for _ in range(n_ref)])
    pr_out = np.stack([probs(d_out) for _ in range(n_ref)])
    pr_z = np.stack([probs(d_z) for _ in range(n_ref)])
    return pt_in, pt_out, pt_z, pr_in, pr_out, pr_z


def test_auc_high_when_members_are_boosted():
    pt_in, pt_out, pt_z, pr_in, pr_out, pr_z = _synthetic(boost=3.0)
    s_in = offline_score(pt_in, pr_in, pt_z, pr_z, a=1.0)           # defender's proxy attack
    s_out = offline_score(pt_out, pr_out, pt_z, pr_z, a=1.0)
    assert auc(s_in, s_out) > 0.9
    s_in = offline_score(pt_in, pr_in, pt_z, pr_z, a=0.5)
    s_out = offline_score(pt_out, pr_out, pt_z, pr_z, a=0.5)
    assert auc(s_in, s_out) > 0.9


def test_auc_half_when_target_and_references_match():
    pt_in, pt_out, pt_z, pr_in, pr_out, pr_z = _synthetic(boost=0.0)
    s_in = offline_score(pt_in, pr_in, pt_z, pr_z, a=1.0)           # defender's proxy attack
    s_out = offline_score(pt_out, pr_out, pt_z, pr_z, a=1.0)
    assert abs(auc(s_in, s_out) - 0.5) < 0.02
    s_in = offline_score(pt_in, pr_in, pt_z, pr_z, a=0.3)
    s_out = offline_score(pt_out, pr_out, pt_z, pr_z, a=0.3)
    assert abs(auc(s_in, s_out) - 0.5) < 0.02
