"""RMIA scores (online §6.2, offline §8.2), AUC and TPR@FPR."""
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

GAMMA = 2.0
A_GRID = [round(0.1 * i, 1) for i in range(11)]


def rmia_score(ratio_x, ratio_z, gamma=GAMMA):
    """score(x) = fraction of z with ratio(x) / ratio(z) ≥ γ, via sorting + binary search.

    For fixed ratio(x) > 0 the predicate is monotone in ratio(z), so it holds on a prefix of the
    sorted ratio(z). searchsorted gives the prefix length; the two loops then move the boundary
    so that it matches the floating-point predicate ratio(x) / ratio(z) ≥ γ exactly.
    """
    rx = np.asarray(ratio_x, dtype=np.float64)
    rz = np.sort(np.asarray(ratio_z, dtype=np.float64))
    n = len(rz)
    idx = np.searchsorted(rz, rx / gamma, side="right")
    while True:
        ok = idx < n
        step = np.zeros(len(rx), dtype=np.int64)
        step[ok] = rx[ok] / rz[idx[ok]] >= gamma
        if not step.any():
            break
        idx += step
    while True:
        ok = idx > 0
        step = np.zeros(len(rx), dtype=np.int64)
        step[ok] = rx[ok] / rz[idx[ok] - 1] < gamma
        if not step.any():
            break
        idx -= step
    return idx / n


def rmia_score_bruteforce(ratio_x, ratio_z, gamma=GAMMA):
    rx = np.asarray(ratio_x, dtype=np.float64)[:, None]
    rz = np.asarray(ratio_z, dtype=np.float64)[None, :]
    return (rx / rz >= gamma).mean(axis=1)


def online_score(p_target_x, p_refs_x, p_target_z, p_refs_z, gamma=GAMMA):
    """§6.2: Pr(x), Pr(z) are the means over the reference models (IN and OUT mixed)."""
    pr_x = np.mean(p_refs_x, axis=0)
    pr_z = np.mean(p_refs_z, axis=0)
    return rmia_score(p_target_x / pr_x, p_target_z / pr_z, gamma)


def offline_score(p_target_x, p_refs_x, p_target_z, p_refs_z, a, gamma=GAMMA):
    """§8.2: Pr(x) = ½((1 + a)·Pr_OUT(x) + (1 − a)); Pr(z) = mean over the reference models."""
    pr_out = np.mean(p_refs_x, axis=0)
    pr_x = 0.5 * ((1 + a) * pr_out + (1 - a))
    pr_z = np.mean(p_refs_z, axis=0)
    return rmia_score(p_target_x / pr_x, p_target_z / pr_z, gamma)


def auc(scores_in, scores_out):
    labels = np.r_[np.ones(len(scores_in)), np.zeros(len(scores_out))]
    return float(roc_auc_score(labels, np.r_[scores_in, scores_out]))


def tpr_at_fpr(scores_in, scores_out, alpha):
    """Largest TPR over all ROC points with FPR ≤ α."""
    labels = np.r_[np.ones(len(scores_in)), np.zeros(len(scores_out))]
    fpr, tpr, _ = roc_curve(labels, np.r_[scores_in, scores_out])
    return float(tpr[fpr <= alpha].max())
