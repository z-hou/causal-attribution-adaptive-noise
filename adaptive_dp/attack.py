"""Attacker (§8.2): reference models on P_att, offline RMIA with a chosen on own data, loss attack.

Index arrays are positions inside P_att (0 … |P_att|−1) or Z_att.
"""
import numpy as np
import torch
import torch.nn.functional as F

from .model import correct, logits, true_prob
from .rmia import A_GRID, auc, offline_score, tpr_at_fpr

K_ATT = 2
A_SELECT_N = 1000
A_SELECT_SEED = 4000


def reference_sets(n_patt, size, K=K_ATT, seed_base=4000):
    """Reference model i is trained on `size` records of P_att drawn with seed 4000 + i."""
    return [np.sort(np.random.default_rng(seed_base + i).choice(n_patt, size, replace=False))
            for i in range(1, K + 1)]


def select_a(ref_models, ref_sets, X_patt, y_patt, X_zatt, y_zatt, n_select=A_SELECT_N,
             seed=A_SELECT_SEED):
    """§8.2 step 3: reference model 1 as a temporary target, model 2 as the only reference.

    Members: n_select original (unperturbed) records of ref-1's training set that are not in ref-2's;
    non-members: n_select records of Z_att; both drawn with one generator seeded by `seed`.
    Returns (best a, {a: AUC}); ties go to the smaller a."""
    rng = np.random.default_rng(seed)
    only1 = np.setdiff1d(ref_sets[0], ref_sets[1])
    mem = np.sort(rng.choice(only1, n_select, replace=False))
    non = np.sort(rng.choice(len(X_zatt), n_select, replace=False))
    dev = X_patt.device
    Xm, ym = X_patt[torch.as_tensor(mem, device=dev)], y_patt[torch.as_tensor(mem, device=dev)]
    Xn, yn = X_zatt[torch.as_tensor(non, device=dev)], y_zatt[torch.as_tensor(non, device=dev)]
    tgt, ref = ref_models
    pt_m, pr_m = true_prob(tgt, Xm, ym), true_prob(ref, Xm, ym)[None]
    pt_n, pr_n = true_prob(tgt, Xn, yn), true_prob(ref, Xn, yn)[None]
    pt_z, pr_z = true_prob(tgt, X_zatt, y_zatt), true_prob(ref, X_zatt, y_zatt)[None]
    aucs = {}
    for a in A_GRID:
        aucs[a] = auc(offline_score(pt_m, pr_m, pt_z, pr_z, a), offline_score(pt_n, pr_n, pt_z, pr_z, a))
    best = max(A_GRID, key=lambda a: (aucs[a], -a))
    return best, aucs


@torch.no_grad()
def evaluate_target(target, ref_models, a, X_tr, y_tr, X_te, y_te, X_zatt, y_zatt):
    """Utility plus offline RMIA and the loss attack on the original challenge points."""
    pt_in, pt_out = true_prob(target, X_tr, y_tr), true_prob(target, X_te, y_te)
    pr_in = np.stack([true_prob(m, X_tr, y_tr) for m in ref_models])
    pr_out = np.stack([true_prob(m, X_te, y_te) for m in ref_models])
    pt_z = true_prob(target, X_zatt, y_zatt)
    pr_z = np.stack([true_prob(m, X_zatt, y_zatt) for m in ref_models])
    s_in = offline_score(pt_in, pr_in, pt_z, pr_z, a)
    s_out = offline_score(pt_out, pr_out, pt_z, pr_z, a)
    l_in = -F.cross_entropy(logits(target, X_tr).double(), y_tr, reduction="none").cpu().numpy()
    l_out = -F.cross_entropy(logits(target, X_te).double(), y_te, reduction="none").cpu().numpy()
    train_acc = float(correct(target, X_tr, y_tr).mean())
    test_acc = float(correct(target, X_te, y_te).mean())
    return {"test_acc": test_acc, "train_acc": train_acc, "gap": train_acc - test_acc,
            "rmia_auc": auc(s_in, s_out), "rmia_tpr1": tpr_at_fpr(s_in, s_out, 0.01),
            "rmia_tpr01": tpr_at_fpr(s_in, s_out, 0.001), "loss_auc": auc(l_in, l_out)}
