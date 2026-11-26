"""Defender: causal effects of the attributes, measured on the target model itself (user change 2026-11-24).

The proxy target is m_D = train(D_train, seed 0), the same model as the nonprivate target of seed 0.
It is attacked by a proxy offline RMIA (γ = 2):

- members: all of D_train; non-members N: a random half of P_def;
- K_REF = 8 reference models, each trained on |D_train| records drawn from the other half R of P_def
  (seeds 2001…2008), so they are OUT for every evaluated record and Pr(x) is their mean (the offline
  formula with a = 1);
- Pr(z) is the mean over the reference models on Z_def; the accuracy is that of m_D on N.

FSE-2022 interventions (§6.3) set attribute j of every record of D_train, N and Z_def to α. ΔMIA_j and
ΔAcc_j are the means over α of the change in AUC and accuracy; there is no averaging over models.
"""
import numpy as np
import pandas as pd
import torch

from .model import correct, true_prob
from .rmia import auc, offline_score
from .train import EPOCHS, train

K_REF = 8
CONT_GRID = 10
PDEF_SPLIT_SEED = 2000
REF_SEED_BASE = 2000
TARGET_SEED = 0


def split_pdef(n_pdef, seed=PDEF_SPLIT_SEED):
    """Random halves of P_def (positions inside P_def): N = non-members, R = reference training set."""
    perm = np.random.default_rng(seed).permutation(n_pdef)
    half = n_pdef // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


def value_set(attr, X_pdef, n_grid=CONT_GRID):
    """§6.3 val_set_j as a list of (label, encoding of length |G_j|).

    Categorical: every category that occurs in P_def (Unknown included if it occurs), in the
    order of the category list. Continuous: n_grid equally spaced points 0, 1/(n−1), …, 1."""
    cols = attr["columns"]
    if attr["type"] == "categorical":
        present = np.asarray(X_pdef[:, cols]).sum(0) > 0
        vals = []
        for i, name in enumerate(attr["categories"]):
            if present[i]:
                vec = np.zeros(len(cols), dtype=np.float32)
                vec[i] = 1.0
                vals.append((name, vec))
        return vals
    return [(float(g), np.full(len(cols), g, dtype=np.float32)) for g in np.linspace(0.0, 1.0, n_grid)]


def set_attribute(X, cols, vec):
    """Set attribute j (all columns in G_j) of every record to the encoding `vec`; other columns
    unchanged. Works on numpy arrays and torch tensors."""
    if isinstance(X, torch.Tensor):
        new = X.clone()
        new[:, torch.as_tensor(cols, device=X.device)] = torch.as_tensor(vec, device=X.device, dtype=X.dtype)
    else:
        new = np.array(X, copy=True)
        new[:, cols] = vec
    return new


def proxy_attack(target, refs, X_mem, y_mem, X_non, y_non, X_z, y_z):
    """Proxy offline RMIA with all-OUT references (a = 1): (AUC members vs non-members, target accuracy on
    the non-members)."""
    def probs(X, y):
        return true_prob(target, X, y), np.stack([true_prob(m, X, y) for m in refs])

    pt_z, pr_z = probs(X_z, y_z)
    s_in = offline_score(*probs(X_mem, y_mem), pt_z, pr_z, a=1.0)
    s_out = offline_score(*probs(X_non, y_non), pt_z, pr_z, a=1.0)
    return auc(s_in, s_out), float(correct(target, X_non, y_non).mean())


def run_defender(X_tr, y_tr, X_pdef, y_pdef, X_zdef, y_zdef, params, n_grid=CONT_GRID, epochs=EPOCHS,
                 log=print):
    """Returns baseline (one row), per-(j, α) interventions and per-attribute effects."""
    N, R = split_pdef(len(X_pdef))
    dev = X_pdef.device
    X_non, y_non = X_pdef[torch.as_tensor(N, device=dev)], y_pdef[torch.as_tensor(N, device=dev)]
    n_ref = len(X_tr)                                  # reference models are trained on |D_train| records
    target = train(X_tr, y_tr, seed=TARGET_SEED, epochs=epochs)
    refs = []
    for k in range(1, K_REF + 1):
        sub = np.sort(np.random.default_rng(REF_SEED_BASE + k).choice(len(R), n_ref, replace=False))
        idx = torch.as_tensor(R[sub], device=dev)
        refs.append(train(X_pdef[idx], y_pdef[idx], seed=REF_SEED_BASE + k, epochs=epochs))

    base_auc, base_acc = proxy_attack(target, refs, X_tr, y_tr, X_non, y_non, X_zdef, y_zdef)
    baseline = pd.DataFrame([{"target_seed": TARGET_SEED, "auc": base_auc, "acc": base_acc,
                              "n_members": len(X_tr), "n_nonmembers": len(N), "n_ref_train": n_ref,
                              "k_ref": K_REF}])
    log(f"defender baseline: AUC={base_auc:.4f} Acc={base_acc:.4f}")

    Xp_np = X_pdef.cpu().numpy()
    records = []
    for a in params["attributes"]:
        cols = a["columns"]
        for label, vec in value_set(a, Xp_np, n_grid):
            au, ac = proxy_attack(target, refs, set_attribute(X_tr, cols, vec), y_tr,
                                  set_attribute(X_non, cols, vec), y_non,
                                  set_attribute(X_zdef, cols, vec), y_zdef)
            records.append({"j": a["j"], "attribute": a["name"], "alpha": label, "auc": au, "acc": ac,
                            "ace_mia": au - base_auc, "ace_acc": ac - base_acc})
    rec = pd.DataFrame(records)

    # ΔMIA_j = mean_α ACE^MIA(j, α); likewise ΔAcc_j
    effects = rec.groupby("j")[["ace_mia", "ace_acc"]].mean().rename(
        columns={"ace_mia": "dMIA", "ace_acc": "dAcc"}).reset_index()
    effects["n_values"] = rec.groupby("j").alpha.nunique().to_numpy()
    return baseline, rec, effects
