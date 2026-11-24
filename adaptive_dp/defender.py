"""Defender (§6): calibration sets/models, proxy online RMIA, FSE-2022 attribute interventions, ΔMIA/ΔAcc.

Uses only P_def and Z_def. Index arrays are positions inside P_def (0 … |P_def|−1).
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .model import correct, true_prob
from .rmia import auc, online_score
from .train import EPOCHS, train

K_DEF = 4
CONT_GRID = 10


def calibration_sets(n_pdef, K=K_DEF, seed_base=2000):
    """S_k: a random half of P_def drawn with seed 2000 + k, k = 1…K."""
    return [np.sort(np.random.default_rng(seed_base + k).choice(n_pdef, n_pdef // 2, replace=False))
            for k in range(1, K + 1)]


def train_calibration_models(X_pdef, y_pdef, sets, epochs=EPOCHS, seed_base=2000):
    out = []
    for k, S in enumerate(sets, start=1):
        idx = torch.as_tensor(S, device=X_pdef.device)
        out.append(train(X_pdef[idx], y_pdef[idx], seed=seed_base + k, epochs=epochs))
    return out


def model_outputs(models, X_pdef, y_pdef, X_zdef, y_zdef):
    """Pr(x | m_k) on P_def, Pr(z | m_k) on Z_def, and correctness on P_def, for every model."""
    P_x = np.stack([true_prob(m, X_pdef, y_pdef) for m in models])
    P_z = np.stack([true_prob(m, X_zdef, y_zdef) for m in models])
    corr = np.stack([correct(m, X_pdef, y_pdef) for m in models])
    return P_x, P_z, corr


def rotation_metrics(P_x, P_z, corr, sets, rotations):
    """§6.2 for each proxy target t in `rotations` (1-based): (AUC_t, Acc_t)."""
    K, n = P_x.shape
    out = {}
    for t in rotations:
        i = t - 1
        refs = [k for k in range(K) if k != i]
        score = online_score(P_x[i], P_x[refs], P_z[i], P_z[refs])
        member = np.zeros(n, dtype=bool)
        member[sets[i]] = True
        out[t] = (auc(score[member], score[~member]), float(corr[i][~member].mean()))
    return out


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


def run_defender(X_pdef, y_pdef, X_zdef, y_zdef, params, rotations, n_grid=CONT_GRID, epochs=EPOCHS,
                 log=print):
    """Returns baseline, per-(j, α, t) interventions, per-attribute effects, split-half info, sets."""
    sets = calibration_sets(len(X_pdef))
    models = train_calibration_models(X_pdef, y_pdef, sets, epochs)
    P_x, P_z, corr = model_outputs(models, X_pdef, y_pdef, X_zdef, y_zdef)
    base = rotation_metrics(P_x, P_z, corr, sets, rotations)
    baseline = pd.DataFrame([{"rotation": t, "auc": base[t][0], "acc": base[t][1],
                              "n_members": len(sets[t - 1]),
                              "n_nonmembers": len(X_pdef) - len(sets[t - 1])} for t in rotations])
    log("defender baseline: " + ", ".join(f"t={t}: AUC={a:.4f} Acc={c:.4f}" for t, (a, c) in base.items()))

    Xp_np = X_pdef.cpu().numpy()
    records = []
    for a in params["attributes"]:
        for label, vec in value_set(a, Xp_np, n_grid):
            Xp_i = set_attribute(X_pdef, a["columns"], vec)
            Xz_i = set_attribute(X_zdef, a["columns"], vec)
            met = rotation_metrics(*model_outputs(models, Xp_i, y_pdef, Xz_i, y_zdef), sets, rotations)
            for t, (au, ac) in met.items():
                records.append({"j": a["j"], "attribute": a["name"], "alpha": label, "t": t, "auc": au,
                                "acc": ac, "ace_mia": au - base[t][0], "ace_acc": ac - base[t][1]})
    rec = pd.DataFrame(records)

    # ΔMIA_j = mean_t mean_α ACE_t^MIA(j, α); likewise ΔAcc_j
    per_t = rec.groupby(["j", "t"])[["ace_mia", "ace_acc"]].mean().reset_index()
    effects = per_t.groupby("j")[["ace_mia", "ace_acc"]].mean().rename(
        columns={"ace_mia": "dMIA", "ace_acc": "dAcc"}).reset_index()
    effects["n_values"] = rec.groupby("j").alpha.nunique().to_numpy()

    half = len(rotations) // 2
    h1, h2 = rotations[:half], rotations[half:]
    d1 = per_t[per_t.t.isin(h1)].groupby("j").ace_mia.mean()
    d2 = per_t[per_t.t.isin(h2)].groupby("j").ace_mia.mean()
    rho = float(spearmanr(d1.to_numpy(), d2.to_numpy()).statistic)
    effects["dMIA_half1"], effects["dMIA_half2"] = d1.to_numpy(), d2.to_numpy()
    split = {"rotations_half1": list(h1), "rotations_half2": list(h2), "spearman": rho}
    return baseline, rec, effects, split, sets
