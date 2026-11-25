"""Reproduce the Adult results: nonprivate / uniform / adaptive input perturbation, evaluated by the attacker.

    python -m adaptive_dp.run --eps-mean 3 --seeds 0 1 2 --out results/epsbar3/

Writes results.csv (one row per method and seed) and allocation.csv (effects and budget of each
attribute), then prints the mean ± sd over the seeds.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import attack, defender
from .allocate import allocate, normalise
from .data import prepare
from .perturb import EPS_MEAN, EPS_Y, perturb
from .train import train

import pdb

METHODS = ["nonprivate", "uniform", "adaptive"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps-mean", type=float, default=EPS_MEAN, help="average attribute budget ε̄ (ε_y = 3)")
    p.add_argument("--eps-total", type=float, default=None,
                   help="total budget instead of --eps-mean: split equally over the 12 attributes and the label, "
                        "ε̄ = ε_y = ε_total / 13")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/")
    p.add_argument("--non-adaptive", action="store_true",
                   help="attack all methods with the same reference models (unperturbed P_att) and the same a")
    p.add_argument("--z-unchanged", action="store_true",
                   help="defender interventions leave Z_def unchanged (only D_train and N are set to α)")
    p.add_argument("--n-train", type=int, default=None, help="use the first n records of D_train (default: all 7000)")
    p.add_argument("--n-test", type=int, default=None, help="use the first n records of D_test (default: all 7000)")
    p.add_argument("--k-att", type=int, default=attack.K_ATT, help="attacker reference models per method (default 2)")
    p.add_argument("--k-def", type=int, default=defender.K_REF, help="defender reference models (default 4)")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, splits, params, _ = prepare()
    splits["train"] = splits["train"][:args.n_train]       # [:None] keeps everything
    splits["test"] = splits["test"][:args.n_test]
    n_tr = len(splits["train"])                            # reference models are trained on n_tr records

    def tensors(Xa, ya):
        return (torch.as_tensor(Xa, dtype=torch.float32, device=device),
                torch.as_tensor(ya, dtype=torch.long, device=device))

    T = {k: tensors(X[idx], y[idx]) for k, idx in splits.items()}

    ##T: *T["train"]、*T["p_def"]、*T["z_def"] 传给防御方分析（第 48 行）；
    ##   *T["p_att"]、*T["z_att"] 传给攻击者选 a（第 76 行）；
    ##   *T["train"]、*T["test"]、*T["z_att"] 传给最终评估（第 84 行）。
    ## T["train"] 用于训练目标模型，T["p_def"]、*T["z_def"] 是防御方的代理数据集，用于构造代理RMIA来计算特征因果归因
    ## T["P_att"]和T["z_att"]用于构造实际的攻击模型
    ## T["test"] 用于检验准确率，和MIA成功率，不参与任何归因和优化

    # defender: causal effects on the target model train(D_train, seed 0), then the budget shares w_j
    _, _, effects = defender.run_defender(*T["train"], *T["p_def"], *T["z_def"], params,
                                          intervene_z=not args.z_unchanged, k_ref=args.k_def,
                                          n_ref_train=n_tr if args.n_train else None)
    V, U = normalise(effects.dMIA, effects.dAcc)

    w, _ = allocate(U, V)

    if args.eps_total is None:
        eps_mean, eps_y = args.eps_mean, EPS_Y
    else:
        eps_mean = eps_y = args.eps_total / (len(w) + 1)
    eps = {"uniform": np.full(len(w), eps_mean), "adaptive": eps_mean * w}
    eps_total = float(eps["uniform"].sum() + eps_y)
    attrs = params["attributes"]
    pd.DataFrame({"j": [a["j"] for a in attrs], "attribute": [a["name"] for a in attrs],
                  "type": [a["type"] for a in attrs], "dMIA": effects.dMIA, "dAcc": effects.dAcc, "w": w,
                  "eps_uniform": eps["uniform"], "eps_adaptive": eps["adaptive"]}
                 ).to_csv(out / "allocation.csv", index=False)

    def fit(method, Xa, ya, seed_perturb, seed_train):
        """Perturb the original records once (uniform / adaptive), then train."""
        if method != "nonprivate":
            Xa, ya = perturb(Xa, ya, params, eps[method], eps_y, seed_perturb)
        return train(*tensors(Xa, ya), seed=seed_train)

    # attacker: 2 reference models per method on P_att, offline RMIA with a chosen on its own data.
    # Adaptive (default): the references are processed like the target. Non-adaptive: every method is
    # attacked with the nonprivate references and their a.
    Xpa, ypa = X[splits["p_att"]], y[splits["p_att"]]
    ref_sets = attack.reference_sets(len(Xpa), n_tr if args.n_train else len(Xpa) // 2, K=args.k_att)
    refs, a_of = {}, {}
    for m in METHODS:
        if args.non_adaptive and m != "nonprivate":
            refs[m], a_of[m] = refs["nonprivate"], a_of["nonprivate"]
            continue
        refs[m] = [fit(m, Xpa[S], ypa[S], 4100 + i, 4000 + i) for i, S in enumerate(ref_sets, start=1)]
        a_of[m], _ = attack.select_a(refs[m][:2], ref_sets[:2], *T["p_att"], *T["z_att"])  # a from refs 1–2

    # targets on D_train, attacked on the original D_train (members) and D_test (non-members)
    Xtr, ytr = X[splits["train"]], y[splits["train"]]
    rows = []
    for m in METHODS:
        for seed in args.seeds:
            target = fit(m, Xtr, ytr, seed, seed)
            r = attack.evaluate_target(target, refs[m], a_of[m], *T["train"], *T["test"], *T["z_att"])
            rows.append({"method": m, "eps_total": float("inf") if m == "nonprivate" else eps_total,
                         "seed": seed, "a": a_of[m], **r})
            print(f"{m:10s} seed={seed}: test_acc={r['test_acc']:.4f} rmia_auc={r['rmia_auc']:.4f} "
                  f"tpr1={r['rmia_tpr1']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)

    g = res.groupby("method", sort=False)[["test_acc", "rmia_auc", "rmia_tpr1", "loss_auc"]]
    print(f"\nε̄ = {eps_mean:g}, ε_y = {eps_y:g}, ε_total = {eps_total:g}; mean ± sd over seeds {args.seeds}:")
    print((g.mean().map("{:.4f}".format) + " ± " + g.std().map("{:.4f}".format)).to_string())


if __name__ == "__main__":
    main()
