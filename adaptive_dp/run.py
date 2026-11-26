"""Main experiment: nonprivate / uniform / adaptive input perturbation, evaluated by a non-adaptive attacker.

    python -m adaptive_dp.run --eps-total 64 --out results/main/epstotal64/

Setting:
- D_train and D_test are the first 2,000 records of their splits.
- The label is not sensitive and stays clean; the whole budget ε_total goes to the J attributes
  (uniform: ε_j = ε_total / J; adaptive: ε_j = ε_total / J · w_j).
- Defender: causal effects of the attributes on train(D_train, seed 0), measured with a proxy offline RMIA
  (8 reference models), turned into the budget shares w_j.
- Attacker (non-adaptive): 8 reference models trained on unperturbed P_att and one a chosen on its own
  data; all three methods are attacked with the same reference models and the same a.

Writes results.csv (one row per method and seed) and allocation.csv (effects and budget of each
attribute), then prints the mean ± sd over the seeds.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

from . import attack, defender
from .allocate import allocate, budgets, normalise
from .data import prepare
from .perturb import perturb
from .train import train

METHODS = ["nonprivate", "uniform", "adaptive"]
N_TRAIN = N_TEST = 2000                 # records of D_train / D_test that are used (the first ones)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps-total", type=float, required=True, help="total budget ε_total over the attributes")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", default="results/")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, splits, params, _ = prepare()
    splits["train"] = splits["train"][:N_TRAIN]
    splits["test"] = splits["test"][:N_TEST]

    def tensors(Xa, ya):
        return (torch.as_tensor(Xa, dtype=torch.float32, device=device),
                torch.as_tensor(ya, dtype=torch.long, device=device))

    # T[k] = (features, labels) of split k on the device:
    #   train, p_def, z_def → defender analysis (causal attribution with the proxy RMIA);
    #   p_att, z_att        → the attacker's reference models and its choice of a;
    #   train, test, z_att  → final evaluation (test is used only here: accuracy and non-members).
    T = {k: tensors(X[idx], y[idx]) for k, idx in splits.items()}

    # defender: causal effects on the target model train(D_train, seed 0), then the budget shares w_j
    _, _, effects = defender.run_defender(*T["train"], *T["p_def"], *T["z_def"], params)
    V, U = normalise(effects.dMIA, effects.dAcc)
    w, _ = allocate(U, V)
    eps = budgets(w, args.eps_total)
    eps_total = float(eps["uniform"].sum())
    attrs = params["attributes"]
    pd.DataFrame({"j": [a["j"] for a in attrs], "attribute": [a["name"] for a in attrs],
                  "type": [a["type"] for a in attrs], "dMIA": effects.dMIA, "dAcc": effects.dAcc, "w": w,
                  "eps_uniform": eps["uniform"], "eps_adaptive": eps["adaptive"]}
                 ).to_csv(out / "allocation.csv", index=False)

    # attacker: reference models on unperturbed P_att; a from reference models 1 and 2
    Xpa, ypa = X[splits["p_att"]], y[splits["p_att"]]
    ref_sets = attack.reference_sets(len(Xpa), N_TRAIN)
    refs = [train(*tensors(Xpa[S], ypa[S]), seed=4000 + i) for i, S in enumerate(ref_sets, start=1)]
    a, _ = attack.select_a(refs[:2], ref_sets[:2], *T["p_att"], *T["z_att"])

    # targets on D_train (attributes perturbed once for uniform / adaptive), attacked on the original
    # D_train (members) and D_test (non-members)
    Xtr, ytr = X[splits["train"]], y[splits["train"]]
    rows = []
    for m in METHODS:
        for seed in args.seeds:
            Xm = Xtr if m == "nonprivate" else perturb(Xtr, params, eps[m], seed)
            target = train(*tensors(Xm, ytr), seed=seed)
            r = attack.evaluate_target(target, refs, a, *T["train"], *T["test"], *T["z_att"])
            rows.append({"method": m, "eps_total": float("inf") if m == "nonprivate" else eps_total,
                         "seed": seed, "a": a, **r})
            print(f"{m:10s} seed={seed}: test_acc={r['test_acc']:.4f} rmia_auc={r['rmia_auc']:.4f} "
                  f"tpr1={r['rmia_tpr1']:.4f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)

    g = res.groupby("method", sort=False)[["test_acc", "rmia_auc", "rmia_tpr1", "loss_auc"]]
    print(f"\nε_total = {eps_total:g} (ε̄ = {eps_total / len(w):g} per attribute, clean labels); "
          f"mean ± sd over seeds {args.seeds}:")
    print((g.mean().map("{:.4f}".format) + " ± " + g.std().map("{:.4f}".format)).to_string())


if __name__ == "__main__":
    main()
