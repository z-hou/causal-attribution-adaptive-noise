"""Summarise the Cardio non-adaptive runs: mean ± sd over seeds, and adaptive − uniform (paired by seed).

    python -B Cardio/summarize.py [results dir, default Cardio/results/nonadaptive_labelclean]

Writes <dir>/summary.csv (one row per ε_total and method) and <dir>/comparison.csv (adaptive − uniform),
and prints both as Markdown tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "nonadaptive_labelclean"
METRICS = [("test_acc", 100), ("rmia_auc", 1), ("rmia_tpr1", 100), ("loss_auc", 1)]   # (column, scale)


def main():
    runs = sorted(OUT.glob("epstotal*/results.csv"), key=lambda p: float(p.parent.name[len("epstotal"):]))
    if not runs:
        raise SystemExit(f"no results under {OUT}")
    rows, comp, allocs = [], [], []
    for path in runs:
        T = float(path.parent.name[len("epstotal"):])
        r = pd.read_csv(path)
        allocs.append(pd.read_csv(path.parent / "allocation.csv"))
        for m, g in r.groupby("method", sort=False):
            row = {"eps_total": "inf" if m == "nonprivate" else f"{T:g}", "method": m, "n_seeds": len(g),
                   "a": ",".join(f"{a:g}" for a in sorted(g.a.unique()))}
            for k, s in METRICS:
                row[f"{k}_mean"], row[f"{k}_sd"] = g[k].mean() * s, g[k].std() * s
            rows.append(row)
        u = r[r.method == "uniform"].sort_values("seed")
        a = r[r.method == "adaptive"].sort_values("seed")
        assert (u.seed.to_numpy() == a.seed.to_numpy()).all()
        c = {"eps_total": T}
        for k, s in METRICS:
            d = (a[k].to_numpy() - u[k].to_numpy()) * s
            c[f"d_{k}"], c[f"p_{k}"] = d.mean(), ttest_rel(a[k], u[k]).pvalue
            c[f"wins_{k}"] = f"{int((d > 0).sum())}/{len(d)}"
        comp.append(c)

    summ = pd.DataFrame(rows)
    nonpriv = summ[summ.method == "nonprivate"]
    cols = [f"{k}_{s}" for k, _ in METRICS for s in ("mean", "sd")]
    if not np.allclose(nonpriv[cols].to_numpy(), nonpriv[cols].to_numpy()[0]):
        print("WARNING: nonprivate results differ between ε_total runs")
    summ = pd.concat([nonpriv.iloc[:1], summ[summ.method != "nonprivate"]], ignore_index=True)
    summ.to_csv(OUT / "summary.csv", index=False)
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "comparison.csv", index=False)

    base = allocs[0][["j", "attribute", "dMIA", "dAcc", "w"]]
    same = all(x[["dMIA", "dAcc", "w"]].equals(base[["dMIA", "dAcc", "w"]]) for x in allocs)
    print(f"allocation identical across ε_total runs: {same}")

    print("\n| ε_total | method | test_acc (%) | rmia_auc | TPR@1%FPR (%) | loss_auc |")
    print("| --- | --- | --- | --- | --- | --- |")
    for _, x in summ.iterrows():
        print(f"| {x.eps_total} | {x.method} | {x.test_acc_mean:.2f} ± {x.test_acc_sd:.2f} | "
              f"{x.rmia_auc_mean:.4f} ± {x.rmia_auc_sd:.4f} | {x.rmia_tpr1_mean:.2f} ± {x.rmia_tpr1_sd:.2f} | "
              f"{x.loss_auc_mean:.4f} ± {x.loss_auc_sd:.4f} |")

    print("\nadaptive − uniform, paired by seed (p: paired t-test; A>U: seeds where adaptive is higher)")
    print("| ε_total | Δtest_acc (pp) | p | A>U | Δrmia_auc | p | A>U | ΔTPR@1% (pp) | p | A>U |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, x in comp.iterrows():
        print(f"| {x.eps_total:g} | {x.d_test_acc:+.2f} | {x.p_test_acc:.3f} | {x.wins_test_acc} | "
              f"{x.d_rmia_auc:+.4f} | {x.p_rmia_auc:.3f} | {x.wins_rmia_auc} | "
              f"{x.d_rmia_tpr1:+.2f} | {x.p_rmia_tpr1:.3f} | {x.wins_rmia_tpr1} |")

    print("\nallocation (defender analysis, same for every ε_total):")
    print(base.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
