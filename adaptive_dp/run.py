"""Command-line entry point for the whole AGENT.md pipeline (input-perturbation version, Adult).

    python -m adaptive_dp.run --smoke
    python -m adaptive_dp.run --seeds 0 1 2 --out results/epsbar3/
    python -m adaptive_dp.run --eps-mean 5 --seeds 0 1 2 --out results/epsbar5/
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import attack, defender
from .allocate import C_SMOOTH, W_HI, W_LO, allocate, normalise
from .data import groups, prepare
from .perturb import EPS_MEAN, EPS_Y, noise_level, perturb
from .train import EPOCHS, train

METHODS = ["nonprivate", "uniform", "adaptive"]
COLUMNS = ["method", "eps_total", "seed", "a", "test_acc", "train_acc", "gap", "rmia_auc", "rmia_tpr1",
           "rmia_tpr01", "loss_auc"]
METRICS = ["test_acc", "train_acc", "gap", "rmia_auc", "rmia_tpr1", "rmia_tpr01", "loss_auc"]

# Choices AGENT.md leaves open, written into the deviation section of summary.md.
IMPLEMENTATION_NOTES = [
    "数据：`data/raw/adult/` 中的 adult.data、adult.test、adult.names 复制自本机已有的 UCI 原始文件"
    "（`privacy-aware-causal-feature-selection/adult/`，与 `feature_selection/DATA/adult/` 的 md5 一致），未重新下载。",
    "运行环境：GH200 节点上的 PyTorch 模块（torch、numpy、pandas、scipy）加上本机 `venv-causal-gh200` 中已装好的 "
    "scikit-learn 1.9.1（通过 PYTHONPATH）；没有新装任何包。AGENT.md 说 CPU 即可，这里用了 1 块 GPU，只影响速度。",
    "随机数：随机排列、所有按种子抽取的子集和输入端加噪都用 `numpy.random.default_rng(seed)`；模型初始化用 "
    "`torch.manual_seed(seed)`，数据打乱用以 seed 初始化的 torch.Generator。目标模型的加噪种子和训练种子都是 seed。",
    "perturb 的抽样顺序：同一个生成器依次处理属性 j = 1…12，再处理标签。广义随机响应先以保留概率抽一次 Bernoulli，"
    "不保留时从其余 k_j − 1 个类别中均匀抽取（实现为在 {0,…,k_j−2} 中抽取后跳过原类别）。k_j 用完整类别表的大小"
    "（adult.names 中的类别加 Unknown），包括数据中从未出现的类别。",
    "攻击者参考模型 i 的训练种子取 4000 + i，与抽样种子相同（AGENT.md 规定了抽样种子 4000 + i 和加噪种子 4100 + i，"
    "没有规定训练种子；§6.1 的校准模型也是抽样与训练用同一个种子）。",
    "选 a 时，1,000 个成员（参考模型 1 独有的训练记录，用加噪前的原始记录和真实标签）和 1,000 个非成员（Z_att）用同一个"
    "种子 4000 的生成器依次抽取；AUC 并列时取较小的 a。这 1,000 个非成员同时也在计算 Pr(z) 的 Z_att 里，AGENT.md 如此规定。",
    "Pr(x | m) 由 float32 logits 转成 float64 后做 softmax；RMIA 的比值与分数用 float64 计算。属性干预在 GPU 上进行，"
    "与单元测试覆盖的 numpy 实现等价（单元测试检查两者相同）。",
    "interventions.csv 中连续属性的 alpha 是缩放后 [0, 1] 中的网格值；类别属性的 alpha 是类别名。",
    "汇总表与对比表中的标准差为样本标准差（ddof = 1）。",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="subsets scaled to 1/5, 2 rotations, 3 continuous grid points, 3 epochs, seed 0")
    p.add_argument("--eps-mean", type=float, default=EPS_MEAN,
                   help="average attribute budget ε̄ (AGENT.md: 3); ε_y stays at 3")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", default="results/")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    if args.smoke:
        args.seeds = [0]
        if args.out == "results/":
            out = Path("results/smoke")
    out.mkdir(parents=True, exist_ok=True)
    scale = 0.2 if args.smoke else 1.0
    epochs = 3 if args.smoke else EPOCHS
    rotations = [1, 2] if args.smoke else [1, 2, 3, 4]
    n_grid = 3 if args.smoke else defender.CONT_GRID
    n_select = int(round(attack.A_SELECT_N * scale))
    log(f"device={device}, smoke={args.smoke}, epochs={epochs}, seeds={args.seeds}, out={out}")

    # ---------------------------------------------------------------- §4 data
    X_np, y_np, splits, params, data_info = prepare(scale=scale, out_dir=out)
    G = groups(params)
    attrs = params["attributes"]
    log(f"data checks passed: {data_info}; split sizes " + str({k: len(v) for k, v in splits.items()}))

    def T_(arr_x, arr_y):
        return (torch.as_tensor(arr_x, dtype=torch.float32, device=device),
                torch.as_tensor(arr_y, dtype=torch.long, device=device))

    X_tr, y_tr = T_(X_np[splits["train"]], y_np[splits["train"]])
    X_te, y_te = T_(X_np[splits["test"]], y_np[splits["test"]])
    X_pd, y_pd = T_(X_np[splits["p_def"]], y_np[splits["p_def"]])
    X_zd, y_zd = T_(X_np[splits["z_def"]], y_np[splits["z_def"]])
    X_pa, y_pa = T_(X_np[splits["p_att"]], y_np[splits["p_att"]])
    X_za, y_za = T_(X_np[splits["z_att"]], y_np[splits["z_att"]])

    # ---------------------------------------------------------------- §6 defender
    t0 = time.time()
    baseline, rec, effects, split, _ = defender.run_defender(
        X_pd, y_pd, X_zd, y_zd, params, rotations, n_grid=n_grid, epochs=epochs, log=log)
    baseline.to_csv(out / "defender_baseline.csv", index=False)
    rec[["j", "attribute", "alpha", "t", "auc", "acc", "ace_mia", "ace_acc"]].to_csv(
        out / "interventions.csv", index=False)
    mean_auc = float(baseline.auc.mean())
    log(f"defender done in {time.time() - t0:.1f}s ({len(rec)} intervention rows): mean proxy AUC={mean_auc:.4f}, "
        f"split-half Spearman(ΔMIA)={split['spearman']:.4f}")
    log("ΔMIA / ΔAcc: " + ", ".join(f"{a['name']}={m:+.4f}/{c:+.4f}" for a, m, c in
                                     zip(attrs, effects.dMIA, effects.dAcc)))
    if not args.smoke:
        if mean_auc <= 0.55:                                            # §10.6
            raise RuntimeError(f"§10.6 failed: mean proxy RMIA AUC = {mean_auc:.4f} ≤ 0.55")
        if not split["spearman"] >= 0.5:                                # §10.7
            raise RuntimeError(f"§10.7 failed: split-half Spearman of ΔMIA = {split['spearman']:.4f} < 0.5")

    # ---------------------------------------------------------------- §7 allocation
    V, U = normalise(effects.dMIA, effects.dAcc)
    w, s = allocate(U, V)
    eps_mean = args.eps_mean
    eps_uni = np.full(len(G), eps_mean)
    eps_ada = eps_mean * w
    for name, e in (("uniform", eps_uni), ("adaptive", eps_ada)):           # §10.4 (Σε_j = 12·ε̄)
        if abs(e.sum() - eps_mean * len(G)) / (eps_mean * len(G)) >= 1e-6:
            raise RuntimeError(f"Σε_j for {name} = {e.sum()} ≠ {eps_mean * len(G)}")
    eps_of = {"uniform": eps_uni, "adaptive": eps_ada}
    eps_total = float(eps_uni.sum() + EPS_Y)
    alloc = pd.DataFrame({
        "j": [a["j"] for a in attrs], "attribute": [a["name"] for a in attrs], "type": [a["type"] for a in attrs],
        "k_j": [len(a["columns"]) if a["type"] == "categorical" else None for a in attrs],
        "n_values": effects.n_values, "dMIA": effects.dMIA, "dAcc": effects.dAcc, "V": V, "U": U, "w": w,
        "eps_uniform": eps_uni, "eps_adaptive": eps_ada,
        "noise_uniform": [noise_level(a["type"], e, len(a["columns"])) for a, e in zip(attrs, eps_uni)],
        "noise_adaptive": [noise_level(a["type"], e, len(a["columns"])) for a, e in zip(attrs, eps_ada)]})
    alloc.to_csv(out / "allocation.csv", index=False)
    log(f"allocation: s={s:.4f}; " + ", ".join(f"{r.attribute}={r.w:.3f}" for r in alloc.itertuples()))

    def fit(method, X, y, seed_perturb, seed_train):
        """Perturb (uniform/adaptive) once, then train; X, y are numpy arrays of original records."""
        if method != "nonprivate":
            X, y = perturb(X, y, params, eps_of[method], EPS_Y, seed_perturb)
        return train(*T_(X, y), seed=seed_train, epochs=epochs)

    # ---------------------------------------------------------------- §8.2 attacker reference models
    Xpa_np, ypa_np = X_np[splits["p_att"]], y_np[splits["p_att"]]
    ref_sets = attack.reference_sets(len(Xpa_np), len(Xpa_np) // 2)
    refs, a_of, a_aucs = {}, {}, {}
    for method in METHODS:
        t0 = time.time()
        refs[method] = [fit(method, Xpa_np[S], ypa_np[S], 4100 + i, 4000 + i)
                        for i, S in enumerate(ref_sets, start=1)]
        a_of[method], a_aucs[method] = attack.select_a(refs[method], ref_sets, X_pa, y_pa, X_za, y_za,
                                                       n_select=n_select)
        log(f"attacker refs [{method}] in {time.time() - t0:.1f}s; a={a_of[method]} "
            f"(AUC {a_aucs[method][a_of[method]]:.4f})")
    pd.DataFrame([{"method": m, "a": a, "auc": v} for m in METHODS for a, v in a_aucs[m].items()]
                 ).to_csv(out / "attacker_a_selection.csv", index=False)

    # ---------------------------------------------------------------- §8.1 targets and evaluation
    Xtr_np, ytr_np = X_np[splits["train"]], y_np[splits["train"]]
    rows = []
    for method in METHODS:
        for seed in args.seeds:
            t0 = time.time()
            target = fit(method, Xtr_np, ytr_np, seed, seed)
            m = attack.evaluate_target(target, refs[method], a_of[method], X_tr, y_tr, X_te, y_te, X_za, y_za)
            rows.append({"method": method, "eps_total": float("inf") if method == "nonprivate" else eps_total,
                         "seed": seed, "a": a_of[method], **m})
            pd.DataFrame(rows, columns=COLUMNS).to_csv(out / "results.csv", index=False)
            log(f"{method:10s} seed={seed}: test={m['test_acc']:.4f} train={m['train_acc']:.4f} "
                f"rmia_auc={m['rmia_auc']:.4f} tpr1={m['rmia_tpr1']:.4f} loss_auc={m['loss_auc']:.4f} "
                f"({time.time() - t0:.1f}s)")
            if not args.smoke and method == "nonprivate" and m["rmia_auc"] <= 0.55:   # §10.8
                raise RuntimeError(f"§10.8 failed: nonprivate rmia_auc = {m['rmia_auc']:.4f} ≤ 0.55")

    meta = {"smoke": args.smoke, "device": str(device), "epochs": epochs, "rotations": rotations,
            "n_grid": n_grid, "scale": scale, "n_select": n_select,
            "split_sizes": {k: len(v) for k, v in splits.items()}, "d_in": params["d_in"],
            "eps_mean": eps_mean, "eps_y": EPS_Y, "eps_total": eps_total, "s": s, "a": a_of,
            "split_half": split, "data": data_info, "n_interventions": int(len(rec) / len(rotations)),
            "total_seconds": time.time() - t_start}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    write_summary(out, pd.read_csv(out / "results.csv"), alloc, baseline, split, meta)
    log(f"done in {time.time() - t_start:.0f}s; wrote {out}")


# ==================================================================== summary.md

def _pm(m, s, d=4):
    return f"{m:.{d}f}" if np.isnan(s) else f"{m:.{d}f} ± {s:.{d}f}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def compare(res, a, b, cols=("test_acc", "rmia_auc", "rmia_tpr1")):
    g = res.groupby("method")[list(cols)]
    mean, sd = g.mean(), g.std(ddof=1)
    out = {}
    for c in cols:
        d = mean.loc[a, c] - mean.loc[b, c]
        thr = sd.loc[a, c] + sd.loc[b, c]
        out[c] = (d, thr, None if np.isnan(thr) else bool(abs(d) > thr))
    return out


def conclusion(res):
    m = res.groupby("method")[METRICS].mean()
    au, an = compare(res, "adaptive", "uniform"), compare(res, "adaptive", "nonprivate")
    s1 = (f"测试准确率：nonprivate {m.loc['nonprivate', 'test_acc'] * 100:.2f}%，uniform "
          f"{m.loc['uniform', 'test_acc'] * 100:.2f}%，adaptive {m.loc['adaptive', 'test_acc'] * 100:.2f}%；"
          f"离线 RMIA 的 AUC 分别为 {m.loc['nonprivate', 'rmia_auc']:.4f}、{m.loc['uniform', 'rmia_auc']:.4f}、"
          f"{m.loc['adaptive', 'rmia_auc']:.4f}，TPR@1%FPR 分别为 {m.loc['nonprivate', 'rmia_tpr1'] * 100:.2f}%、"
          f"{m.loc['uniform', 'rmia_tpr1'] * 100:.2f}%、{m.loc['adaptive', 'rmia_tpr1'] * 100:.2f}%。")
    s2 = (f"adaptive − uniform：Δtest_acc = {au['test_acc'][0] * 100:+.2f} 个百分点，Δrmia_auc = "
          f"{au['rmia_auc'][0]:+.4f}，Δrmia_tpr1 = {au['rmia_tpr1'][0] * 100:+.2f} 个百分点；adaptive − nonprivate："
          f"Δtest_acc = {an['test_acc'][0] * 100:+.2f} 个百分点，Δrmia_auc = {an['rmia_auc'][0]:+.4f}，"
          f"Δrmia_tpr1 = {an['rmia_tpr1'][0] * 100:+.2f} 个百分点。")
    if any(v[2] is None for comp in (au, an) for v in comp.values()):
        s3 = "每组只有 1 个种子，无法比较差值与标准差之和。"
    else:
        big = [f"{name} 的 {c}" for name, comp in (("adaptive − uniform", au), ("adaptive − nonprivate", an))
               for c, v in comp.items() if v[2]]
        s3 = ("绝对值大于两组标准差之和的差值为：" + "、".join(big) + "。") if big else \
            "这 6 个差值的绝对值都没有超过两组标准差之和。"
    return s1 + s2 + s3


def write_summary(out, res, alloc, baseline, split, meta):
    g = res.groupby("method", sort=False)
    mean, sd = g[METRICS].mean(), g[METRICS].std(ddof=1)
    s_rows = [[m, res[res.method == m].eps_total.iloc[0], int(g.size()[m]), res[res.method == m].a.iloc[0]] +
              [_pm(mean.loc[m, c], sd.loc[m, c]) for c in METRICS] for m in METHODS if m in mean.index]
    c_rows = []
    for a, b in (("adaptive", "nonprivate"), ("adaptive", "uniform")):
        comp = compare(res, a, b)
        row = [f"{a} − {b}"]
        for c in ("test_acc", "rmia_auc", "rmia_tpr1"):
            d, thr, big = comp[c]
            row += [f"{d:+.4f}", f"{thr:.4f}", "n/a" if big is None else ("是" if big else "否")]
        c_rows.append(row)
    b_rows = [[int(r.rotation), f"{r.auc:.4f}", f"{r.acc:.4f}", int(r.n_members), int(r.n_nonmembers)]
              for r in baseline.itertuples()]
    al = alloc.sort_values("w")
    a_rows = [[int(r.j), r.attribute, r.type, "" if pd.isna(r.k_j) else int(r.k_j), int(r.n_values),
               f"{r.dMIA:+.5f}", f"{r.dAcc:+.5f}", f"{r.V:.4f}", f"{r.U:.4f}", f"{r.w:.4f}",
               f"{r.eps_uniform:.3f}", f"{r.eps_adaptive:.3f}", f"{r.noise_uniform:.4f}", f"{r.noise_adaptive:.4f}"]
              for r in al.itertuples()]
    notes = []
    if meta["eps_mean"] != EPS_MEAN:
        notes.append(f"按用户指示（2026-11-18），属性平均预算 ε̄ 从 {EPS_MEAN:g} 调大到 {meta['eps_mean']:g}："
                     f"uniform 中所有 ε_j = {meta['eps_mean']:g}，adaptive 中 ε_j = {meta['eps_mean']:g}·w_j，"
                     f"Σ_j ε_j = {meta['eps_mean'] * len(alloc):g}；标签预算 ε_y = {meta['eps_y']:g} 不变；"
                     f"ε_total = {meta['eps_total']:g}。AGENT.md §5.4 把 ε̄ = 3 定为固定值，§2.1 不允许自行增加预算取值，"
                     "这是用户的决定。防御方分析与 w_j 不依赖 ε̄，与 ε̄ = 3 的运行相同。")
    if meta["smoke"]:
        notes.append("冒烟测试：6 个子集、校准集、攻击者参考集和选 a 用的 1,000/1,000 条都按 1/5 缩小；只做 2 次轮换"
                     "（t = 1, 2，参考模型仍是其余 3 个校准模型），连续属性取 3 个网格点，只跑种子 0。折半 Spearman 用轮换 "
                     "{1} 与 {2}。§10.6–10.8 的阈值检查只在完整实验中执行。")
    notes += IMPLEMENTATION_NOTES

    text = "\n".join([
        "# 不加噪 / 输入端均匀加噪 / 输入端自适应加噪：Adult 上的最小验证" + ("（冒烟测试）" if meta["smoke"] else ""),
        "",
        f"设置：|D_train| = |D_test| = {meta['split_sizes']['train']:,}，d_in = {meta['d_in']}，epoch = {meta['epochs']}；"
        f"属性平均预算 ε̄ = {meta['eps_mean']:g}（Σ_j ε_j = {meta['eps_mean'] * len(alloc):g}），标签预算 ε_y = {meta['eps_y']:g}，"
        f"总预算 ε_total = {meta['eps_total']:g}（本地差分隐私，δ = 0）；c = {C_SMOOTH}，w_j ∈ [{W_LO}, {W_HI}]。"
        f"属性干预共 {meta['n_interventions']} 个（j, α）组合。",
        "",
        "## 1. 汇总表（3 个种子的均值 ± 标准差）",
        "",
        _table(["method", "eps_total", "n", "a"] + METRICS, s_rows),
        "",
        f"注：每个方法只有 {meta['split_sizes']['test']:,} 个非成员，TPR@0.1%FPR（rmia_tpr01）只对应 "
        f"{int(meta['split_sizes']['test'] * 0.001)} 个误报点，数值不稳定。a 为攻击者在自己的数据上为该方法选出的值。"
        "所有指标都在加噪前的原始 D_train / D_test（真实标签）上计算。",
        "",
        "## 2. 对比表",
        "",
        "Δ 为两组均值之差；sd_sum 为两组标准差之和；最后一列标出 abs(Δ) 是否大于 sd_sum。",
        "",
        _table(["对比"] + [h for c in ("Δtest_acc", "Δrmia_auc", "Δrmia_tpr1")
                           for h in (c, "sd_sum", "abs(Δ)>sd_sum")], c_rows),
        "",
        "## 3. 防御方分析结果",
        "",
        "不干预时各次轮换的代理在线 RMIA（γ = 2）：",
        "",
        _table(["轮换 t", "AUC_t", "Acc_t", "成员数", "非成员数"], b_rows),
        "",
        f"- 平均 AUC = {baseline.auc.mean():.4f}，平均 Acc = {baseline.acc.mean():.4f}。",
        f"- ΔMIA 的折半 Spearman 相关（轮换 {split['rotations_half1']} 对 {split['rotations_half2']}）= "
        f"{split['spearman']:.4f}。",
        f"- 分配的缩放因子 s = {meta['s']:.6f}。ΔMIA < 0 表示把属性固定后攻击变弱（属性携带成员信号），"
        "ΔAcc < 0 表示固定后准确率下降（属性有用）。",
        "",
        "分配表（按 w_j 从小到大，即噪声从大到小；noise 对类别属性是保留原值的概率，对连续属性是 Laplace 尺度 1/ε_j）：",
        "",
        _table(["j", "attribute", "type", "k_j", "n_values", "dMIA", "dAcc", "V", "U", "w", "eps_uniform",
                "eps_adaptive", "noise_uniform", "noise_adaptive"], a_rows),
        "",
        "## 4. 结论",
        "",
        conclusion(res),
        "",
        "## 5. 偏离记录",
        "",
        "\n".join(f"{i}. {n}" for i, n in enumerate(notes, start=1)),
        "",
    ])
    (out / "summary.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
