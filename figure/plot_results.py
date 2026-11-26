"""Line charts of the non-adaptive attack results: one figure per dataset (Adult, ACSEmployment).

Each figure has three panels against the total budget ε_total: test accuracy, RMIA AUC and TPR@1%FPR.
Lines are the mean over the 10 seeds, the bands ±1 sample sd; the grey line is the no-noise baseline.
Labels are clean: the whole budget goes to the attributes.

    python -B figure/plot_results.py          # writes figure/{adult,acs_employment}.{png,pdf}
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figure"
DATASETS = [("adult", "Adult", ROOT / "results" / "n2000_k8_total_10seeds_labelclean" / "nonadaptive"),
            ("acs_employment", "ACSEmployment (California, 2018)",
             ROOT / "ACSEmployment" / "results" / "nonadaptive_labelclean")]
PANELS = [("test_acc", 100, "Test accuracy (%)  ↑ better"),
          ("rmia_auc", 1, "RMIA AUC  ↓ better"),
          ("rmia_tpr1", 100, "TPR at 1% FPR (%)  ↓ better")]
# categorical slots 1–2 of the reference palette (validated all-pairs); markers as secondary encoding
SERIES = [("uniform", "Uniform", "#2a78d6", "o"), ("adaptive", "Adaptive", "#eb6834", "s")]
BASELINE = "#52514e"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(d):
    runs = sorted(d.glob("epstotal*/results.csv"), key=lambda p: float(p.parent.name[len("epstotal"):]))
    df = pd.concat([pd.read_csv(p).assign(T=float(p.parent.name[len("epstotal"):])) for p in runs])
    stats = df[df.method != "nonprivate"].groupby(["method", "T"]).agg(["mean", "std"])
    base = df[df.method == "nonprivate"].groupby("T").mean(numeric_only=True)
    assert base[[c for c, _, _ in PANELS]].nunique().max() == 1, "nonprivate differs across runs"
    return stats, base.iloc[0], sorted(df["T"].unique()), df.seed.nunique()


def plot(key, title, d):
    stats, base, Ts, n_seeds = load(d)
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
    for ax, (col, scale, ylabel) in zip(axes, PANELS):
        for meth, label, color, marker in SERIES:
            m = stats.loc[meth, (col, "mean")].reindex(Ts) * scale
            s = stats.loc[meth, (col, "std")].reindex(Ts) * scale
            ax.fill_between(Ts, m - s, m + s, color=color, alpha=0.14, linewidth=0)
            ax.plot(Ts, m, color=color, linewidth=1.6, marker=marker, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8, label=label, solid_capstyle="round")
        b = base[col] * scale
        ax.axhline(b, color=BASELINE, linewidth=1.1, label="No noise")
        ax.annotate(f"no noise {b:.3f}" if scale == 1 else f"no noise {b:.2f}", (Ts[0], b),
                    xytext=(0, -4), textcoords="offset points", fontsize=8.5, color=MUTED, va="top")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, max(hi, b + 0.04 * (b - lo)))
        ax.set_title(ylabel, fontsize=10.5, color=TEXT, loc="left")
        ax.set_xlabel("Total privacy budget ε_total")
        ax.set_xlim(0, 104)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, bbox_to_anchor=(0.995, 1.0))
    fig.suptitle(f"{title}: non-adaptive attack, clean labels, mean ± 1 sd over {n_seeds} seeds",
                 x=0.01, y=0.985, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{key}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote figure/{key}.png and .pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for key, title, d in DATASETS:
        plot(key, title, d)
