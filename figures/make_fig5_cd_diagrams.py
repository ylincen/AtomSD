"""
make_fig5_cd_diagrams.py
========================
Reproduces Figure 5 of the paper (Appendix C.5): critical-difference diagrams
(Demšar, 2006) for constrained subgroup discovery, one panel per support threshold.

For each threshold the methods are ranked per dataset by mean held-out KL divergence
(higher is better, missing subgroups count as 0), a Friedman test with the Iman-Davenport
correction is run, and the Nemenyi critical difference is drawn.

Input: the per-fold result tables written by ``reproduce_results.ipynb`` when it builds
Tables 1 and 3::

    results/Experiment3/result_exp3_fixed_mixint_raw.csv   (absolute thresholds alpha)
    results/Experiment3/result_exp3_ratio_mixint_raw.csv   (ratio thresholds alpha')

Run the notebook first, then::

    python figures/make_fig5_cd_diagrams.py

Output: ``figures/output/cd_diagram_{fixed,ratio}.{png,pdf}``.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "Experiment3"
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "output"

# Method labels as written by reproduce_results.ipynb -> labels used in the figure.
DISPLAY_METHOD = {
    "AtomSD": "AtomSD",
    "RSD": "RSD",
    r"SD$-\text{KL}$": "SD-KL",
    r"SD$-{\mu}$": r"SD-$\mu$",
    "SyFlow": "SyFlow",
}
METHOD_ORDER = list(DISPLAY_METHOD)

# Nemenyi critical values q_alpha for alpha = 0.05 (Demšar 2006, Table 5), k = number of methods.
QALPHA_005 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def load_and_pivot(csv_path, method_order):
    """Average held-out KL over folds, pivot to (dataset, alpha) x method, missing -> 0."""
    df = pd.read_csv(csv_path)
    avg = df.groupby(["dataset", "alpha", "method"])["KL_test"].mean().reset_index()
    pivot = avg.pivot_table(index=["dataset", "alpha"], columns="method", values="KL_test")
    mo = [m for m in method_order if m in pivot.columns]
    return pivot.fillna(0.0)[mo], mo


def compute_ranks(pivot):
    """Rank per row; higher KL = rank 1."""
    return pivot.rank(axis=1, ascending=False, method="average")


def friedman_test(rank_matrix):
    """Friedman test with the Iman-Davenport F correction. rank_matrix: (n_datasets, k_methods)."""
    n, k = rank_matrix.shape
    rank_means = rank_matrix.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * np.sum((rank_means - (k + 1) / 2) ** 2)
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
    p_value = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (n - 1))
    return chi2, ff, p_value


def nemenyi_cd(k, n, alpha=0.05):
    return QALPHA_005[k] * np.sqrt(k * (k + 1) / (6 * n))


def get_cliques(avg_ranks, cd):
    """Maximal groups of methods whose average ranks differ by less than CD."""
    sorted_indices = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_indices]
    k = len(avg_ranks)
    cliques = [(sorted_indices[i], sorted_indices[j])
               for i in range(k) for j in range(i + 1, k) if sorted_ranks[j] - sorted_ranks[i] < cd]
    maximal = []
    for i, (a, b) in enumerate(cliques):
        ra, rb = avg_ranks[a], avg_ranks[b]
        subsumed = any(
            avg_ranks[c] <= ra and avg_ranks[d] >= rb and (avg_ranks[c] < ra or avg_ranks[d] > rb)
            for j, (c, d) in enumerate(cliques) if j != i
        )
        if not subsumed:
            maximal.append((a, b))
    return maximal


def draw_cd_diagram(avg_ranks, cd, method_names, title, ax):
    k = len(avg_ranks)
    sorted_idx = np.argsort(avg_ranks)
    low, high = 1, k
    width = high - low

    def _pos(rank):
        return (rank - low) / width

    ax.set_xlim(-0.1, 1.12)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=9, fontweight="bold", transform=ax.transAxes)

    y_axis = 0.6
    ax.plot([_pos(low), _pos(high)], [y_axis, y_axis], "k-", lw=1.0, clip_on=False)
    for tick in range(1, k + 1):
        x = _pos(tick)
        ax.plot([x, x], [y_axis - 0.02, y_axis + 0.02], "k-", lw=1.0, clip_on=False)
        ax.text(x, y_axis + 0.05, str(tick), ha="center", va="bottom", fontsize=7)

    cd_x_start, cd_x_end, y_cd = _pos(low), _pos(low + cd), y_axis + 0.15
    ax.plot([cd_x_start, cd_x_end], [y_cd, y_cd], "k-", lw=1.5, clip_on=False)
    for xx in (cd_x_start, cd_x_end):
        ax.plot([xx, xx], [y_cd - 0.02, y_cd + 0.02], "k-", lw=1.5, clip_on=False)
    ax.text((cd_x_start + cd_x_end) / 2, y_cd + 0.03, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)

    n_left = (k + 1) // 2
    left_indices, right_indices = sorted_idx[:n_left], sorted_idx[n_left:]
    y_start, y_step = y_axis - 0.06, 0.12

    for i, idx in enumerate(left_indices):
        rank, x_rank, y_name, x_name = avg_ranks[idx], _pos(avg_ranks[idx]), y_start - i * y_step, -0.05
        ax.plot([x_rank, x_rank], [y_axis, y_name], "k-", lw=0.7, clip_on=False)
        ax.plot([x_name, x_rank], [y_name, y_name], "k-", lw=0.7, clip_on=False)
        ax.text(x_name - 0.02, y_name, f"{method_names[idx]} ({rank:.2f})", ha="right", va="center", fontsize=7.5)

    for i, idx in enumerate(right_indices):
        rank, x_rank, y_name, x_name = avg_ranks[idx], _pos(avg_ranks[idx]), y_start - i * y_step, 1.07
        ax.plot([x_rank, x_rank], [y_axis, y_name], "k-", lw=0.7, clip_on=False)
        ax.plot([x_name, x_rank], [y_name, y_name], "k-", lw=0.7, clip_on=False)
        ax.text(x_name + 0.02, y_name, f"({rank:.2f}) {method_names[idx]}", ha="left", va="center", fontsize=7.5)

    y_bar = y_start - max(len(left_indices), len(right_indices)) * y_step - 0.02
    for ci, (a, b) in enumerate(get_cliques(avg_ranks, cd)):
        ax.plot([_pos(avg_ranks[a]), _pos(avg_ranks[b])], [y_bar - ci * 0.04] * 2, "k-", lw=3.0, alpha=0.7, clip_on=False)


def make_cd_diagrams(csv_path, alpha_symbol, alpha_label_fmt, output_prefix):
    pivot, mo = load_and_pivot(csv_path, METHOD_ORDER)
    method_names = [DISPLAY_METHOD[m] for m in mo]
    alphas = sorted(pivot.index.get_level_values(1).unique())
    k = len(mo)

    fig, axes = plt.subplots(len(alphas), 1, figsize=(5.5, 1.6 * len(alphas)))
    if len(alphas) == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alphas):
        sub = pivot.xs(alpha, level="alpha")
        rank_matrix = compute_ranks(sub).values
        _, _, p_val = friedman_test(rank_matrix)
        cd = nemenyi_cd(k, len(sub))
        title = f"${alpha_symbol} = {alpha_label_fmt(alpha)}$  (Friedman $p = {p_val:.1e}$)"
        draw_cd_diagram(rank_matrix.mean(axis=0), cd, method_names, title, ax)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = OUTPUT_DIR / f"{output_prefix}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    make_cd_diagrams(RESULTS_DIR / "result_exp3_fixed_mixint_raw.csv", r"\alpha", lambda a: str(int(a)), "cd_diagram_fixed")
    make_cd_diagrams(RESULTS_DIR / "result_exp3_ratio_mixint_raw.csv", r"\alpha'", lambda a: f"{a:.2f}", "cd_diagram_ratio")
