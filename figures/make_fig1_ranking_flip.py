"""
make_fig1_ranking_flip.py
=========================
Reproduces Figure 1 of the paper (also discussed in Appendix C.1): ranking error of
support-weighted KL estimation with finite samples.

Two subgroups are compared against a 5-mode Gaussian-mixture background:

* Subgroup A (true deviation, support 8%): same 5 modes as the background but with
  different mixture weights, so KL(A || background) > 0.
* Subgroup B (spurious, support 15%): drawn from the background itself, so KL = 0.

For each sample size k of subgroup A (B gets round(k * 0.15 / 0.08) points), the
support-weighted KL score of both subgroups is estimated from fresh i.i.d. draws, and the
"ranking error rate" is the fraction of repetitions in which the spurious subgroup B
outranks the true subgroup A.

Usage (from the repository root or from this directory)::

    python figures/make_fig1_ranking_flip.py

Output: ``figures/output/fig1_multimodal_ranking_flip.{png,pdf}``.
Requires numpy, pandas, matplotlib and seaborn only.
"""

from __future__ import annotations

import pathlib
import warnings
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kl_estimation import build_histogram_density, evaluate_density_at_points, sample_wise_kl

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent / "output"
N_REPETITIONS = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, name: str, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = output_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved {path}")


def _sample_gmm(rng: np.random.Generator, n: int, modes: list[float], stds: list[float],
                weights: list[float]) -> np.ndarray:
    """Draw *n* samples from a 1-D Gaussian mixture (weights are normalised internally)."""
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    component = rng.choice(len(modes), size=n, p=weights)
    return np.array([rng.normal(modes[c], stds[c]) for c in component])


def _gmm_pdf(x: np.ndarray, modes: list[float], stds: list[float], weights: list[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    density = np.zeros_like(x, dtype=float)
    for m, s, wi in zip(modes, stds, w):
        density += wi * np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
    return density


# ---------------------------------------------------------------------------
# Fresh-draw KL experiment
# ---------------------------------------------------------------------------

def draw_experiment(
    subgroup_modes: list[float], subgroup_stds: list[float], subgroup_weights: list[float],
    bg_modes: list[float], bg_stds: list[float], bg_weights: list[float],
    histogram_options: dict[str, Any],
    k_values: list[int],
    n_repetitions: int = N_REPETITIONS,
    n_bg_reference: int = 10_000,
    seed: int = 42,
) -> tuple[pd.DataFrame, float]:
    """For each k and repetition, draw k points from the subgroup mixture P and estimate
    KL(P || Q) against a fixed histogram density Q built from a large background draw.

    Returns a DataFrame with columns [k, repetition, kl_train] and a large-sample reference KL.
    """
    rng = np.random.default_rng(seed)

    # Q always uses Laplace smoothing so that no bin of Q has zero density.
    q_options = {**histogram_options, "smoothing": "laplace"}
    bg_data = _sample_gmm(rng, n_bg_reference, bg_modes, bg_stds, bg_weights)
    q_density, bin_edges_q = build_histogram_density(bg_data, histogram_options=q_options)

    ref_data = _sample_gmm(rng, n_bg_reference, subgroup_modes, subgroup_stds, subgroup_weights)
    ref_p_density, _ = build_histogram_density(ref_data, bin_edges=bin_edges_q, histogram_options=histogram_options)
    reference_kl, _ = sample_wise_kl(
        evaluate_density_at_points(ref_data, ref_p_density, bin_edges_q),
        evaluate_density_at_points(ref_data, q_density, bin_edges_q),
    )

    rows: list[dict[str, Any]] = []
    for k in k_values:
        for rep in range(n_repetitions):
            p_data = _sample_gmm(rng, k, subgroup_modes, subgroup_stds, subgroup_weights)
            # P is binned on Q's grid so that both densities share the same support.
            p_density, _ = build_histogram_density(p_data, bin_edges=bin_edges_q, histogram_options=histogram_options)
            kl_hat, _ = sample_wise_kl(
                evaluate_density_at_points(p_data, p_density, bin_edges_q),
                evaluate_density_at_points(p_data, q_density, bin_edges_q),
            )
            rows.append({"k": k, "repetition": rep, "kl_train": kl_hat})

    return pd.DataFrame(rows, columns=["k", "repetition", "kl_train"]), reference_kl


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_ranking_flip_rate(
    results_a: pd.DataFrame, results_b: pd.DataFrame,
    coverage_a: float, coverage_b: float,
    label_a: str, label_b: str,
    output_dir: pathlib.Path, fig_name: str,
    bg_modes: list[float], bg_stds: list[float], bg_weights: list[float],
    a_modes: list[float], a_stds: list[float], a_weights: list[float],
) -> plt.Figure:
    """Three panels: (1) subgroup vs. background densities, (2) support-weighted KL estimates
    with 95% quantile bands, (3) ranking error rate (fraction of repetitions with B > A)."""
    k_values_a = sorted(results_a["k"].unique())
    k_set_b = set(results_b["k"].unique())

    plot_k, flip_rates = [], []
    mean_score_a, mean_score_b = [], []
    q025_a, q975_a, q025_b, q975_b = [], [], [], []

    for k_a in k_values_a:
        k_b = max(1, round(k_a * coverage_b / coverage_a))
        if k_b not in k_set_b:
            continue
        plot_k.append(k_a)
        score_a = coverage_a * results_a.loc[results_a["k"] == k_a, "kl_train"].to_numpy()
        score_b = coverage_b * results_b.loc[results_b["k"] == k_b, "kl_train"].to_numpy()
        mean_score_a.append(float(score_a.mean()))
        mean_score_b.append(float(score_b.mean()))
        q025_a.append(float(np.percentile(score_a, 2.5)))
        q975_a.append(float(np.percentile(score_a, 97.5)))
        q025_b.append(float(np.percentile(score_b, 2.5)))
        q975_b.append(float(np.percentile(score_b, 97.5)))
        flip_rates.append(float((score_b > score_a).mean()))

    COLOR_A, COLOR_B, COLOR_FLIP = "#762a83", "#1b7837", "#b2182b"
    FONT_TICK, FONT_LEGEND = 7.5, 6.5
    rc = {"font.size": 5, "axes.titlesize": FONT_LEGEND, "axes.labelsize": FONT_LEGEND,
          "xtick.labelsize": FONT_TICK, "ytick.labelsize": FONT_TICK, "legend.fontsize": FONT_LEGEND,
          "lines.linewidth": 0.8, "lines.markersize": 2, "axes.linewidth": 0.4, "grid.linewidth": 0.3,
          "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 2, "ytick.major.size": 2}

    with plt.rc_context(rc):
        fig, (ax_dens, ax_kl, ax_flip) = plt.subplots(1, 3, figsize=(5.5, 2.2))

        x = np.linspace(min(bg_modes) - 1.2, max(bg_modes) + 1.2, 800)
        ax_dens.plot(x, _gmm_pdf(x, bg_modes, bg_stds, bg_weights), color=COLOR_B, linestyle=":",
                     label=f"Background / {label_b}", zorder=3)
        ax_dens.plot(x, _gmm_pdf(x, a_modes, a_stds, a_weights), color=COLOR_A, alpha=0.6, linestyle="--",
                     label=label_a, zorder=4)
        ax_dens.set_xlabel("Value")
        ax_dens.set_ylabel("(Conditional) density of Y")
        ax_dens.set_title("Subgroup vs. Background", y=1.22)
        ax_dens.legend(framealpha=0.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)

        ax_kl.fill_between(plot_k, q025_a, q975_a, color=COLOR_A, alpha=0.15)
        ax_kl.fill_between(plot_k, q025_b, q975_b, color=COLOR_B, alpha=0.15)
        ax_kl.plot(plot_k, mean_score_a, marker="o", color=COLOR_A, label=f"{label_a} (s={coverage_a}, KL>0)")
        ax_kl.plot(plot_k, mean_score_b, marker="s", color=COLOR_B, linestyle="--",
                   label=f"{label_b} (s={coverage_b}, KL=0)")
        ax_kl.set_xlabel("Subgroup A sample size k")
        ax_kl.set_ylabel("support × KL estimate")
        ax_kl.set_title("Support-weighted KL (mean ± 95%)", y=1.22)
        ax_kl.legend(framealpha=0.8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1)

        ax_flip.plot(plot_k, flip_rates, marker="D", color=COLOR_FLIP)
        ax_flip.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax_flip.set_xlabel("Subgroup A sample size k")
        ax_flip.set_ylabel("Error rate")
        ax_flip.set_title("Ranking Error Rate", y=1.22)
        ax_flip.set_ylim(-0.02, 1.05)

        fig.tight_layout()
        _save_fig(fig, fig_name, output_dir)
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(output_dir: pathlib.Path = OUTPUT_DIR, n_repetitions: int = N_REPETITIONS) -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})

    coverage_a = 0.08   # support of the true subgroup A
    coverage_b = 0.15   # support of the spurious subgroup B

    bg_modes, bg_stds, bg_weights = [-3.0, -1.5, 0.0, 1.5, 3.0], [0.3] * 5, [1.0] * 5
    a_modes, a_stds, a_weights = bg_modes, [0.3] * 5, [0.35, 0.05, 0.20, 0.05, 0.35]

    histogram_options: dict[str, Any] = {"bin_method": "fd", "n_bins": 20, "smoothing": "laplace", "epsilon": 1e-6}
    k_values = [10, 20, 30, 40, 50, 60, 70, 80, 100]
    k_values_b = [max(1, round(k * coverage_b / coverage_a)) for k in k_values]

    print("Subgroup A (true deviation) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        results_a, ref_kl_a = draw_experiment(a_modes, a_stds, a_weights, bg_modes, bg_stds, bg_weights,
                                              histogram_options, k_values, n_repetitions, seed=42)
    print(f"  large-sample KL_A = {ref_kl_a:.4f}")

    print("Subgroup B (spurious, drawn from the background) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        results_b, ref_kl_b = draw_experiment(bg_modes, bg_stds, bg_weights, bg_modes, bg_stds, bg_weights,
                                              histogram_options, k_values_b, n_repetitions, seed=43)
    print(f"  large-sample KL_B = {ref_kl_b:.4f}")

    fig = plot_ranking_flip_rate(
        results_a, results_b, coverage_a, coverage_b,
        label_a="Subgroup A (true)", label_b="Subgroup B (spurious)",
        output_dir=output_dir, fig_name="fig1_multimodal_ranking_flip",
        bg_modes=bg_modes, bg_stds=bg_stds, bg_weights=bg_weights,
        a_modes=a_modes, a_stds=a_stds, a_weights=a_weights,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
