"""
make_fig3_bootstrap.py
======================
Reproduces Figure 3 of the paper (Appendix C.2): bootstrap stability of KL estimates.

For every algorithm and every discovered subgroup (regardless of size), this
script bootstraps the full dataset and re-estimates KL divergence using the
exact same ``score_subgroups`` function from ``src/utils_search.py``.

The goal: show that small subgroups produce highly variable KL estimates
under natural data perturbation (bootstrap).

Subgroups are bucketed by training size:
    <50, 50–100, 100–200, 200–300, 300+

Input: the per-fold subgroup membership files written by Experiment 1 under
``results/Experiment1/results/<method>/subgroup_ids/`` (run ``shell/run_exp1.sh`` first; these
raw outputs are not shipped with the repository).

Usage::

    python figures/make_fig3_bootstrap.py

Outputs are written to ``figures/output/`` (``bootstrap_kl_stability_per_algorithm.{png,pdf}``
is the figure in the paper).
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
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
ROOT = pathlib.Path(__file__).resolve().parents[1]
# Load src/utils_search.py directly so that importing it does not pull in torch/flowtorch via src/__init__.py
import importlib.util
_spec = importlib.util.spec_from_file_location("utils_search", ROOT / "src" / "utils_search.py")
_utils_search = importlib.util.module_from_spec(_spec)
sys.modules["utils_search"] = _utils_search
_spec.loader.exec_module(_utils_search)
score_subgroups = _utils_search.score_subgroups

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR  = ROOT / "dataset"
RESULTS_DIR  = ROOT / "results" / "Experiment1" / "results"
OUTPUT_DIR   = pathlib.Path(__file__).resolve().parent / "output"
ALGORITHMS   = ["cdtree", "rsd", "sd_kl", "sd_mean", "syflow"]

N_BOOTSTRAP    = 300
SEED           = 42

SIZE_BUCKETS = [
    (0,    50,  "<50"),
    (50,  100,  "50–100"),
    (100, 200,  "100–200"),
    (200, 300,  "200–300"),
    (300, np.inf, "300+"),
]

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataset_name_from_file(json_path: pathlib.Path) -> str:
    stem = json_path.stem
    m = re.match(r"^(.+?)_cv\d+_rep\d+_subgroups_ids$", stem)
    if m:
        return m.group(1)
    return stem.split("_cv")[0]


def _load_dataset(dataset_name: str) -> pd.DataFrame:
    path = DATASET_DIR / f"{dataset_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _reconstruct_indices(d: dict, n_rows: int):
    test_idx  = np.array(sorted(d["test_indices"]), dtype=int)
    train_idx = np.setdiff1d(np.arange(n_rows), test_idx)
    return train_idx, test_idx


def _absolute_subgroup_indices(sg: dict, train_idx, test_idx):
    abs_train = train_idx[np.array(sg["train"], dtype=int)]
    abs_test  = test_idx[np.array(sg["test"],  dtype=int)]
    return abs_train, abs_test


def _size_bucket(n: int) -> str:
    for lo, hi, label in SIZE_BUCKETS:
        if lo <= n < hi:
            return label
    return SIZE_BUCKETS[-1][2]


# ---------------------------------------------------------------------------
# Bootstrap for one subgroup
# ---------------------------------------------------------------------------

def bootstrap_one_subgroup(
    y_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    sg_train_relative: np.ndarray,
    sg_test_relative: np.ndarray,
    y_lim: tuple[float, float],
    n_bootstrap: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap the training and test sets, re-estimate KL each time.

    Parameters
    ----------
    y_all : array of shape (n_total,)
        Full target variable.
    train_idx, test_idx : arrays
        Absolute row indices for the original train/test split.
    sg_train_relative, sg_test_relative : arrays
        Relative positions (within train_idx / test_idx) of subgroup members.
    y_lim : (lo, hi)
        Global histogram range.
    n_bootstrap : int
        Number of bootstrap replicates.
    seed : int
        RNG seed.

    Returns
    -------
    DataFrame with columns [bootstrap, kl_train, kl_test, sg_train_size, sg_test_size].
    """
    rng = np.random.default_rng(seed)
    n_train = len(train_idx)
    n_test  = len(test_idx)

    # Precompute boolean masks for subgroup membership (relative to train/test)
    train_sg_mask = np.zeros(n_train, dtype=bool)
    train_sg_mask[sg_train_relative] = True
    test_sg_mask = np.zeros(n_test, dtype=bool)
    test_sg_mask[sg_test_relative] = True

    rows = []
    for b in range(n_bootstrap):
        # Bootstrap: resample positions with replacement
        boot_train_pos = rng.choice(n_train, size=n_train, replace=True)
        boot_test_pos  = rng.choice(n_test,  size=n_test,  replace=True)

        # Target values for this bootstrap
        y_train_boot = y_all[train_idx[boot_train_pos]]
        y_test_boot  = y_all[test_idx[boot_test_pos]]

        # Subgroup membership carries over from original positions
        boot_train_sg = train_sg_mask[boot_train_pos]
        boot_test_sg  = test_sg_mask[boot_test_pos]

        n_sg_train = int(boot_train_sg.sum())
        n_sg_test  = int(boot_test_sg.sum())

        if n_sg_train < 2 or n_sg_test < 2:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # KL on training split — densities estimated from y_train_boot
            res_train, edges_all, edges_sg, p_s, p = score_subgroups(
                y=y_train_boot, y_train=y_train_boot,
                idx=boot_train_sg, split="train", y_lim=y_lim,
            )

            # KL on test split — edges recomputed (edges_all from y_train_boot,
            # edges_subgroup from test subgroup)
            res_test, _, _, _, _ = score_subgroups(
                y=y_test_boot, y_train=y_train_boot,
                idx=boot_test_sg, split="test", y_lim=y_lim,
            )

        rows.append({
            "bootstrap": b,
            "kl_train": res_train["KL_train"],
            "kl_test":  res_test["KL_test"],
            "sg_train_size": n_sg_train,
            "sg_test_size":  n_sg_test,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_bootstrap_experiment(
    output_dir: pathlib.Path | str = OUTPUT_DIR,
    algorithms: list[str] = ALGORITHMS,
    n_bootstrap: int = N_BOOTSTRAP,
) -> pd.DataFrame:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for alg in algorithms:
        sg_dir = RESULTS_DIR / alg / "subgroup_ids"
        if not sg_dir.exists():
            log.warning("Directory not found, skipping: %s", sg_dir)
            continue

        json_files = sorted(sg_dir.glob("*.json"))
        log.info("[%s] %d JSON files found.", alg, len(json_files))

        for jf in json_files:
            dataset_name = _dataset_name_from_file(jf)
            m = re.search(r"_cv(\d+)_", jf.name)
            cv = int(m.group(1)) if m else -1

            try:
                df = _load_dataset(dataset_name)
            except FileNotFoundError:
                log.warning("Dataset file missing for '%s', skipping.", dataset_name)
                continue

            target_col = df.columns[-1]
            y_all = df[target_col].to_numpy(dtype=float)
            n_rows = len(df)
            y_lim = (float(y_all.min()) - 1e-3, float(y_all.max()) + 1e-3)

            with open(jf) as f:
                d = json.load(f)

            train_idx, test_idx = _reconstruct_indices(d, n_rows)
            subgroup_keys = [k for k in d.keys() if k != "test_indices"]

            for sg_id in subgroup_keys:
                sg = d[sg_id]
                sg_train_rel = np.array(sg["train"], dtype=int)
                sg_test_rel  = np.array(sg["test"],  dtype=int)

                n_sg_train = len(sg_train_rel)
                n_sg_test  = len(sg_test_rel)

                if n_sg_train < 2 or n_sg_test < 2:
                    continue

                bucket = _size_bucket(n_sg_train)

                log.info(
                    "  %s | %s | cv%d | sg%s  train=%d test=%d [%s]",
                    alg, dataset_name, cv, sg_id, n_sg_train, n_sg_test, bucket,
                )

                # --- Original (non-bootstrap) KL ---
                y_train_orig = y_all[train_idx]
                y_test_orig  = y_all[test_idx]
                train_sg_mask_orig = np.zeros(len(train_idx), dtype=bool)
                train_sg_mask_orig[sg_train_rel] = True
                test_sg_mask_orig = np.zeros(len(test_idx), dtype=bool)
                test_sg_mask_orig[sg_test_rel] = True

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    orig_train, edges_all, edges_sg, p_s, p = score_subgroups(
                        y=y_train_orig, y_train=y_train_orig,
                        idx=train_sg_mask_orig, split="train", y_lim=y_lim,
                    )
                    orig_test, _, _, _, _ = score_subgroups(
                        y=y_test_orig, y_train=y_train_orig,
                        idx=test_sg_mask_orig, split="test", y_lim=y_lim,
                    )

                # --- Bootstrap ---
                boot_df = bootstrap_one_subgroup(
                    y_all=y_all,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    sg_train_relative=sg_train_rel,
                    sg_test_relative=sg_test_rel,
                    y_lim=y_lim,
                    n_bootstrap=n_bootstrap,
                    seed=SEED,
                )

                if boot_df.empty:
                    continue

                # Summarize
                kl_train_orig = orig_train["KL_train"]
                kl_test_orig  = orig_test["KL_test"]
                kl_train_std  = boot_df["kl_train"].std()
                kl_test_std   = boot_df["kl_test"].std()
                kl_train_iqr  = boot_df["kl_train"].quantile(0.75) - boot_df["kl_train"].quantile(0.25)
                kl_test_iqr   = boot_df["kl_test"].quantile(0.75) - boot_df["kl_test"].quantile(0.25)

                all_rows.append({
                    "algorithm": alg,
                    "dataset": dataset_name,
                    "cv": cv,
                    "subgroup_id": sg_id,
                    "train_sg_size": n_sg_train,
                    "test_sg_size": n_sg_test,
                    "size_bucket": bucket,
                    "kl_train_orig": kl_train_orig,
                    "kl_test_orig":  kl_test_orig,
                    "kl_train_boot_mean": boot_df["kl_train"].mean(),
                    "kl_train_boot_std":  kl_train_std,
                    "kl_train_boot_iqr":  kl_train_iqr,
                    "kl_test_boot_mean":  boot_df["kl_test"].mean(),
                    "kl_test_boot_std":   kl_test_std,
                    "kl_test_boot_iqr":   kl_test_iqr,
                    "kl_train_boot_cv": kl_train_std / max(abs(kl_train_orig), 1e-10),
                    "kl_test_boot_cv":  kl_test_std / max(abs(kl_test_orig), 1e-10),
                })

    if not all_rows:
        log.warning("No subgroups found.")
        return pd.DataFrame()

    master_df = pd.DataFrame(all_rows)

    csv_path = output_dir / "bootstrap_kl_summary.csv"
    master_df.to_csv(csv_path, index=False)
    log.info("Saved summary → %s  (%d subgroups)", csv_path, len(master_df))

    # --- Plot ---
    _plot_bootstrap_stability(master_df, output_dir)
    _plot_bootstrap_stability_per_algorithm(master_df, output_dir)

    return master_df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _save_fig(fig: plt.Figure, name: str, output_dir: pathlib.Path) -> None:
    for ext in ("png", "pdf"):
        p = output_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"  Saved {p}")


def _plot_bootstrap_stability(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Box plots of bootstrap KL variability (std) by subgroup size bucket."""
    KEEP_DATASETS = {
        "ailerons", "bikesharing", "california", "concrete", "elevators",
        "energy", "forestfires", "house", "NavalPropolsion", "puma32h",
        "SkillCraft", "sml2010", "superconductivity", "support2",
        "synchronous", "Thermography", "toxicity", "treasury", "wizmir",
    }
    df = df[df["dataset"].isin(KEEP_DATASETS)].copy()
    bucket_order = [b[2] for b in SIZE_BUCKETS]
    # Only keep buckets that actually have data
    bucket_order = [b for b in bucket_order if b in df["size_bucket"].values]

    rc = {"font.size": 6, "axes.titlesize": 7, "axes.labelsize": 6.5,
          "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
          "legend.fontsize": 5.5, "lines.linewidth": 0.8,
          "axes.linewidth": 0.4, "grid.linewidth": 0.3,
          "xtick.major.width": 0.4, "ytick.major.width": 0.4,
          "xtick.major.size": 2, "ytick.major.size": 2,
          "boxplot.flierprops.markersize": 2,
          "boxplot.medianprops.linewidth": 0.8}

    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))

        for ax, col, title in zip(
            axes,
            ["kl_train_boot_std", "kl_test_boot_std"],
            ["KL (train) bootstrap std", "KL (test) bootstrap std"],
        ):
            sns.boxplot(
                data=df, x="size_bucket", y=col, order=bucket_order,
                palette="viridis", width=0.5, linewidth=0.6,
                fliersize=2, ax=ax,
            )
            ax.set_xlabel("Subgroup training size")
            ax.set_ylabel("Bootstrap std of KL estimate")
            ax.set_title(title, y=1.02)

        fig.tight_layout()
        _save_fig(fig, "bootstrap_kl_stability_by_size", output_dir)
        plt.close(fig)


def _plot_bootstrap_stability_per_algorithm(df: pd.DataFrame, output_dir: pathlib.Path) -> None:
    """Per-algorithm box plots: one row per algorithm, two columns (train/test)."""
    bucket_order = [b[2] for b in SIZE_BUCKETS]
    bucket_order = [b for b in bucket_order if b in df["size_bucket"].values]

    KEEP_DATASETS = {
        "ailerons", "bikesharing", "california", "concrete", "elevators",
        "energy", "forestfires", "house", "NavalPropolsion", "puma32h",
        "SkillCraft", "sml2010", "superconductivity", "support2",
        "synchronous", "Thermography", "toxicity", "treasury", "wizmir",
    }
    ALG_DISPLAY = {
        "cdtree": "AtomSD", "rsd": "RSD", "sd_kl": "SD-KL",
        "sd_mean": r"SD-$\mu$", "syflow": "SyFlow",
    }

    df = df[df["dataset"].isin(KEEP_DATASETS)].copy()
    algorithms = sorted(df["algorithm"].unique())
    n_alg = len(algorithms)

    FONT = 6.5
    BOX_COLOR = "#5a5a8a"

    rc = {"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
          "xtick.labelsize": FONT, "ytick.labelsize": FONT,
          "lines.linewidth": 0.8,
          "axes.linewidth": 0.4, "grid.linewidth": 0.3,
          "xtick.major.width": 0.4, "ytick.major.width": 0.4,
          "xtick.major.size": 2, "ytick.major.size": 2}

    with plt.rc_context(rc):
        fig, axes = plt.subplots(n_alg, 2, figsize=(5.5, 1.4 * n_alg),
                                 squeeze=False)

        for row, alg in enumerate(algorithms):
            alg_df = df[df["algorithm"] == alg]
            alg_buckets = [b for b in bucket_order if b in alg_df["size_bucket"].values]

            for col_idx, (metric, label) in enumerate([
                ("kl_train_boot_std", "KL (train)"),
                ("kl_test_boot_std",  "KL (test)"),
            ]):
                ax = axes[row, col_idx]
                # Always use full bucket_order; empty buckets get an empty array
                box_data = []
                for b in bucket_order:
                    vals = alg_df.loc[alg_df["size_bucket"] == b, metric].dropna().values
                    box_data.append(vals if len(vals) > 0 else np.array([]))
                bp = ax.boxplot(
                    box_data,
                    labels=bucket_order,
                    widths=0.35,
                    patch_artist=True,
                    showfliers=True,
                    flierprops=dict(marker=".", markersize=1.5,
                                    markerfacecolor="#999999",
                                    markeredgecolor="#999999", alpha=0.4),
                    boxprops=dict(facecolor=BOX_COLOR, alpha=0.35,
                                  linewidth=0.5, edgecolor=BOX_COLOR),
                    medianprops=dict(color=BOX_COLOR, linewidth=0.8),
                    whiskerprops=dict(color=BOX_COLOR, linewidth=0.5),
                    capprops=dict(color=BOX_COLOR, linewidth=0.5),
                )

                ax.set_xlabel("Subgroup training size" if row == n_alg - 1 else "")
                ax.set_ylabel("Bootstrap std" if col_idx == 0 else "")
                ax.set_ylim(-0.02, 1.2)
                ax.set_title(f"{ALG_DISPLAY.get(alg, alg)} — {label}", y=1.02)

                if row < n_alg - 1:
                    ax.set_xticklabels([])

        fig.tight_layout()
        _save_fig(fig, "bootstrap_kl_stability_per_algorithm", output_dir)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_bootstrap_experiment()
