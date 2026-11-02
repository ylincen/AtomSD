from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns

# utility function to get statistics for Experiment 3 (Tables 1 and 3)
def get_statistics(df):
    method_cols = [
        "AtomSD",
        "RSD",
        r"SD$-\text{KL}$",
        r"SD$-{\mu}$",
        "SyFlow",
    ]

    out = df.copy()

    # ranks
    ranks = out[method_cols].rank(axis=1, method="min", ascending=False)
    ranks.columns = [f"{c}_rank" for c in method_cols]
    out = pd.concat([out, ranks], axis=1)

    # formatted table with bold best values
    formatted = out.copy()
    best_mask = ranks.eq(1).rename(columns={f"{c}_rank": c for c in method_cols})

    for c in method_cols:
        formatted[c] = [
            rf"\textbf{{{v:.3f}}}" if is_best else f"{v:.3f}"
            for v, is_best in zip(out[c], best_mask[c])
        ]

    # average rank per alpha
    avg_rank_per_alpha = (
        out.groupby("alpha")[[f"{c}_rank" for c in method_cols]]
        .mean()
        .rename(columns={f"{c}_rank": c for c in method_cols})
        .reset_index()
        .round(2)
    )

    # count of best entries per alpha
    best_count_per_alpha = (
        pd.concat([out[["alpha"]], best_mask], axis=1)
        .groupby("alpha")[method_cols]
        .sum()
        .reset_index()
    )
    total = best_count_per_alpha[method_cols].sum(axis=0).to_frame().T
    best_count_per_alpha['dataset'] = r"\#Best"
    total['dataset'] = 'Total'
    total['alpha'] = None
    avg_rank_per_alpha['dataset'] = r"Avg.\ Rank"
    final_df = pd.concat([formatted[['dataset', 'alpha']+method_cols],avg_rank_per_alpha, best_count_per_alpha, total], axis=0)
    if (final_df['alpha'] > 1).all():
        final_df['alpha'] = final_df['alpha'].astype(int)
    final_df.replace("nan", "0.000", inplace=True)
    return final_df


METRICS = ["A", "B", "C", "D", ""]
DEFAULT_PALETTE = {
    "A": "#2D6A6F",
    "B": "#6C7AAB",
    "C": "#C27A3A",
    "D": "#8B5E83",
    "E": "#B85C4B",
}
RESERVED_COLUMNS = {"name", "dataset", "alpha"}

BACKGROUND_COLOR = "#FFFFFF"
AXIS_BACKGROUND = "#FCFAF5"
TEXT_COLOR = "#231F1C"
MUTED_TEXT = "#6C625A"
GRID_COLOR = "#DDD2C3"
SPINE_COLOR = "#9D8F83"
ZERO_LINE_COLOR = "#7E6B5C"

def configure_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        rc={
            "font.family": "DejaVu Serif",
            "figure.facecolor": BACKGROUND_COLOR,
            "axes.facecolor": AXIS_BACKGROUND,
            "axes.edgecolor": SPINE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.9,
        },
    )


def infer_metrics(df: pd.DataFrame, metrics: Sequence[str] | None = None) -> list[str]:
    if "alpha" not in df.columns:
        raise KeyError("Dataframe must include an 'alpha' column.")

    inferred = list(metrics) if metrics is not None else [col for col in df.columns if col not in RESERVED_COLUMNS]
    missing = [metric for metric in inferred if metric not in df.columns]
    if missing:
        raise KeyError(f"Missing metric columns: {missing}")
    if not inferred:
        raise ValueError("No metric columns were provided or inferred.")
    return inferred


def alpha_label(alpha: float) -> str:
    if alpha >= 1:
        return f"{alpha:.0f}"
    return f"{alpha:.2f}"


def melt_for_plot(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    id_vars = [column for column in ["name", "dataset", "alpha"] if column in df.columns]
    plot_df = df.melt(
        id_vars=id_vars,
        value_vars=list(metrics),
        var_name="metric",
        value_name="value",
    )
    plot_df["alpha"] = pd.to_numeric(plot_df["alpha"], errors="coerce")
    plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
    return plot_df


def resolve_palette(
    metrics: Sequence[str],
    palette: dict[str, str] | None = None,
) -> dict[str, str]:
    merged = {}
    palette = palette or {}
    for metric in metrics:
        if metric in palette:
            merged[metric] = palette[metric]
        elif metric in DEFAULT_PALETTE:
            merged[metric] = DEFAULT_PALETTE[metric]

    missing = [metric for metric in metrics if metric not in merged]
    if missing:
        fallback_colors = sns.color_palette("flare", n_colors=len(missing)).as_hex()
        for metric, color in zip(missing, fallback_colors):
            merged[metric] = color

    return merged


def resolve_legend_names(
    metrics: Sequence[str],
    legend_names: dict[str, str] | None = None,
) -> dict[str, str]:
    legend_names = legend_names or {}
    return {metric: legend_names.get(metric, metric) for metric in metrics}


def compute_offsets(metrics: Sequence[str], span: float = 0.72) -> dict[str, float]:
    metrics = list(metrics)
    if not metrics:
        return {}
    if len(metrics) == 1:
        return {metrics[0]: 0.0}

    centers = np.linspace(-span / 2, span / 2, len(metrics))
    return dict(zip(metrics, centers))


def metric_geometry(metrics: Sequence[str], span: float = 0.72) -> dict[str, float | dict[str, float]]:
    offsets = compute_offsets(metrics, span=span)
    offset_values = np.array(list(offsets.values()), dtype=float)
    spacing = np.diff(np.sort(offset_values)).min() if len(offset_values) > 1 else span * 0.5

    return {
        "offsets": offsets,
        "box_width": min(0.12, spacing * 0.72),
        "point_jitter": min(0.028, spacing * 0.18),
    }


def build_legend_handles(
    metrics: Sequence[str],
    palette: dict[str, str],
    legend_names: dict[str, str],
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=palette[metric],
            markeredgecolor=AXIS_BACKGROUND,
            markeredgewidth=0.8,
            markersize=8,
            label=legend_names[metric],
        )
        for metric in metrics
    ]


def finite_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def draw_slim_boxplot(
    ax: plt.Axes,
    values: np.ndarray,
    x_center: float,
    color: str,
    width: float,
) -> None:
    if len(values) == 0:
        return

    ax.boxplot(
        [values],
        positions=[x_center],
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
        boxprops={
            "facecolor": mcolors.to_rgba(color, 0.32),
            "edgecolor": color,
            "linewidth": 1.15,
            "zorder": 3,
        },
        whiskerprops={"color": color, "linewidth": 1.0, "zorder": 3},
        capprops={"color": color, "linewidth": 1.0, "zorder": 3},
        medianprops={"color": TEXT_COLOR, "linewidth": 1.25, "zorder": 4},
    )


def draw_centered_points(
    ax: plt.Axes,
    values: np.ndarray,
    x_center: float,
    color: str,
    *,
    jitter: float,
    rng: np.random.Generator,
    size: float = 16,
    edge_width: float = 0.5,
) -> None:
    if len(values) == 0:
        return

    x_values = x_center + rng.uniform(-jitter, jitter, size=len(values))
    ax.scatter(
        x_values,
        values,
        s=size,
        color=color,
        alpha=0.50,
        edgecolors=AXIS_BACKGROUND,
        linewidths=edge_width,
        zorder=4,
    )


def style_publication_axis(
    ax: plt.Axes,
    alpha_levels: Sequence[float],
    *,
    x_label: str = "alpha",
    y_label: str = "metric value",
    tick_size: float = 11,
) -> None:
    ax.set_facecolor(AXIS_BACKGROUND)
    ax.set_xticks(range(len(alpha_levels)))
    ax.set_xticklabels([alpha_label(alpha) for alpha in alpha_levels], fontsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.grid(axis="y", linestyle=(0, (2, 4)), linewidth=0.9, alpha=0.95)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=1.6, alpha=0.95, zorder=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE_COLOR)
    ax.spines["bottom"].set_color(SPINE_COLOR)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.set_xlabel(x_label, fontsize=tick_size, fontweight="semibold")
    ax.set_ylabel(y_label, fontsize=tick_size, fontweight="semibold")
    sns.despine(ax=ax, top=True, right=True)


def apply_scientific_formatter(ax: plt.Axes, *, power_limits: tuple[int, int] = (-2, 2)) -> None:
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits(power_limits)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=power_limits)


def draw_grouped_boxplot_panel(
    ax: plt.Axes,
    plot_df: pd.DataFrame,
    alpha_levels: Sequence[float],
    metrics: Sequence[str],
    palette: dict[str, str],
    *,
    span: float = 0.5,
    show_points: bool = True,
    point_size: float = 16,
    edge_width: float = 0.5,
    seed: int = 123,
) -> None:
    geometry = metric_geometry(metrics, span=span)
    offsets = geometry["offsets"]
    pos_map = {alpha: idx for idx, alpha in enumerate(alpha_levels)}
    rng = np.random.default_rng(seed)

    for metric in metrics:
        metric_df = plot_df.loc[plot_df["metric"] == metric]
        for alpha in alpha_levels:
            values = finite_values(metric_df.loc[metric_df["alpha"] == alpha, "value"])
            if len(values) == 0:
                continue

            x_center = pos_map[alpha] + float(offsets[metric])
            draw_slim_boxplot(
                ax,
                values,
                x_center,
                palette[metric],
                float(geometry["box_width"]),
            )
            if show_points:
                draw_centered_points(
                    ax,
                    values,
                    x_center,
                    palette[metric],
                    jitter=float(geometry["point_jitter"]),
                    rng=rng,
                    size=point_size,
                    edge_width=edge_width,
                )

    ax.set_xlim(-0.6, len(alpha_levels) - 0.4)


def make_boxplot(
    df: pd.DataFrame,
    output_path: Path,
    *,
    metrics: Sequence[str] | None = None,
    palette: dict[str, str] | None = None,
    legend_names: dict[str, str] | None = None,
    x_label: str = "alpha",
    y_label: str = "metric value",
    tick_size: float = 11,
    show_points: bool = True,
) -> None:
    configure_theme()

    metrics = infer_metrics(df, metrics)
    plot_df = melt_for_plot(df, metrics)
    plot_df = plot_df.loc[plot_df["alpha"].notna()].copy()

    active_metrics = [
        metric
        for metric in metrics
        if np.isfinite(plot_df.loc[plot_df["metric"] == metric, "value"]).any()
    ]
    if not active_metrics:
        raise ValueError("No finite values are available to plot.")

    plot_df = plot_df.loc[plot_df["metric"].isin(active_metrics)].copy()
    alpha_levels = sorted(plot_df["alpha"].dropna().unique().tolist())
    palette = resolve_palette(active_metrics, palette)
    legend_names = resolve_legend_names(active_metrics, legend_names)

    values = finite_values(plot_df["value"])
    y_max = float(values.max())
    y_min = float(values.min())
    lower_margin = y_max * 0.05 if y_max > 0 and y_min >= 0 else max(abs(y_min) * 0.07, 1e-12)
    upper_limit = y_max * 1.08 if y_max > 0 else 1.0
    lower_limit = min(0.0, y_min) - lower_margin

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor(BACKGROUND_COLOR)

    draw_grouped_boxplot_panel(
        ax,
        plot_df,
        alpha_levels,
        active_metrics,
        palette,
        show_points=show_points,
    )
    style_publication_axis(
        ax,
        alpha_levels,
        x_label=x_label,
        y_label=y_label,
        tick_size=tick_size,
    )
    apply_scientific_formatter(ax, power_limits=(-2, 2))
    ax.set_ylim(lower_limit, upper_limit)
    ax.margins(x=0.03)

    legend_handles = build_legend_handles(active_metrics, palette, legend_names)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=min(len(active_metrics), 5),
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.6,
        fontsize=13,
    )

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    # plt.close(fig)
