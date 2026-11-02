"""
kl_estimation.py
================
Histogram density estimation and sample-wise KL divergence helpers used by
``make_fig1_ranking_flip.py`` (Figure 1 of the paper).

All functions are pure; randomness is handled by the caller.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def _compute_bin_edges(data: np.ndarray, bin_method: str, n_bins: int | None) -> np.ndarray:
    """Compute histogram bin edges from *data*.

    bin_method is one of ``"fixed"`` (uses *n_bins*), ``"sqrt"``, ``"sturges"`` or ``"fd"``
    (Freedman-Diaconis). Returns a monotonically increasing array of length n_bins + 1.
    """
    n = len(data)
    if n == 0:
        raise ValueError("Cannot compute bin edges from empty data array.")

    if bin_method == "fixed":
        if n_bins is None or n_bins < 1:
            raise ValueError("n_bins must be a positive integer when bin_method='fixed'.")
        k = n_bins
    elif bin_method == "sqrt":
        k = max(1, int(np.ceil(np.sqrt(n))))
    elif bin_method == "sturges":
        k = max(1, int(np.ceil(np.log2(n))) + 1)
    elif bin_method == "fd":
        iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
        if iqr == 0.0:
            k = max(1, int(np.ceil(np.log2(n))) + 1)  # fall back to Sturges
        else:
            h = 2.0 * iqr * (n ** (-1.0 / 3.0))
            data_range = float(data.max() - data.min())
            k = max(1, int(np.ceil(data_range / h)))
    else:
        raise ValueError(f"Unknown bin_method '{bin_method}'. Choose from 'fixed', 'sqrt', 'sturges', 'fd'.")

    return np.linspace(float(data.min()), float(data.max()), k + 1)


def _apply_smoothing(
    counts: np.ndarray,
    smoothing: str,
    epsilon: float,
    prior_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth raw bin counts.

    ``"none"``: unchanged; ``"laplace"``: +1 per bin; ``"epsilon"``: +epsilon per bin;
    ``"bayes"``: Dirichlet prior proportional to *prior_counts* with total pseudo-count equal to
    the number of bins (the correction used in ``src/score.py``).
    """
    counts = counts.astype(float)
    if smoothing == "none":
        return counts
    if smoothing == "laplace":
        return counts + 1.0
    if smoothing == "epsilon":
        return counts + epsilon
    if smoothing == "bayes":
        if prior_counts is None:
            raise ValueError("prior_counts must be provided when smoothing='bayes'.")
        n_bins = len(counts)
        prior = np.asarray(prior_counts, dtype=float)
        return counts + prior / prior.sum() * n_bins
    raise ValueError(f"Unknown smoothing '{smoothing}'. Choose from 'none', 'laplace', 'epsilon', 'bayes'.")


def build_histogram_density(
    data: np.ndarray,
    bin_edges: np.ndarray | None = None,
    histogram_options: dict[str, Any] | None = None,
    prior_data: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a (smoothed) histogram density to *data*.

    Returns ``(density_per_bin, bin_edges)``; the density integrates to 1 over the bin support.
    Pass *bin_edges* explicitly whenever two densities must live on the same grid.
    """
    if histogram_options is None:
        histogram_options = {"bin_method": "sqrt", "n_bins": 10, "smoothing": "laplace", "epsilon": 1e-6}

    data = np.asarray(data, dtype=float)
    smoothing = histogram_options.get("smoothing", "none")
    epsilon = float(histogram_options.get("epsilon", 1e-6))

    if bin_edges is None:
        bin_edges = _compute_bin_edges(
            data,
            bin_method=histogram_options.get("bin_method", "sqrt"),
            n_bins=histogram_options.get("n_bins", None),
        )
    bin_edges = np.asarray(bin_edges, dtype=float)

    counts, _ = np.histogram(data, bins=bin_edges)
    if smoothing == "bayes":
        if prior_data is None:
            raise ValueError("prior_data must be provided when smoothing='bayes'.")
        prior_counts, _ = np.histogram(np.asarray(prior_data, dtype=float), bins=bin_edges)
        smoothed = _apply_smoothing(counts, smoothing, epsilon, prior_counts=prior_counts.astype(float) + 1.0)
    else:
        smoothed = _apply_smoothing(counts, smoothing, epsilon)

    widths = np.diff(bin_edges)
    safe_widths = np.where(widths > 0, widths, 1.0)
    density_per_bin = smoothed / (safe_widths * float(smoothed.sum()))
    return density_per_bin, bin_edges


def evaluate_density_at_points(eval_points: np.ndarray, density_per_bin: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Evaluate a piecewise-constant histogram density at *eval_points*.

    Points outside the bin support take the density of the nearest edge bin.
    """
    eval_points = np.asarray(eval_points, dtype=float)
    bin_edges = np.asarray(bin_edges, dtype=float)
    bin_idx = np.searchsorted(bin_edges, eval_points, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, len(density_per_bin) - 1)
    return density_per_bin[bin_idx]


def sample_wise_kl(p_values: np.ndarray, q_values: np.ndarray, epsilon: float = 1e-300) -> tuple[float, np.ndarray]:
    """Sample-wise KL estimate (1/N) * sum_i [log p(x_i) - log q(x_i)] at points drawn from P.

    Terms with p(x_i) == 0 contribute 0. Returns ``(kl, per_point_terms)``.
    """
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.asarray(q_values, dtype=float)

    if np.any(q_values <= 0):
        warnings.warn(
            f"{int(np.sum(q_values <= 0))} q_value(s) are <= 0 after smoothing; using the epsilon floor {epsilon}.",
            RuntimeWarning,
            stacklevel=2,
        )

    zero_p_mask = p_values == 0.0
    safe_p = np.where(zero_p_mask, 1.0, np.maximum(p_values, epsilon))
    safe_q = np.maximum(q_values, epsilon)
    per_point = np.where(zero_p_mask, 0.0, np.log(safe_p) - np.log(safe_q))
    kl_scalar = float(per_point.mean()) if len(per_point) > 0 else 0.0
    return kl_scalar, per_point
