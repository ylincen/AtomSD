# CDTree/hist.py
import numpy as np
from .data_encoding import get_cl_data_from_hist
from .model_encoding import rissanen_integer_cl
from .regret import regret as regret_default


def r_hist_counts(x: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    """
    Mimic R hist(..., right=TRUE, include.lowest=TRUE):
    bins: (breaks[i], breaks[i+1]] and include.lowest for breaks[0]
    """
    x = np.asarray(x, dtype=float)
    breaks = np.asarray(breaks, dtype=float)
    k = len(breaks) - 1
    idx = np.searchsorted(breaks, x, side="left") - 1
    # idx = np.searchsorted(breaks, x, side="right") - 1
    idx = np.where(x == breaks[0], 0, idx)
    mask = (idx >= 0) & (idx < k)
    return np.bincount(idx[mask], minlength=k)


def _build_hist_fixed(x, left_bound, right_bound, len_bins):
    cuts = np.linspace(left_bound, right_bound, len_bins)
    counts = r_hist_counts(x, cuts)
    widths = np.diff(cuts)
    n = int(len(x))
    p_hat = counts / max(n, 1)
    density = np.divide(p_hat, widths, out=np.zeros_like(p_hat, dtype=float), where=widths > 0)
    return {"breaks": cuts, "counts": counts, "density": density, "widths": widths, "n": n}


def get_hist_cl_components(len_bins, x, left_bound, right_bound,
                           cl_boundary=0.0, inside_ratio=1, regret_fn=None):
    """
    R-aligned decomposition:
      cl_data  = negloglike + regret(n,k)
      cl_model = rissanen_integer_cl(k) + cl_boundary
      score    = cl_data + cl_model
      sc       = cl_data
    inside_ratio: kept only for API compatibility (R passes it), not used by R's get_cl_data either. :contentReference[oaicite:2]{index=2}
    """
    if regret_fn is None:
        regret_fn = regret_default

    x = np.asarray(x, dtype=float).ravel()
    hist_fixed = _build_hist_fixed(x, left_bound, right_bound, len_bins)

    counts = hist_fixed["counts"]
    breaks = hist_fixed["breaks"]
    n = int(hist_fixed["n"])
    k = int(len(counts))

    negloglike = float(get_cl_data_from_hist(counts=counts, bin_edges=breaks))
    nml_regret = float(regret_fn(n, k))
    cl_data = negloglike + nml_regret

    cl_num_bins = float(rissanen_integer_cl(k))
    cl_model = cl_num_bins + float(cl_boundary)

    score = cl_data + cl_model

    return {
        "hist_fixed": hist_fixed,
        "negloglike": negloglike,
        "regret": nml_regret,
        "cl_data": cl_data,      # == sc
        "cl_num_bins": cl_num_bins,
        "cl_boundary": float(cl_boundary),
        "cl_model": cl_model,
        "sc": cl_data,
        "score": score,
        "len_bins": int(len_bins),
        "num_bins": int(k),
    }


def get_hist_cl_total(len_bins, x, left_bound, right_bound,
                      cl_boundary=0.0, inside_ratio=1, regret_fn=None):
    return float(
        get_hist_cl_components(
            len_bins, x, left_bound, right_bound,
            cl_boundary=cl_boundary, inside_ratio=inside_ratio, regret_fn=regret_fn
        )["score"]
    )


def search_best_num_bin(x, eps, left_bound, right_bound,
                        cl_boundary=0, inside_ratio=1, regret_fn=None,
                        num_gap=30, init_len=2):
    """
    Match R call style:
    search_best_num_bin(x, eps, left_bound, right_bound, cl_boundary=0, inside_ratio=1)
    - keep inside_ratio for API compatibility
    - default num_gap=30, init_len=2 (R behavior)
    """
    if regret_fn is None:
        regret_fn = regret_default

    x = np.asarray(x, dtype=float).ravel()

    prev_cl = np.inf
    length = int(init_len)

    while (x.max() - x.min()) / max(length, 1) > eps:
        cl = get_hist_cl_total(length, x, left_bound, right_bound,
                               cl_boundary=cl_boundary, inside_ratio=inside_ratio, regret_fn=regret_fn)
        if cl < prev_cl:
            prev_cl = cl
            length += int(num_gap)
        else:
            break

    start = max(2, length - 2 * int(num_gap))
    end = length

    best_len = None
    best_comp = None
    best_score = np.inf

    for l in range(start, end + 1):
        comp = get_hist_cl_components(l, x, left_bound, right_bound,
                                      cl_boundary=cl_boundary, inside_ratio=inside_ratio, regret_fn=regret_fn)
        if comp["score"] < best_score:
            best_score = comp["score"]
            best_len = l
            best_comp = comp

    best_comp["best_len"] = int(best_len)
    return best_comp
