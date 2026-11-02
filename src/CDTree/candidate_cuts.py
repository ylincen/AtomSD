import numpy as np
from typing import Dict, Any, List
from .model_encoding import rissanen_integer_cl

def generate_cuts_and_clforcuts(z: np.ndarray, l: int = 7, C: int = 5) -> Dict[str, Any]:
    """
    Python equivalent of R generate_cuts_and_clforcuts(z, l=7, C=5).
    Returns:
      cuts_list: list of numpy arrays (candidate cuts for each layer)
      cuts_cl: numpy array (MDL encoding cost per layer)
    """
    z = np.asarray(z, dtype=float).ravel()
    cut_points_length = C * (2 ** np.arange(0, l + 1))  # e.g., [5,10,20,...]

    uniq = np.unique(z[~np.isnan(z)])
    cuts_list: List[np.ndarray] = []
    cuts_cl = np.zeros(len(cut_points_length), dtype=float)

    # Binary feature special case
    if len(uniq) == 2:
        cut = float(np.mean(uniq))
        for i in range(len(cut_points_length)):
            cuts_list.append(np.array([cut], dtype=float))
            cuts_cl[i] = 0.0
        return {"cuts_list": cuts_list, "cuts_cl": cuts_cl}

    # General case: quantile cuts
    # General case: per-level quantile cuts + per-level encoding cost
    for level, m in enumerate(cut_points_length):
        # 1) generate cuts for this level (ONLY generate, no cl inside)
        cuts = get_cuts_for_z_quantile(z, int(m))

        # 2) R-like: use the number of UNIQUE cuts as "true_length"
        #    and CAP it by m (this is where 'level' concept matters)
        true_length = int(min(len(cuts), m))  # <-- this replaces your undefined 'level' usage

        # 3) encoding costs (same formulas you already used)
        # granularity = log2(true_length) + 1
        granularity = (np.log2(true_length) + 1.0) if true_length > 0 else 1.0
        cl_granularity = rissanen_integer_cl(granularity)

        # cl_cut = log2(true_length)
        cl_cut = np.log2(true_length) if true_length > 0 else 0.0

        cuts_list.append(cuts)
        cuts_cl[level] = float(cl_cut + cl_granularity)

    return {"cuts_list": cuts_list, "cuts_cl": cuts_cl}


def get_cuts_for_z(z: np.ndarray, cut_points_length: int) -> Dict[str, Any]:
    """Wrapper to choose which strategy to generate cuts."""
    return get_cuts_for_z_quantile(z, cut_points_length)


def get_cuts_for_z_quantile(z: np.ndarray, cut_points_length: int) -> Dict[str, Any]:
    """
    Python equivalent of R get_cuts_for_z_quantile().
    Uses quantiles to generate candidate cuts.
    """
    z = np.asarray(z, dtype=float).ravel()
    z = z[~np.isnan(z)]

    # probs = seq(0,1,length.out=cut_points_length+2) then take inner probs
    ps = np.linspace(0.0, 1.0, cut_points_length + 2)
    inner_ps = ps[1:-1]

    # R quantile default type differs slightly from numpy percentile interpolation.
    # We'll use numpy quantile (linear) for now; we can match R's exact type later if needed.
    cuts = r_quantile_type7(z, inner_ps)  # numpy>=1.22 uses 'method'
    cuts = np.unique(cuts)
    # R uses true_length = length(unique(cuts))
    #true_length = min(len(cuts), cut_points_length[level])

    # granularity = log2(true_length) + 1
    # Note: in R, granularity can be non-integer; they pass it into Rissanen_integer_cl().
    # TR passes a non-integer granularity to Rissanen_integer_cl; we do the same.
    #granularity = np.log2(true_length) + 1.0 if true_length > 0 else 1.0
    #cl_granularity = rissanen_integer_cl(granularity)

    #cl_cut = np.log2(true_length) if true_length > 0 else 0.0

    #return {"cuts": cuts, "cl": cl_cut + cl_granularity}
    return cuts

def r_quantile_type7(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    probs = np.asarray(probs, dtype=float)

    if n == 0:
        return np.full_like(probs, np.nan, dtype=float)

    # R type=7:
    # h = (n - 1) * p + 1
    h = (n - 1) * probs + 1.0
    j = np.floor(h).astype(int)          # 1..n
    g = h - j

    # clamp j to [1, n]
    j = np.clip(j, 1, n)

    # convert to 0-index
    j0 = j - 1
    j1 = np.clip(j0 + 1, 0, n - 1)

    return (1.0 - g) * x[j0] + g * x[j1]

