import numpy as np
from typing import Dict, Any, Optional

from .candidate_cuts import generate_cuts_and_clforcuts
from .model_encoding import get_cl_model_tree_change
from .hist import search_best_num_bin  # should return dict with 'score' etc.


def search_best_hist_and_boundary(x: np.ndarray,
                                 eps: float,
                                 default_bound,
                                 regret_fn) -> Dict[str, Any]:
    left_boundary = float(default_bound[0])
    right_boundary = float(default_bound[1])

    # Try to match R call signature first
    try:
        return search_best_num_bin(
            x=x,
            eps=eps,
            left_bound=left_boundary,
            right_bound=right_boundary,
            cl_boundary=0,
            inside_ratio=1,
            regret_fn=regret_fn
        )
    except TypeError:
        # fallback if hist.search_best_num_bin doesn't accept those args yet
        return search_best_num_bin(
            x=x,
            eps=eps,
            left_bound=left_boundary,
            right_bound=right_boundary,
            regret_fn=regret_fn
        )



def search_split(x: np.ndarray,
                 z: np.ndarray,
                 z_num_dim: int,
                 node_to_split_hist_score: float,
                 current_num_leaf_nodes: int,
                 eps: float,
                 default_bound,
                 regret_fn) -> Dict[str, Any]:
    """
    Python equivalent of R search_split() for ONE feature vector z.

    Returns:
      best_hist: dict with left/right hist results
      best_gain: float
      best_z_cut: float or None
      tree_cl_model_change: dict with current/potential tree costs
    """
    x = np.asarray(x, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()

    cuts_and_cl = generate_cuts_and_clforcuts(z)
    cuts_list = cuts_and_cl["cuts_list"]
    cuts_cl = cuts_and_cl["cuts_cl"]

    best_gain_all_layers = -np.inf
    best_hist_all_layers: Optional[Dict[str, Any]] = None
    best_z_cut_all_layers: Optional[float] = None
    tree_cl_model_change = None

    for i_list, cuts in enumerate(cuts_list):
        cut_cl = float(cuts_cl[i_list])

        best_gain_this_layer = -np.inf
        cuts = np.asarray(cuts, dtype=float).ravel()
        if len(np.unique(cuts)) <= 2 and i_list != (len(cuts_list) - 1):
            continue

        # compute ONCE per node (match R semantics)
        tree_cl_model_change = get_cl_model_tree_change(current_num_leaf_nodes)

        for cut in cuts:
            left_mask = z < cut
            right_mask = ~left_mask

            x_left = x[left_mask]
            x_right = x[right_mask]

            # Mimic R: skip tiny children
            if len(x_left) <= 2 or len(x_right) <= 2:
                continue

            x_left_hist = search_best_hist_and_boundary(
                x_left, eps=eps, default_bound=default_bound, regret_fn=regret_fn
            )
            x_right_hist = search_best_hist_and_boundary(
                x_right, eps=eps, default_bound=default_bound, regret_fn=regret_fn
            )

            total_cl_after_split = (
                tree_cl_model_change["potential_cl_model_tree"]
                + float(x_left_hist["score"]) + float(x_right_hist["score"])
                + cut_cl + np.log2(z_num_dim)
            )

            total_cl_before_split = (
                tree_cl_model_change["current_cl_model_tree"]
                + float(node_to_split_hist_score)
            )

            gain = total_cl_before_split - total_cl_after_split
            if gain > best_gain_this_layer:
                best_gain_this_layer = gain
                best_hist = {"x_left_hist": x_left_hist, "x_right_hist": x_right_hist}
                best_z_cut = float(cut)


        if best_gain_this_layer > best_gain_all_layers:
            best_gain_all_layers = best_gain_this_layer
            best_hist_all_layers = best_hist
            best_z_cut_all_layers = best_z_cut
        else:
            # Same as R: break if no improvement at this layer
            break


    return {
        "best_hist": best_hist_all_layers,
        "best_gain": float(best_gain_all_layers),
        "best_z_cut": best_z_cut_all_layers,
        "tree_cl_model_change": tree_cl_model_change
    }
