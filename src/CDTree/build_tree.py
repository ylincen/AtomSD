# build_tree.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np 
from .node import LeafNode
from .hist import search_best_num_bin
from .search import search_split
#from .regret import regret

# ============================
# Debug control
# ============================
DEBUG = False


def construct_leaf_node(
    indices: np.ndarray,
    x_hist: Dict[str, Any],
    indices_test: Optional[np.ndarray] = None,
    cut_value_list: Optional[List[float]] = None,
    cut_icol_list: Optional[List[int]] = None,
    cut_operator_list: Optional[List[str]] = None,
) -> LeafNode:
    """
    Small helper to mimic R's construct_leaf_node().
    If you already have a separate construct_leaf_node() function in your project,
    you can replace this with an import.
    """
    return LeafNode(
        indices=np.asarray(indices, dtype=int),
        x_hist=x_hist,
        indices_test=None if indices_test is None else np.asarray(indices_test, dtype=int),
        cut_value_list=[] if cut_value_list is None else list(cut_value_list),
        cut_icol_list=[] if cut_icol_list is None else list(cut_icol_list),
        cut_operator_list=[] if cut_operator_list is None else list(cut_operator_list),
    )


def build_tree(
    x: np.ndarray,
    z_matrix: np.ndarray,
    eps: float,
    regret_fn: Callable[[int, int], float],
    x_test: Optional[np.ndarray] = None,
    z_test: Optional[np.ndarray] = None,
    left_bound: Optional[float] = None,
    right_bound: Optional[float] = None,
    print_process: bool = True,
) -> Dict[str, Any]:
    """
    Python translation of build_tree.R::build_tree (best-first greedy splitting)

    Core loop:
      1) build default histogram for root
      2) for each iteration:
         - for each leaf: compute its best split (gain, cut, child hists)
         - pick the leaf with max gain and actually apply the split
         - stop if max gain < 0

    Validation tip (convert-one-validate-one):
      Compare iteration-1 outputs with R:
        - which leaf split (root)
        - icol_split
        - z_cut_value
        - gain_after_split
    """
    x = np.asarray(x, dtype=float).ravel()
    z_matrix = np.asarray(z_matrix, dtype=float)

    if (x_test is None) != (z_test is None):
        raise ValueError("x_test and z_test must be both None or both provided (match R behavior).")

    # --- decide global bounds (match R logic) ---
    if x_test is None and z_test is None:
        if left_bound is None or right_bound is None:
            raise ValueError(
                "x_test and z_test are both None. In R, you must provide left_bound and right_bound."
            )
        left_bound_use = float(left_bound)
        right_bound_use = float(right_bound)
    else:
        x_test = np.asarray(x_test, dtype=float).ravel()
        z_test = np.asarray(z_test, dtype=float)
        if left_bound is None or right_bound is None:
            combined = np.concatenate([x, x_test])
            left_bound_use = float(np.min(combined) - eps)
            right_bound_use = float(np.max(combined) + eps)
        else:
            if print_process:
                print("use the left_bound & right_bound provided instead of inferring from x and x_test")
            left_bound_use = float(left_bound)
            right_bound_use = float(right_bound)

    default_bound = (left_bound_use, right_bound_use)

    # --- root default histogram ---
    default_x_hist = search_best_num_bin(
        x=x,
        eps=eps,
        left_bound=left_bound_use,
        right_bound=right_bound_use,
        regret_fn=regret_fn,
        cl_boundary=0.0,
    )

    # --- init node list with root ---
    node_list: List[LeafNode] = []
    which_nodes_splitted: List[int] = []
    icol_split_list: List[Optional[int]] = []

    root_indices = np.arange(len(x), dtype=int)
    root_test_indices = None if x_test is None else np.arange(len(x_test), dtype=int)

    node_list.append(
        construct_leaf_node(
            indices=root_indices,
            x_hist=default_x_hist,
            indices_test=root_test_indices,
        )
    )

    iter_idx = 1
    while True:
        if print_process:
            print(f"iter: {iter_idx}",flush=True) #测试打印

        # 1) compute best split for each leaf (if not already computed)
        for i in range(len(node_list)):
            node_list[i] = search_z_split_with_test(
                node=node_list[i],
                x=x,
                z_matrix=z_matrix,
                current_num_leaf_nodes=len(node_list),
                eps=eps,
                default_bound=default_bound,
                regret_fn=regret_fn,
                x_test=x_test,
                z_test=z_test,
            )

        # 2) apply the single best split among all leaves
        res_update = update_node_list(node_list)

        # stop condition (MUST have, otherwise infinite loop)
        if res_update["stop_splitting"]:
            break

        # update node_list first (after split)
        node_list = res_update["node_list"]

        if DEBUG:
            if iter_idx == 1:
                print("V1 which_node_splitted:", res_update["which_node_splitted"])
                print("V1 icol_split:", res_update["icol_split"])
                print("V1 leaf0 mother_split:", node_list[0].mother_split)
                if len(node_list) > 1:
                    print("V1 leaf1 mother_split:", node_list[1].mother_split)

            which_nodes_splitted.append(res_update["which_node_splitted"])
            icol_split_list.append(res_update["icol_split"])
            iter_idx += 1


    return {
        "node_list": node_list,
        "which_nodes_splitted": which_nodes_splitted,
        "default_hist": default_x_hist,
        "icol_split_list": icol_split_list,
    }


def search_z_split_with_test(
    node: LeafNode,
    x: np.ndarray,
    z_matrix: np.ndarray,
    current_num_leaf_nodes: int,
    eps: float,
    default_bound: Tuple[float, float],
    regret_fn: Callable[[int, int], float],
    x_test: Optional[np.ndarray],
    z_test: Optional[np.ndarray],
) -> LeafNode:
    """
    Compute best split for ONE leaf node across ALL feature columns.

    IMPORTANT: This function matches your current search_split() signature:
      search_split(x, z, z_num_dim, node_to_split_hist_score,
                   current_num_leaf_nodes, eps, default_bound, regret_fn)
    """
    # mimic R: if node already has children, skip
    if node.left_child is not None:
        return node

    z_num_dim = z_matrix.shape[1]

    best_gain = float("-inf")
    best_res: Optional[Dict[str, Any]] = None
    best_icol: Optional[int] = None

    # Search best feature column
    for icol in range(z_num_dim):
        idx = node.indices
        z_col = z_matrix[idx, icol]

        split_res = search_split(
            x=x[idx],
            z=z_col,
            z_num_dim=z_num_dim,
            node_to_split_hist_score=float(node.x_hist["score"]),  # critical: matches your search.py
            current_num_leaf_nodes=current_num_leaf_nodes,
            eps=eps,
            default_bound=default_bound,
            regret_fn=regret_fn,
        )

        if split_res["best_gain"] > best_gain:
            best_gain = split_res["best_gain"]
            best_res = split_res
            best_icol = icol

    if best_res is None or best_icol is None:
        # no valid split found for this node
        node.gain_after_split = -np.inf
        node.z_cut_icol = None
        node.z_cut_value = None
        node.left_child = None
        node.right_child = None
        return node


    # record best split info on this node
    node.z_cut_icol = int(best_icol)
    node.z_cut_value = float(best_res["best_z_cut"]) if best_res["best_z_cut"] is not None else None
    node.gain_after_split = float(best_res["best_gain"])
    node.cl_model_change_after_split = best_res.get("tree_cl_model_change", None)

    # If no valid cut found, just keep children None (gain likely -inf or <=0)
    if node.z_cut_value is None:
        return node

    # --- split train indices ---
    idx = node.indices
    z_vals = z_matrix[idx, node.z_cut_icol]
    left_mask = z_vals < node.z_cut_value
    right_mask = ~left_mask  # >=

    left_indices = idx[left_mask]
    right_indices = idx[right_mask]

    # --- split test indices (if exists) ---
    left_indices_test = None
    right_indices_test = None
    if node.indices_test is not None:
        assert z_test is not None, "z_test must exist if indices_test exists"
        idx_t = node.indices_test
        zt_vals = z_test[idx_t, node.z_cut_icol]
        left_mask_t = zt_vals < node.z_cut_value
        right_mask_t = ~left_mask_t
        left_indices_test = idx_t[left_mask_t]
        right_indices_test = idx_t[right_mask_t]

    # --- extend decision path lists ---
    cut_value_list = list(node.cut_value_list) + [node.z_cut_value]
    cut_icol_list = list(node.cut_icol_list) + [node.z_cut_icol]
    left_ops = list(node.cut_operator_list) + ["<"]
    right_ops = list(node.cut_operator_list) + [">="]

    # --- build children using best histograms from search_split ---
    x_left_hist = best_res["best_hist"]["x_left_hist"]
    x_right_hist = best_res["best_hist"]["x_right_hist"]

    node.left_child = construct_leaf_node(
        indices=left_indices,
        x_hist=x_left_hist,
        indices_test=left_indices_test,
        cut_value_list=cut_value_list,
        cut_icol_list=cut_icol_list,
        cut_operator_list=left_ops,
    )
    node.right_child = construct_leaf_node(
        indices=right_indices,
        x_hist=x_right_hist,
        indices_test=right_indices_test,
        cut_value_list=cut_value_list,
        cut_icol_list=cut_icol_list,
        cut_operator_list=right_ops,
    )

    # mother_split info (useful for tracing)
    node.left_child.mother_split = {"z_cut_icol": node.z_cut_icol, "z_cut_value": node.z_cut_value}
    node.right_child.mother_split = {"z_cut_icol": node.z_cut_icol, "z_cut_value": node.z_cut_value}

    return node


def update_node_list(node_list: List[LeafNode]) -> Dict[str, Any]:
    """
    Apply ONE best split among all leaves.

    R behavior:
      - pick leaf with maximum gain_after_split
      - stop if max_gain < 0
      - else replace that node by its left_child, append right_child
    """
    gains = [float(nd.gain_after_split) for nd in node_list]
    which_max_gain = int(np.argmax(gains))
    max_gain = gains[which_max_gain]

    #测试验证
    if DEBUG:
        print("V1 max_gain:", max_gain)

    if max_gain <= 0:
        return {
            "stop_splitting": True,
            "node_list": node_list,
            "which_node_splitted": which_max_gain,
            "icol_split": None,
        }

    chosen = node_list[which_max_gain]
    icol_split = chosen.z_cut_icol

    # Must have children if gain is positive
    if chosen.left_child is None or chosen.right_child is None:
        # Defensive: if something went wrong upstream, stop rather than corrupt the tree
        return {
            "stop_splitting": True,
            "node_list": node_list,
            "which_node_splitted": which_max_gain,
            "icol_split": None,
        }

    if DEBUG:
        print(
            "[ACCEPT] "
            f"leaf_index={which_max_gain} "
            f"icol={chosen.z_cut_icol} "
            f"cut={chosen.z_cut_value:.10f} "
            f"gain={chosen.gain_after_split:.6f} "
            f"left_n={len(chosen.left_child.indices)} "
            f"right_n={len(chosen.right_child.indices)}"
        )


    # replace with left child and append right child
    node_list.append(chosen.right_child)
    node_list[which_max_gain] = chosen.left_child

    return {
        "stop_splitting": False,
        "node_list": node_list,
        "which_node_splitted": which_max_gain,
        "icol_split": icol_split,
    }
    




