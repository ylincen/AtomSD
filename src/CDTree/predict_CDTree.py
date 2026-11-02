# predict_CDTree.py
from __future__ import annotations

from typing import List, Sequence, Dict, Any
import numpy as np
import pandas as pd


def predict_CDTree(tree: Any, x_test: np.ndarray, z_test: np.ndarray) -> np.ndarray:
    """
    Predict log conditional density log p(x_test | z_test) for each test sample.

    R equivalent: predict_CDTree(tree, x_test, z_test)
    - tree['node_list']: list of leaf nodes (aligned with your Python port) ????
    - returns: log_density_x (np.ndarray, shape=(n_test,))
    """
    x_test = np.asarray(x_test, dtype=float)
    z_test = np.asarray(z_test, dtype=float)

    test_indices_list = cover_test_data(tree, x_test, z_test)

    # R: log_density_x = rep(NA, length(x_test))
    log_density_x = np.full(shape=(len(x_test), 1), fill_value=np.nan, dtype=float)

    # R loops leaf-by-leaf and fills the corresponding test indices
    for i, test_idx in enumerate(test_indices_list):
        if len(test_idx) == 0:
            continue
        leaf_node = tree.node_list[i] if hasattr(tree, "node_list") else tree["node_list"][i]
        log_density_x[test_idx] = predict_x_given_z_leaf(leaf_node, x_test[test_idx])

    return log_density_x


def predict_x_given_z_leaf(leaf_node: Any, x_: np.ndarray) -> np.ndarray:
    """
    Predict log density values for x_ assuming they fall into this leaf.

    R equivalent: predict_x_given_z_leaf(leaf_node, x_)
    Uses leaf_node.x_hist.hist_fixed with Laplace smoothing (counts + 1).
    """
    x_ = np.asarray(x_, dtype=float)

    # R: hist_ = leaf_node$x_hist$hist_fixed
    # Your Python hist objects commonly store keys like:
    # hist_fixed["counts"], hist_fixed["breaks"]
    hist_fixed = leaf_node.x_hist["hist_fixed"]

    counts = np.asarray(hist_fixed["counts"], dtype=float) + 1.0  # Laplace smoothing
    breaks_ = np.asarray(hist_fixed["breaks"], dtype=float)

    widths = np.diff(breaks_)
    if widths.size == 0:
        raise ValueError("Histogram breaks must contain at least 2 points.")
    bin_width = float(widths[0])

    density_ = counts / counts.sum() / bin_width
    log_density_ = np.log(density_)

    idx = np.searchsorted(breaks_, x_, side="right") - 1

    if np.any(idx < 0) or np.any(idx >= (len(breaks_) - 1)):
        raise ValueError(
            "The target variable is outside the range of the histogram; "
            "re-train the model by updating the range of the target variable."
        )

    # idx is 0-based bin index, directly index log_density_
    return log_density_[idx]


def cover_test_data(tree: Any, x_test: np.ndarray, z_test: np.ndarray) -> List[np.ndarray]:
    """
    For each leaf in the tree, compute the indices of test samples covered by that leaf
    (i.e., satisfy all root-to-leaf split conditions).

    R equivalent: cover_test_data(tree, x_test, z_test)
    """
    x_test = np.asarray(x_test, dtype=float)
    z_test = np.asarray(z_test, dtype=float)

    n_test = len(x_test)
    test_indices_list: List[np.ndarray] = []

    for node in tree['node_list']:
        split_icol_list = node.cut_icol_list
        split_value_list = node.cut_value_list
        split_operator_list = node.cut_operator_list

        test_indices = np.arange(n_test, dtype=int)

        for split_icol, split_value, split_op in zip(
            split_icol_list, split_value_list, split_operator_list
        ):
            # IMPORTANT:
            # If your split_icol was kept 1-based (R style), uncomment the next line:
            # split_icol = split_icol - 1

            col = int(split_icol)
            val = float(split_value)

            if split_op == "<":
                mask = z_test[test_indices, col] < val
            else:
                # R uses ">=" for the else branch
                mask = z_test[test_indices, col] >= val

            test_indices = test_indices[mask]

            # Early stop if empty
            if test_indices.size == 0:
                break

        test_indices_list.append(test_indices)

    return test_indices_list


def cover_test_data_with_rules(tree: Any, x_test: np.ndarray, z_test: np.ndarray, feature_names: List[str]) -> List[pd.DataFrame]:
    """
    For each leaf in the tree, extract the rules
    """
    x_test = np.asarray(x_test, dtype=float)
    z_test = np.asarray(z_test, dtype=float)

    n_test = len(x_test)
    # test_indices_list: List[np.ndarray] = []

    rules_df = [[None]] * len(tree['node_list'])

    for idx, node in enumerate(tree['node_list']):
        rules = pd.DataFrame(columns=feature_names, index=['lower', 'upper', '='])
        split_icol_list = node.cut_icol_list
        split_value_list = node.cut_value_list
        split_operator_list = node.cut_operator_list

        test_indices = np.arange(n_test, dtype=int)

        for split_icol, split_value, split_op in zip(
            split_icol_list, split_value_list, split_operator_list
        ):
            # IMPORTANT:
            # If your split_icol was kept 1-based (R style), uncomment the next line:
            # split_icol = split_icol - 1

            col = int(split_icol)
            val = float(split_value)

            if split_op == "<":
                temp = rules.at['upper', feature_names[col]]
                if pd.isna(temp) or (temp > val):
                    rules.at['upper', feature_names[col]] = val
                mask = z_test[test_indices, col] < val
            else:
                # R uses ">=" for the else branch
                mask = z_test[test_indices, col] >= val
                temp = rules.at['lower', feature_names[col]]
                if pd.isna(temp) or (temp < val):
                    rules.at['lower', feature_names[col]] = val

            test_indices = test_indices[mask]

            # Early stop if empty
            if test_indices.size == 0:
                break

        # test_indices_list.append(test_indices)
        if not rules.empty:
            rules_df[idx] = rules

    return rules_df