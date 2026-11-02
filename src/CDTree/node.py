# node.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class LeafNode:
    """
    Python equivalent of construct_leaf_node() in Node.R
    """

    # ---- original fields (already in your file) ----
    indices: np.ndarray
    x_hist: Dict

    indices_test: Optional[np.ndarray] = None

    # root-to-leaf path information (parallel lists)
    cut_value_list: List[float] = field(default_factory=list)
    cut_icol_list: List[int] = field(default_factory=list)
    cut_operator_list: List[str] = field(default_factory=list)

    # ---- added fields (dynamic in R; predefined in Python) ----
    # children pointers (set after best split is computed)
    left_child: Optional["LeafNode"] = None
    right_child: Optional["LeafNode"] = None

    # chosen split info for this node (set after split search)
    z_cut_value: Optional[float] = None
    z_cut_icol: Optional[int] = None

    # best gain if this node were split (used by update_node_list)
    gain_after_split: float = float("-inf")

    # optional: store tree model encoding change returned by search_split
    cl_model_change_after_split: Optional[Dict[str, Any]] = None

    # record split info from parent (useful for tracing / debugging)
    mother_split: Optional[Dict[str, Any]] = None

    
