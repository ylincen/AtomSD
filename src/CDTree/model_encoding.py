import math
from typing import Dict

def log2(x: float) -> float:
    """Safe log2 wrapper."""
    return math.log2(x)

def rissanen_integer_cl(n: float) -> float:
    """
    Universal code length for positive integers (Rissanen).
    Matches the R implementation exactly.
    Reference: Grunwald (2007), p.100 (as in the R comment).
    """
    if n < 1:
        raise ValueError("Integer code only applies to positive integers >= 1")

    c0 = 2.865
    cl = log2(c0)

    to_add = float(n)
    while True:
        v = log2(to_add)
        if v > 0:
            cl += v
            to_add = v
        else:
            return cl

def log2_catalan(n_internal_nodes: int) -> float:
    """
    log2 of Catalan number-related term used to encode tree shape.
    Matches the R implementation:
      if n_internal_nodes <= 1: return 0
      else sum_{k=2..n} log2((n+k)/k)
    """
    if n_internal_nodes <= 1:
        return 0.0

    n = n_internal_nodes
    total = 0.0
    for k in range(2, n + 1):
        total += log2((n + k) / k)
    return total

def get_cl_model_tree_change(num_leaf_nodes: int) -> Dict[str, float]:
    """
    Computes current and potential MDL model cost for tree structure
    when adding one split (i.e., increasing leaf count by 1).
    """
    current = rissanen_integer_cl(num_leaf_nodes) + log2_catalan(num_leaf_nodes - 1)
    potential = rissanen_integer_cl(num_leaf_nodes + 1) + log2_catalan(num_leaf_nodes)
    return {
        "potential_cl_model_tree": potential,
        "current_cl_model_tree": current,
    }
