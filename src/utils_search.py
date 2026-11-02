from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import glob
import json
# from sklearn.preprocessing import StandardScaler

def dkl_density(p_train_subgroup, p_train, y_test_subgroup, edges_all, edges_subgroup):
    bin_ind_all = np.digitize(y_test_subgroup, bins=edges_all[1:-1])
    bin_ind_subgroup = np.digitize(y_test_subgroup, bins=edges_subgroup[1:-1])
    
    p_per_instance = p_train[bin_ind_all]
    p_subgroup_per_instance = p_train_subgroup[bin_ind_subgroup]
    kl_test = np.mean(np.log(p_subgroup_per_instance) - np.log(p_per_instance))
    return kl_test

def score_subgroups(y, y_train, idx, split: str, y_lim, edges_all=None, edges_subgroup=None, p_s_train=None, p_train=None, gamma=1.0):
    """
    y: The target variable (y_test or the same y_train).
    y_train: The training target variable.
    idx: subgroup indices.
    split: The data split to score ("train" or "test").
    gamma: Size correction exponent.
    """
    y = np.asarray(y)
    n = len(y)

    # Build subgroup mask
    # Assuming idx is a boolean mask (Syflow, RSD, SD_mu, SD_KL)
    if isinstance(idx, (list, tuple, np.ndarray)) and np.asarray(idx).dtype == bool:
        mask = np.asarray(idx)
        if len(mask) != n:
            raise ValueError("Boolean mask must match y length.")
    else:
        mask = np.zeros(n, dtype=bool)
        mask[np.asarray(idx, dtype=int)] = True

    y_s = y[mask]
    n_s = len(y_s)

    if n_s == 0:
        raise ValueError("Subgroup is empty.")

    # Edges `learned' from training data
    if edges_all is None:
        assert edges_subgroup is None
        edges_all = np.histogram_bin_edges(y_train, bins="fd", range=(y_lim[0], y_lim[1]))
        edges_subgroup = np.histogram_bin_edges(y_s, bins="fd", range=(y_lim[0], y_lim[1]))

    # # Histograms
    # c_all, _ = np.histogram(y, bins=edges_all, range=(y_lim[0], y_lim[1]))
    # c_s, _   = np.histogram(y_s, bins=edges_subgroup, range=(y_lim[0], y_lim[1]))

    # # Add a pseudo-count (Bayes correction)
    # c_all = c_all + 1
    # c_s = c_s + 1
    # Histograms
    c_all, _ = np.histogram(y, bins=edges_all)
    c_prior, _ = np.histogram(y, bins = edges_subgroup)
    c_s, _   = np.histogram(y_s, bins=edges_subgroup)

    # Add a pseudo-count (Laplace correction for all and bayes correction for subgroup)
    c_all = c_all + 1
    c_prior = c_prior + 1
    bayes_correction = c_prior / c_prior.sum() * (len(edges_subgroup) - 1) # so the \alpha_0 corresponds to the Conjugate prior is equal to the number of bins
    c_s = c_s + bayes_correction

    if p_s_train is None and p_train is None:
        # Convert to density
        p = c_all / c_all.sum() / (edges_all[1:] - edges_all[:-1])
        p_s = c_s / c_s.sum() / (edges_subgroup[1:] - edges_subgroup[:-1])
 
        # p = np.clip(p, 1e-100, None)
        # p_s = np.clip(p_s, 1e-100, None)
    else:
        p = p_train
        p_s = p_s_train

    kl = dkl_density(p_s, p, y_s, edges_all, edges_subgroup)
    # -----------------------------
    # Size correction
    # -----------------------------
    support = (n_s / n) ** gamma
    kl_corr = support * kl

    return {
        f"KL_{split}": float(kl),
        f"support_{split}": n_s,
        f"support_ratio_{split}": float(n_s / n),
        f"KL_corrected_{split}": float(kl_corr),
    }, edges_all, edges_subgroup, p_s, p

def load_data(res_root_path: str, 
              data_root_path: str = "./dataset/", return_data: bool = True, dataset_name: str = None):
    """
    Load dataset from a JSON file.
    """
    subgroup_id_suffix = "_subgroups_ids.json"
    template = rf"{res_root_path}/subgroup_ids/*_*_*{subgroup_id_suffix}"
    json_paths = list(glob.glob(template))

    all_data = {}
    if dataset_name is None:
        dataset_name = ""
    for json_path in json_paths:
        if json_path.endswith(subgroup_id_suffix) and dataset_name in json_path:
            meta = str(json_path).split(subgroup_id_suffix)[0].split("subgroup_ids/")[-1].split("_")
            if "bikesharing_ohe" not in json_path:
                dname = meta[0] 
                cv = int(meta[1].split("cv")[1])
                rep = int(meta[2].split("rep")[1])
            else:
                dname = meta[0]+"_ohe"
                cv = int(meta[2].split("cv")[1])
                rep = int(meta[3].split("rep")[1])
            with open(json_path, "r") as f:
                subgroup_ids = json.load(f)
        else:
            continue
        
        if return_data:
            df = pd.read_csv(data_root_path + f"/{dname}.csv")
            X = df.iloc[:, :-1].to_numpy(dtype=float)
            Y = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)  # keep 2D like R matrix
            n = Y.shape[0]

            test_index = subgroup_ids['test_indices']
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_index] = False

            Y_train = Y[train_mask, :]
            X_train = X[train_mask, :]
            Y_test = Y[test_index, :]
            X_test = X[test_index, :]

            all_data[(dname, cv, rep)] = {
                "X_train": X_train,
                "Y_train": Y_train,
                "X_test": X_test,
                "Y_test": Y_test,
                "subgroups": subgroup_ids
            }
        else:
            all_data[(dname, cv, rep)] = {
                "subgroups": subgroup_ids
            }

    return all_data


# Convert pandas dataframe rule to conditions in list representation (variable index, lower bound, upper bound)
def convert_rule(rule: pd.DataFrame):
    conditions = []
    # Iter over column
    for col in rule.columns:
        cur_col = rule[col]
        assert len(cur_col) == 3, f"Invalid rule: {rule}"
        if not pd.isna(cur_col['=']):
            conditions.append((int(col[1:]), int(cur_col['=']), int(cur_col['='])))
            continue
        elif sum(pd.isna(cur_col)) == 3:
            continue
        elif sum(pd.isna(cur_col)) == 2:
            if not pd.isna(cur_col['lower']):
                conditions.append((int(col[1:]), float(cur_col['lower']), np.infty))
            elif not pd.isna(cur_col['upper']):
                conditions.append((int(col[1:]), - np.infty, float(cur_col['upper'])))
            else:
                # This wont happen, but just in case
                raise ValueError(f"Invalid rule: {rule}")
        elif sum(pd.isna(cur_col)) == 1:
            if not pd.isna(cur_col['lower']) and not pd.isna(cur_col['upper']):
                conditions.append((int(col[1:]), float(cur_col['lower']), float(cur_col['upper'])))
            else:
                # This wont happen, but just in case
                raise ValueError(f"Invalid rule: {rule}")
    return conditions

Condition = tuple[int, float, float]
ConditionList = list[Condition]
ScoreFn = Callable[..., tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
UnionFn = Callable[[np.ndarray, list[ConditionList]], np.ndarray]


@dataclass
class ExpansionOptimizationResult:
    score: dict[str, Any]
    edges_all: np.ndarray
    edges_subgroup: np.ndarray
    p_s_train: np.ndarray
    p_train: np.ndarray
    complete_set_ids: np.ndarray
    test_subgroup_ids: np.ndarray
    condition_union_payload: list[dict[str, Any]]
    selected_full_conditions: list[ConditionList]
    boundary_condition: Optional[ConditionList]
    selected_full_indices: tuple[int, ...]
    selected_full_names: tuple[str, ...]
    boundary_parent_index: Optional[int]
    boundary_parent_name: Optional[str]


@dataclass
class _EvaluatedState:
    full_indices: tuple[int, ...]
    boundary_parent_index: Optional[int]
    boundary_condition: Optional[ConditionList]
    score: dict[str, Any]


def _extract_condition_list(condition: Optional[Sequence[Condition]]) -> Optional[ConditionList]:
    if condition is None:
        return None
    normalized: ConditionList = []
    for var, low, high in sorted(condition, key=lambda item: int(item[0])):
        low = float(low)
        high = float(high)
        if np.isneginf(low) and np.isposinf(high):
            continue
        normalized.append((int(var), low, high))
    return normalized


def _condition_subgroup(condition: Optional[Sequence[Condition]]) -> tuple[Condition, ...]:
    normalized = _extract_condition_list(condition)
    if normalized is None:
        return tuple()
    return tuple(normalized)


def _condition_mask(X: np.ndarray, condition: Sequence[Condition]) -> np.ndarray:
    mask = np.ones((X.shape[0],), dtype=bool)
    for var, low, high in condition:
        if low == high:
            mask = np.logical_and(mask, X[:, int(var)] == low)
        else:
            mask = np.logical_and(mask, np.logical_and(X[:, int(var)] >= low, X[:, int(var)] < high))
    return mask


def _condition_to_bounds(condition: Sequence[Condition], n_features: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.full(n_features, -np.inf, dtype=float)
    upper = np.full(n_features, np.inf, dtype=float)
    equality = np.zeros(n_features, dtype=bool)
    for var, low, high in condition:
        var = int(var)
        lower[var] = float(low)
        upper[var] = float(high)
        equality[var] = float(low) == float(high)
    return lower, upper, equality


def _bounds_to_condition_list(lower: np.ndarray, upper: np.ndarray) -> ConditionList:
    condition: ConditionList = []
    for var in range(lower.shape[0]):
        low = float(lower[var])
        high = float(upper[var])
        if np.isneginf(low) and np.isposinf(high):
            continue
        condition.append((var, low, high))
    return _extract_condition_list(condition) or []


def _nan_test_score() -> dict[str, float]:
    return {
        "KL_test": np.nan,
        "support_test": np.nan,
        "support_ratio_test": np.nan,
        "KL_corrected_test": np.nan,
    }

def _pearson_feature_ranking(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    feature_indices: Sequence[int],
) -> list[int]:
    """Rank ``feature_indices`` by |Pearson correlation| with the target inside ``mask`` (descending).

    Features whose correlation is undefined (constant column, or fewer than two samples) are placed
    last, in index order. Used only by ``partial_search="pearson"``.
    """
    y_sub = np.asarray(y[mask], dtype=float).ravel()
    strengths = np.full(len(feature_indices), -1.0, dtype=float)
    for pos, feature_idx in enumerate(feature_indices):
        x = np.asarray(X[mask, feature_idx], dtype=float).ravel()
        if x.shape[0] < 2 or np.std(x) == 0 or np.std(y_sub) == 0:
            continue
        corr = np.corrcoef(x, y_sub)[0, 1]
        if np.isfinite(corr):
            strengths[pos] = abs(corr)
    order = np.argsort(-strengths, kind="stable")
    return [int(feature_indices[pos]) for pos in order]


def optimize_search_expansion_union_v2(
    condition_union: Sequence[Sequence[Condition]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_min_samples: int,
    y_lim: tuple[float, float],
    score_subgroups_fn: ScoreFn,
    get_union_subgroup_ids_fn: UnionFn,
    candidate_names: Optional[Sequence[str]] = None,
    tol: float = 1e-12,
    partial_search: Literal["best", "pearson"] = "best",
) -> Optional[ExpansionOptimizationResult]:
    """Greedy Atomic Pruning (GAP): alternate SearchFull / SearchPartial until no move improves KL_train.

    ``partial_search`` selects how SearchPartial shrinks the bounding box of the partial atom:

    * ``"best"`` (default, as described in the paper): best-improvement search. Every single-bound
      cut over every non-equality feature is scored, and the one that most improves the score is
      committed. The result does not depend on the order in which features are visited.
    * ``"pearson"``: Pearson-guided first-improvement search. Features are visited in decreasing
      |Pearson correlation| with the target inside the current region (the correlation acts as a
      surrogate score for which feature to try first), and the best cut of the first feature that
      yields any improvement is committed without scoring the remaining features. The search space
      per iteration is smaller, so this is faster, but it may return a different, typically
      slightly worse, local optimum than ``"best"``.
    """
    _ = get_union_subgroup_ids_fn
    if partial_search not in ("best", "pearson"):
        raise ValueError(f"partial_search must be 'best' or 'pearson', got {partial_search!r}")

    candidate_conditions = [_extract_condition_list(condition) or [] for condition in condition_union]
    if len(candidate_conditions) == 0:
        return None

    n_candidates = len(candidate_conditions)
    n_features = X_train.shape[1]
    if candidate_names is None:
        candidate_names = tuple(str(idx) for idx in range(n_candidates))
    else:
        candidate_names = tuple(str(name) for name in candidate_names)

    atomic_train_masks = [_condition_mask(X_train, condition) for condition in candidate_conditions]
    parent_boundaries = [_condition_to_bounds(condition, n_features) for condition in candidate_conditions]
    full_condition_subgroups = [_condition_subgroup(condition) for condition in candidate_conditions]
    evaluation_cache: dict[tuple[tuple[int, ...], Optional[int], tuple[Condition, ...]], _EvaluatedState] = {}

    def build_train_mask(
        full_indices: Sequence[int],
        boundary_condition: Optional[Sequence[Condition]],
    ) -> np.ndarray:
        train_mask = np.zeros((X_train.shape[0],), dtype=bool)
        for idx in full_indices:
            train_mask = np.logical_or(train_mask, atomic_train_masks[int(idx)])
        if boundary_condition is not None:
            train_mask = np.logical_or(train_mask, _condition_mask(X_train, boundary_condition))
        return train_mask

    def build_test_mask(
        full_indices: Sequence[int],
        boundary_condition: Optional[Sequence[Condition]],
    ) -> np.ndarray:
        test_mask = np.zeros((X_test.shape[0],), dtype=bool)
        for idx in full_indices:
            test_mask = np.logical_or(test_mask, _condition_mask(X_test, candidate_conditions[int(idx)]))
        if boundary_condition is not None:
            test_mask = np.logical_or(test_mask, _condition_mask(X_test, boundary_condition))
        return test_mask

    def evaluate_state(
        full_indices: Sequence[int],
        boundary_parent_index: Optional[int],
        boundary_condition: Optional[Sequence[Condition]],
        normalize_full_boundary: bool = True,
    ) -> Optional[_EvaluatedState]:
        normalized_full = tuple(sorted(set(int(idx) for idx in full_indices)))
        normalized_boundary = _extract_condition_list(boundary_condition)
        normalized_parent = None if boundary_parent_index is None else int(boundary_parent_index)

        if normalize_full_boundary and normalized_parent is not None:
            if normalized_parent in normalized_full:
                normalized_parent = None
                normalized_boundary = None
            elif _condition_subgroup(normalized_boundary) == full_condition_subgroups[normalized_parent]:
                normalized_full = tuple(sorted(normalized_full + (normalized_parent,)))
                normalized_parent = None
                normalized_boundary = None

        cache_key = (normalized_full, normalized_parent, _condition_subgroup(normalized_boundary))
        if cache_key in evaluation_cache:
            return evaluation_cache[cache_key]

        if len(normalized_full) == 0 and normalized_parent is None:
            return None

        train_mask = build_train_mask(normalized_full, normalized_boundary)
        if int(np.count_nonzero(train_mask)) < max(n_min_samples, 50):
            return None

        try:
            score_train, _, _, _, _ = score_subgroups_fn(
                y_train,
                y_train,
                train_mask,
                "train",
                y_lim,
            )
        except ValueError:
            return None

        state = _EvaluatedState(
            full_indices=normalized_full,
            boundary_parent_index=normalized_parent,
            boundary_condition=normalized_boundary,
            score=dict(score_train),
        )
        evaluation_cache[cache_key] = state
        return state

    def materialize_state(state: _EvaluatedState) -> ExpansionOptimizationResult:
        train_mask = build_train_mask(state.full_indices, state.boundary_condition)
        test_mask = build_test_mask(state.full_indices, state.boundary_condition)
        complete_set_ids = np.flatnonzero(train_mask)
        score_train, edges_all, edges_subgroup, p_s_train, p_train = score_subgroups_fn(
            y_train,
            y_train,
            train_mask,
            "train",
            y_lim,
        )

        score = dict(score_train)
        test_subgroup_ids = np.flatnonzero(test_mask)
        if test_subgroup_ids.shape[0] == 0:
            score.update(_nan_test_score())
        else:
            score_test, _, _, _, _ = score_subgroups_fn(
                y_test,
                y_train,
                test_mask,
                "test",
                y_lim,
                edges_all=edges_all,
                edges_subgroup=edges_subgroup,
                p_s_train=p_s_train,
                p_train=p_train,
            )
            score.update(score_test)

        selected_full_conditions = [list(candidate_conditions[idx]) for idx in state.full_indices]
        selected_full_names = tuple(candidate_names[idx] for idx in state.full_indices)
        boundary_condition = None if state.boundary_condition is None else list(state.boundary_condition)
        boundary_parent_index = state.boundary_parent_index
        boundary_parent_name = None if boundary_parent_index is None else candidate_names[boundary_parent_index]

        condition_union_payload: list[dict[str, Any]] = []
        for idx in state.full_indices:
            condition_union_payload.append(
                {
                    "kind": "full",
                    "source_index": idx,
                    "source_name": candidate_names[idx],
                    "condition": list(candidate_conditions[idx]),
                }
            )
        if boundary_parent_index is not None and boundary_condition is not None:
            condition_union_payload.append(
                {
                    "kind": "boundary",
                    "source_index": boundary_parent_index,
                    "source_name": boundary_parent_name,
                    "condition": boundary_condition,
                }
            )

        return ExpansionOptimizationResult(
            score=score,
            edges_all=edges_all,
            edges_subgroup=edges_subgroup,
            p_s_train=p_s_train,
            p_train=p_train,
            complete_set_ids=complete_set_ids,
            test_subgroup_ids=test_subgroup_ids,
            condition_union_payload=condition_union_payload,
            selected_full_conditions=selected_full_conditions,
            boundary_condition=boundary_condition,
            selected_full_indices=state.full_indices,
            selected_full_names=selected_full_names,
            boundary_parent_index=boundary_parent_index,
            boundary_parent_name=boundary_parent_name,
        )

    def state_sort_key(state: _EvaluatedState) -> tuple[Any, ...]:
        boundary_parent = -1 if state.boundary_parent_index is None else state.boundary_parent_index
        return (state.full_indices, boundary_parent, _condition_subgroup(state.boundary_condition))

    def choose_improvement(
        baseline: _EvaluatedState,
        candidates: Sequence[Optional[_EvaluatedState]],
    ) -> Optional[_EvaluatedState]:
        best: Optional[_EvaluatedState] = None
        baseline_kl = baseline.score["KL_train"]
        for candidate in candidates:
            if candidate is None:
                continue
            candidate_kl = candidate.score["KL_train"]
            if candidate_kl <= baseline_kl + tol:
                continue
            if best is None:
                best = candidate
                continue
            best_kl = best.score["KL_train"]
            if candidate_kl > best_kl + tol:
                best = candidate
                continue
            if abs(candidate_kl - best_kl) <= tol and state_sort_key(candidate) < state_sort_key(best):
                best = candidate
        return best

    def refine_boundary(full_indices: Sequence[int], parent_index: int) -> Optional[_EvaluatedState]:
        current_condition = list(candidate_conditions[parent_index])
        current_state = evaluate_state(
            full_indices,
            parent_index,
            current_condition,
            normalize_full_boundary=False,
        )
        if current_state is None:
            return None

        parent_lower, parent_upper, equality = parent_boundaries[parent_index]
        scan_features = [idx for idx in range(n_features) if not equality[idx]]
        if partial_search == "pearson":
            subset_data_mask = build_train_mask(tuple(sorted(full_indices + (parent_index,))), None)
            scan_features = _pearson_feature_ranking(X_train, y_train, subset_data_mask, scan_features)

        while True:
            lower, upper, _ = _condition_to_bounds(current_condition, n_features)
            current_mask = _condition_mask(X_train, current_condition)
            if int(np.count_nonzero(current_mask)) <= 1:
                break
            best_state: Optional[_EvaluatedState] = None
            best_condition: Optional[ConditionList] = None

            def consider(trial_lower: np.ndarray, trial_upper: np.ndarray) -> None:
                # Keep the trial if it improves on the current box and beats the best cut seen so far.
                nonlocal best_state, best_condition
                trial_condition = _bounds_to_condition_list(trial_lower, trial_upper)
                trial_state = evaluate_state(
                    full_indices,
                    parent_index,
                    trial_condition,
                    normalize_full_boundary=False,
                )
                if trial_state is None:
                    return
                trial_kl = trial_state.score["KL_train"]
                if trial_kl <= current_state.score["KL_train"] + tol:
                    return
                if best_state is None:
                    best_state, best_condition = trial_state, trial_condition
                    return
                best_kl = best_state.score["KL_train"]
                if trial_kl > best_kl + tol or (
                    abs(trial_kl - best_kl) <= tol and state_sort_key(trial_state) < state_sort_key(best_state)
                ):
                    best_state, best_condition = trial_state, trial_condition

            for feature_idx in scan_features:
                values = np.unique(X_train[current_mask, feature_idx])
                if values.shape[0] <= 1:
                    continue

                # Raise the lower bound to an observed value inside the current box.
                for new_low in values[1:]:
                    if new_low <= lower[feature_idx]:
                        continue
                    if np.isfinite(upper[feature_idx]) and new_low >= upper[feature_idx]:
                        continue
                    if float(new_low) < parent_lower[feature_idx]:
                        continue
                    trial_lower = lower.copy()
                    trial_lower[feature_idx] = float(new_low)
                    consider(trial_lower, upper)

                # Lower the upper bound to an observed value inside the current box.
                for new_high in values[1:]:
                    if new_high >= upper[feature_idx]:
                        continue
                    if np.isfinite(lower[feature_idx]) and new_high <= lower[feature_idx]:
                        continue
                    if float(new_high) > parent_upper[feature_idx]:
                        continue
                    trial_upper = upper.copy()
                    trial_upper[feature_idx] = float(new_high)
                    consider(lower, trial_upper)

                if partial_search == "pearson" and best_state is not None:
                    # First improvement: commit the best cut of the highest-ranked improving feature.
                    break

            if best_state is None or best_condition is None:
                break
            current_condition = best_condition
            current_state = best_state
        return current_state

    def optimize_full_core(initial_state: _EvaluatedState) -> _EvaluatedState:
        current_state = initial_state
        while True:
            selected = set(current_state.full_indices)
            candidate_states: list[Optional[_EvaluatedState]] = []
            for idx in range(n_candidates):
                if idx == current_state.boundary_parent_index:
                    continue
                if idx in selected:
                    next_full = tuple(full_idx for full_idx in current_state.full_indices if full_idx != idx)
                else:
                    next_full = tuple(sorted(current_state.full_indices + (idx,)))
                candidate_states.append(
                    evaluate_state(
                        next_full,
                        current_state.boundary_parent_index,
                        current_state.boundary_condition,
                        normalize_full_boundary=True,
                    )
                )

            next_state = choose_improvement(current_state, candidate_states)
            if next_state is None:
                return current_state
            current_state = next_state

    def optimize_boundary(initial_state: _EvaluatedState) -> _EvaluatedState:
        core_full_indices = initial_state.full_indices
        candidate_states: list[Optional[_EvaluatedState]] = [
            evaluate_state(core_full_indices, None, None, normalize_full_boundary=True)
        ]
        for parent_index in range(n_candidates):
            boundary_core = tuple(idx for idx in core_full_indices if idx != parent_index)
            refined_state = refine_boundary(boundary_core, parent_index)
            if refined_state is None:
                candidate_states.append(None)
                continue
            candidate_states.append(
                evaluate_state(
                    refined_state.full_indices,
                    refined_state.boundary_parent_index,
                    refined_state.boundary_condition,
                    normalize_full_boundary=True,
                )
            )

        next_state = choose_improvement(initial_state, candidate_states)
        if next_state is None:
            return initial_state
        return next_state

    current_state = evaluate_state(tuple(range(n_candidates)), None, None, normalize_full_boundary=True)
    if current_state is None:
        return None

    current_state = optimize_full_core(current_state)
    while True:
        improved = False
        boundary_state = optimize_boundary(current_state)
        if boundary_state.score["KL_train"] > current_state.score["KL_train"] + tol:
            current_state = boundary_state
            improved = True

        full_core_state = optimize_full_core(current_state)
        if full_core_state.score["KL_train"] > current_state.score["KL_train"] + tol:
            current_state = full_core_state
            improved = True

        if not improved:
            break

    return materialize_state(current_state)