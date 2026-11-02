from .utils_search import score_subgroups
from fnmatch import fnmatch
import pandas as pd
import numpy as np
import glob
import json
from pathlib import Path
from .utils import setup_logger
from .utils_search import optimize_search_expansion_union_v2

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

def load_rules(res_root_path: str, return_condition: bool = False, dataset_name: str = None):
    """
    Load dataset from a JSON file.
    """
    subgroup_rules_suffix = "_subgroups*_rules.csv"
    template = rf"{res_root_path}/subgroup_rules/*_*_*{subgroup_rules_suffix}"
    csv_paths = list(glob.glob(template))

    all_rules = {}
    if dataset_name is None:
        dataset_name = ""
    for csv_path in csv_paths:
        if fnmatch(csv_path, f"*{subgroup_rules_suffix}") and (dataset_name in csv_path):
            meta = str(csv_path).split(subgroup_rules_suffix)[0].split("subgroup_rules/")[-1].split("_")
            if "bikesharing_ohe" not in csv_path:
                dname = meta[0] 
                cv = int(meta[1].split("cv")[1])
                rep = int(meta[2].split("rep")[1])
                subgroup_id = int(meta[3].split("subgroups")[1])
            else:
                dname = meta[0]+"_ohe"
                cv = int(meta[2].split("cv")[1])
                rep = int(meta[3].split("rep")[1])
                subgroup_id = int(meta[4].split("subgroups")[1])
            rules_df = pd.read_csv(csv_path, index_col=0)
            rules_df = rules_df.drop(columns=["cv", "rep", "dataset", "subgroup_id"], errors="raise") # redundant info
        else:
            continue
        
        if return_condition:
            rules_df = convert_rule(rules_df)
        if (dname, cv, rep) in all_rules:
            all_rules[(dname, cv, rep)].update({str(subgroup_id): rules_df})
        else:
            all_rules[(dname, cv, rep)] = {str(subgroup_id): rules_df}

    return all_rules


from typing import List, Tuple, Literal
def get_subgroup_ids(X: np.array, condition: List[Tuple[int, float, float]]):
    candidate_ids = np.ones((X.shape[0],),dtype=bool)
    for cond in condition:
        var, low, high = cond
        var = int(var)
        candidate_ids = np.logical_and(candidate_ids, np.logical_and(X[:,var]>=low, X[:,var]<high) if low != high else X[:,var]==low)
    return np.where(candidate_ids)[0]


from sklearn.preprocessing import StandardScaler
def get_meta(res_path: str, data_root_path: str, dataset_name: str = None):
    try:
        loaded_data = load_data(res_path, data_root_path = data_root_path, dataset_name=dataset_name)
        loaded_rules = load_rules(res_path, dataset_name=dataset_name)
    except Exception as e:
        print(f"Error loading results from {res_path}: {e}")
        return None

    all_meta = {}
    for kk in loaded_data:
        # dname, cv, rep = kk
        try:
            y_train = loaded_data[kk]["Y_train"].flatten() if "Y_train" in loaded_data[kk] else None
            y_test = loaded_data[kk]["Y_test"].flatten() if "Y_test" in loaded_data[kk] else None
            X_train = loaded_data[kk]["X_train"] if "X_train" in loaded_data[kk] else None
            X_test = loaded_data[kk]["X_test"] if "X_test" in loaded_data[kk] else None
            subgroup_ids = loaded_data[kk]["subgroups"]
                
        except Exception as e:
            print(f"Error processing {kk} of {res_path}: {e}")
            continue
        try:
            rules = loaded_rules[kk]
        except Exception as e:
            print(f"Warning: Error loading rules of {kk} from {res_path}: {e}, considered entire as a subgroup.")
            # print(subgroup_ids)
            all_meta[kk] = None # Placeholder for entire data as a subgroup
            continue
        left_bound = np.min(np.concatenate((y_train, y_test), axis=0)) - 1e-3
        right_bound = np.max(np.concatenate((y_train, y_test), axis=0)) + 1e-3
        # out = []
        all_subgroups_meta = {}
        for id in subgroup_ids:
            if id=="test_indices":
                continue
            train_subgroup_ids = subgroup_ids[id]["train"]
            test_subgroup_ids = subgroup_ids[id]["test"]
            if id in rules:
                rule = rules[id]
            else:
                rule = None
                print(f"Warning: no rule found for subgroup {id} of {kk} of {res_path}")
            if len(train_subgroup_ids) > 0:
                score, edges_all, edges_subgroup, p_s_train, p_train = score_subgroups(
                    y_train, y_train, train_subgroup_ids, "train", (left_bound, right_bound))

                # score['rule_df'] = rule
                condition = convert_rule(rule)
                score['condition'] = condition
                # verified_train = verify_rule_id_match(X_train, train_subgroup_ids, condition)
                post_train_subgroup_ids = get_subgroup_ids(X_train, condition)
                verified_train = np.array_equal(np.sort(train_subgroup_ids), np.sort(post_train_subgroup_ids))
                score['verified_train'] = verified_train
                score['train_subgroup_ids'] = train_subgroup_ids
                score['test_subgroup_ids'] = test_subgroup_ids
                score.pop("KL_corrected_train")
                if len(test_subgroup_ids) > 0:
                    score_test, _, _, _, _ = score_subgroups(
                        y_test, y_train, test_subgroup_ids, "test", (left_bound, right_bound), 
                        edges_all=edges_all, edges_subgroup=edges_subgroup,
                        p_s_train=p_s_train, p_train=p_train)
                    post_test_subgroup_ids = get_subgroup_ids(X_test, condition)
                    verified_test = np.array_equal(np.sort(test_subgroup_ids), np.sort(post_test_subgroup_ids))
                    score.update(score_test)
                    score['verified_test'] = verified_test
                    score.pop("KL_corrected_test")
            else:
                score = None
            all_subgroups_meta[id] = score
        all_meta[kk] = all_subgroups_meta
    return all_meta

def get_union_subgroup_ids(X: np.array, conditions: List[List[Tuple[int, float, float]]]):
    set_ids = np.zeros((X.shape[0],),dtype=bool)
    for condition in conditions:
        if len(condition) == 0:
            continue
        candidate_ids = np.ones((X.shape[0],),dtype=bool)
        for cond in condition:
            var, low, high = cond
            var = int(var)
            candidate_ids = np.logical_and(candidate_ids, np.logical_and(X[:,var]>=low, X[:,var]<high) if low != high else X[:,var]==low)
        set_ids = np.logical_or(set_ids, candidate_ids)
    return np.where(set_ids)[0]

def search_expansion_mixed(method_name: Literal["syflow", "rsd", "sd_kl", "sd_mean", "cdtree"], 
                     alpha: float, res_root_path: str, 
                     data_root_path : str,
                     output_path: str, dataset_name: str = None,
                     partial_search: str = "best"):
    assert method_name in ["syflow", "rsd", "sd_kl", "sd_mean", "cdtree"]
    assert partial_search in ["best", "pearson"], partial_search
    if method_name == "cdtree":
        other_meta = None
    else:
        other_meta = get_meta(f"{res_root_path}/{method_name}", data_root_path=data_root_path, dataset_name=dataset_name)
    cdtree_res = load_data(f"{res_root_path}/cdtree", data_root_path=data_root_path, dataset_name=dataset_name)
    cdtree_meta = get_meta(f"{res_root_path}/cdtree", data_root_path=data_root_path, dataset_name=dataset_name)

    # logger = setup_logger(method_name+"-"+dataset_name, output_path, os.getpid())
    results_collection = []
    aux_results = []
    if other_meta is None:
        # PLACEHOLDER for Exp 3, self-search (constrained on support size) using CDTree's subgroups
        other_meta = {key: None for key in cdtree_meta} # Consider the entire dataset as a subgroup for each experiment key
    for exp_key in other_meta:
        _at_exp_key_meta = other_meta[exp_key]
        _at_exp_key_res = cdtree_res[exp_key]
        cdtree_at_exp_key_meta = cdtree_meta[exp_key]
        X_train = _at_exp_key_res['X_train']
        y_train = _at_exp_key_res['Y_train']
        X_test = _at_exp_key_res['X_test']
        y_test = _at_exp_key_res['Y_test']

        # Check if alpha is float or int
        if alpha <= 1.0:
            # alpha = float(alpha)
            n_min_samples = np.round(X_train.shape[0] * alpha).astype(int)
        else:
            # alpha = float(alpha)
            n_min_samples = int(alpha)

        # n_min_samples = np.round(X_train.shape[0] * alpha).astype(int)
        left_bound = np.min(np.concatenate((y_train, y_test), axis=0)) - 1e-3
        right_bound = np.max(np.concatenate((y_train, y_test), axis=0)) + 1e-3

        best_score = {}
        best_original_score = {}
        best_aux_result = None
        good_origin = False
        good_cdtree = False
        complete_set_ids = np.array([], dtype=int)
        test_subgroup_ids = np.array([], dtype=int)
        condition_union = []

        FLAG_empty_other_subgroup = False
        if _at_exp_key_meta is None: # The entire dataset as a subgroup
            _at_exp_key_meta = {"-1": None}
            FLAG_empty_other_subgroup = True
        for subgroup_id in _at_exp_key_meta:
            # Set other to full set as the entire subgroup
            if (FLAG_empty_other_subgroup) or (_at_exp_key_meta[subgroup_id] is None):
                split = "train"
                score_train_original = {
                    f"KL_{split}": 0.0, # The entire dataset is considered to be the subgroup
                    f"support_{split}": None,
                    f"support_ratio_{split}": 1.0,
                    f"KL_corrected_{split}": 0.0,
                }
                split = "test"
                score_test_original = {
                    f"KL_{split}": 0.0,
                    f"support_{split}": None,
                    f"support_ratio_{split}": 1.0,
                    f"KL_corrected_{split}": 0.0,
                }
                best_original_score.update(score_train_original)
                best_original_score.update(score_test_original)
                good_origin = True
            else:
                train_subgroup_ids_other= _at_exp_key_meta[subgroup_id]['train_subgroup_ids']

                score_original = _at_exp_key_meta[subgroup_id]
                if score_original['support_train'] < max(n_min_samples, 50):
                    continue

                if best_original_score == {}:
                    # or (best_original_score['support_train'] >= n_min_samples)
                    best_original_score = score_original
                    # if score_original['support_train'] >= n_min_samples:
                    #     good_origin = True
                    good_origin = True
                elif score_original['KL_train'] > best_original_score['KL_train']:
                    # if score_original['support_train'] >= n_min_samples:
                    best_original_score = score_original
                    good_origin = True

            intersected_list = []
            condition_union = []
            # CDTree wont have None rule set only empty []
            for tr_sub_cdtree in cdtree_at_exp_key_meta:
                temp = cdtree_at_exp_key_meta[tr_sub_cdtree]

                if (FLAG_empty_other_subgroup) or (_at_exp_key_meta[subgroup_id] is None): # No intersection is needed
                    intersected_list.append(tr_sub_cdtree)
                    condition = temp['condition']
                    condition_union.append(condition)
                else:
                    # Check if the temp is overlapped with train_subgroup_ids_other
                    if set(temp['train_subgroup_ids']).intersection(set(train_subgroup_ids_other)):
                        intersected_list.append(tr_sub_cdtree)
                        condition = temp['condition']
                        condition_union.append(condition)
            

            if len(condition_union) == 0:
                split = "train"
                score_train = {
                    f"KL_{split}": 0.0, # The entire dataset is considered to be the subgroup
                    f"support_{split}": None,
                    f"support_ratio_{split}": 1.0,
                    f"KL_corrected_{split}": 0.0,
                }
                split = "test"
                score_test = {
                    f"KL_{split}": 0.0,
                    f"support_{split}": None,
                    f"support_ratio_{split}": 1.0,
                    f"KL_corrected_{split}": 0.0,
                }
                best_score = score_train
                best_score.update(score_test)
                best_aux_result = {
                    "new_rules": None,
                    "new_rule_payload": None,
                    "selected_full_rule_sets": None,
                    "selected_full_rule_set_indices": None,
                    "boundary_parent_rule_set": None,
                    "boundary_parent_rule_set_index": None,
                    "boundary_condition": None,
                    "new_train_subgroup_ids": None,
                    "new_test_subgroup_ids": None,
                }
                good_cdtree = True
                continue
            elif len(condition_union) == 1:
                if len(condition_union[0]) == 0:
                    # Empty CDTree, means entire dataset as a subgroup
                    split = "train"
                    score_train = {
                        f"KL_{split}": 0.0, # The entire dataset is considered to be the subgroup
                        f"support_{split}": None,
                        f"support_ratio_{split}": 1.0,
                        f"KL_corrected_{split}": 0.0,
                    }
                    split = "test"
                    score_test = {
                        f"KL_{split}": 0.0,
                        f"support_{split}": None,
                        f"support_ratio_{split}": 1.0,
                        f"KL_corrected_{split}": 0.0,
                    }
                    best_score = score_train
                    best_score.update(score_test)
                    best_aux_result = {
                        "new_rules": None,
                        "new_rule_payload": None,
                        "selected_full_rule_sets": None,
                        "selected_full_rule_set_indices": None,
                        "boundary_parent_rule_set": None,
                        "boundary_parent_rule_set_index": None,
                        "boundary_condition": None,
                        "new_train_subgroup_ids": None,
                        "new_test_subgroup_ids": None,
                    }
                    good_cdtree = True
                    continue

            optimized_result = optimize_search_expansion_union_v2(
                condition_union=condition_union,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_min_samples=n_min_samples,
                y_lim=(left_bound, right_bound),
                score_subgroups_fn=score_subgroups,
                get_union_subgroup_ids_fn=get_union_subgroup_ids,
                candidate_names=intersected_list,
                partial_search=partial_search,
            )
            if optimized_result is None:
                continue

            score = dict(optimized_result.score)
            complete_set_ids = optimized_result.complete_set_ids
            test_subgroup_ids = optimized_result.test_subgroup_ids
            condition_union = [entry["condition"] for entry in optimized_result.condition_union_payload]

            good_cdtree = good_cdtree or len(complete_set_ids) >= n_min_samples
            if best_score == {} or score['KL_train'] > best_score['KL_train']:
                best_score = score
                best_aux_result = {
                    "new_rules": condition_union,
                    "new_rule_payload": optimized_result.condition_union_payload,
                    "selected_full_rule_sets": list(optimized_result.selected_full_names),
                    "selected_full_rule_set_indices": list(optimized_result.selected_full_indices),
                    "boundary_parent_rule_set": optimized_result.boundary_parent_name,
                    "boundary_parent_rule_set_index": optimized_result.boundary_parent_index,
                    "boundary_condition": optimized_result.boundary_condition,
                    "new_train_subgroup_ids": complete_set_ids.tolist(),
                    "new_test_subgroup_ids": test_subgroup_ids.tolist(),
                }

        # Handling empty CDTree case
        if best_score == {}:
            # No subgroup passed the support threshold or optimizer; emit metadata-only row.
            pass

        best_score.update({
            "dataset": exp_key[0],
            "cv": exp_key[1],
            "rep": exp_key[2],
            "alpha": alpha,
            "base": f"cdtree-{method_name}",
            "satisfy_alpha": good_cdtree
        })
        if best_original_score == {}:
            pass
        best_original_score.update({
            "dataset": exp_key[0],
            "cv": exp_key[1],
            "rep": exp_key[2],
            "alpha": alpha,
            "base": f"original-{method_name}",
            "satisfy_alpha": good_origin
        })
        aux_results.append({
            "verified_train": best_original_score['verified_train'] if 'verified_train' in best_original_score else None,
            "train_subgroup_ids": best_original_score['train_subgroup_ids'] if 'train_subgroup_ids' in best_original_score else None,
            "test_subgroup_ids": best_original_score['test_subgroup_ids'] if 'test_subgroup_ids' in best_original_score else None,
            "verified_test": best_original_score['verified_test'] if 'verified_test' in best_original_score else None,
            "new_rules": best_aux_result["new_rules"] if best_aux_result is not None else [],
            "new_rule_payload": best_aux_result["new_rule_payload"] if best_aux_result is not None else [],
            "selected_full_rule_sets": best_aux_result["selected_full_rule_sets"] if best_aux_result is not None else [],
            "selected_full_rule_set_indices": best_aux_result["selected_full_rule_set_indices"] if best_aux_result is not None else [],
            "boundary_parent_rule_set": best_aux_result["boundary_parent_rule_set"] if best_aux_result is not None else None,
            "boundary_parent_rule_set_index": best_aux_result["boundary_parent_rule_set_index"] if best_aux_result is not None else None,
            "boundary_condition": best_aux_result["boundary_condition"] if best_aux_result is not None else None,
            "new_train_subgroup_ids": best_aux_result["new_train_subgroup_ids"] if best_aux_result is not None else [],
            "new_test_subgroup_ids": best_aux_result["new_test_subgroup_ids"] if best_aux_result is not None else [],
            "dataset": exp_key[0],
            "cv": exp_key[1],
            "rep": exp_key[2],
            "alpha": alpha,
        })
        results_collection.append(best_score)
        results_collection.append(best_original_score)
        # logger.info("Completed job for dataset=%s, cv=%d, rep=%d", exp_key[0], exp_key[1], exp_key[2])
    results_df = pd.DataFrame(results_collection)
    results_df.drop(columns=["condition", "verified_train", "train_subgroup_ids", "test_subgroup_ids", "verified_test"], inplace=True, errors="ignore")

    # Save subgroup scores
    if dataset_name is None:
        dataset_name = "All"
    output_folder = Path(f"{output_path}/results/") / "subgroup_scores"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_{method_name}_{alpha}_subgroups_scores.csv"
    results_df.to_csv(output_file, index=False)

    output_folder = Path(f"{output_path}/results/") / "subgroup_aux"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_{method_name}_{alpha}_subgroups_aux.jsonl"
    with open(output_file, "w") as f:
        for r in aux_results:
            f.write(json.dumps(r) + "\n")
    return results_df, aux_results