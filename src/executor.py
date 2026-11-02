from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from . import build_tree, predict_CDTree, cover_test_data, cover_test_data_with_rules, score_subgroups, setup_logger
from . import regret as regret_fn

from .utils import *
import random

from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

from pathlib import Path
from copy import deepcopy

from .syflow import *
from .syflow.methods import replace_feature_names, convert, QF_WKL
from sklearn.preprocessing import StandardScaler
import pysubgroup as ps
from collections import namedtuple
import warnings
import time
from .syflow.RSD.rulelist_class import MDLRuleList, reduce

warnings.filterwarnings("ignore")
device = torch.device("cpu")
verbose = False

ROOT_PATH = "./"

def set_random(seed):
    np.random.seed(seed)
    random.seed(seed)

def make_folds_indices(n: int, n_folds: int = 5, random_state: int = 42) -> List[np.ndarray]:
    """
    Cross-validation fold indices
    """
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)
    return np.array_split(indices, n_folds)

def drive_cdtree(arguments):
    rep = arguments['rep']
    cv = arguments['cv']
    n_cv = arguments['n_cv']
    dataset_index = arguments['dataset_index']
    dataset_paths = arguments['dataset_paths']
    root_path = arguments['root_path']

    data_path = dataset_paths[dataset_index]
    dataset_name = str(data_path).split(".csv")[0].split("/")[-1]

    df = pd.read_csv(data_path)
    z = df.iloc[:, :-1].to_numpy(dtype=float)
    x = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)  # keep 2D like R matrix
    n = x.shape[0]
    
    # Finish this for parallelism
    set_random(1000 * rep + 10 * cv + dataset_index)
    folds = make_folds_indices(n, n_folds=n_cv, random_state=1000 * rep + 10 * n_cv + dataset_index)

    test_index = folds[cv - 1]
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_index] = False

    x_train = x[train_mask, :]
    z_train = z[train_mask, :]
    x_test = x[test_index, :]
    z_test = z[test_index, :]

    left_bound = np.min(x) - 1e-3
    right_bound = np.max(x) + 1e-3

    t0 = time.time()
    tree = build_tree(
        x=x_train,
        z_matrix=z_train,
        x_test=None,
        z_test=None,
        eps=1e-3,
        print_process=False,
        regret_fn=regret_fn,
        left_bound=left_bound,
        right_bound=right_bound,
    )
    t1 = time.time()

    log_predicted_density = predict_CDTree(
        tree, x_test, z_test
    )

    log_predicted_density_train = predict_CDTree(
        tree, x_train, z_train
    )

    test_indices_list = cover_test_data(tree, x_test, z_test)
    leaves = tree.node_list if hasattr(tree, "node_list") else tree["node_list"]
    L = len(leaves)
    out = []
    ids_subgroup = {}

    for k in range(L):
        ids_subgroup_local = {}
        leaf = leaves[k]
        idx_train = leaf.indices if leaf.indices is not None else []
        # score, edges, p_s_train, p_train = score_subgroups(x_train, x_train, idx_train, "train")
        score, edges_all, edges_subgroup, p_s_train, p_train = score_subgroups(x_train, x_train, idx_train, "train", (left_bound, right_bound))
        # temp = deepcopy(score)
        idx_test = test_indices_list[k]
        ids_subgroup_local["train"] = idx_train.ravel().tolist()
        supp_te = len(idx_test)
        if supp_te > 0:
            ids_subgroup_local["test"] = idx_test.ravel().tolist()
            score_test, _, _, _, _ = score_subgroups(x_test, x_train, idx_test, "test", (left_bound, right_bound), edges_all=edges_all, edges_subgroup=edges_subgroup, p_s_train=p_s_train, p_train=p_train)
        else:
            ids_subgroup_local["test"] = []
            split = "test"
            score_test = {
                f"KL_{split}": np.nan,
                f"support_{split}": np.nan,
                f"support_ratio_{split}": np.nan,
                f"KL_corrected_{split}": np.nan,
            }
        
        score.update(score_test)
        score.update({
            "leaf_id": k
        })
        out.append(score)
        ids_subgroup[k] = ids_subgroup_local
    ids_subgroup["test_indices"] = test_index.ravel().tolist()

    scores_both = pd.DataFrame(out)
    ord_idx = scores_both.sort_values(
        by=["KL_train", "KL_test"], # This is fair
        ascending=[False, False],
        na_position="last"
    ).index.to_numpy()
    scores_both = scores_both.loc[ord_idx].reset_index(drop=True)

    # Save subgroup scores for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/cdtree") / "subgroup_scores"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_scores.csv"
    scores_both['cv'] = cv
    scores_both['rep'] = rep
    scores_both['dataset'] = dataset_name
    scores_both.to_csv(output_file, index=False)

    # Save subgroup IDs for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/cdtree") / "subgroup_ids"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_ids.json"
    with open(output_file, "w") as f:
        json.dump(ids_subgroup, f)

    negloglike = -np.mean(log_predicted_density / np.log2(np.exp(1)))
    runtime = float(t1 - t0)
    negloglikes_train = -np.mean(log_predicted_density_train / np.log2(np.exp(1)))
    node_list = tree["node_list"] if isinstance(tree, dict) else tree.node_list
    numLeafs = len(node_list)
    meanNumBins = float(np.mean([len(leaf.x_hist["hist_fixed"]["counts"]) for leaf in node_list])) if node_list else np.nan
    res = dict(
        dataset=dataset_name,
        cv=cv,
        rep=rep,
        negloglike=negloglike,
        negloglikes_train=negloglikes_train,
        runtime=runtime,
        numLeafs=numLeafs,
        meanNumBins=meanNumBins,
    )

    feature_name_dummy = [f"X{i}" for i in range(z.shape[1])]
    rules_dfs = cover_test_data_with_rules(tree, x_test, z_test, feature_name_dummy)

    # Save rules for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/cdtree") / "subgroup_rules"
    output_folder.mkdir(parents=True, exist_ok=True)
    for idn in range(len(rules_dfs)):
        if rules_dfs[idn] is None:
            continue
        rules_dfs[idn]['cv'] = cv
        rules_dfs[idn]['rep'] = rep
        rules_dfs[idn]['dataset'] = dataset_name
        rules_dfs[idn]['subgroup_id'] = idn
        output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups{idn}_rules.csv"
        rules_dfs[idn].to_csv(output_file, index=True)

    return res, scores_both


def drive_syflow(arguments):
    rep = arguments['rep']
    cv = arguments['cv']
    n_cv = arguments['n_cv']
    dataset_index = arguments['dataset_index']
    dataset_paths = arguments['dataset_paths']
    root_path = arguments['root_path']
    config = arguments['config']()

    data_path = dataset_paths[dataset_index]
    dataset_name = str(data_path).split(".csv")[0].split("/")[-1]

    df = pd.read_csv(data_path)
    z = df.iloc[:, :-1].to_numpy(dtype=float)
    x = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)  # keep 2D like R matrix
    feature_names = df.columns[:-1].tolist()
    feature_names_dummy = [f"X{i}" for i in range(len(feature_names))]
    n = x.shape[0]

    seed = 1000 * rep + 10 * cv + dataset_index
    config.seed = seed
    n_subgroups = config.n_subgroups
    set_random(seed)

    folds = make_folds_indices(n, n_folds=n_cv, random_state=1000 * rep + 10 * n_cv + dataset_index)

    test_index = folds[cv - 1]
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_index] = False

    Y = x[train_mask, :]
    X = z[train_mask, :] # Naming is different from CDTree
    Y_test = x[test_index, :]
    X_test = z[test_index, :]

    # subgroups, rules = run_syflow(features, target.reshape(-1,1), config, config.n_subgroups, feature_names)
    cut_points = torch.zeros((X.shape[1],2))
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    Y_train = scaler_y.fit_transform(Y)
    X_tensor = torch.tensor(X_train,dtype=torch.float64)
    Y_tensor = torch.tensor(Y_train,dtype=torch.float64)

    X_test = scaler_x.fit_transform(X_test)
    Y_test = scaler_y.fit_transform(Y_test)
    left_bound = np.min(np.concatenate((Y_train, Y_test), axis=0)) - 1e-3
    right_bound = np.max(np.concatenate((Y_train, Y_test), axis=0)) + 1e-3
    X_tensor_test = torch.tensor(X_test, dtype=torch.float64)

    # subgroups = []
    priors = []
    rules = []
    pop_flow = None

    ids_subgroup = {}
    out = []

    train_times = []
    model_times = []
    test_times = []

    rules_dfs = [[None]] * n_subgroups
    for idn in range(n_subgroups):
        ids_subgroup_local = {}
        # print("Discovering Subgroup #{}".format(idn+1))
        t0 = time.time()
        for i in range(X.shape[1]):
            cut_points[i,0] = torch.quantile(X_tensor[:,i],0)
            cut_points[i,1] = torch.quantile(X_tensor[:,i],1)
        cut_points = torch.sort(cut_points,dim=1)[0]
        classifier = And_Finder(cut_points,temperature=config.temperature,use_weights=config.use_weights,bin_deviation=config.bin_deviation)
        flows, classifier = syflow(X_tensor,Y_tensor,classifier,flow_population=pop_flow,subgroup_priors=priors,
                                            pop_train_epochs=config.pop_train_epochs,subgroup_train_epochs=config.subgroup_train_epochs,final_fit_epochs=config.final_fit_epochs,
                                        device=device,verbose=verbose,lr_flow=config.lr_flow,alpha=config.alpha,
                                    lr_classifier=config.lr_classifier,lambd=config.lambd,config=config)
        t1 = time.time()
        model_times.append(float(t1 - t0))
        pop_flow = flows[0]
        priors.append(flows[1])
        classifier = classifier.to(torch.device("cpu"))
        subgroup_train_bool = torch.argmax(classifier(X_tensor),dim=1).detach().numpy()==1
        t1_5 = time.time()
        train_times.append(float(t1_5-t1))
        rules.append(classifier.get_rules(cut_points,scaler=scaler_x,feature_names=feature_names,X=X))
        rules_dfs[idn] = classifier.get_rules_df(cut_points,scaler=scaler_x,feature_names=feature_names_dummy,X=X)

        # Y_tensor_test = torch.tensor(Y_test,dtype=torch.float64)

        t2 = time.time()
        subgroup_test_bool = torch.argmax(classifier(X_tensor_test),dim=1).detach().numpy()==1
        t3 = time.time()
        test_times.append(float(t3-t2))
        ids_subgroup_local["test"] = np.argwhere(subgroup_test_bool).ravel().tolist()
        ids_subgroup_local["train"] = np.argwhere(subgroup_train_bool).ravel().tolist()
        ids_subgroup[idn] = ids_subgroup_local
        score, edges_all, edges_subgroup, p_s_train, p_train = score_subgroups(Y_train, Y_train, subgroup_train_bool, "train", (left_bound, right_bound))
        # temp = deepcopy(score)
        if len(ids_subgroup_local["test"]) > 0:
            score_test, _, _, _, _ = score_subgroups(Y_test, Y_train, subgroup_test_bool, "test", (left_bound, right_bound), edges_all=edges_all, edges_subgroup=edges_subgroup, p_s_train=p_s_train, p_train=p_train)
        else:
            split = "test"
            score_test = {
                f"KL_{split}": np.nan,
                f"support_{split}": np.nan,
                f"support_ratio_{split}": np.nan,
                f"KL_corrected_{split}": np.nan,
    
            }
        score.update(score_test)
        score.update({
            "subgroup_id": idn,
            "rule": rules[-1]
        })
        out.append(score)
    ids_subgroup["test_indices"] = test_index.ravel().tolist()

    scores_both = pd.DataFrame(out)
    ord_idx = scores_both.sort_values(
        by=["KL_train", "KL_test"], # This is fair
        ascending=[False, False],
        na_position="last"
    ).index.to_numpy()
    scores_both = scores_both.loc[ord_idx].reset_index(drop=True)

    # Save subgroup scores for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/syflow") / "subgroup_scores"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_scores.csv"
    scores_both['cv'] = cv
    scores_both['rep'] = rep
    scores_both['dataset'] = dataset_name
    scores_both.to_csv(output_file, index=False)

    # Save subgroup IDs for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/syflow") / "subgroup_ids"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_ids.json"
    with open(output_file, "w") as f:
        json.dump(ids_subgroup, f)

    res = dict(
        dataset=dataset_name,
        cv=cv,
        rep=rep,
        model_time = np.sum(model_times),
        n_model = n_subgroups,
        train_time=np.sum(train_times),
        test_time=np.sum(test_times),
        n_subgroups=n_subgroups,
    )

    # Save rules for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/syflow") / "subgroup_rules"
    output_folder.mkdir(parents=True, exist_ok=True)
    for idn in range(len(rules_dfs)):
        if rules_dfs[idn] is None:
            continue
        rules_dfs[idn]['cv'] = cv
        rules_dfs[idn]['rep'] = rep
        rules_dfs[idn]['dataset'] = dataset_name
        rules_dfs[idn]['subgroup_id'] = idn
        output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups{idn}_rules.csv"
        rules_dfs[idn].to_csv(output_file, index=True)

    return res, scores_both


def drive_sd_kl(arguments):
    rep = arguments['rep']
    cv = arguments['cv']
    n_cv = arguments['n_cv']
    dataset_index = arguments['dataset_index']
    dataset_paths = arguments['dataset_paths']
    root_path = arguments['root_path']
    config = arguments['config']()

    data_path = dataset_paths[dataset_index]
    dataset_name = str(data_path).split(".csv")[0].split("/")[-1]

    df = pd.read_csv(data_path)
    z = df.iloc[:, :-1].to_numpy(dtype=float)
    x = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)  # keep 2D like R matrix
    feature_names = df.columns[:-1].tolist()
    n = x.shape[0]

    seed = 1000 * rep + 10 * cv + dataset_index
    config.seed = seed
    n_subgroups = config.n_subgroups
    set_random(seed)

    folds = make_folds_indices(n, n_folds=n_cv, random_state=1000 * rep + 10 * n_cv + dataset_index)

    test_index = folds[cv - 1]
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_index] = False

    Y_raw = x[train_mask, :]
    X_raw = z[train_mask, :] # Naming is different from CDTree
    Y_test_raw = x[test_index, :]
    X_test_raw = z[test_index, :]

    left_bound = np.min(x) - 1e-3
    right_bound = np.max(x) + 1e-3

    X = pd.DataFrame(X_raw)
    Y = pd.DataFrame(Y_raw)
    Y.columns = ["Y"]
    X.columns = [f"X{i}" for i in range(X.shape[1])]

    X_test = pd.DataFrame(X_test_raw)
    Y_test = pd.DataFrame(Y_test_raw)
    Y_test.columns = ["Y"]
    X_test.columns = [f"X{i}" for i in range(X_test.shape[1])]

    data = pd.concat([X,Y],axis=1)
    target = ps.NumericTarget("Y")

    alpha = config.alpha  # size-weighting exponent of the pysubgroup quality function (not the support threshold)
    search_space = ps.create_selectors(data, ignore=["Y"],nbins=config.ncutpoints, intervals_only=False)
    task = ps.SubgroupDiscoveryTask (
        data, 
        target, 
        search_space, 
        result_set_size=n_subgroups, 
        depth=config.sd_depth, 
        qf=QF_WKL(alpha))

    t0 = time.time()
    result = ps.BeamSearch(beam_width=config.beam_width,beam_width_adaptive=False).execute(task)
    t1 = time.time()
    result.to_dataframe()
    # subgroups = []
    out = []
    rules = []
    ids_subgroup = {}

    train_times = []
    test_times = []

    rules_dfs = [[None]] * n_subgroups
    feature_names_dummy = [f"X{i}" for i in range(z.shape[1])]
    for i in range(n_subgroups):
        df_rules = pd.DataFrame(columns=feature_names_dummy, index=['lower', 'upper', '='])
        ids_subgroup_local = {}
        t2 = time.time()
        result_string = str(result.to_dataframe().iloc[i]["subgroup"])
        rules.append(replace_feature_names(result_string,feature_names))
        parts = result_string.split(" AND ")
        conditions = []
        for part in parts:
            # parse this: "X0>=0.80" or "X0<0.80"
            if "==" in part:
                var, val = part.split("==")
                var = int(var[1:])
                val = convert(data,var, val)
                conditions.append((var,val,val))
                df_rules.at['=', f"X{var}"] = val
                continue
            elif "<" in part:
                var, high = part.split("<")
                low = - np.infty
                var = int(var[1:])
                high = float(high)
                temp = df_rules.at['upper', f"X{var}"]
                if temp > high or pd.isna(temp):
                    df_rules.at['upper', f"X{var}"] = high
            else:
                var, low = part.split(">=")
                high = np.infty
                var = int(var[1:])
                low = float(low)
                temp = df_rules.at['lower', f"X{var}"]
                if temp < low or pd.isna(temp):
                    df_rules.at['lower', f"X{var}"] = low
            conditions.append((var,low,high))
        
        rules_dfs[i] = df_rules
        # This is how to get the subgroups on train split
        subgroup_member = np.ones((X.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member = np.logical_and(subgroup_member, np.logical_and(X.iloc[:,var]>=low, X.iloc[:,var]<high) if low != high else X.iloc[:,var]==low)
        ids_subgroup_local["train"] = np.argwhere(subgroup_member.to_numpy()).ravel().tolist()
        t3 = time.time()
        train_times.append(float(t3 - t2))
        # Apply the same rule to the test split (pysubgroup intervals are [lower, upper))
        t4 = time.time()
        subgroup_member_test = np.ones((X_test.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member_test = np.logical_and(subgroup_member_test, np.logical_and(X_test.iloc[:,var]>=low, X_test.iloc[:,var]<high) if low != high else X_test.iloc[:,var]==low)
        t5 = time.time()
        test_times.append(float(t5 - t4))
        ids_subgroup_local["test"] = np.argwhere(subgroup_member_test.to_numpy()).ravel().tolist()
        ids_subgroup[i] = ids_subgroup_local

        score, edges_all, edges_subgroup, p_s_train, p_train = score_subgroups(Y_raw, Y_raw, subgroup_member, "train", (left_bound, right_bound))
        if len(ids_subgroup_local["test"]) > 0:
            score_test, _, _, _, _ = score_subgroups(Y_test_raw, Y_raw, 
                                                  subgroup_member_test, "test", (left_bound, right_bound), edges_all=edges_all, 
                                                  edges_subgroup=edges_subgroup, p_s_train=p_s_train, p_train=p_train)
        else:
            split = "test"
            score_test = {
                f"KL_{split}": np.nan,
                f"support_{split}": np.nan,
                f"support_ratio_{split}": np.nan,
                f"KL_corrected_{split}": np.nan,
            }
        score.update(score_test)
        score.update({
            "subgroup_id": i
        })
        out.append(score)
    ids_subgroup["test_indices"] = test_index.ravel().tolist()

    scores_both = pd.DataFrame(out)
    ord_idx = scores_both.sort_values(
        by=["KL_train", "KL_test"], # This is fair
        ascending=[False, False],
        na_position="last"
    ).index.to_numpy()
    scores_both = scores_both.loc[ord_idx].reset_index(drop=True)

    # Save subgroup scores for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/sd_kl") / "subgroup_scores"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_scores.csv"
    scores_both['cv'] = cv
    scores_both['rep'] = rep
    scores_both['dataset'] = dataset_name
    scores_both.to_csv(output_file, index=False)

    # Save subgroup IDs for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/sd_kl") / "subgroup_ids"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_ids.json"
    with open(output_file, "w") as f:
        json.dump(ids_subgroup, f)

    # Save rules for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/sd_kl") / "subgroup_rules"
    output_folder.mkdir(parents=True, exist_ok=True)
    for idn in range(len(rules_dfs)):
        if rules_dfs[idn] is None:
            continue
        rules_dfs[idn]['cv'] = cv
        rules_dfs[idn]['rep'] = rep
        rules_dfs[idn]['dataset'] = dataset_name
        rules_dfs[idn]['subgroup_id'] = idn
        output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups{idn}_rules.csv"
        rules_dfs[idn].to_csv(output_file, index=True)
        
    res = dict(
        dataset=dataset_name,
        cv=cv,
        rep=rep,
        model_time = float(t1 - t0),
        n_model = 1,
        train_time=np.sum(train_times),
        test_time=np.sum(test_times),
        n_subgroups=n_subgroups
    )
    return res, scores_both


def drive_sd_mean(arguments):
    rep = arguments['rep']
    cv = arguments['cv']
    n_cv = arguments['n_cv']
    dataset_index = arguments['dataset_index']
    dataset_paths = arguments['dataset_paths']
    root_path = arguments['root_path']
    config = arguments['config']()

    data_path = dataset_paths[dataset_index]
    dataset_name = str(data_path).split(".csv")[0].split("/")[-1]

    df = pd.read_csv(data_path)
    z = df.iloc[:, :-1].to_numpy(dtype=float)
    x = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)  # keep 2D like R matrix
    feature_names = df.columns[:-1].tolist()
    n = x.shape[0]

    seed = 1000 * rep + 10 * cv + dataset_index
    config.seed = seed
    n_subgroups = config.n_subgroups
    set_random(seed)

    # folds = make_folds_indices(n, n_folds=n_cv)
    folds = make_folds_indices(n, n_folds=n_cv, random_state=1000 * rep + 10 * n_cv + dataset_index)

    test_index = folds[cv - 1]
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_index] = False

    Y_raw = x[train_mask, :]
    X_raw = z[train_mask, :] # Naming is different from CDTree
    Y_test_raw = x[test_index, :]
    X_test_raw = z[test_index, :]

    left_bound = np.min(x) - 1e-3
    right_bound = np.max(x) + 1e-3

    X = pd.DataFrame(X_raw)
    Y = pd.DataFrame(Y_raw)
    Y.columns = ["Y"]
    X.columns = [f"X{i}" for i in range(X.shape[1])]

    X_test = pd.DataFrame(X_test_raw)
    Y_test = pd.DataFrame(Y_test_raw)
    Y_test.columns = ["Y"]
    X_test.columns = [f"X{i}" for i in range(X_test.shape[1])]

    data = pd.concat([X,Y],axis=1)
    target = ps.NumericTarget("Y")
    alpha = config.alpha  # size-weighting exponent of the pysubgroup quality function (not the support threshold)

    search_space = ps.create_selectors(data, ignore=["Y"],nbins=config.ncutpoints, intervals_only=False)
    task = ps.SubgroupDiscoveryTask (
        data, 
        target, 
        search_space, 
        result_set_size=n_subgroups, 
        depth=config.sd_depth, 
        qf=ps.StandardQFNumeric(alpha))
    t0 = time.time()
    result = ps.BeamSearch(beam_width=config.beam_width,beam_width_adaptive=False).execute(task)
    t1 = time.time()
    out = []
    rules = []
    ids_subgroup = {}

    train_times = []
    test_times = []

    rules_dfs = [[None]] * n_subgroups
    feature_names_dummy = [f"X{i}" for i in range(z.shape[1])]
    for i in range(n_subgroups):
        ids_subgroup_local = {}
        df_rules = pd.DataFrame(columns=feature_names_dummy, index=['lower', 'upper', '='])
        t2 = time.time()
        result_string = str(result.to_dataframe().iloc[i]["subgroup"])
        rules.append(replace_feature_names(result_string,feature_names))
        parts = result_string.split(" AND ")
        conditions = []
        for part in parts:
            # parse this: "X0>=0.80" or "X0<0.80"
            if "==" in part:
                var, val = part.split("==")
                var = int(var[1:])
                #if isinstance(val,str):
                val = convert(data,var, val)
                conditions.append((var,val,val))
                df_rules.at['=', f"X{var}"] = val
                continue
            elif "<" in part:
                var, high = part.split("<")
                low = - np.infty
                var = int(var[1:])
                high = float(high)
                # df_rules.at['upper', f"X{var}"] = high
                temp = df_rules.at['upper', f"X{var}"]
                if temp > high or pd.isna(temp):
                    df_rules.at['upper', f"X{var}"] = high
            else:
                var, low = part.split(">=")
                high = np.infty
                var = int(var[1:])
                low = float(low)
                # df_rules.at['lower', f"X{var}"] = low
                temp = df_rules.at['lower', f"X{var}"]
                if temp < low or pd.isna(temp):
                    df_rules.at['lower', f"X{var}"] = low
            conditions.append((var,low,high))
        
        rules_dfs[i] = df_rules
        # This is how to get the subgroups on train split
        subgroup_member = np.ones((X.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member = np.logical_and(subgroup_member, np.logical_and(X.iloc[:,var]>=low, X.iloc[:,var]<=high) if low != high else X.iloc[:,var]==low)
        t3 = time.time()
        train_times.append(float(t3 - t2))
        ids_subgroup_local["train"] = np.argwhere(subgroup_member.to_numpy()).ravel().tolist()
        
        # subgroups.append(subgroup_member)
        t4 = time.time()
        # Apply the same rule to the test split (pysubgroup intervals are [lower, upper))
        subgroup_member_test = np.ones((X_test.shape[0],),dtype=bool)
        for cond in conditions:
            var, low, high = cond
            var = int(var)
            subgroup_member_test = np.logical_and(subgroup_member_test, np.logical_and(X_test.iloc[:,var]>=low, X_test.iloc[:,var]<high) if low != high else X_test.iloc[:,var]==low)
        t5 = time.time()
        test_times.append(float(t5 - t4))
        # subgroups.append(subgroup_member_test)
        ids_subgroup_local["test"] = np.argwhere(subgroup_member_test.to_numpy()).ravel().tolist()
        ids_subgroup[i] = ids_subgroup_local

        score, edges_all, edges_subgroup, p_s_train, p_train = score_subgroups(Y_raw, Y_raw, subgroup_member, "train", (left_bound, right_bound))
        if len(ids_subgroup_local["test"]) > 0:
            # score_test, _ = score_subgroups(Y_test_raw, Y_raw, subgroup_member_test, "test", edges=edges)
            score_test, _, _, _, _ = score_subgroups(Y_test_raw, Y_raw,
                                                  subgroup_member_test, "test", (left_bound, right_bound), edges_all=edges_all,
                                                  edges_subgroup=edges_subgroup, p_s_train=p_s_train, p_train=p_train)
        else:
            split = "test"
            score_test = {
                f"KL_{split}": np.nan,
                f"support_{split}": np.nan,
                f"support_ratio_{split}": np.nan,
                f"KL_corrected_{split}": np.nan,
            }
        score.update(score_test)
        score.update({
            "subgroup_id": i
        })
        out.append(score)

    ids_subgroup['test_indices'] = test_index.ravel().tolist()
    scores_both = pd.DataFrame(out)
    ord_idx = scores_both.sort_values(
        by=["KL_train", "KL_test"], # This is fair
        ascending=[False, False],
        na_position="last"
    ).index.to_numpy()
    scores_both = scores_both.loc[ord_idx].reset_index(drop=True)

    # Save subgroup scores for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/sd_mean") / "subgroup_scores"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_scores.csv"
    scores_both['cv'] = cv
    scores_both['rep'] = rep
    scores_both['dataset'] = dataset_name
    scores_both.to_csv(output_file, index=False)

    # Save subgroup IDs for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/sd_mean") / "subgroup_ids"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_ids.json"
    with open(output_file, "w") as f:
        json.dump(ids_subgroup, f)

    # Save rules for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/sd_mean") / "subgroup_rules"
    output_folder.mkdir(parents=True, exist_ok=True)
    for idn in range(len(rules_dfs)):
        if rules_dfs[idn] is None:
            continue
        rules_dfs[idn]['cv'] = cv
        rules_dfs[idn]['rep'] = rep
        rules_dfs[idn]['dataset'] = dataset_name
        rules_dfs[idn]['subgroup_id'] = idn
        output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups{idn}_rules.csv"
        rules_dfs[idn].to_csv(output_file, index=True)

    res = dict(
        dataset=dataset_name,
        cv=cv,
        rep=rep,
        model_time = float(t1 - t0),
        n_model = 1,
        train_time=np.sum(train_times),
        test_time=np.sum(test_times),
        n_subgroups=n_subgroups
    )

    return res, scores_both


def drive_rsd(arguments):
    rep = arguments['rep']
    cv = arguments['cv']
    n_cv = arguments['n_cv']
    dataset_index = arguments['dataset_index']
    dataset_paths = arguments['dataset_paths']
    root_path = arguments['root_path']
    config = arguments['config']()

    data_path = dataset_paths[dataset_index]
    dataset_name = str(data_path).split(".csv")[0].split("/")[-1]

    df = pd.read_csv(data_path)
    z = df.iloc[:, :-1].to_numpy(dtype=float)
    x = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)  # keep 2D like R matrix
    feature_names = df.columns[:-1].tolist()
    n = x.shape[0]

    seed = 1000 * rep + 10 * cv + dataset_index
    config.seed = seed
    n_subgroups = config.n_subgroups
    set_random(seed)

    folds = make_folds_indices(n, n_folds=n_cv, random_state=1000 * rep + 10 * n_cv + dataset_index)

    test_index = folds[cv - 1]
    train_mask = np.ones(n, dtype=bool)
    train_mask[test_index] = False

    Y_raw = x[train_mask, :]
    X_raw = z[train_mask, :] # Naming is different from CDTree
    Y_test_raw = x[test_index, :]
    X_test_raw = z[test_index, :]

    left_bound = np.min(x) - 1e-3
    right_bound = np.max(x) + 1e-3
    # subgroups = []
    out = []
    rules = []
    ids_subgroup = {}

    X = pd.DataFrame(X_raw)
    X.columns = feature_names
    Y = pd.DataFrame(Y_raw)
    Y.columns = ["Y"]
    
    X_test = pd.DataFrame(X_test_raw)
    Y_test = pd.DataFrame(Y_test_raw)
    Y_test.columns = ["Y"]
    X_test.columns = feature_names
    
    target_model = "gaussian"
    task = "discovery"
    t0 = time.time()
    model = MDLRuleList(task = task, target_model = target_model,max_rules=n_subgroups, n_cutpoints=config.ncutpoints)
    model.fit(X, Y)
    t1 = time.time()
    ids_subgroup = {}
    covered = np.zeros(X_test.shape[0], dtype=bool)
    train_times = []
    test_times = []
    rules_dfs = [[None]] * len(model._rulelist.subgroups)
    if len(model._rulelist.subgroups) == 0:
        # Handle case with no discovered subgroups...
        ids_subgroup[0] = {"train": [], "test": []}
        split = "train"
        score = {
                f"KL_{split}": np.nan,
                f"support_{split}": np.nan,
                f"support_ratio_{split}": np.nan,
                f"KL_corrected_{split}": np.nan,
    
            }
        split = "test"
        score_test = {
            f"KL_{split}": np.nan,
            f"support_{split}": np.nan,
            f"support_ratio_{split}": np.nan,
            f"KL_corrected_{split}": np.nan,

        }
        score.update(score_test)
        score.update({
            "subgroup_id": 0
        })
        out.append(score)
    else:
        # feature_names_dummy = [f"X{i}" for i in range(z.shape[1])]
        rules_textual = model._rulelist.get_rules()
        for idx, subgroup in enumerate(model._rulelist.subgroups):
            ids_subgroup_local = {}
            df_rules = pd.DataFrame(columns=feature_names, index=['lower', 'upper', '='])
            rule = rules_textual[idx].split("THEN")[0].strip()
            rule_items = rule.split(" AND ")
            for item in rule_items:
                item_trimmed = item.replace(" ", "")
                if ">=" in item_trimmed: # X?>=value
                    var, low = item_trimmed.split(">=")
                    temp = df_rules.at['lower', var]
                    if pd.isna(temp) or (temp < low):
                        df_rules.at['lower', var] = float(low)
                elif ("<=" in item_trimmed) and ("<" in item_trimmed): # lower<=X<upper
                    lower_val, var_upper_val = item_trimmed.split("<=")
                    var, upper_val = var_upper_val.split("<")
                    temp = df_rules.at['lower', var]
                    if pd.isna(temp) or (temp < low):
                        df_rules.at['lower', var] = float(lower_val)
                    temp = df_rules.at['upper', var]
                    if pd.isna(temp) or (temp > upper_val):
                        df_rules.at['upper', var] = float(upper_val)
                    # df_rules.loc['lower', var] = float(lower_val)
                    # df_rules.loc['upper', var] = float(upper_val)
                elif ("<" in item_trimmed) and ("<=" not in item_trimmed): # X<upper only
                    var, high = item_trimmed.split("<")
                    temp = df_rules.at['upper', var]
                    if pd.isna(temp) or (temp > high):
                        df_rules.at['upper', var] = float(high)
                elif ("=" in item_trimmed): # Categorical case, X=value
                    var, value = item_trimmed.split("=")
                    df_rules.at['=', var] = value
                else:
                    raise NotImplementedError(f"Unknown rule format (RSD): {item_trimmed}")
            df_rules.columns = [f"X{i}" for i in range(len(df_rules.columns))]
            rules_dfs[idx] = df_rules
            t2 = time.time()
            subgroup_member = reduce(lambda x,y: x & y, [item.activation_function(X).values for item in subgroup.pattern])
            t3 = time.time()
            train_times.append(float(t3 - t2))
            # subgroups.append(subgroup_member)
            ids_subgroup_local["train"] = np.argwhere(subgroup_member).ravel().tolist()

            # Each RSD rule is evaluated as an independent subgroup (rule set), not sequentially as a rule list
            t4 = time.time()
            subgroup_member_test = reduce(
                lambda a, b: a & b,
                [item.activation_function(X_test).values for item in subgroup.pattern],
                np.ones(X_test.shape[0], dtype=bool)  # safe initializer in case pattern is empty
            )
            # a = (~covered) & subgroup_member_test
            # covered |= a
            t5 = time.time()
            test_times.append(float(t5 - t4))

            # subgroup_member_test = a

            ids_subgroup_local["test"] = np.argwhere(subgroup_member_test).ravel().tolist()
            ids_subgroup[idx] = ids_subgroup_local
            score, edges_all, edges_subgroup, p_s_train, p_train = score_subgroups(Y_raw, Y_raw, subgroup_member, "train", (left_bound, right_bound))
            if len(ids_subgroup_local["test"]) > 0:
                # score_test, _ = score_subgroups(Y_test_raw, Y_raw, subgroup_member_test, "test", edges=edges)
                score_test, _, _, _, _ = score_subgroups(Y_test_raw, Y_raw,
                                                    subgroup_member_test, "test", (left_bound, right_bound), edges_all=edges_all,
                                                    edges_subgroup=edges_subgroup, p_s_train=p_s_train, p_train=p_train)
            else:
                split = "test"
                score_test = {
                    f"KL_{split}": np.nan,
                    f"support_{split}": np.nan,
                    f"support_ratio_{split}": np.nan,
                    f"KL_corrected_{split}": np.nan,
                }
            score.update(score_test)
            score.update({
                "subgroup_id": idx
            })
            out.append(score)

    ids_subgroup["test_indices"] = test_index.ravel().tolist()
    scores_both = pd.DataFrame(out)
    ord_idx = scores_both.sort_values(
        by=["KL_train", "KL_test"], # This is fair
        ascending=[False, False],
        na_position="last"
    ).index.to_numpy()
    scores_both = scores_both.loc[ord_idx].reset_index(drop=True)

    # Save subgroup scores for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/rsd") / "subgroup_scores"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_scores.csv"
    scores_both['cv'] = cv
    scores_both['rep'] = rep
    scores_both['dataset'] = dataset_name
    scores_both.to_csv(output_file, index=False)

    # Save subgroup IDs for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/rsd") / "subgroup_ids"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups_ids.json"
    with open(output_file, "w") as f:
        json.dump(ids_subgroup, f)

    # Save rules for each fold
    output_folder = Path(f"{root_path}/results/Experiment1/results/rsd") / "subgroup_rules"
    output_folder.mkdir(parents=True, exist_ok=True)
    for idn in range(len(rules_dfs)):
        if rules_dfs[idn] is None:
            continue
        rules_dfs[idn]['cv'] = cv
        rules_dfs[idn]['rep'] = rep
        rules_dfs[idn]['dataset'] = dataset_name
        rules_dfs[idn]['subgroup_id'] = idn
        output_file = output_folder / f"{dataset_name}_cv{cv}_rep{rep}_subgroups{idn}_rules.csv"
        rules_dfs[idn].to_csv(output_file, index=True)

    res = dict(
        dataset=dataset_name,
        cv=cv,
        rep=rep,
        model_time = float(t1 - t0),
        n_model = 1,
        train_time=np.sum(train_times),
        test_time=np.sum(test_times),
        n_subgroups=n_subgroups
    )

    # return None, scores_both
    return res, scores_both

DRIVER_MAP = {
    "cdtree": drive_cdtree,
    "syflow": drive_syflow,
    "sd_kl": drive_sd_kl,
    "sd_mean": drive_sd_mean,
    "rsd": drive_rsd
}

# Main caller for all methods
def caller(args):
    n_cv = args['n_cv']
    rep_num = args['rep_num']
    max_cpu = args['max_cpu']
    TEST_MODE = args['TEST_MODE']
    root_path = args.get('ROOT_PATH', ROOT_PATH)
    method = args['method']
    config = args.get('params', None)
    data_path = args.get('data_path', "/dataset/")

    try:
        driver = DRIVER_MAP[method]
    except KeyError:
        logger.error("Unknown method in caller (executor.py): %s", method)
        exit(1)

    logger = setup_logger(method, root_path, os.getpid())
    folder_path = Path(f"{root_path}/{data_path}")

    dataset_paths_meta = folder_path.glob("*.csv")
    if TEST_MODE:
        n_cv = 2
        rep_num = 1
        max_cpu = 4
        dataset_paths = list(dataset_paths_meta)[:1]
    else:
        dataset_paths = list(dataset_paths_meta)

    para_args = []
    for combo in product(range(1, rep_num + 1), range(1, n_cv + 1), range(len(dataset_paths))):
        rep_, cv, dataset_index = combo
        arg_dict = dict(
            rep=rep_,
            cv=cv,
            n_cv=n_cv,
            dataset_index=dataset_index,
            dataset_paths=dataset_paths,
            root_path=root_path,
            config=config
        )
        para_args.append(arg_dict)
    
    with ProcessPoolExecutor(max_workers=max_cpu) as pool:
        future_to_x = {}
        for x in para_args:
            future = pool.submit(driver, x)
            future_to_x[future] = deepcopy(x)
            data_path_temp = x['dataset_paths']
            d_name = data_path_temp[x['dataset_index']]
            dname = str(d_name).split(".csv")[0].split("/")[-1]

            logger.info("Submitted job for dataset=%s, cv=%d, rep=%d", dname, x['cv'], x['rep'])

        results = []
        scores = []
        for future in as_completed(future_to_x):
            x = future_to_x[future]
            # Pop the last of x
            data_path_temp = x.pop('dataset_paths')
            d_name = data_path_temp[x['dataset_index']]
            x['dataset_name'] = str(d_name).split(".csv")[0].split("/")[-1]
            try:
                res, score_df = future.result()
                if res is not None:
                    results.append(res)
                if score_df is not None:
                    scores.append(score_df)
                logger.info(f"Completed: dataset={x['dataset_name']}, cv={x['cv']}, rep={x['rep']}")
            except Exception as e:
                # Log the error with the parameter of the task that spawned this future
                logger.error(f"Error in processing: dataset={x['dataset_name']}, cv={x['cv']}, rep={x['rep']}, error=%r", e)
                # results.append(None)
    scores_subgroup = pd.concat(scores, ignore_index=True)
    Path(f"{root_path}/results/Experiment1/results/{method}").mkdir(parents=True, exist_ok=True)
    scores_subgroup.to_csv(Path(f"{root_path}/results/Experiment1/results/{method}/subgroup_scores.csv"), index=False)
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df.to_csv(Path(f"{root_path}/results/Experiment1/results/{method}/aux.csv"), index=False)