from itertools import product
from src import setup_logger, search_expansion_mixed
from pathlib import Path
import pandas as pd
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from copy import deepcopy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default="./")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--output_path", default="./")
    parser.add_argument("--data_name", default=None)
    parser.add_argument("--max_cpu", type=int, default=128)
    parser.add_argument("--partial_search", choices=["best", "pearson"], default="best",
                        help="SearchPartial strategy: 'best' = best-improvement over all cuts of all "
                             "features (paper default); 'pearson' = first-improvement, visiting features "
                             "in decreasing |Pearson correlation| with the target (faster, greedier)")
    args = parser.parse_args()

    root_path = args.root_path
    alphas = args.alphas
    output_path = args.output_path
    data_name = args.data_name
    max_cpu = args.max_cpu

    res_root_path = Path(root_path) / "results" / "Experiment1" / "results" # path to results of exp1
    res_root_path = str(res_root_path)
    data_root_path = Path(root_path) / "dataset"
    output_path = str(Path(root_path) / "results"/ "Experiment3" / output_path)
    os.makedirs(output_path, exist_ok=True)

    logger = setup_logger(data_name if data_name is not None else "All", output_path, os.getpid())

    if data_name is None:
        # Get all the name of the dataset csv under the data_root_path
        dataset_paths = list(data_root_path.glob("*.csv"))
        # Extract the datanames from dataset_paths
        dataset_names = [path.stem for path in dataset_paths]
    else:
        dataset_names = [data_name]
    data_root_path = str(data_root_path)
    method_names = ["cdtree"]

    para_args = []
    for combo in product(method_names, alphas, dataset_names):
        method_name, alpha, dataset_name = combo
        arg_dict = dict(
            method_name=method_name,
            alpha=alpha,
            res_root_path=res_root_path,
            data_root_path=data_root_path,
            output_path=output_path,
            dataset_name=dataset_name,
        )
        para_args.append(arg_dict)
    
    output_folder = Path(f"{output_path}/results/")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"All_subgroups_on_time.csv"

    with ProcessPoolExecutor(max_workers=max_cpu) as pool:
        future_to_x = {}
        for x in para_args:
            future = pool.submit(search_expansion_mixed, 
                                 x['method_name'], 
                                 x['alpha'], 
                                 x['res_root_path'], 
                                 x['data_root_path'], 
                                 x['output_path'], 
                                 x['dataset_name'],
                                 partial_search=args.partial_search)
            future_to_x[future] = deepcopy(x)
            dname = x['dataset_name']
            method = x['method_name']

            logger.info("Submitted job for dataset=%s, method=%s", dname, method)

        results = []
        aux_results = []
        first_write = True
        for future in as_completed(future_to_x):
            x = future_to_x[future]
            try:
                res, aux = future.result()
                if res is not None:
                    results.append(res)
                    res.to_csv(
                        output_file,
                        mode="w" if first_write else "a",
                        header=first_write,
                        index=False,
                    )
                    first_write = False
                if aux is not None:
                    aux_results.extend(aux)
                logger.info(f"Completed: dataset={x['dataset_name']}, method={x['method_name']}")
            except Exception as e:
                # Log the error with the parameter of the task that spawned this future
                logger.error(f"Error in processing: dataset={x['dataset_name']}, method={x['method_name']}, e={e}")
                # results.append(None)

    results_df = pd.concat(results)
    # Save subgroup scores
    if dataset_name is None:
        dataset_name = "All"
    output_folder = Path(f"{output_path}/results/")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"All_subgroups_scores.csv"
    results_df.to_csv(output_file, index=False)

    output_folder = Path(f"{output_path}/results/")
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / f"All_subgroups_aux.jsonl"
    with open(output_file, "w") as f:
        for r in aux_results:
            f.write(json.dumps(r) + "\n")
    logger.info("All done")