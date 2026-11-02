from __future__ import annotations

import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
import json

from src.utils import *
from src import caller
from pathlib import Path
from copy import deepcopy

ROOT_PATH = Path(__file__).resolve().parent

CONFIGS = {
    "Config_Real_World" : Config_Real_World,
    "Config_SD": Config_SD
}

if __name__ == "__main__":
    DEFAULTS = {
        "n_cv": 2,
        "rep_num": 1,
        "TEST_MODE": False,
        "max_cpu": 4,
        "method": "CDTree",  # By default
        "ROOT_PATH": ROOT_PATH
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        help="Method to use: CDTree, Syflow, SD_KL, SD_Mean, or RSD."
    )
    args = parser.parse_args()
    
    all_configs = {}
    if args.config:
        with open(args.config, "r") as f:
            for line in f:
                config = json.loads(line)
                params = DEFAULTS.copy()
                method_name = config.get("method", "").lower()
                params.update(config)
                all_configs[method_name] = params
    else:
        raise ValueError("Please provide a config file path with --config or -c argument")

    if args.method:
        method = args.method.lower()
        if method in all_configs:
            if "cdtree" == method:
                all_configs['cdtree']['method'] = 'cdtree'
                caller(all_configs[method])
            elif "syflow" == method:
                config_name = all_configs[method].get("config", "Config_Real_World")
                syflow_config = CONFIGS.get(config_name)
                if syflow_config:
                    config = all_configs[method]
                    config['params'] = syflow_config
                    config['method'] = 'syflow'
                    caller(config)
                else:
                    raise ValueError(f"Config {config_name} not found for method Syflow")
            elif "sd_kl" == method:
                config_name = all_configs[method].get("config", "Config_SD")
                sd_kl_config = CONFIGS.get(config_name)
                if sd_kl_config:
                    config = all_configs[method]
                    config['params'] = sd_kl_config
                    config['method'] = 'sd_kl'
                    caller(config)
                else:
                    raise ValueError(f"Config {config_name} not found for method SD_KL")
            elif "sd_mean" == method:
                config_name = all_configs[method].get("config", "Config_SD")
                sd_mean_config = CONFIGS.get(config_name)
                if sd_mean_config:
                    config = all_configs[method]
                    config['params'] = sd_mean_config
                    config['method'] = 'sd_mean'
                    caller(config)
                else:
                    raise ValueError(f"Config {config_name} not found for method SD_Mean")
            elif "rsd" == method:
                config_name = all_configs[method].get("config", "Config_SD")
                rsd_config = CONFIGS.get(config_name)
                if rsd_config:
                    config = all_configs[method]
                    config['params'] = rsd_config
                    config['method'] = 'rsd'
                    caller(config)
                else:
                    raise ValueError(f"Config {config_name} not found for method RSD")
            else:
                raise ValueError(f"Method {args.method} is not implemented yet")
        else:
            raise ValueError(f"Config of method {args.method} not found in config file")