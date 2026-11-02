# AtomSD: Subgroup Discovery @ Arbitrary Support Thresholds
This is the official repository of the NeurIPS 2026 paper **"Optimal Subgroup Discovery at Every Support Threshold"**. 

Note on naming: the method is called **AtomSD** in the paper. In the code, configuration files, and result files it sometimes appears as `cdtree` / `CDTree`, after the conditional density tree it builds on.

## 🔥 Fast reproduction of main empirical results at [reproduce_results.ipynb](reproduce_results.ipynb)
This helps reproduce the main tables and figures reported in the paper.

## Reproducing the Main Experiments
Please follow the instructions below **step by step**.

All empirical experiments were conducted on a Linux cluster with Python 3.10. A CUDA-enabled GPU is NOT required.

The [requirements.txt](requirements.txt) file specifies the minimal set of Python dependencies required for reproducing the results reported in this paper.

### Raw Results Used to Generate Table 4 (Validating Theorem 3.3)
First configure your Python environment in ```./shell/run_exp1.sh```
```bash
source YOUR_CONDA_ENVIRONMENT_PATH
```
Then execute the following commands
```bash
chmod u+x ./shell/run_exp1.sh
./shell/run_exp1.sh
```
The raw results will be saved under ```./results/Experiment1/```

Note: Experiments 2 and 3 depend on the outputs of Experiment 1.
### Raw Results Used to Generate Figures 2 and 4 (Validating Corollary 3.6)
First configure your Python environment in ```./shell/run_exp2.sh```
```bash
source YOUR_CONDA_ENVIRONMENT_PATH
```
Then execute the following commands
```bash
chmod u+x ./shell/run_exp2.sh
./shell/run_exp2.sh
```
The raw results will be saved under ```./results/Experiment2/```

### Raw Results Used to Generate Tables 1 and 3 (Supporting Theorem 3.5 in Finite Sample Setting)
First configure your Python environment in ```./shell/run_exp3.sh```
```bash
source YOUR_CONDA_ENVIRONMENT_PATH
```
Then execute the following commands
```bash
chmod u+x ./shell/run_exp3.sh
./shell/run_exp3.sh
```
The raw results will be saved under ```./results/Experiment3/```

Note that these are raw results and do not directly correspond to the fine-grained tables and figures presented in the paper. For convenience, we provide a streamlined reproduction of the main results in [reproduce_results.ipynb](reproduce_results.ipynb), avoiding the need for cumbersome utility functions.

### Quick smoke test
`config/unconstrained_test.jsonl` runs Experiment 1 in test mode (one dataset, 2 folds, 4 CPUs) and can be used to check the installation:
```bash
python run_exp1.py -c ./config/unconstrained_test.jsonl -m CDTree
```

## Reproducing Figures 1, 3 and 5
The scripts under [`figures/`](figures/) write to `figures/output/`.
- **Figure 1** (ranking error of support-weighted KL on synthetic data, Appendix C.1): `python figures/make_fig1_ranking_flip.py`. Self-contained; needs no experiment outputs.
- **Figure 5** (critical-difference diagrams, Appendix C.5): run `reproduce_results.ipynb` first (it writes `results/Experiment3/result_exp3_*_raw.csv`), then `python figures/make_fig5_cd_diagrams.py`.
- **Figure 3** (bootstrap stability of KL estimates, Appendix C.2): `python figures/make_fig3_bootstrap.py`. Requires the raw per-fold subgroup files of Experiment 1 under `results/Experiment1/results/<method>/subgroup_ids/`, i.e. `shell/run_exp1.sh` must have been run first.

## Datasets
The 19 benchmark datasets used in the paper are provided as CSV files under [`dataset/`](dataset/) (last column = target). They are public regression benchmarks commonly used in the subgroup discovery literature, mostly from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/); see Table 2 of the paper for a summary. The copyright of each dataset remains with its original provider.

## Third-party code
- `src/syflow/` (except `RSD/`) is adapted from the official [SyFlow](https://github.com/nilspwalter/syflow) implementation (Xu, Walter, Kalofolias, and Vreeken, ICML 2024), licensed under [CC BY-NC-SA 4.0](src/syflow/LICENSE).
- `src/syflow/RSD/` is the [Robust Subgroup Discovery](https://github.com/HMProenca/RobustSubgroupDiscovery) implementation by Hugo Proença (Proença et al., DAMI 2022), licensed under [MIT](src/syflow/RSD/LICENSE).
- `src/CDTree/` is our Python port of the conditional density tree from [Yang and van Leeuwen, NeurIPS 2024](https://openreview.net/forum?id=5SUP6vUVkP).
- The SD-KL and SD-μ baselines use [pysubgroup](https://github.com/flemmerich/pysubgroup) (Lemmerich and Becker, 2018).

## License
Our own code is released under the [MIT License](LICENSE). The third-party directories listed above keep their original licenses; in particular, `src/syflow/` is CC BY-NC-SA 4.0 (non-commercial, share-alike).

## Citation
```bibtex
@inproceedings{yang2026atomsd,
  title     = {Optimal Subgroup Discovery at Every Support Threshold},
  author    = {Yang, Lincen and Huang, Qi and van Stein, Niki and B{\"a}ck, Thomas and van Leeuwen, Matthijs and Li, Zhong},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
