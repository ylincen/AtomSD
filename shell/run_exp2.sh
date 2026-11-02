#!/bin/bash
source YOUR_CONDA_ENVIRONMENT_PATH
SCRIPT="./run_exp2.py"
mkdir -p logs

LOGFILE="logs/Experiment2_alpha_ratio_run.log"
nohup python -u "$SCRIPT" --root_path "./" --max_cpu 64 --output_path "Experiment2_alpha_ratio_mixint/" \
        > "$LOGFILE" 2>&1 &
LOGFILE="logs/Experiment2_alpha_fixed_run.log"
nohup python -u "$SCRIPT" --root_path "./" --max_cpu 64 --alphas 50 100 150 200 300 --output_path "Experiment2_alpha_fixed_mixint/"\
        > "$LOGFILE" 2>&1 &
echo "All jobs started."