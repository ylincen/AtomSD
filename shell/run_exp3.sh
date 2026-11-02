#!/bin/bash
source YOUR_CONDA_ENVIRONMENT_PATH
SCRIPT="./run_exp3.py"
mkdir -p logs

LOGFILE="logs/Experiment3_alpha_ratio_run.log"
nohup python -u "$SCRIPT" --root_path "./" --max_cpu 120 --output_path "Experiment3_alpha_ratio_mixint/" \
        > "$LOGFILE" 2>&1 &
LOGFILE="logs/Experiment3_alpha_fixed_run.log"
nohup python -u "$SCRIPT" --root_path "./" --max_cpu 120 --alphas 50 100 150 200 300 --output_path "Experiment3_alpha_fixed_mixint/"\
        > "$LOGFILE" 2>&1 &
echo "All jobs started."