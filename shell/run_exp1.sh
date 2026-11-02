#!/bin/bash
source YOUR_CONDA_ENVIRONMENT_PATH
CONFIG_PATH="./config/unconstrained.jsonl"
SCRIPT="./run_exp1.py"
METHODS=("CDTree" "Syflow" "SD_KL" "SD_Mean" "RSD")
mkdir -p logs

for METHOD in "${METHODS[@]}"
do
    LOGFILE="logs/${METHOD}_run.log"
    
    echo "Starting $METHOD (log: $LOGFILE)"
    
    nohup python "$SCRIPT" -c "$CONFIG_PATH" -m "$METHOD" \
        > "$LOGFILE" 2>&1 &
done
echo "All jobs started."