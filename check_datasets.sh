#!/bin/bash

# Dataset status check script
# Usage: bash check_datasets.sh

SEEDS=(0 42 64)
SEARCH_METHODS=("best_of_n.yaml" "beam_search.yaml" "dvts.yaml")
METHOD_NAMES=("bon" "beam_search" "dvts")
DATASET_NAME="HuggingFaceH4/MATH-500"
MODEL_NAME="Qwen2.5-3B-Instruct"

echo "========================================================================"
echo "Dataset Status Check"
echo "========================================================================"
printf "%-60s | %-8s | %6s | %s\n" "DATASET" "FILTER" "COV" "STATUS"
echo "------------------------------------------------------------------------"

for i in "${!SEARCH_METHODS[@]}"; do
    METHOD_NAME="${METHOD_NAMES[$i]}"
    REPO="ENSEONG/default-${DATASET_NAME##*/}-$MODEL_NAME-$METHOD_NAME"

    for SEED in "${SEEDS[@]}"; do
        python exp/ds_chehck.py "$REPO" --filter "seed-$SEED" --format oneline 2>/dev/null
    done
done

echo "========================================================================"
