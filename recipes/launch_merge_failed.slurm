#!/bin/bash
#SBATCH --job-name=merge_failed
#SBATCH --qos=nogpu
#SBATCH --partition=cpu-max24
#SBATCH --exclude=n80,n79
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=24

#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --time=12:00:00

source ~/slurm_common.sh

DATASET_REPO="ENSEONG/full-math-private-n256-Llama-3.2-3B-Instruct-bon"

# 23 (T, seed) configs that failed previously due to -gen duplicate
FAILED_CONFIGS=(
    "0.7 0"   "0.7 128"
    "0.8 64"  "0.8 128"  "0.8 256"
    "0.9 0"   "0.9 42"   "0.9 256"  "0.9 512"
    "1.0 42"  "1.0 256"
    "1.1 0"   "1.1 42"   "1.1 64"   "1.1 128"  "1.1 256"  "1.1 512"
    "1.2 0"   "1.2 42"   "1.2 64"   "1.2 128"  "1.2 256"  "1.2 512"
)

for params in "${FAILED_CONFIGS[@]}"; do
    read T SEED <<< "$params"
    FILTER_STR="T-${T}--top_p-1.0--n-256--seed-${SEED}"
    echo "Merging: T=$T seed=$SEED filter=$FILTER_STR"
    python scripts/merge_chunks.py \
        --dataset_name=$DATASET_REPO \
        --filter_strings $FILTER_STR \
        --exclude_strings=-gen
done

echo "All merge operations completed!"
