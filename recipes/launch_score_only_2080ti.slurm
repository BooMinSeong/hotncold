#!/bin/bash
#SBATCH --job-name=score_only
#SBATCH --partition=RTX2080Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --time=4:00:00

# Usage:
#   sbatch recipes/launch_score_only_2080ti.slurm <recipe.yaml> [--seed=N --temperature=T --hub_dataset_id=... --dataset_name=... ...]

source ~/slurm_common.sh

STEP=5000
DATASET_START=0
DATASET_END=$STEP

python scripts/score_only.py "$@" \
    --dataset_start=$DATASET_START \
    --dataset_end=$DATASET_END \
    --push_to_hub=true
