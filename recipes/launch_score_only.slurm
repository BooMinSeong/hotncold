#!/bin/bash
#SBATCH --job-name=score_only
#SBATCH --qos=nogpu
#SBATCH --partition=cpu-max24
#SBATCH --exclude=n80,n79
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=24

#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --time=4:00:00

# Usage:
#   sbatch recipes/launch_score_only.slurm <recipe.yaml> [--seed=N --temperature=T --hub_dataset_id=... --dataset_name=... ...]

source ~/slurm_common.sh

STEP=5000
DATASET_START=0
DATASET_END=$STEP

python scripts/score_only.py "$@" \
    --dataset_start=$DATASET_START \
    --dataset_end=$DATASET_END \
    --push_to_hub=true
