#!/bin/bash
#SBATCH --job-name=compute_api
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10%10
#SBATCH --output=logs/%x/%A/task_%a.out
#SBATCH --error=logs/%x/%A/task_%a.err
#SBATCH --time=12:00:00

# Compute Job Template with vLLM API Backend
# This job connects to an external PRM server for scoring.
#
# Usage:
#   # With direct API URL
#   sbatch --export=ALL,PRM_API_URL=http://node001:8001 \
#       recipes/launch_array_api.slurm recipes/Qwen2.5-3B-Instruct/best_of_n.yaml
#
#   # With service discovery file
#   sbatch --export=ALL,PRM_SERVICE_FILE=/path/to/service.json \
#       recipes/launch_array_api.slurm recipes/Qwen2.5-3B-Instruct/best_of_n.yaml

# Module setup
module purge
module load gnu12/12.2.0
module load cuda/12.6

# Environment
source ~/.bashrc
source .venv/bin/activate

export CC=$(which gcc)
export CXX=$(which g++)

# Dataset chunking configuration
STEP=${STEP:-50}
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
DATASET_START=$((TASK_INDEX * STEP))
DATASET_END=$((DATASET_START + STEP))

# PRM configuration
PRM_MODEL=${PRM_MODEL:-"Qwen/Qwen2.5-Math-PRM-7B"}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

# Determine PRM backend arguments
PRM_ARGS="--prm_backend=vllm_api --prm_path=$PRM_MODEL"

if [ -n "$PRM_SERVICE_FILE" ]; then
    PRM_ARGS="$PRM_ARGS --prm_service_file=$PRM_SERVICE_FILE"
elif [ -n "$PRM_API_URL" ]; then
    PRM_ARGS="$PRM_ARGS --prm_api_base_url=$PRM_API_URL"
else
    echo "ERROR: Either PRM_SERVICE_FILE or PRM_API_URL must be set"
    exit 1
fi

echo "========================================"
echo "Compute Job Configuration (API Mode)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "Dataset Range: $DATASET_START - $DATASET_END"
echo "PRM Model: $PRM_MODEL"
echo "PRM Backend: vllm_api"
if [ -n "$PRM_SERVICE_FILE" ]; then
    echo "Service File: $PRM_SERVICE_FILE"
fi
if [ -n "$PRM_API_URL" ]; then
    echo "API URL: $PRM_API_URL"
fi
echo "========================================"

python scripts/test_time_compute.py "$@" \
    --dataset_start=$DATASET_START \
    --dataset_end=$DATASET_END \
    --num_samples=500 \
    --gpu_memory_utilization=$GPU_MEM_UTIL \
    $PRM_ARGS \
    --push_to_hub=true
