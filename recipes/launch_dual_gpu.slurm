#!/bin/bash
#SBATCH --job-name=dual_gpu_compute
#SBATCH --partition=A100-80GB
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x/%A.out
#SBATCH --error=logs/%x/%A.err
#SBATCH --time=12:00:00

# Dual GPU Compute Job Template
# This job runs both LLM and PRM on the same node using vLLM offline inference.
# GPU 0: Generation LLM
# GPU 1: Process Reward Model
#
# Usage:
#   sbatch recipes/launch_dual_gpu.slurm recipes/Qwen2.5-3B-Instruct/best_of_n.yaml

# Module setup
module purge
module load gnu12/12.2.0
module load cuda/12.6

# Environment
source ~/.bashrc
source .venv/bin/activate

export CC=$(which gcc)
export CXX=$(which g++)

# GPU allocation
export CUDA_VISIBLE_DEVICES=0,1

# Configuration
PRM_MODEL=${PRM_MODEL:-"Qwen/Qwen2.5-Math-PRM-7B"}
LLM_GPU_UTIL=${LLM_GPU_UTIL:-0.5}
PRM_GPU_UTIL=${PRM_GPU_UTIL:-0.4}

echo "========================================"
echo "Dual GPU Compute Configuration"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "PRM Model: $PRM_MODEL"
echo "LLM GPU Utilization: $LLM_GPU_UTIL"
echo "PRM GPU Utilization: $PRM_GPU_UTIL"
echo "========================================"

python scripts/test_time_compute.py "$@" \
    --num_samples=500 \
    --gpu_memory_utilization=$LLM_GPU_UTIL \
    --prm_backend=vllm_offline \
    --prm_path=$PRM_MODEL \
    --prm_gpu_memory_utilization=$PRM_GPU_UTIL \
    --prm_tensor_parallel_size=1 \
    --push_to_hub=true
