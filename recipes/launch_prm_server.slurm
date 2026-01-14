#!/bin/bash
#SBATCH --job-name=prm_server
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/prm_server/%A.out
#SBATCH --error=logs/prm_server/%A.err
#SBATCH --time=24:00:00

# PRM Server Job Template
# This job runs a vLLM server for Process Reward Model (PRM) scoring.
# Other compute jobs can connect to this server via API.
#
# Usage:
#   sbatch recipes/launch_prm_server.slurm
#   sbatch --export=ALL,PRM_MODEL=Qwen/Qwen2.5-Math-PRM-7B recipes/launch_prm_server.slurm

# Module setup
module purge
module load gnu12/12.2.0
module load cuda/12.6

# Environment
source ~/.bashrc
source .venv/bin/activate

export CC=$(which gcc)
export CXX=$(which g++)

# Configuration (can be overridden via environment variables)
PRM_MODEL=${PRM_MODEL:-"Qwen/Qwen2.5-Math-PRM-7B"}
PRM_PORT=${PRM_PORT:-8001}
SERVICE_DIR=${SERVICE_DIR:-"/home/b.ms/projects/hotncold/services"}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.9}

# Create directories
mkdir -p $SERVICE_DIR
mkdir -p logs/prm_server

echo "========================================"
echo "PRM Server Configuration"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: $PRM_MODEL"
echo "Port: $PRM_PORT"
echo "Service Directory: $SERVICE_DIR"
echo "GPU Memory Utilization: $GPU_MEM_UTIL"
echo "========================================"

# Start the PRM server
python scripts/start_prm_server.py \
    --model $PRM_MODEL \
    --port $PRM_PORT \
    --service-dir $SERVICE_DIR \
    --gpu-memory-utilization $GPU_MEM_UTIL
