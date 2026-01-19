#!/bin/bash
#SBATCH --job-name=default_run # Job name
# cpu, gpu Setting
# SBATCH --qos=hpgpu
#SBATCH --partition=L40S
#SBATCH --exclude=n80
#SBATCH --gres=gpu:1                   # Number of GPUs (per node)
#SBATCH --cpus-per-task=8              # CPU cores/threads

#SBATCH --array=1-10%10 # Array range (this creates 10 tasks with IDs from 1 to 10, with max 10 tasks concurrently)
#SBATCH --output=logs/%x/%A/task_%a.out
#SBATCH --error=logs/%x/%A/task_%a.err
#SBATCH --time=12:00:00

#  '''Usage:
#  # Best-of-N on the MATH-500 dataset
#  
#  sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
#      --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-bon-completions
#  '''

# 모듈 설정
module purge
module load gnu12/12.2.0
module load cuda/12.6
echo "module load complete"
# 환경 설정
source ~/.bashrc
source .venv/bin/activate
echo "venv activate complete"

# 컴파일러 명시 설정
export CC=$(which gcc)
export CXX=$(which g++)
echo "compiler set compelete"

STEP=50
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
DATASET_START=$((TASK_INDEX * STEP))
DATASET_END=$((DATASET_START + STEP))

export PRM=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

# Temperature settings (comma-separated for argparse)

python scripts/test_time_compute.py "$@" \
    --dataset_start=$DATASET_START \
    --dataset_end=$DATASET_END \
    --num_samples=500 \
    --prm_path=$PRM \
    --push_to_hub=true
