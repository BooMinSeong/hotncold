#!/bin/bash
#SBATCH --job-name=single_default # Job name
# cpu, gpu Setting
#SBATCH --qos=hpgpu
#SBATCH --partition=A100-80GB
#SBATCH --exclude=n80
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8              # CPU cores/threads

#SBATCH --output=logs/%x/%A/task_%a.out
#SBATCH --error=logs/%x/%A/task_%a.err
#SBATCH --time=12:00:00

 '''Usage:
 # Individual run with specific dataset range (default temperature settings)

 # Example 1: Run best-of-n on full dataset with seed-42
 sbatch recipes/launch_single_default.slurm recipes/Qwen2.5-1.5B-Instruct/best_of_n.yaml \
     --hub_dataset_id=ENSEONG/default-Qwen2.5-1.5B-Instruct-bon \
     --dataset_start=0 \
     --dataset_end=500 \
     --seed=42

 # Example 2: Run beam_search on full dataset with seed-42
 sbatch recipes/launch_single_default.slurm recipes/Qwen2.5-1.5B-Instruct/beam_search.yaml \
     --hub_dataset_id=ENSEONG/default-Qwen2.5-1.5B-Instruct-beam_search \
     --dataset_start=0 \
     --dataset_end=500 \
     --seed=42

 # Example 3: Run dvts on custom range
 sbatch recipes/launch_single_default.slurm recipes/Qwen2.5-1.5B-Instruct/dvts.yaml \
     --hub_dataset_id=ENSEONG/default-Qwen2.5-1.5B-Instruct-dvts \
     --dataset_start=100 \
     --dataset_end=150 \
     --seed=42
 '''

# 모듈 설정
module purge
module load gnu12/12.2.0
module load cuda/12.6

# 환경 설정
source ~/.bashrc
source .venv/bin/activate

# 컴파일러 명시 설정
export CC=$(which gcc)
export CXX=$(which g++)

# PRM 모델 설정
export PRM=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B
DATASET_NAME="ENSEONG/math-private"

# Note: dataset_start, dataset_end, and seed should be passed as command-line arguments
# via --dataset_start, --dataset_end, and --seed when calling this script

python scripts/test_time_compute.py "$@" \
	--dataset_name=$DATASET_NAME \
	--prm_path=$PRM \
	--push_to_hub=true
