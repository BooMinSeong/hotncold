#!/bin/bash
#SBATCH --job-name=single_beam # Job name
# cpu, gpu Setting
#SBATCH --qos=hpgpu
#SBATCH --partition=A100-80GB
#SBATCH --exclude=n80
#SBATCH --gres=gpu:1                   # Number of GPUs (per node)
#SBATCH --cpus-per-task=8              # CPU cores/threads

#SBATCH --output=logs/%x-%j.out   # Standard output (%j is replaced by job ID)
#SBATCH --error=logs/%x-%j.err    # Standard error
#SBATCH --time=12:00:00                  # Time limit hrs:min:sec

 '''Usage:
 # Individual run with specific dataset range

 # Example 1: Run on chunk 350-400 (missing chunk from beam_search seed-42)
 sbatch recipes/launch_single.slurm recipes/Qwen2.5-1.5B-Instruct/beam_search.yaml \
     --hub_dataset_id=ENSEONG/hnc-Qwen2.5-1.5B-Instruct-beam_search \
     --dataset_start=350 \
     --dataset_end=400

 # Example 2: Run on full dataset (0-500)
 sbatch recipes/launch_single.slurm recipes/Qwen2.5-1.5B-Instruct/best_of_n.yaml \
     --hub_dataset_id=ENSEONG/default-Qwen2.5-1.5B-Instruct-bon \
     --dataset_start=0 \
     --dataset_end=500

 # Example 3: Custom range
 sbatch recipes/launch_single.slurm recipes/Qwen2.5-1.5B-Instruct/dvts.yaml \
     --hub_dataset_id=ENSEONG/hnc-Qwen2.5-1.5B-Instruct-dvts \
     --dataset_start=100 \
     --dataset_end=150
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

# Temperature settings (comma-separated for argparse)
TEMPS="0.6,0.8,1.0,1.2"
TEMP_RATIOS="0.25,0.25,0.25,0.25"

# Note: dataset_start and dataset_end should be passed as command-line arguments
# via --dataset_start and --dataset_end when calling this script

python scripts/test_time_compute.py "$@" \
    --num_samples=500 \
    --prm_path=$PRM \
    --temperatures=$TEMPS \
    --temperature_ratios=$TEMP_RATIOS \
    --push_to_hub=true
