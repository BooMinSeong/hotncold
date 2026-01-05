#!/bin/bash
#SBATCH --job-name=beam_search10 # Job name
# cpu, gpu Setting
#SBATCH --qos=hpgpu
#SBATCH --partition=A100-80GB
#SBATCH --exclude=n80
#SBATCH --gres=gpu:1                   # Number of GPUs (per node)
#SBATCH --cpus-per-task=8              # CPU cores/threads

#SBATCH --array=10-10%10 # Array range (this creates 20 tasks with IDs from 1 to 20, with max 8 tasks concurrently)
#SBATCH --output=logs/%x-%j_%A_%a.out   # Standard output (%A is replaced by job ID, %a by task ID)
#SBATCH --error=logs/%x-%j_%A_%a.err    # Standard error
#SBATCH --time=12:00:00                  # Time limit hrs:min:sec

 '''Usage:
 # Best-of-N on the MATH-500 dataset
 
 sbatch recipes/launch_array.slurm recipes/Llama-3.2-1B-Instruct/best_of_n.yaml \
     --hub_dataset_id=<YOUR_ORG>/Llama-3.2-1B-Instruct-bon-completions
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


# # Define the array of input files (assuming 10 input files)
# STEP=50
# ENDPOINT=$((SLURM_ARRAY_TASK_COUNT * STEP - STEP))
# STARTS=($(seq 0 $STEP $ENDPOINT))  # Generate sequence from 0 to ENDPOINT with a step size of 100
# # Use the SLURM_ARRAY_TASK_ID to pick the correct input file
# INPUT_FILE=${INPUT_FILES[$SLURM_ARRAY_TASK_ID-1]}
# 
# DATASET_START=${STARTS[$SLURM_ARRAY_TASK_ID-1]}
# DATASET_END=$((${STARTS[$SLURM_ARRAY_TASK_ID-1]}+$STEP))

STEP=50
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
DATASET_START=$((TASK_INDEX * STEP))
DATASET_END=$((DATASET_START + STEP))

export PRM=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

# Temperature settings (comma-separated for argparse)
TEMPS="0.6,0.8,1.0,1.2"
TEMP_RATIOS="0.25,0.25,0.25,0.25"

python scripts/test_time_compute.py "$@" \
    --dataset_start=$DATASET_START \
    --dataset_end=$DATASET_END \
    --num_samples=500 \
    --prm_path=$PRM \
    --temperatures=$TEMPS \
    --temperature_ratios=$TEMP_RATIOS \
    --push_to_hub=true
