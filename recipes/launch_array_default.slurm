#!/bin/bash
#SBATCH --job-name=default_run # Job name
# cpu, gpu Setting
# SBATCH --qos=hpgpu
# SBATCH --partition=A100-80GB
#SBATCH --partition=L40S
#SBATCH --exclude=n52
#SBATCH --gres=gpu:1                   # Number of GPUs (per node)
#SBATCH --cpus-per-task=8              # CPU cores/threads

#SBATCH --array=1-1%10 # Array range (this creates 10 tasks with IDs from 1 to 10, with max 10 tasks concurrently)
#SBATCH --output=logs/%x/%A/task_%a.out
#SBATCH --error=logs/%x/%A/task_%a.err
#SBATCH --time=24:00:00

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
echo "compiler set complete"


# Prolog이 생성한 로컬 SSD 환경변수 로드 (HF_HOME, HF_DATASETS_CACHE, WANDB_DIR 등)
ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "sourced prolog env: $ENV_FILE"
    # HF_HOME이 로컬 SSD로 가면 모델을 매번 다시 다운로드하므로,
    # hub 캐시만 홈 디렉토리를 사용하도록 복원
    # prolog source 후에 추가
    export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_BASE/torchinductor"
    export VLLM_CACHE_ROOT="$LOCAL_JOB_BASE/vllm"
    export TRITON_CACHE_DIR="$LOCAL_JOB_BASE/triton"

    mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
    mkdir -p "$VLLM_CACHE_ROOT"
    mkdir -p "$TRITON_CACHE_DIR"

    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
    mkdir -p "$HF_DATASETS_CACHE"
    echo "HF_HUB_CACHE restored to $HF_HUB_CACHE"
else
    echo "[WARN] prolog env not found: $ENV_FILE (local SSD caching disabled)"
fi

STEP=5000
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
DATASET_START=$((TASK_INDEX * STEP))
DATASET_END=$((DATASET_START + STEP))

export PRM=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

# Temperature settings (comma-separated for argparse)

python scripts/test_time_compute.py "$@" \
    --dataset_start=$DATASET_START \
    --dataset_end=$DATASET_END \
    --prm_path=$PRM \
    --push_to_hub=true
