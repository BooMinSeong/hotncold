#!/bin/bash
#SBATCH --job-name=verify_score_opt
#SBATCH --qos=nogpu
#SBATCH --partition=cpu-max24
#SBATCH --exclude=n80,n79
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x/%A/task_%a.out
#SBATCH --error=logs/%x/%A/task_%a.err
#SBATCH --time=03:00:00

# Usage:
#   sbatch recipes/verify_score_optimization.slurm
#
# Optional overrides:
#   sbatch recipes/verify_score_optimization.slurm \
#       --dataset_name=ENSEONG/... --n=64 --num_proc=8

# 모듈 설정
module purge
module load gnu12/12.2.0
module load cuda/12.6

# 환경 설정
source ~/.bashrc
source .venv/bin/activate

export CC=$(which gcc)
export CXX=$(which g++)

# Prolog이 생성한 로컬 SSD 환경변수 로드
ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "sourced prolog env: $ENV_FILE"
    export TORCHINDUCTOR_CACHE_DIR="$LOCAL_JOB_BASE/torchinductor"
    export VLLM_CACHE_ROOT="$LOCAL_JOB_BASE/vllm"
    export TRITON_CACHE_DIR="$LOCAL_JOB_BASE/triton"
    mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
    mkdir -p "$HF_DATASETS_CACHE"
    echo "HF_HUB_CACHE restored to $HF_HUB_CACHE"
else
    echo "[WARN] prolog env not found: $ENV_FILE (local SSD caching disabled)"
fi

python scripts/verify_score_optimization.py \
    --num_proc=16\
    "$@"
