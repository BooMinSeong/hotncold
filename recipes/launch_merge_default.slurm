#!/bin/bash
#SBATCH --job-name=merge_default
#SBATCH --qos=nogpu
#SBATCH --partition=cpu-max24
#SBATCH --exclude=n80,n79
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=24

#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err
#SBATCH --time=12:00:00

# 모듈 설정
module purge
module load gnu12/12.2.0

# 환경 설정
source ~/.bashrc
source .venv/bin/activate

# 컴파일러 명시 설정
export CC=$(which gcc)
export CXX=$(which g++)

# Prolog이 생성한 로컬 SSD 환경변수 로드
ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
    echo "sourced prolog env: $ENV_FILE"
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
    export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null)
    mkdir -p "$HF_DATASETS_CACHE"
    echo "HF_HUB_CACHE restored to $HF_HUB_CACHE"
else
    echo "[WARN] prolog env not found: $ENV_FILE (local SSD caching disabled)"
fi

# merge 스크립트 실행
bash merge_default.sh
