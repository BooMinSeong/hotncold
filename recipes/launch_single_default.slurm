#!/bin/bash
#SBATCH --job-name=single_default # Job name
# cpu, gpu Setting
#SBATCH --qos=hpgpu
#SBATCH --partition=A100-80GB
#SBATCH --exclude=n80,n79
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8              # CPU cores/threads

#SBATCH --output=logs/%x/%A/task_%a.out
#SBATCH --error=logs/%x/%A/task_%a.err
#SBATCH --time=24:00:00

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

# 이전 job에서 상속된 로컬 SSD 환경변수 제거 (sbatch --export=ALL 기본 동작으로 인한 오염 방지)
unset HF_HOME HF_DATASETS_CACHE WANDB_DIR TMPDIR LOCAL_JOB_BASE

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

# PRM 모델 설정
export PRM=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

# Note: dataset_start, dataset_end, dataset_name, and seed should be passed as command-line arguments
# via --dataset_start, --dataset_end, --dataset_name, and --seed when calling this script

python scripts/test_time_compute.py "$@" \
	--prm_path=$PRM \
	--push_to_hub=true
