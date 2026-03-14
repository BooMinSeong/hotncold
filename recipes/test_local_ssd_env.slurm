#!/bin/bash
#SBATCH --job-name=test_local_ssd
#SBATCH --qos=hpgpu
#SBATCH --partition=A100-80GB
#SBATCH --exclude=n79,n80
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

# Usage: sbatch recipes/test_local_ssd_env.slurm

echo "=== SLURM BASIC INFO ==="
echo "JOB ID   : $SLURM_JOB_ID"
echo "USER     : $USER"
echo "NODE     : $(hostname)"
echo

# === BEFORE: prolog source 전 환경변수 ===
echo "=== BEFORE prolog source ==="
echo "HF_HOME           = ${HF_HOME:-(unset)}"
echo "HF_HUB_CACHE      = ${HF_HUB_CACHE:-(unset)}"
echo "HF_DATASETS_CACHE  = ${HF_DATASETS_CACHE:-(unset)}"
echo "WANDB_DIR         = ${WANDB_DIR:-(unset)}"
echo "TMPDIR            = ${TMPDIR:-(unset)}"
echo "LOCAL_JOB_BASE    = ${LOCAL_JOB_BASE:-(unset)}"
echo

# === Prolog env 파일 확인 및 로드 ===
ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
echo "=== PROLOG ENV FILE ==="
if [ -f "$ENV_FILE" ]; then
    echo "[OK] $ENV_FILE exists"
    echo "--- contents ---"
    cat "$ENV_FILE"
    echo "----------------"
    source "$ENV_FILE"
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
    mkdir -p "$HF_DATASETS_CACHE"
else
    echo "[FAIL] $ENV_FILE not found — prolog may not be configured"
    exit 1
fi
echo

# === AFTER: prolog source 후 환경변수 ===
echo "=== AFTER prolog source ==="
echo "HF_HOME           = ${HF_HOME:-(unset)}"
echo "HF_HUB_CACHE      = ${HF_HUB_CACHE:-(unset)}"
echo "HF_DATASETS_CACHE  = ${HF_DATASETS_CACHE:-(unset)}"
echo "WANDB_DIR         = ${WANDB_DIR:-(unset)}"
echo "TMPDIR            = ${TMPDIR:-(unset)}"
echo "LOCAL_JOB_BASE    = ${LOCAL_JOB_BASE:-(unset)}"
echo

# === 디렉토리 존재 확인 ===
echo "=== VERIFY DIRECTORIES ==="
for dir in "$LOCAL_JOB_BASE" "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$TMPDIR" "$HF_HUB_CACHE"; do
    if [ -d "$dir" ]; then
        echo "[OK]   $dir"
    else
        echo "[FAIL] $dir does not exist"
    fi
done
echo

# === 쓰기 테스트 ===
echo "=== WRITE TEST (local SSD) ==="
echo "test" > "$HF_DATASETS_CACHE/test_write.txt" && echo "[OK] write to HF_DATASETS_CACHE" || echo "[FAIL] write to HF_DATASETS_CACHE"
echo "test" > "$TMPDIR/test_write.txt" && echo "[OK] write to TMPDIR" || echo "[FAIL] write to TMPDIR"
echo

echo "=== DONE ==="
