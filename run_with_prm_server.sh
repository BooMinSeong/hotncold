#!/bin/bash
# Orchestration script for running experiments with a separate PRM server
#
# This script:
# 1. Launches a PRM server job
# 2. Waits for the service file to be created
# 3. Launches compute array jobs that connect to the PRM server
# 4. Sets up automatic cleanup when compute jobs finish
#
# Usage:
#   ./run_with_prm_server.sh recipes/Qwen2.5-3B-Instruct/best_of_n.yaml
#   ./run_with_prm_server.sh recipes/Qwen2.5-3B-Instruct/best_of_n.yaml --seed=42

set -e

# Configuration
PRM_MODEL=${PRM_MODEL:-"Qwen/Qwen2.5-Math-PRM-7B"}
SERVICE_DIR=${SERVICE_DIR:-"/home/b.ms/projects/hotncold/services"}
MAX_WAIT=${MAX_WAIT:-300}  # 5 minutes

# Parse arguments
CONFIG_FILE=$1
if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <config_file> [additional_args...]"
    echo "Example: $0 recipes/Qwen2.5-3B-Instruct/best_of_n.yaml --seed=42"
    exit 1
fi
shift  # Remove first argument, pass rest to compute job

# Ensure directories exist
mkdir -p $SERVICE_DIR
mkdir -p logs/prm_server

echo "========================================"
echo "PRM Server + Compute Orchestration"
echo "========================================"
echo "Config: $CONFIG_FILE"
echo "PRM Model: $PRM_MODEL"
echo "Service Directory: $SERVICE_DIR"
echo "========================================"

# Step 1: Launch PRM server job
echo "[1/4] Launching PRM server job..."
PRM_JOB_ID=$(sbatch --parsable \
    --export=ALL,PRM_MODEL=$PRM_MODEL,SERVICE_DIR=$SERVICE_DIR \
    recipes/launch_prm_server.slurm)

echo "PRM server job ID: $PRM_JOB_ID"

# Step 2: Wait for service file
SERVICE_FILE="${SERVICE_DIR}/${PRM_JOB_ID}_prm_service.json"
echo "[2/4] Waiting for service file: $SERVICE_FILE"

WAITED=0
while [ ! -f "$SERVICE_FILE" ] && [ $WAITED -lt $MAX_WAIT ]; do
    sleep 10
    WAITED=$((WAITED + 10))
    echo "  Waited ${WAITED}s / ${MAX_WAIT}s..."
done

if [ ! -f "$SERVICE_FILE" ]; then
    echo "ERROR: Service file not found after ${MAX_WAIT}s"
    echo "Cancelling PRM server job..."
    scancel $PRM_JOB_ID
    exit 1
fi

echo "Service file found!"
cat $SERVICE_FILE

# Step 3: Launch compute array job with dependency
echo "[3/4] Launching compute array job..."
COMPUTE_JOB_ID=$(sbatch --parsable \
    --dependency=afterok:$PRM_JOB_ID \
    --export=ALL,PRM_SERVICE_FILE=$SERVICE_FILE,PRM_MODEL=$PRM_MODEL \
    recipes/launch_array_api.slurm \
    $CONFIG_FILE \
    "$@")

echo "Compute job ID: $COMPUTE_JOB_ID"

# Step 4: Set up cleanup job
echo "[4/4] Setting up cleanup job..."
CLEANUP_JOB_ID=$(sbatch --parsable \
    --dependency=afterany:$COMPUTE_JOB_ID \
    --job-name=cleanup_prm_${PRM_JOB_ID} \
    --output=logs/cleanup_%j.out \
    --time=00:05:00 \
    --wrap="scancel $PRM_JOB_ID 2>/dev/null || true; rm -f $SERVICE_FILE; echo 'Cleanup completed'")

echo "Cleanup job ID: $CLEANUP_JOB_ID"

echo ""
echo "========================================"
echo "Jobs submitted successfully!"
echo "========================================"
echo "  PRM Server:  $PRM_JOB_ID"
echo "  Compute:     $COMPUTE_JOB_ID"
echo "  Cleanup:     $CLEANUP_JOB_ID"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f logs/prm_server/${PRM_JOB_ID}.out"
echo "========================================"
