#!/bin/bash

# SEEDS=(0 42 64 128 256 512)
SEEDS=(128 256 512)
SEARCH_METHODS=("best_of_n.yaml") # "beam_search.yaml" "dvts.yaml"
METHOD_NAMES=("bon") # "beam_search"  "dvts"
DATASET_NAME="ENSEONG/math-private" # math-ai/aime25 "HuggingFaceH4/MATH-500"
# DATASET_NAME="ENSEONG/gsm8k-private" # math-ai/aime25 "HuggingFaceH4/MATH-500" "ENSEONG/math-private"
# MODEL_NAME="Llama-3.2-3B-Instruct" # meta-llama/Llama-3.2-3B-Instruct # Qwen2.5-3B-Instruct # Qwen3-4B-Instruct-2507
MODEL_NAME="Qwen2.5-3B-Instruct"
N=256

# GPU partition: "A100-80GB" or "L40S"
GPU_PARTITION="L40S"

JOB_COUNT=0
QOS_LIMIT=16


# Initialize Log
rm -r logs/default_run/*

# TEMPERATURES=(1.0)
# for TEMPERATURE in "${TEMPERATURES[@]}"; do
for TEMPERATURE in $(seq 0.1 0.1 1.2); do
	# 각 search method와 dataset name을 순회
	for i in "${!SEARCH_METHODS[@]}"; do
	    METHOD="${SEARCH_METHODS[$i]}"
	    METHOD_NAME="${METHOD_NAMES[$i]}"

	    # 각 seed에 대해 실행
	    for SEED in "${SEEDS[@]}"; do
		JOB_COUNT=$((JOB_COUNT + 1))
		QOS_FLAG=""
		if [ "$GPU_PARTITION" = "A100-80GB" ] || [ "$GPU_PARTITION" = "H200" ]; then
		    if [ "$JOB_COUNT" -gt "$QOS_LIMIT" ]; then
			QOS_FLAG="--qos=add_hpgpu"
		    else
			QOS_FLAG="--qos=hpgpu"
		    fi
		fi
		echo "Running (#$JOB_COUNT): Method=$METHOD, Dataset=$DATASET_NAME, Seed=$SEED, Temp=$TEMPERATURE, GPU=$GPU_PARTITION ${QOS_FLAG:+(QOS: add_hpgpu)}"
		sbatch --partition=$GPU_PARTITION $QOS_FLAG recipes/launch_array_default.slurm \
		    recipes/$MODEL_NAME/$METHOD \
		    --hub_dataset_id=ENSEONG/full-${DATASET_NAME##*/}-n${N}-$MODEL_NAME-$METHOD_NAME \
		    --seed=$SEED \
		    --n=$N \
		    --temperature=$TEMPERATURE \
		    --dataset_name=$DATASET_NAME
	    done
	done
done
