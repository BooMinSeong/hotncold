#!/bin/bash

SEEDS=(0 42 64)
SEARCH_METHODS=("best_of_n.yaml") # "beam_search.yaml" "dvts.yaml"
METHOD_NAMES=("bon") # "beam_search"  "dvts"
# TEMPERATURE=0.1
# DATASET_NAME="HuggingFaceH4/MATH-500" # math-ai/aime25 "HuggingFaceH4/MATH-500"
DATASET_NAME="ENSEONG/math-private" # math-ai/aime25 "HuggingFaceH4/MATH-500" "ENSEONG/math-private"
MODEL_NAME="Llama-3.2-3B-Instruct" # meta-llama/Llama-3.2-3B-Instruct # Qwen2.5-3B-Instruct # Qwen3-4B-Instruct-2507

JOB_COUNT=0
QOS_LIMIT=16

for TEMPERATURE in $(seq 0.1 0.1 1.2); do
	# 각 search method와 dataset name을 순회
	for i in "${!SEARCH_METHODS[@]}"; do
	    METHOD="${SEARCH_METHODS[$i]}"
	    METHOD_NAME="${METHOD_NAMES[$i]}"

	    # 각 seed에 대해 실행
	    for SEED in "${SEEDS[@]}"; do
		JOB_COUNT=$((JOB_COUNT + 1))
		QOS_FLAG=""
		if [ "$JOB_COUNT" -gt "$QOS_LIMIT" ]; then
		    QOS_FLAG="--qos=add_hpgpu"
		fi
		echo "Running (#$JOB_COUNT): Method=$METHOD, Dataset=$DATASET_NAME, Seed=$SEED, Temp=$TEMPERATURE ${QOS_FLAG:+(QOS: add_hpgpu)}"
		sbatch $QOS_FLAG recipes/launch_array_default.slurm \
		    recipes/$MODEL_NAME/$METHOD \
		    --hub_dataset_id=ENSEONG/full-${DATASET_NAME##*/}-$MODEL_NAME-$METHOD_NAME \
		    --seed=$SEED \
		    --temperature=$TEMPERATURE \
		    --dataset_name=$DATASET_NAME
	    done
	done
done
