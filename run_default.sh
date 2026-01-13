#!/bin/bash

SEEDS=(0 42 64)
SEARCH_METHODS=("best_of_n.yaml" "beam_search.yaml" "dvts.yaml")
METHOD_NAMES=("bon" "beam_search" "dvts")
TEMPERATURE=0.8
DATASET_NAME="math-ai/aime25" # "HuggingFaceH4/MATH-500"

# 각 search method와 dataset name을 순회
for i in "${!SEARCH_METHODS[@]}"; do
    METHOD="${SEARCH_METHODS[$i]}"
    METHOD_NAME="${METHOD_NAMES[$i]}"
    
    # 각 seed에 대해 실행
    for SEED in "${SEEDS[@]}"; do
        echo "Running: Method=$METHOD, Dataset=$DATASET, Seed=$SEED, Temp=$TEMPERATURE"
        sbatch recipes/launch_array_default.slurm \
            recipes/Qwen2.5-1.5B-Instruct/$METHOD \
            --hub_dataset_id=ENSEONG/default-${DATASET_NAME##*/}-Qwen2.5-3B-Instruct-$METHOD_NAME \
            --seed=$SEED \
            --temperature=$TEMPERATURE \
	    --dataset_name=$DATASET_NAME
    done
done
