#!/bin/bash

SEEDS=(0 42 64)
SEARCH_METHODS=("best_of_n.yaml" "beam_search.yaml" "dvts.yaml")
DATASET_NAMES=("bon" "beam_search" "dvts")
TEMPERATURE=0.4

# Dataset Range per seed and METHOD
# Each range is defined as "start end" pairs
BON_MISSING=()
BEAM_MISSING=("150 200")
DVTS_MISSING=("0 50" "150 200" "200 250" "250 300")

# Combine all missing ranges into a single array for iteration
declare -a ALL_MISSING_RANGES
ALL_MISSING_RANGES[0]="${BON_MISSING[@]}"
ALL_MISSING_RANGES[1]="${BEAM_MISSING[@]}"
ALL_MISSING_RANGES[2]="${DVTS_MISSING[@]}"

# 각 search method와 dataset name을 순회
for i in "${!SEARCH_METHODS[@]}"; do
    METHOD="${SEARCH_METHODS[$i]}"
    DATASET="${DATASET_NAMES[$i]}"

    # Get missing ranges for this method
    if [ $i -eq 0 ]; then
        RANGES=("${BON_MISSING[@]}")
    elif [ $i -eq 1 ]; then
        RANGES=("${BEAM_MISSING[@]}")
    else
        RANGES=("${DVTS_MISSING[@]}")
    fi

    # Skip if no missing ranges for this method
    if [ ${#RANGES[@]} -eq 0 ]; then
        echo "Skipping $METHOD: No missing ranges"
        continue
    fi

    # 각 빠진 범위에 대해 실행
    for RANGE in "${RANGES[@]}"; do
        read -r START END <<< "$RANGE"

        # 각 seed에 대해 실행
        for SEED in "${SEEDS[@]}"; do
            echo "Running: Method=$METHOD, Dataset=$DATASET, Range=$START-$END, Seed=$SEED, Temp=$TEMPERATURE"
            sbatch recipes/launch_single_default.slurm \
                recipes/Qwen2.5-1.5B-Instruct/$METHOD \
                --hub_dataset_id=ENSEONG/default-Qwen2.5-1.5B-Instruct-$DATASET \
                --dataset_start=$START \
                --dataset_end=$END \
                --seed=$SEED \
                --temperature=$TEMPERATURE
        done
    done
done


