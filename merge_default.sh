#!/bin/bash

SEEDS=(0 42 64)
METHOD_NAMES=("bon" "beam_search" "dvts")
# TEMPERATURES=(0.1 0.2 0.4 0.8)
TEMPERATURES=(0.1 0.2)
DATASET_NAME="HuggingFaceH4/MATH-500" # math-ai/aime25" # HuggingFaceH4/MATH-500" # "HuggingFaceH4/MATH-500" math-ai/aime25"
MODEL_NAME="Qwen2.5-3B-Instruct"

# Default parameters from config (must match src/sal/config.py)
TOP_P=1.0
N=64
BEAM_WIDTH=4          # m in the paper
NUM_ITERATIONS=40
LOOKAHEAD=0

# 각 temperature에 대해 순회
for TEMPERATURE in "${TEMPERATURES[@]}"; do
    # 각 method에 대해 순회
    for METHOD_NAME in "${METHOD_NAMES[@]}"; do
        # 각 seed에 대해 merge 실행
        for SEED in "${SEEDS[@]}"; do
            DATASET_REPO="ENSEONG/default-${DATASET_NAME##*/}-$MODEL_NAME-$METHOD_NAME"

            # best_of_n는 m, iters, look 파라미터가 없음
            if [ "$METHOD_NAME" = "bon" ]; then
                FILTER_STR="T-${TEMPERATURE}--top_p-${TOP_P}--n-${N}--seed-${SEED}"
            else
                # beam_search, dvts는 m, iters, look 파라미터 포함
                FILTER_STR="T-${TEMPERATURE}--top_p-${TOP_P}--n-${N}--m-${BEAM_WIDTH}--iters-${NUM_ITERATIONS}--look-${LOOKAHEAD}--seed-${SEED}"
            fi

            echo "Merging: Temp=$TEMPERATURE, Method=$METHOD_NAME, Dataset=$DATASET_REPO, Filter=$FILTER_STR"
            python scripts/merge_chunks.py \
                --dataset_name=$DATASET_REPO \
                --filter_strings $FILTER_STR 
        done
    done
done

# 모든 백그라운드 작업이 완료될 때까지 대기
echo "All merge operations completed!"
