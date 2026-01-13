#!/bin/bash

SEEDS=(0 42 64)
METHOD_NAMES=("bon" "beam_search" "dvts")
TEMPERATURES=(0.4 0.8)
DATASET_NAME="math-ai/aime25" # "HuggingFaceH4/MATH-500"

# Default parameters from config
TOP_P=1.0
N=64
LOOKAHEAD=0

# 각 temperature에 대해 순회
for TEMPERATURE in "${TEMPERATURES[@]}"; do
    # 각 method에 대해 순회
    for METHOD_NAME in "${METHOD_NAMES[@]}"; do
        # 각 seed에 대해 merge 실행
        for SEED in "${SEEDS[@]}"; do
            DATASET_REPO="ENSEONG/default-${DATASET_NAME##*/}-Qwen2.5-1.5B-Instruct-$METHOD_NAME"

            # best_of_n는 look 파라미터가 없음
            if [ "$METHOD_NAME" = "bon" ]; then
                FILTER_STR="T-${TEMPERATURE}--top_p-${TOP_P}--n-${N}--seed-${SEED}"
            else
                # beam_search, dvts는 look 파라미터 포함
                FILTER_STR="T-${TEMPERATURE}--top_p-${TOP_P}--n-${N}--look-${LOOKAHEAD}--seed-${SEED}"
            fi

            echo "Merging: Temp=$TEMPERATURE, Method=$METHOD_NAME, Dataset=$DATASET_REPO, Filter=$FILTER_STR"
            python scripts/merge_chunks.py \
                --dataset_name=$DATASET_REPO \
                --filter_strings $FILTER_STR &
        done
    done
done

# 모든 백그라운드 작업이 완료될 때까지 대기
wait
echo "All merge operations completed!"
