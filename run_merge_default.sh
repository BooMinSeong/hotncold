#!/bin/bash

SEEDS=(0 42 64)
METHOD_NAMES=("bon" "beam_search" "dvts")
TEMPERATURE=0.4
DATASET_NAME="math-ai/aime25" # "HuggingFaceH4/MATH-500"
MODEL_NAME="Qwen2.5-3B-Instruct"

# Default parameters from config
TOP_P=1.0
N=64
LOOKAHEAD=0

# 각 method에 대해 순회
for METHOD_NAME in "${METHOD_NAMES[@]}"; do
    # 각 seed에 대해 merge 실행
    for SEED in "${SEEDS[@]}"; do
        DATASET_REPO="ENSEONG/default-${DATASET_NAME##*/}-$MODEL_NAME-$METHOD_NAME"

        # best_of_n는 look 파라미터가 없음
        if [ "$METHOD_NAME" = "bon" ]; then
            FILTER_STR="T-${TEMPERATURE}--top_p-${TOP_P}--n-${N}--seed-${SEED}"
        else
            # beam_search, dvts는 look 파라미터 포함
            FILTER_STR="T-${TEMPERATURE}--top_p-${TOP_P}--n-${N}--m-4--iters-40--look-0--seed-${SEED}"
        fi

        echo "Merging: Method=$METHOD_NAME, Dataset=$DATASET_REPO, Filter=$FILTER_STR"
        uv run python scripts/merge_chunks.py \
            --dataset_name=$DATASET_REPO \
            --filter_strings $FILTER_STR 
    done
done

# 모든 백그라운드 작업이 완료될 때까지 대기
echo "All merge operations completed!"
