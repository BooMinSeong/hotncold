#!/bin/bash

MODEL_NAME="Qwen2.5-3B-Instruct"
DATASET_NAME="MATH-500"
SEEDS=(0 42 64)
SEARCH_METHODS=("best_of_n.yaml" "beam_search.yaml" "dvts.yaml")
DATASET_NAMES=("bon" "beam_search" "dvts")
TEMPERATURES=(0.4 0.8)

# 시드별로 missing 범위 정의 (연관 배열 사용)
declare -A BON_MISSING
BON_MISSING[0]="150 200, 450 500"
# BON_MISSING[42]=""
# BON_MISSING[64]=""

declare -A BEAM_MISSING
# BEAM_MISSING[0]=""
BEAM_MISSING[42]="100 150"
BEAM_MISSING[64]="200 250"

declare -A DVTS_MISSING
# DVTS_MISSING[0]=""
# DVTS_MISSING[42]=""
DVTS_MISSING[64]="250 300, 300 350, 350 400, 400 450, 450 500"

for TEMPERATURE in "${TEMPERATURES[@]}"; do
	# 각 search method와 dataset name을 순회
	for i in "${!SEARCH_METHODS[@]}"; do
	    METHOD="${SEARCH_METHODS[$i]}"
	    DATASET="${DATASET_NAMES[$i]}"
	    
	    echo "Processing $METHOD ($DATASET)..."
	    
	    # 각 seed에 대해 실행
	    for SEED in "${SEEDS[@]}"; do
		# 현재 method와 seed에 해당하는 missing 범위 가져오기
		if [ $i -eq 0 ]; then
		    RANGES="${BON_MISSING[$SEED]}"
		elif [ $i -eq 1 ]; then
		    RANGES="${BEAM_MISSING[$SEED]}"
		else
		    RANGES="${DVTS_MISSING[$SEED]}"
		fi
		
		# 해당 시드에 missing 범위가 없으면 스킵
		if [ -z "$RANGES" ]; then
		    echo "  Skipping Seed=$SEED: No missing ranges"
		    continue
		fi
		
		# 콤마로 구분된 범위들을 처리
		IFS=',' read -ra RANGE_ARRAY <<< "$RANGES"
		
		for RANGE in "${RANGE_ARRAY[@]}"; do
		    # 앞뒤 공백 제거
		    RANGE=$(echo "$RANGE" | xargs)
		    read -r START END <<< "$RANGE"
		    
		    echo "  Running: Method=$METHOD, Dataset=$DATASET, Range=$START-$END, Seed=$SEED, Temp=$TEMPERATURE"
		    
		    sbatch recipes/launch_single_default.slurm \
			recipes/$MODEL_NAME/$METHOD \
			--hub_dataset_id=ENSEONG/default-$DATASET_NAME-$MODEL_NAME-$DATASET \
			--dataset_start=$START \
			--dataset_end=$END \
			--seed=$SEED \
			--temperature=$TEMPERATURE
		done
	    done
	done

echo "All jobs submitted!"
