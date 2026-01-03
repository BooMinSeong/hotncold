#!/bin/bash

# Test script for hot and cold temperature strategy
# Tests best_of_n, beam_search, and dvts with multi-temperature

set -e  # Exit on error

echo "========================================="
echo "Testing Hot and Cold Temperature Strategy"
echo "========================================="
echo ""

# Configuration
GPU_ID=${GPU_ID:-1}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-1.5B-Instruct"}
DATASET_NAME=${DATASET_NAME:-"HuggingFaceH4/MATH-500"}
NUM_SAMPLES=${NUM_SAMPLES:-2}
OUTPUT_BASE="./test_outputs_hot_cold"

# Temperature settings (space-separated for argparse)
TEMPS="0.6 0.8 1.0"
TEMP_RATIOS="0.33 0.34 0.33"

echo "Configuration:"
echo "  GPU: $GPU_ID"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_NAME"
echo "  Samples: $NUM_SAMPLES"
echo "  Temperatures: $TEMPS"
echo "  Temperature Ratios: $TEMP_RATIOS"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Test 1: best_of_n with multi-temperature
echo "========================================="
echo "Test 1: best_of_n with multi-temperature"
echo "========================================="
OUTPUT_DIR="$OUTPUT_BASE/best_of_n"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run python scripts/test_time_compute.py \
    --approach best_of_n \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --num_samples $NUM_SAMPLES \
    --n 12 \
    --temperatures $TEMPS \
    --temperature_ratios $TEMP_RATIOS \
    --output_dir "$OUTPUT_DIR" \
    --custom_chat_template none

echo "✓ best_of_n completed"
echo ""

# Test 2: best_of_n with equal distribution (no ratios)
echo "========================================="
echo "Test 2: best_of_n with equal distribution"
echo "========================================="
OUTPUT_DIR="$OUTPUT_BASE/best_of_n_equal"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run python scripts/test_time_compute.py \
    --approach best_of_n \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --num_samples $NUM_SAMPLES \
    --n 12 \
    --temperatures $TEMPS \
    --output_dir "$OUTPUT_DIR" \
    --custom_chat_template none

echo "✓ best_of_n (equal) completed"
echo ""

# Test 3: beam_search with multi-temperature
echo "========================================="
echo "Test 3: beam_search with multi-temperature"
echo "========================================="
OUTPUT_DIR="$OUTPUT_BASE/beam_search"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run python scripts/test_time_compute.py \
    --approach beam_search \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --num_samples $NUM_SAMPLES \
    --n 12 \
    --beam_width 3 \
    --num_iterations 3 \
    --temperatures $TEMPS \
    --temperature_ratios $TEMP_RATIOS \
    --output_dir "$OUTPUT_DIR" \
    --custom_chat_template none

echo "✓ beam_search completed"
echo ""

# Test 4: beam_search with single temperature (baseline)
echo "========================================="
echo "Test 4: beam_search with single temperature (baseline)"
echo "========================================="
OUTPUT_DIR="$OUTPUT_BASE/beam_search_baseline"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run python scripts/test_time_compute.py \
    --approach beam_search \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --num_samples $NUM_SAMPLES \
    --n 12 \
    --beam_width 3 \
    --num_iterations 3 \
    --temperature 0.8 \
    --output_dir "$OUTPUT_DIR" \
    --custom_chat_template none

echo "✓ beam_search (baseline) completed"
echo ""

# Test 5: DVTS with multi-temperature
echo "========================================="
echo "Test 5: DVTS with multi-temperature"
echo "========================================="
OUTPUT_DIR="$OUTPUT_BASE/dvts"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run python scripts/test_time_compute.py \
    --approach dvts \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --num_samples $NUM_SAMPLES \
    --n 12 \
    --beam_width 4 \
    --num_iterations 3 \
    --temperatures $TEMPS \
    --temperature_ratios $TEMP_RATIOS \
    --output_dir "$OUTPUT_DIR" \
    --custom_chat_template none

echo "✓ DVTS completed"
echo ""

# Test 6: DVTS with single temperature (baseline)
echo "========================================="
echo "Test 6: DVTS with single temperature (baseline)"
echo "========================================="
OUTPUT_DIR="$OUTPUT_BASE/dvts_baseline"
mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run python scripts/test_time_compute.py \
    --approach dvts \
    --model_path "$MODEL_PATH" \
    --dataset_name "$DATASET_NAME" \
    --num_samples $NUM_SAMPLES \
    --n 12 \
    --beam_width 4 \
    --num_iterations 3 \
    --temperature 0.8 \
    --output_dir "$OUTPUT_DIR" \
    --custom_chat_template none

echo "✓ DVTS (baseline) completed"
echo ""

# Summary
echo "========================================="
echo "All Tests Completed Successfully!"
echo "========================================="
echo ""
echo "Results saved in: $OUTPUT_BASE"
echo ""
echo "Test Summary:"
echo "  1. best_of_n (multi-temp)     : $OUTPUT_BASE/best_of_n"
echo "  2. best_of_n (equal dist)     : $OUTPUT_BASE/best_of_n_equal"
echo "  3. beam_search (multi-temp)   : $OUTPUT_BASE/beam_search"
echo "  4. beam_search (baseline)     : $OUTPUT_BASE/beam_search_baseline"
echo "  5. DVTS (multi-temp)          : $OUTPUT_BASE/dvts"
echo "  6. DVTS (baseline)            : $OUTPUT_BASE/dvts_baseline"
echo ""
echo "To compare results, check the 'pred' field in each results.json"
