#!/bin/bash
# Preprocess all experiments in registry
# This script preprocesses all experiments and pushes to Hub with 'preprocessed-' prefix
#
# Usage:
#   bash preprocess_all.sh
#
# Estimated time: 2-4 hours for all experiments
# Each category will be processed sequentially

set -e  # Exit on error

echo "=========================================="
echo "Preprocessing All Experiments"
echo "=========================================="
echo ""
echo "This will preprocess all experiments in exp/configs/registry.yaml"
echo "and push them to Hub with 'preprocessed-' prefix"
echo ""
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Category 1: MATH-500 Qwen2.5-1.5B (HNC Strategy)
echo ""
echo "=========================================="
echo "[1/5] Preprocessing: math500_Qwen2.5-1.5B_hnc"
echo "=========================================="
uv run python exp/scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-1.5B_hnc \
    --push-to-hub \
    --validate

# Category 2: MATH-500 Qwen2.5-1.5B (Default Strategy)
echo ""
echo "=========================================="
echo "[2/5] Preprocessing: math500_Qwen2.5-1.5B"
echo "=========================================="
uv run python exp/scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-1.5B \
    --push-to-hub \
    --validate

# Category 3: MATH-500 Qwen2.5-3B (Default Strategy)
echo ""
echo "=========================================="
echo "[3/5] Preprocessing: math500_Qwen2.5-3B"
echo "=========================================="
uv run python exp/scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-3B \
    --push-to-hub \
    --validate

# Category 4: AIME25 Qwen2.5-1.5B
echo ""
echo "=========================================="
echo "[4/5] Preprocessing: aime25_Qwen2.5-1.5B"
echo "=========================================="
uv run python exp/scripts/preprocess_dataset.py \
    --category aime25_Qwen2.5-1.5B \
    --push-to-hub \
    --validate

# Category 5: AIME25 Qwen2.5-3B
echo ""
echo "=========================================="
echo "[5/5] Preprocessing: aime25_Qwen2.5-3B"
echo "=========================================="
uv run python exp/scripts/preprocess_dataset.py \
    --category aime25_Qwen2.5-3B \
    --push-to-hub \
    --validate

echo ""
echo "=========================================="
echo "All preprocessing complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Update exp/configs/registry.yaml to use preprocessed- paths"
echo "2. Run analysis scripts with preprocessed datasets"
echo ""
