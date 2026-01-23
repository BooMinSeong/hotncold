#!/bin/bash
# Run difficulty-temperature analysis for all experiments in registry
#
# Usage:
#   bash analyze_difficulty_temperature_all.sh

set -e  # Exit on error

echo "=========================================="
echo "Difficulty-Temperature Analysis (All)"
echo "=========================================="
echo ""

# MATH-500 Qwen2.5-1.5B - bon
echo "[1/9] math500_Qwen2.5-1.5B - bon"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category math500_Qwen2.5-1.5B \
    --approach bon \
    --output-dir exp/analysis_output-MATH500-Qwen2.5-1.5B-bon-difficulty \
    --verbose

# MATH-500 Qwen2.5-1.5B - beam_search
echo "[2/9] math500_Qwen2.5-1.5B - beam_search"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category math500_Qwen2.5-1.5B \
    --approach beam_search \
    --output-dir exp/analysis_output-MATH500-Qwen2.5-1.5B-beam_search-difficulty \
    --verbose

# MATH-500 Qwen2.5-1.5B - dvts
echo "[3/9] math500_Qwen2.5-1.5B - dvts"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category math500_Qwen2.5-1.5B \
    --approach dvts \
    --output-dir exp/analysis_output-MATH500-Qwen2.5-1.5B-dvts-difficulty \
    --verbose

# MATH-500 Qwen2.5-3B - bon
echo "[4/9] math500_Qwen2.5-3B - bon"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category math500_Qwen2.5-3B \
    --approach bon \
    --output-dir exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty \
    --verbose

# MATH-500 Qwen2.5-3B - beam_search
echo "[5/9] math500_Qwen2.5-3B - beam_search"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category math500_Qwen2.5-3B \
    --approach beam_search \
    --output-dir exp/analysis_output-MATH500-Qwen2.5-3B-beam_search-difficulty \
    --verbose

# MATH-500 Qwen2.5-3B - dvts
echo "[6/9] math500_Qwen2.5-3B - dvts"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category math500_Qwen2.5-3B \
    --approach dvts \
    --output-dir exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty \
    --verbose

# AIME25 Qwen2.5-1.5B - bon
echo "[7/9] aime25_Qwen2.5-1.5B - bon"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category aime25_Qwen2.5-1.5B \
    --approach bon \
    --output-dir exp/analysis_output-AIME25-Qwen2.5-1.5B-bon-difficulty \
    --verbose

# AIME25 Qwen2.5-1.5B - beam_search
echo "[8/9] aime25_Qwen2.5-1.5B - beam_search"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category aime25_Qwen2.5-1.5B \
    --approach beam_search \
    --output-dir exp/analysis_output-AIME25-Qwen2.5-1.5B-beam_search-difficulty \
    --verbose

# AIME25 Qwen2.5-1.5B - dvts
echo "[9/12] aime25_Qwen2.5-1.5B - dvts"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category aime25_Qwen2.5-1.5B \
    --approach dvts \
    --output-dir exp/analysis_output-AIME25-Qwen2.5-1.5B-dvts-difficulty \
    --verbose

# AIME25 Qwen2.5-3B - bon
echo "[10/12] aime25_Qwen2.5-3B - bon"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category aime25_Qwen2.5-3B \
    --approach bon \
    --output-dir exp/analysis_output-AIME25-Qwen2.5-3B-bon-difficulty \
    --verbose

# AIME25 Qwen2.5-3B - beam_search
echo "[11/12] aime25_Qwen2.5-3B - beam_search"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category aime25_Qwen2.5-3B \
    --approach beam_search \
    --output-dir exp/analysis_output-AIME25-Qwen2.5-3B-beam_search-difficulty \
    --verbose

# AIME25 Qwen2.5-3B - dvts
echo "[12/12] aime25_Qwen2.5-3B - dvts"
uv run python exp/scripts/analyze_difficulty_temperature.py \
    --category aime25_Qwen2.5-3B \
    --approach dvts \
    --output-dir exp/analysis_output-AIME25-Qwen2.5-3B-dvts-difficulty \
    --verbose

echo ""
echo "=========================================="
echo "All difficulty-temperature analyses complete!"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  - exp/analysis_output-MATH500-*-difficulty/"
echo "  - exp/analysis_output-AIME25-*-difficulty/"
echo ""
