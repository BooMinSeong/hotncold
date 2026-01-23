# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Search and Learn (sal) is a research project for scaling test-time compute with open LLMs. The project implements search algorithms (Best-of-N, Beam Search, DVTS) that use Process Reward Models (PRMs) to guide language models to solve complex math problems by allowing them to "think longer" on harder problems.

## Key Commands

### Setup
```bash
# Create environment and install dependencies
conda create -n sal python=3.11 && conda activate sal
pip install -e '.[dev]'

# Login to Hugging Face Hub
huggingface-cli login

# Install Git LFS for pushing models
sudo apt-get install git-lfs
```

### Code Quality
```bash
make style      # Format and fix imports with ruff
make quality    # Check code quality with ruff
```

### Running Test-Time Compute

All test-time compute experiments use the main script with YAML configs:

```bash
# Basic usage with a recipe config
export CONFIG=recipes/Llama-3.2-1B-Instruct/best_of_n.yaml
python scripts/test_time_compute.py $CONFIG

# Override model, dataset, or PRM
python scripts/test_time_compute.py $CONFIG \
    --model_path=meta-llama/Llama-3.2-8B-Instruct \
    --dataset_name=AI-MO/aimo-validation-aime \
    --dataset_split=train \
    --prm_path=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

# Push results to Hub as dataset branch
python scripts/test_time_compute.py $CONFIG --push_to_hub=true
```

### Merging Parallel Results
```bash
# After running parallelized jobs (e.g., via Slurm array)
python scripts/merge_chunks.py \
    --dataset_name=<YOUR_ORG>/Llama-3.2-1B-Instruct-best_of_n-completions \
    --filter_strings seed-0
```

### Preprocessing Results for Fast Analysis

**IMPORTANT**: Before running analysis scripts, experiment results must be preprocessed. This is a **one-time operation** per experiment that pre-computes all prediction evaluations (using `math_verify`), providing **20-30x faster** analysis.

```bash
# Preprocess a single experiment
python exp/scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --push-to-hub

# Preprocess all experiments in a category (recommended)
python exp/scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-1.5B \
    --push-to-hub

# Save locally instead of pushing to Hub
python exp/scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --output-dir /tmp/preprocessed

# Validate preprocessing correctness
python exp/scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --push-to-hub \
    --validate
```

**How it works**:
- **Input**: Original experiment dataset with `pred_naive@N`, `pred_weighted@N`, `pred_maj@N` fields
- **Process**: Evaluates all predictions using `math_verify.parse()` + `verify()`
- **Output**: New dataset with added `is_correct_naive@N`, `is_correct_weighted@N`, `is_correct_maj@N` boolean fields
- **Storage**: Preprocessed datasets are saved to Hub with `preprocessed-` prefix (e.g., `ENSEONG/preprocessed-default-MATH-500-...`)

**Why preprocess?**
- Analysis scripts need to evaluate thousands of predictions (e.g., 384,000 for MATH-500)
- `math_verify` parsing/verification is computationally expensive
- Preprocessing once eliminates this bottleneck from analysis
- Result: Analysis that took 5-10 minutes now takes 10-30 seconds

### Analyzing Results

After preprocessing, use the unified analysis scripts:

```bash
# Analyze a single experiment
python exp/scripts/analyze_results.py ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-bon

# Analyze a category from registry
python exp/scripts/analyze_results.py --category math500_Qwen2.5-1.5B

# Temperature comparison analysis
python exp/scripts/analyze_results.py --category math500_Qwen2.5-1.5B --analysis-type temperature_comparison

# Difficulty-based temperature analysis
python exp/scripts/analyze_difficulty_temperature.py \
    --category math500_Qwen2.5-1.5B \
    --approach bon \
    --output-dir exp/analysis_output-MATH500-Qwen2.5-1.5B-difficulty
```

**Note**: Analysis scripts **require** preprocessed datasets. If you try to analyze a non-preprocessed dataset, you'll get an error message telling you to run preprocessing first.

### Training PRMs (requires TRL)
```bash
# Install TRL dependencies
pip install -e '.[trl]'

# See recipes/training/ for training scripts
# Example structure: recipes/training/Qwen2.5-Math-1.5B-Instruct-PRM/train.sh
```

## Architecture

### Core Module Structure (`src/sal/`)

- **config.py**: Central configuration dataclass for all experiments. Controls approach (best_of_n/beam_search/dvts), model paths, PRMs, dataset settings, and search hyperparameters.

- **models/reward_models.py**: PRM loading and inference. Supports multiple PRMs including RLHFlow, math-shepherd, and Skywork-o1 models. Handles tokenization and scoring of reasoning steps.

- **search/**: Three search algorithms that use PRMs to guide generation:
  - `best_of_n.py`: Sample N completions, select best by PRM score
  - `beam_search.py`: Stepwise beam search guided by PRM
  - `diverse_verifier_tree_search.py`: DVTS algorithm combining diversity and verification

- **utils/**: Supporting utilities
  - `data.py`: Dataset loading/filtering from Hub
  - `math.py`: Math-specific utilities for step extraction and formatting
  - `parser.py`: YAML config parsing (H4ArgumentParser)
  - `score.py`: Scoring completions with PRMs
  - `qwen_math_parser.py`: Answer extraction for MATH dataset evaluation

### Experiment Analysis Structure (`exp/`)

- **scripts/**: Analysis and preprocessing scripts
  - `preprocess_dataset.py`: Preprocess experiment results by pre-computing evaluations (adds `is_correct_*` fields)
  - `analyze_results.py`: Unified analysis script with auto-discovery (summary, HNC comparison, temperature comparison, model comparison)
  - `analyze_difficulty_temperature.py`: Difficulty-stratified temperature performance analysis

- **analysis/**: Core analysis modules
  - `core.py`: Fundamental evaluation functions (`evaluate_answer`, `evaluate_result`)
  - `metrics.py`: Accuracy computation from preprocessed datasets (uses `is_correct_*` fields for 20-30x speedup)
  - `difficulty.py`: Problem difficulty stratification and baseline computation
  - `difficulty_temperature.py`: Temperature-difficulty analysis utilities
  - `preprocessing.py`: Preprocessing utilities (`preprocess_single_subset`, validation functions)
  - `datasets.py`: Dataset loading with auto-discovery from Hub
  - `discovery.py`: Automatic configuration discovery from Hub metadata
  - `visualization.py`: Plotting utilities for reports

- **configs/**: Configuration files
  - `registry.yaml`: Registry of experiment Hub paths organized by category

### Search Algorithm Flow

1. `scripts/test_time_compute.py` loads config, LLM (via vLLM), and PRM
2. Dataset is processed through the selected search algorithm (`approach_fn`)
3. Search algorithm generates multiple completions using the LLM
4. PRM scores reasoning steps to guide search
5. Results are scored and saved (locally or pushed to Hub as dataset branches)

### Config System

All experiments are driven by YAML configs in `recipes/`. Configs specify:
- `approach`: Which search algorithm to use
- `model_path`, `prm_path`: Model identifiers from Hub
- Search parameters: `n` (compute budget), `batch_size`, `temperature`, etc.
- Dataset settings: `dataset_name`, `num_samples`, start/end indices
- Output: local `output_dir` or `hub_dataset_id` for Hub uploads

Command-line arguments override config values.

### Supported PRMs

- `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` (default)
- `peiyi9979/math-shepherd-mistral-7b-prm`
- `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B`
- `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`
- Custom PRMs trained with TRL (see `recipes/training/`)

## Important Notes

- **Preprocessing Required**: Analysis scripts require preprocessed datasets. Always run `exp/scripts/preprocess_dataset.py` on experiment results before analysis. This one-time preprocessing adds `is_correct_*` boolean fields that enable 20-30x faster analysis by eliminating expensive `math_verify` calls.

- **Dataset Naming**: Preprocessed datasets use `preprocessed-` prefix (e.g., `ENSEONG/preprocessed-default-MATH-500-...`). Update registry paths to point to preprocessed versions after preprocessing.

- **Chat Templates**: Default configs use a custom Llama 3 chat template optimized for math reasoning. For non-Llama models, set `--custom_chat_template=none`.

- **GPU Memory**: The `gpu_memory_utilization` parameter (default 0.5) controls how much GPU memory vLLM uses. PRMs use remaining memory. Adjust for your hardware.

- **Dataset Branches**: When pushing to Hub, results are pushed as branches/revisions on the dataset repo (not main). Use `load_dataset(repo, revision=branch)` to load specific results.

- **Parallelization**: For large-scale experiments, use `recipes/launch_array.slurm` to shard dataset across multiple jobs, then merge with `scripts/merge_chunks.py`.

- **Evaluation**: Current evaluation uses an external fork of Qwen2.5-Math's parser. A standalone evaluation script is planned.
