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
uv run python scripts/test_time_compute.py $CONFIG

# Override model, dataset, or PRM
uv run python scripts/test_time_compute.py $CONFIG \
    --model_path=meta-llama/Llama-3.2-8B-Instruct \
    --dataset_name=AI-MO/aimo-validation-aime \
    --dataset_split=train \
    --prm_path=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B

# Push results to Hub as dataset branch
uv run python scripts/test_time_compute.py $CONFIG --push_to_hub=true
```

### Multi-Temperature Sampling

All search algorithms support multi-temperature sampling to diversify completions. See `TEST_GUIDE.md` for full details.

```bash
# best_of_n: distribute n completions across temperatures
uv run python scripts/test_time_compute.py \
    --approach best_of_n \
    --temperatures "0.6,0.8,1.0" \
    --temperature_ratios "0.33,0.34,0.33" \
    --n 12 ...

# beam_search / dvts: each beam/path uses a cycled temperature
uv run python scripts/test_time_compute.py \
    --approach beam_search \
    --temperatures "0.6,0.8,1.0" \
    --beam_width 3 --n 12 ...
```

Config constraints: `n` must be divisible by the number of temperatures; for beam_search/dvts, `n` must also be divisible by `beam_width`.

### Merging Parallel Results
```bash
# After running parallelized jobs (e.g., via Slurm array)
python scripts/merge_chunks.py \
    --dataset_name=<YOUR_ORG>/Llama-3.2-1B-Instruct-best_of_n-completions \
    --filter_strings seed-0
```

### Automation Scripts

Convenience shell scripts for large-scale runs (configure seeds/temperatures/methods inside each):

```bash
./run_default.sh        # Submit all default jobs via sbatch array
./run_hnc.sh            # Submit hot/cold temperature experiments
./merge_default.sh      # Merge all completed parallel job results
./run_missing.sh        # Submit missing dataset ranges
```

For automated missing-job detection and submission:
```bash
# Dry run: show what would be submitted
python scripts/run_missing_auto.py --dry-run

# Interactive: confirm before submitting
python scripts/run_missing_auto.py --interactive

# Filter by method, seed, or temperature
python scripts/run_missing_auto.py --method bon --seed 42
```

### Training PRMs (requires TRL)
```bash
pip install -e '.[trl]'
# See recipes/training/ for training scripts
```

## Architecture

### Core Module Structure (`src/sal/`)

- **config.py**: Central configuration dataclass. Controls approach (best_of_n/beam_search/dvts), model paths, PRMs, dataset settings, search hyperparameters, and multi-temperature parameters (`temperatures`, `temperature_ratios`).

- **models/reward_models.py**: PRM loading and inference. Supports RLHFlow, math-shepherd, and Skywork-o1 models. Handles tokenization and scoring of reasoning steps.

- **search/**: Three search algorithms:
  - `best_of_n.py`: Sample N completions, select best by PRM score
  - `beam_search.py`: Stepwise beam search guided by PRM
  - `diverse_verifier_tree_search.py`: DVTS — combines diversity and verification; includes token length validation to prevent `max_model_len` errors

- **utils/**:
  - `data.py`: Dataset loading/filtering from Hub
  - `math.py`: Math-specific utilities for step extraction and formatting
  - `parser.py`: YAML config parsing (H4ArgumentParser)
  - `score.py`: Scoring completions with PRMs
  - `temperature.py`: Multi-temperature scheduling and ratio distribution
  - `qwen_math_parser.py`: Answer extraction for MATH dataset evaluation
  - `ds_check.py`: Dataset validation/range checking
  - `hub.py`: Hugging Face Hub utilities

- **prm-toolkit/**: Git submodule providing PRM server infrastructure for inference.

### Search Algorithm Flow

1. `scripts/test_time_compute.py` loads config, LLM (via vLLM), and PRM
2. Dataset is processed through the selected search algorithm (`approach_fn`)
3. Search algorithm generates completions using the LLM (with optional multi-temperature cycling)
4. PRM scores reasoning steps to guide search
5. Results are scored and saved (locally or pushed to Hub as dataset branches)

### Config System

All experiments are driven by YAML configs in `recipes/`. Configs specify:
- `approach`: Which search algorithm to use
- `model_path`, `prm_path`: Model identifiers from Hub
- Search parameters: `n` (compute budget), `batch_size`, `temperature`, `temperatures`, `temperature_ratios`
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

- **Chat Templates**: Default configs use a custom Llama 3 chat template optimized for math reasoning. For non-Llama models, set `--custom_chat_template=none`.

- **GPU Memory**: The `gpu_memory_utilization` parameter (default 0.5) controls how much GPU memory vLLM uses. PRMs use remaining memory. Adjust for your hardware.

- **Dataset Branches**: When pushing to Hub, results are pushed as branches/revisions on the dataset repo (not main). Use `load_dataset(repo, revision=branch)` to load specific results.

- **Parallelization**: For large-scale experiments, use `recipes/launch_array_default.slurm` to shard dataset across multiple jobs, then merge with `scripts/merge_chunks.py`. The `run_missing_auto.py` script automates detection of missing ranges.

- **Evaluation**: Current evaluation uses an external fork of Qwen2.5-Math's parser (`sal/utils/qwen_math_parser.py`).
