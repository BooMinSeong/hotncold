<p align="center">
  <img style="width:200px" src="https://raw.githubusercontent.com/huggingface/search-and-learn/main/assets/logo.png">
</p>

<p align="center">
      🤗 <a href="https://huggingface.co/collections/HuggingFaceH4/scaling-test-time-compute-with-open-models-675c3b475a0d6eb4528fec23" target="_blank">Models & Datasets</a> |
      📃 <a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute" target="_blank">Blog Post</a>
</p>

# Search and Learn

Recipes to enhance LLM capabilities by scaling inference-time compute. Name inspired by Rich Sutton's [Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf):

> The two methods that seem to scale arbitrarily in this way are _**search**_ and _**learning**_.

## What is this?

Test-time compute scaling lets models "think longer" on harder problems without increasing pretraining cost. This repo implements search algorithms guided by Process Reward Models (PRMs) to solve complex math problems, extending the [original HuggingFace work](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) with multi-temperature sampling and large-scale experiment automation.

## Search Algorithms

Three algorithms are supported, all driven by YAML configs in [`recipes/`](./recipes/):

| Algorithm | Key idea |
| :--- | :--- |
| **Best-of-N** | Sample N completions, select the one with the highest PRM score |
| **Beam Search** | Stepwise tree expansion guided by PRM scores at each step |
| **DVTS** | Diverse Verifier Tree Search — balances exploration diversity with PRM verification |

Best-of-N and DVTS only need a single run at `n=256` (completions can be subsampled). Beam search requires a separate run for each value of `n`.

## Supported Models & PRMs

**Generator models** (recipe configs provided):

| Model | Recipes |
| :--- | :--- |
| `Qwen/Qwen2.5-3B-Instruct` | best_of_n, beam_search, dvts |
| `Qwen/Qwen2.5-1.5B-Instruct` | best_of_n, beam_search, dvts |
| `meta-llama/Llama-3.2-3B-Instruct` | best_of_n, beam_search, dvts |
| `meta-llama/Llama-3.2-1B-Instruct` | best_of_n, beam_search, dvts |
| `nvidia/AceMath-7B-Instruct` | best_of_n, beam_search, dvts |

Any model with a compatible chat template can be used via `--model_path`.

**Process Reward Models:**

- `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` (default)
- `peiyi9979/math-shepherd-mistral-7b-prm`
- `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B`
- `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`
- Custom PRMs trained with TRL (see [`recipes/training/`](recipes/training/))

## Installation

```shell
conda create -n sal python=3.11 && conda activate sal
pip install -e '.[dev]'
huggingface-cli login
```

## Quick Start

Run a search algorithm using a recipe config:

```shell
export CONFIG=recipes/Qwen2.5-1.5B-Instruct/best_of_n.yaml
uv run python scripts/test_time_compute.py $CONFIG
```

By default this runs Best-of-N with `n=4` over the first 10 problems of [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) and saves results to `data/`. To push results to the Hub as a dataset branch:

```shell
uv run python scripts/test_time_compute.py $CONFIG --push_to_hub=true
```

Override any config value from the command line:

```shell
uv run python scripts/test_time_compute.py $CONFIG \
    --model_path=meta-llama/Llama-3.2-8B-Instruct \
    --prm_path=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --dataset_name=AI-MO/aimo-validation-aime \
    --dataset_split=train \
    --n=64 \
    --seed=42
```

> **Note:** Default configs use a Llama 3 chat template optimized for math reasoning. For other model families, set `--custom_chat_template=none`.

## Multi-Temperature Sampling

All three algorithms support distributing completions across multiple temperatures to increase diversity. See [`TEST_GUIDE.md`](TEST_GUIDE.md) for full details.

```shell
# Best-of-N: split n completions proportionally across temperatures
uv run python scripts/test_time_compute.py $CONFIG \
    --temperatures "0.6,0.8,1.0" \
    --temperature_ratios "0.33,0.34,0.33" \
    --n 12

# Beam search / DVTS: each beam cycles through the temperature list
uv run python scripts/test_time_compute.py $CONFIG \
    --approach beam_search \
    --temperatures "0.6,0.8,1.0" \
    --beam_width 3 --n 12
```

## Large-Scale Experiments

Running full experiments (500 problems, `n=256`, multiple seeds) requires parallelization. The repo includes Slurm array job scripts and automation utilities.

### Parallelized generation

```shell
# Submit an array job (shards dataset across tasks)
sbatch recipes/launch_array_default.slurm recipes/Qwen2.5-3B-Instruct/best_of_n.yaml \
    --n=256 --seed=0 \
    --hub_dataset_id=<YOUR_ORG>/Qwen2.5-3B-best_of_n-completions

# Merge results after all tasks complete
python scripts/merge_chunks.py \
    --dataset_name=<YOUR_ORG>/Qwen2.5-3B-best_of_n-completions \
    --filter_strings seed-0
```

### Automation scripts

Convenience scripts for bulk experiment management:

```shell
./run_default.sh          # Submit all default experiment jobs
./run_hnc.sh              # Submit hot/cold temperature experiments
./merge_default.sh        # Merge all completed parallel results
python scripts/run_missing_auto.py --dry-run   # Find and submit missing ranges
```

## Training PRMs

Fine-tune your own PRM with TRL:

```shell
pip install -e '.[trl]'
# See recipes/training/ for model-specific training scripts
```

The [`training` README](recipes/training/README.md) covers training on custom data and evaluating on [ProcessBench](https://arxiv.org/abs/2412.06559).

## Project Structure

```
├── src/sal/               # Core library (config, search algorithms, PRM inference)
│   ├── config.py          # Central Config dataclass
│   ├── models/            # PRM loading and inference
│   ├── search/            # best_of_n, beam_search, dvts algorithms
│   └── utils/             # Data loading, scoring, temperature scheduling
├── scripts/               # Experiment entry points and automation
│   ├── test_time_compute.py   # Main experiment runner
│   ├── merge_chunks.py        # Merge parallel job results
│   └── run_missing_auto.py    # Automated missing-job detection/submission
├── recipes/               # YAML configs per model/algorithm + Slurm launchers
├── prm-toolkit/           # PRM server infrastructure (git submodule)
└── TEST_GUIDE.md          # Multi-temperature testing guide
```

## Citation

```bibtex
@misc{beeching2024scalingtesttimecompute,
      title={Scaling test-time compute with open models},
      author={Edward Beeching and Lewis Tunstall and Sasha Rush},
      url={https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute},
}
```

```bibtex
@misc{snell2024scalingllmtesttimecompute,
      title={Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters},
      author={Charlie Snell and Jaehoon Lee and Kelvin Xu and Aviral Kumar},
      year={2024},
      eprint={2408.03314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.03314},
}
```
