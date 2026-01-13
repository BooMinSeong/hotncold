"""
Analyze AIME25 experimental results across multiple seeds.
Computes metrics for different approaches (bon, beam_search, dvts) with
default temperature strategy (T=0.4), averaging across seeds 0, 42, 64.
"""

from math_verify import parse, verify
from datasets import load_dataset, Dataset, get_dataset_config_names
from sal.utils.math import extract_completion_answers
from sal.utils.score import compute_pass_at_k
import re
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime


def evaluate_result(data, key="answer"):
    """Evaluate a single prediction against gold answer."""
    gold_answer = data["answer"]
    gold_answer = parse("\\boxed{" + gold_answer + "}")

    prediction = data[key]
    prediction = parse(prediction)

    return verify(gold_answer, prediction)


def score_pass_at_k(dataset: Dataset, n: int = 256) -> Dataset:
    """Compute pass@k metrics for different k values."""
    dataset = dataset.map(
        extract_completion_answers,
        fn_kwargs={"n": None},
        num_proc=4,
        desc=f"Extract answers for Pass@k",
    )

    subsets = [2**i for i in range(n) if 2**i <= n]
    for k in subsets:
        dataset = dataset.map(
            compute_pass_at_k,
            fn_kwargs={"k": k},
            num_proc=4,
            desc=f"Compute Pass@{k}",
        )
    return dataset


def analyze_single_dataset(dataset, dataset_name, seed):
    """Analyze a single dataset and return results."""
    results_by_method = {
        'naive': {},
        'weighted': {},
        'maj': {}
    }

    if 'train' in dataset:
        dataset = dataset['train']

    # Find all prediction keys
    pred_keys = [key for key in dataset.features.keys() if key.startswith('pred_')]
    print(f"  Found {len(pred_keys)} prediction keys to evaluate")

    # Accumulate results for each key
    results_accumulator = {key: [] for key in pred_keys}

    # Evaluate all predictions
    print(f"  Evaluating predictions...")
    for data in tqdm(dataset, desc=f"  {dataset_name} (seed {seed})", leave=False):
        for key in pred_keys:
            result = evaluate_result(data, key)
            results_accumulator[key].append(result)

    # Calculate accuracy for each key
    for key in pred_keys:
        results = results_accumulator[key]
        accuracy = sum(results) / len(results)

        # Parse key: pred_method@number format
        match = re.match(r'pred_(naive|weighted|maj)@(\d+)', key)
        if match:
            method = match.group(1)
            n_samples = int(match.group(2))
            results_by_method[method][n_samples] = accuracy

    return results_by_method


def analyze_pass_at_k(dataset, dataset_name, seed):
    """
    Extract pass@k metrics from dataset.

    The dataset should already contain pass@{k} fields computed during generation.
    This function aggregates them across problems.

    Args:
        dataset: HuggingFace dataset with pass@k fields
        dataset_name: Name of dataset (for logging)
        seed: Seed value (for logging)

    Returns:
        Dictionary mapping k values to mean pass@k probabilities:
        {1: 0.45, 2: 0.62, 4: 0.75, ...}
    """
    if 'train' in dataset:
        dataset = dataset['train']

    # Find all pass@k fields
    pass_k_fields = {}
    for key in dataset.features.keys():
        if key.startswith('pass@'):
            # Extract k value from field name (e.g., 'pass@1' -> 1)
            try:
                k = int(key.split('@')[1])
                pass_k_fields[k] = key
            except (IndexError, ValueError):
                continue

    if not pass_k_fields:
        print(f"  Warning: No pass@k fields found in {dataset_name} (seed {seed})")
        return {}

    print(f"  Found {len(pass_k_fields)} pass@k fields: {sorted(pass_k_fields.keys())}")

    # Aggregate pass@k values across problems
    results = {}
    for k, field_name in pass_k_fields.items():
        pass_k_values = [row[field_name] for row in dataset if field_name in row]
        if pass_k_values:
            results[k] = sum(pass_k_values) / len(pass_k_values)

    return results


def plot_temperature_comparison(all_results, model_size, approach, method, output_dir, seeds):
    """
    Compare T=0.4 vs T=0.8 for same model size, approach and method.
    Lines with error bands. T=0.8 shown as dashed baseline.
    """
    temperatures = [f'{model_size}-T0.4', f'{model_size}-T0.8']

    # Set style
    sns.set_style("whitegrid")

    # Collect all n values
    all_n_values = set()
    for temp in temperatures:
        dataset_name = f"{temp}-{approach}"
        if dataset_name in all_results:
            for seed in seeds:
                if seed in all_results[dataset_name]:
                    all_n_values.update(all_results[dataset_name][seed][method].keys())

    if not all_n_values:
        return

    n_values = sorted(all_n_values)

    # Prepare data
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {f'{model_size}-T0.4': '#2ca02c', f'{model_size}-T0.8': '#ff7f0e'}

    for temp in temperatures:
        dataset_name = f"{temp}-{approach}"

        if dataset_name not in all_results:
            continue

        means = []
        stds = []
        valid_n = []

        for n in n_values:
            values = []
            for seed in seeds:
                if seed in all_results[dataset_name]:
                    if n in all_results[dataset_name][seed][method]:
                        values.append(all_results[dataset_name][seed][method][n])

            if values:
                valid_n.append(n)
                means.append(np.mean(values))
                stds.append(np.std(values))

        if valid_n:
            means = np.array(means)
            stds = np.array(stds)

            # T=0.4: solid line with markers, T=0.8: dashed line (baseline)
            if temp == f'{model_size}-T0.4':
                ax.plot(valid_n, means, 'o-', label=f'T=0.4',
                       color=colors[temp], linewidth=2, markersize=8)
            else:
                ax.plot(valid_n, means, '--', label=f'T=0.8 (baseline)',
                       color=colors[temp], linewidth=2)

            ax.fill_between(valid_n, means - stds, means + stds,
                           alpha=0.2, color=colors[temp])

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title(f'AIME25 - {model_size} - {approach.replace("_", " ").title()} - {method.upper()}\nTemperature Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'aime25-{model_size}-{approach}-{method}-temp_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_curves(all_results, model_size, approach, output_dir, seeds):
    """
    Scaling curve showing all methods for an approach, comparing T=0.4 vs T=0.8.
    X-axis in log scale. T=0.8 shown as dashed baseline.
    """
    temperatures = [f'{model_size}-T0.4', f'{model_size}-T0.8']
    methods = ['naive', 'weighted', 'maj']

    # Set style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color mapping
    colors = {
        f'{model_size}-T0.4': {'naive': '#2ca02c', 'weighted': '#98df8a', 'maj': '#d4f1d4'},
        f'{model_size}-T0.8': {'naive': '#ff7f0e', 'weighted': '#ffbb78', 'maj': '#ffd9b3'}
    }

    for temp in temperatures:
        dataset_name = f"{temp}-{approach}"

        if dataset_name not in all_results:
            continue

        for method in methods:
            # Collect all n values
            all_n_values = set()
            for seed in seeds:
                if seed in all_results[dataset_name]:
                    all_n_values.update(all_results[dataset_name][seed][method].keys())

            if not all_n_values:
                continue

            n_values = sorted(all_n_values)
            means = []
            stds = []

            for n in n_values:
                values = []
                for seed in seeds:
                    if seed in all_results[dataset_name]:
                        if n in all_results[dataset_name][seed][method]:
                            values.append(all_results[dataset_name][seed][method][n])

                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))

            if means:
                # T=0.4: solid line with markers, T=0.8: dashed line
                temp_label = 'T=0.4' if temp == f'{model_size}-T0.4' else 'T=0.8 (baseline)'
                linestyle = 'o-' if temp == f'{model_size}-T0.4' else '--'
                label = f'{temp_label}-{method}'
                ax.plot(n_values, means, linestyle, label=label,
                       color=colors[temp][method], linewidth=2, markersize=6 if temp == f'{model_size}-T0.4' else 0)

                means = np.array(means)
                stds = np.array(stds)
                ax.fill_between(n_values, means - stds, means + stds,
                               alpha=0.15, color=colors[temp][method])

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title(f'AIME25 - {model_size} - {approach.replace("_", " ").title()}\nScaling Curves (T=0.4 vs T=0.8)',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'aime25-{model_size}-{approach}-scaling_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pass_at_k_curves(all_results, model_size, approach, output_dir, seeds):
    """
    Plot pass@k scaling curves comparing T=0.4 vs T=0.8.

    X-axis: k values (log scale)
    Y-axis: Pass@k probability [0, 1.0]
    Lines: T=0.4 vs T=0.8
    Error bands: ± 1 std across seeds
    """
    temperatures = [f'{model_size}-T0.4', f'{model_size}-T0.8']

    # Set style
    sns.set_style("whitegrid")

    # Collect all k values
    all_k_values = set()
    for temp in temperatures:
        dataset_name = f"{temp}-{approach}"
        if dataset_name in all_results:
            for seed in seeds:
                if seed in all_results[dataset_name] and 'pass@k' in all_results[dataset_name][seed]:
                    all_k_values.update(all_results[dataset_name][seed]['pass@k'].keys())

    if not all_k_values:
        print(f"  No pass@k data found for {model_size}-{approach}")
        return

    k_values = sorted(all_k_values)

    # Prepare data
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {f'{model_size}-T0.4': '#2ca02c', f'{model_size}-T0.8': '#ff7f0e'}

    for temp in temperatures:
        dataset_name = f"{temp}-{approach}"

        if dataset_name not in all_results:
            continue

        means = []
        stds = []
        valid_k = []

        for k in k_values:
            values = []
            for seed in seeds:
                if seed in all_results[dataset_name]:
                    if 'pass@k' in all_results[dataset_name][seed]:
                        if k in all_results[dataset_name][seed]['pass@k']:
                            values.append(all_results[dataset_name][seed]['pass@k'][k])

            if values:
                valid_k.append(k)
                means.append(np.mean(values))
                stds.append(np.std(values))

        if valid_k:
            means = np.array(means)
            stds = np.array(stds)

            # T=0.4: solid line with markers, T=0.8: dashed line (baseline)
            if temp == f'{model_size}-T0.4':
                ax.plot(valid_k, means, 'o-', label=f'T=0.4',
                       color=colors[temp], linewidth=2, markersize=8)
            else:
                ax.plot(valid_k, means, '--', label=f'T=0.8 (baseline)',
                       color=colors[temp], linewidth=2)

            ax.fill_between(valid_k, means - stds, means + stds,
                           alpha=0.2, color=colors[temp])

    ax.set_xlabel('k (number of samples)', fontsize=12)
    ax.set_ylabel('Pass@k', fontsize=12)
    ax.set_title(f'AIME25 - {model_size} - {approach.replace("_", " ").title()}\nPass@k Curves (T=0.4 vs T=0.8)',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'aime25-{model_size}-{approach}-pass_at_k_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pass_at_k_comparison(all_results, model_size, approach, output_dir, seeds):
    """
    Bar chart comparing pass@k at specific k values (T=0.4 vs T=0.8).

    X-axis: Selected k values [1, 8, 32, 64]
    Y-axis: Pass@k probability
    Bars: T=0.4 vs T=0.8 side-by-side for each k
    """
    temperatures = [f'{model_size}-T0.4', f'{model_size}-T0.8']
    selected_k = [1, 8, 32, 64]  # Representative k values

    # Set style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {f'{model_size}-T0.4': '#2ca02c', f'{model_size}-T0.8': '#ff7f0e'}
    bar_width = 0.35
    x_positions = np.arange(len(selected_k))

    for idx, temp in enumerate(temperatures):
        dataset_name = f"{temp}-{approach}"

        if dataset_name not in all_results:
            continue

        means = []
        stds = []
        valid_k = []

        for k in selected_k:
            values = []
            for seed in seeds:
                if seed in all_results[dataset_name]:
                    if 'pass@k' in all_results[dataset_name][seed]:
                        if k in all_results[dataset_name][seed]['pass@k']:
                            values.append(all_results[dataset_name][seed]['pass@k'][k])

            if values:
                valid_k.append(k)
                means.append(np.mean(values))
                stds.append(np.std(values))
            else:
                valid_k.append(k)
                means.append(0)
                stds.append(0)

        if means:
            # Position bars: T=0.4 left, T=0.8 right
            offset = -bar_width/2 if idx == 0 else bar_width/2
            label = 'T=0.4' if temp == f'{model_size}-T0.4' else 'T=0.8'
            ax.bar(x_positions + offset, means, bar_width,
                   yerr=stds, label=label, color=colors[temp],
                   capsize=5, alpha=0.8)

    ax.set_xlabel('k (number of samples)', fontsize=12)
    ax.set_ylabel('Pass@k', fontsize=12)
    ax.set_title(f'AIME25 - {model_size} - {approach.replace("_", " ").title()}\nPass@k Comparison (T=0.4 vs T=0.8)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(selected_k)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'aime25-{model_size}-{approach}-pass_at_k_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(all_results, approach, method, temperature, output_dir, seeds):
    """
    Compare 1.5B vs 3B model for same approach, method, and temperature.
    Lines with error bands showing model scaling.
    """
    model_sizes = ['1.5B', '3B']

    # Set style
    sns.set_style("whitegrid")

    # Collect all n values
    all_n_values = set()
    for model_size in model_sizes:
        dataset_name = f"{model_size}-{temperature}-{approach}"
        if dataset_name in all_results:
            for seed in seeds:
                if seed in all_results[dataset_name]:
                    all_n_values.update(all_results[dataset_name][seed][method].keys())

    if not all_n_values:
        return

    n_values = sorted(all_n_values)

    # Prepare data
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'1.5B': '#1f77b4', '3B': '#ff7f0e'}

    for model_size in model_sizes:
        dataset_name = f"{model_size}-{temperature}-{approach}"

        if dataset_name not in all_results:
            continue

        means = []
        stds = []
        valid_n = []

        for n in n_values:
            values = []
            for seed in seeds:
                if seed in all_results[dataset_name]:
                    if n in all_results[dataset_name][seed][method]:
                        values.append(all_results[dataset_name][seed][method][n])

            if values:
                valid_n.append(n)
                means.append(np.mean(values))
                stds.append(np.std(values))

        if valid_n:
            means = np.array(means)
            stds = np.array(stds)

            ax.plot(valid_n, means, 'o-', label=f'{model_size}',
                   color=colors[model_size], linewidth=2, markersize=8)

            ax.fill_between(valid_n, means - stds, means + stds,
                           alpha=0.2, color=colors[model_size])

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title(f'AIME25 - {approach.replace("_", " ").title()} - {method.upper()}\nModel Size Comparison ({temperature.upper()})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'aime25-{approach}-{method}-{temperature}-model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_scaling_curves(all_results, approach, temperature, output_dir, seeds):
    """
    Scaling curves comparing 1.5B vs 3B models for all methods.
    X-axis in log scale.
    """
    model_sizes = ['1.5B', '3B']
    methods = ['naive', 'weighted', 'maj']

    # Set style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color mapping
    colors = {
        '1.5B': {'naive': '#1f77b4', 'weighted': '#aec7e8', 'maj': '#c6dbef'},
        '3B': {'naive': '#ff7f0e', 'weighted': '#ffbb78', 'maj': '#ffd9b3'}
    }

    for model_size in model_sizes:
        dataset_name = f"{model_size}-{temperature}-{approach}"

        if dataset_name not in all_results:
            continue

        for method in methods:
            # Collect all n values
            all_n_values = set()
            for seed in seeds:
                if seed in all_results[dataset_name]:
                    all_n_values.update(all_results[dataset_name][seed][method].keys())

            if not all_n_values:
                continue

            n_values = sorted(all_n_values)
            means = []
            stds = []

            for n in n_values:
                values = []
                for seed in seeds:
                    if seed in all_results[dataset_name]:
                        if n in all_results[dataset_name][seed][method]:
                            values.append(all_results[dataset_name][seed][method][n])

                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))

            if means:
                label = f'{model_size}-{method}'
                ax.plot(n_values, means, 'o-', label=label,
                       color=colors[model_size][method], linewidth=2, markersize=6)

                means = np.array(means)
                stds = np.array(stds)
                ax.fill_between(n_values, means - stds, means + stds,
                               alpha=0.15, color=colors[model_size][method])

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title(f'AIME25 - {approach.replace("_", " ").title()}\nModel Size Comparison ({temperature.upper()})',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'aime25-{approach}-{temperature}-model_scaling_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_pass_at_k_comparison(all_results, approach, temperature, output_dir, seeds):
    """
    Compare pass@k curves for 1.5B vs 3B models.

    X-axis: k values (log scale)
    Y-axis: Pass@k probability
    Lines: 1.5B vs 3B
    """
    model_sizes = ['1.5B', '3B']

    # Set style
    sns.set_style("whitegrid")

    # Collect all k values
    all_k_values = set()
    for model_size in model_sizes:
        dataset_name = f"{model_size}-{temperature}-{approach}"
        if dataset_name in all_results:
            for seed in seeds:
                if seed in all_results[dataset_name] and 'pass@k' in all_results[dataset_name][seed]:
                    all_k_values.update(all_results[dataset_name][seed]['pass@k'].keys())

    if not all_k_values:
        print(f"  No pass@k data found for {approach} at {temperature}")
        return

    k_values = sorted(all_k_values)

    # Prepare data
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'1.5B': '#1f77b4', '3B': '#ff7f0e'}

    for model_size in model_sizes:
        dataset_name = f"{model_size}-{temperature}-{approach}"

        if dataset_name not in all_results:
            continue

        means = []
        stds = []
        valid_k = []

        for k in k_values:
            values = []
            for seed in seeds:
                if seed in all_results[dataset_name]:
                    if 'pass@k' in all_results[dataset_name][seed]:
                        if k in all_results[dataset_name][seed]['pass@k']:
                            values.append(all_results[dataset_name][seed]['pass@k'][k])

            if values:
                valid_k.append(k)
                means.append(np.mean(values))
                stds.append(np.std(values))

        if valid_k:
            means = np.array(means)
            stds = np.array(stds)

            ax.plot(valid_k, means, 'o-', label=f'{model_size}',
                   color=colors[model_size], linewidth=2, markersize=8)

            ax.fill_between(valid_k, means - stds, means + stds,
                           alpha=0.2, color=colors[model_size])

    ax.set_xlabel('k (number of samples)', fontsize=12)
    ax.set_ylabel('Pass@k', fontsize=12)
    ax.set_title(f'AIME25 - {approach.replace("_", " ").title()}\nModel Size Pass@k Comparison ({temperature.upper()})',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'aime25-{approach}-{temperature}-model_pass_at_k_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Dataset configurations
    datasets_config = {
        # 1.5B model configurations
        '1.5B-T0.4-bon': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-bon',
            'subset_template': 'math-ai_aime25--T-0.4--top_p-1.0--n-64--seed-{seed}--agg_strategy-last',
            'model_size': '1.5B'
        },
        '1.5B-T0.4-beam_search': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-beam_search',
            'subset_template': 'math-ai_aime25--T-0.4--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'model_size': '1.5B'
        },
        '1.5B-T0.4-dvts': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-dvts',
            'subset_template': 'math-ai_aime25--T-0.4--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'model_size': '1.5B'
        },
        '1.5B-T0.8-bon': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-bon',
            'subset_template': 'math-ai_aime25--T-0.8--top_p-1.0--n-64--seed-{seed}--agg_strategy-last',
            'model_size': '1.5B'
        },
        '1.5B-T0.8-beam_search': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-beam_search',
            'subset_template': 'math-ai_aime25--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'model_size': '1.5B'
        },
        '1.5B-T0.8-dvts': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-dvts',
            'subset_template': 'math-ai_aime25--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'model_size': '1.5B'
        },
        # 3B model configurations
        '3B-T0.4-bon': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-3B-Instruct-bon',
            'subset_template': 'math-ai_aime25--T-0.4--top_p-1.0--n-64--seed-{seed}--agg_strategy-last',
            'model_size': '3B'
        },
        '3B-T0.4-beam_search': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-3B-Instruct-beam_search',
            'subset_template': 'math-ai_aime25--T-0.4--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'model_size': '3B'
        },
        '3B-T0.4-dvts': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-3B-Instruct-dvts',
            'subset_template': 'math-ai_aime25--T-0.4--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'model_size': '3B'
        },
        '3B-T0.8-bon': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-3B-Instruct-bon',
            'subset_template': 'math-ai_aime25--T-0.8--top_p-1.0--n-64--seed-{seed}--agg_strategy-last',
            'model_size': '3B'
        },
        '3B-T0.8-beam_search': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-3B-Instruct-beam_search',
            'subset_template': 'math-ai_aime25--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'model_size': '3B'
        },
        '3B-T0.8-dvts': {
            'path': 'ENSEONG/default-aime25-Qwen2.5-3B-Instruct-dvts',
            'subset_template': 'math-ai_aime25--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'model_size': '3B'
        }
    }

    # Seed configurations
    seeds = [0, 42, 64]

    # Store all results: dataset_name -> seed -> method -> n_samples -> accuracy
    all_results = defaultdict(lambda: defaultdict(dict))

    # Process each dataset
    for dataset_name, config in datasets_config.items():
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}")

        for seed in seeds:
            print(f"\nLoading seed {seed}...")

            try:
                # Load dataset
                subset_name = config['subset_template'].format(seed=seed)
                print(f"  Using subset: {subset_name}")
                dataset = load_dataset(config['path'], subset_name)

                # Analyze pred_* methods
                results = analyze_single_dataset(dataset, dataset_name, seed)
                all_results[dataset_name][seed] = results

                # Analyze pass@k
                pass_at_k_results = analyze_pass_at_k(dataset, dataset_name, seed)
                all_results[dataset_name][seed]['pass@k'] = pass_at_k_results

            except Exception as e:
                print(f"  Error processing {dataset_name} seed {seed}: {e}")
                continue

    # Generate visualizations
    print("\n\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Create output directory
    seed_str = '-'.join(map(str, seeds))
    output_dir = f"exp/aime25_results_analysis_seeds_{seed_str}_T0.4_vs_T0.8_with_model_comparison"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    approaches = ['bon', 'beam_search', 'dvts']
    model_sizes = ['1.5B', '3B']
    temperatures = ['T0.4', 'T0.8']

    # Generate temperature comparison plots for each model size
    print("\n  Generating temperature comparison plots...")
    for model_size in model_sizes:
        for approach in approaches:
            print(f"\n    {model_size} - {approach}...")

            # Scaling curves for each approach and model size
            plot_scaling_curves(all_results, model_size, approach, output_dir, seeds)
            print(f"      - Scaling curves: aime25-{model_size}-{approach}-scaling_curves.png")

            # Temperature comparison for each method
            for method in ['naive', 'weighted', 'maj']:
                plot_temperature_comparison(all_results, model_size, approach, method, output_dir, seeds)
                print(f"      - Temperature comparison ({method}): aime25-{model_size}-{approach}-{method}-temp_comparison.png")

            # Pass@k visualizations
            plot_pass_at_k_curves(all_results, model_size, approach, output_dir, seeds)
            print(f"      - Pass@k curves: aime25-{model_size}-{approach}-pass_at_k_curves.png")

            plot_pass_at_k_comparison(all_results, model_size, approach, output_dir, seeds)
            print(f"      - Pass@k comparison: aime25-{model_size}-{approach}-pass_at_k_comparison.png")

    # Generate model size comparison plots
    print("\n  Generating model size comparison plots...")
    for approach in approaches:
        for temperature in temperatures:
            print(f"\n    {approach} - {temperature}...")

            # Model scaling curves
            plot_model_scaling_curves(all_results, approach, temperature, output_dir, seeds)
            print(f"      - Model scaling: aime25-{approach}-{temperature}-model_scaling_comparison.png")

            # Model comparison for each method
            for method in ['naive', 'weighted', 'maj']:
                plot_model_comparison(all_results, approach, method, temperature, output_dir, seeds)
                print(f"      - Model comparison ({method}): aime25-{approach}-{method}-{temperature}-model_comparison.png")

            # Model pass@k comparison
            plot_model_pass_at_k_comparison(all_results, approach, temperature, output_dir, seeds)
            print(f"      - Model pass@k: aime25-{approach}-{temperature}-model_pass_at_k_comparison.png")

    print(f"\nAll visualizations saved to {output_dir}/")
    print("="*80)

    # Generate markdown report
    print("\n\n" + "="*80)
    print("GENERATING ANALYSIS REPORT")
    print("="*80)

    # Build markdown report
    md_lines = ["# AIME25 Analysis Report\n"]

    # Metadata section
    md_lines.append("## Metadata\n")
    md_lines.append(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append(f"- **Dataset**: AIME25\n")
    md_lines.append(f"- **Model Sizes**: 1.5B, 3B\n")
    md_lines.append(f"- **Temperatures**: 0.4 vs 0.8\n")
    md_lines.append(f"- **Seeds**: {seeds}\n")
    md_lines.append(f"- **Approaches**: bon, beam_search, dvts\n")
    md_lines.append("\n")

    # Temperature comparison section
    md_lines.append("## Temperature Comparison (T=0.4 vs T=0.8)\n")
    md_lines.append("\n")

    approaches = ['bon', 'beam_search', 'dvts']
    methods = ['naive', 'weighted', 'maj']
    model_sizes_report = ['1.5B', '3B']
    temperatures_report = ['T0.4', 'T0.8']

    for model_size in model_sizes_report:
        md_lines.append(f"### MODEL: {model_size}\n")
        md_lines.append("\n")

        for approach in approaches:
            md_lines.append(f"#### Approach: {approach.upper()}\n")
            md_lines.append("\n")

            for method in methods:
                md_lines.append(f"##### Method: {method.upper()}\n")
                md_lines.append("\n")

                # Collect all n values for this approach and method
                all_n_values = set()
                for temp in temperatures_report:
                    dataset_name = f"{model_size}-{temp}-{approach}"
                    if dataset_name in all_results:
                        for seed in seeds:
                            if seed in all_results[dataset_name]:
                                all_n_values.update(all_results[dataset_name][seed][method].keys())

                if not all_n_values:
                    md_lines.append(f"*No data available for {method}*\n")
                    md_lines.append("\n")
                    continue

                # Table header
                md_lines.append("| n | T=0.4 Acc | T=0.8 Acc | T=0.4 vs T=0.8 |\n")
                md_lines.append("|---|-----------|-----------|----------------|\n")

                for n in sorted(all_n_values):
                    t04_values = []
                    t08_values = []

                    # Collect T=0.4 values
                    t04_dataset = f"{model_size}-T0.4-{approach}"
                    if t04_dataset in all_results:
                        for seed in seeds:
                            if seed in all_results[t04_dataset]:
                                if n in all_results[t04_dataset][seed][method]:
                                    t04_values.append(all_results[t04_dataset][seed][method][n])

                    # Collect T=0.8 values
                    t08_dataset = f"{model_size}-T0.8-{approach}"
                    if t08_dataset in all_results:
                        for seed in seeds:
                            if seed in all_results[t08_dataset]:
                                if n in all_results[t08_dataset][seed][method]:
                                    t08_values.append(all_results[t08_dataset][seed][method][n])

                    if t04_values or t08_values:
                        t04_mean = np.mean(t04_values) if t04_values else 0
                        t08_mean = np.mean(t08_values) if t08_values else 0

                        diff = t04_mean - t08_mean
                        improvement = (diff / t08_mean * 100) if t08_mean > 0 else 0

                        t04_str = f"{t04_mean:.4f}" if t04_values else "N/A"
                        t08_str = f"{t08_mean:.4f}" if t08_values else "N/A"
                        diff_str = f"{diff:+.4f} ({improvement:+.2f}%)" if (t04_values and t08_values) else "N/A"

                        md_lines.append(f"| {n} | {t04_str} | {t08_str} | {diff_str} |\n")

                md_lines.append("\n")

    # Pass@k Analysis section
    md_lines.append("## Pass@k Analysis (Model Upper Bound)\n")
    md_lines.append("\n")
    md_lines.append("Pass@k measures the probability that at least one of k samples is correct. ")
    md_lines.append("This represents the theoretical upper bound of model performance.\n")
    md_lines.append("\n")

    for model_size in model_sizes_report:
        md_lines.append(f"### MODEL: {model_size}\n")
        md_lines.append("\n")

        for approach in approaches:
            md_lines.append(f"#### Approach: {approach.upper()}\n")
            md_lines.append("\n")

            # Collect all k values for this approach
            all_k_values = set()
            for temp in temperatures_report:
                dataset_name = f"{model_size}-{temp}-{approach}"
                if dataset_name in all_results:
                    for seed in seeds:
                        if seed in all_results[dataset_name]:
                            if 'pass@k' in all_results[dataset_name][seed]:
                                all_k_values.update(all_results[dataset_name][seed]['pass@k'].keys())

            if not all_k_values:
                md_lines.append(f"*No pass@k data available for {approach}*\n")
                md_lines.append("\n")
                continue

            # Table header
            md_lines.append("| k | T=0.4 Pass@k | T=0.8 Pass@k | T=0.4 vs T=0.8 |\n")
            md_lines.append("|---|--------------|--------------|----------------|\n")

            for k in sorted(all_k_values):
                t04_values = []
                t08_values = []

                # Collect T=0.4 values
                t04_dataset = f"{model_size}-T0.4-{approach}"
                if t04_dataset in all_results:
                    for seed in seeds:
                        if seed in all_results[t04_dataset]:
                            if 'pass@k' in all_results[t04_dataset][seed]:
                                if k in all_results[t04_dataset][seed]['pass@k']:
                                    t04_values.append(all_results[t04_dataset][seed]['pass@k'][k])

                # Collect T=0.8 values
                t08_dataset = f"{model_size}-T0.8-{approach}"
                if t08_dataset in all_results:
                    for seed in seeds:
                        if seed in all_results[t08_dataset]:
                            if 'pass@k' in all_results[t08_dataset][seed]:
                                if k in all_results[t08_dataset][seed]['pass@k']:
                                    t08_values.append(all_results[t08_dataset][seed]['pass@k'][k])

                if t04_values or t08_values:
                    t04_mean = np.mean(t04_values) if t04_values else 0
                    t08_mean = np.mean(t08_values) if t08_values else 0

                    diff = t04_mean - t08_mean
                    improvement = (diff / t08_mean * 100) if t08_mean > 0 else 0

                    t04_str = f"{t04_mean:.4f}" if t04_values else "N/A"
                    t08_str = f"{t08_mean:.4f}" if t08_values else "N/A"
                    diff_str = f"{diff:+.4f} ({improvement:+.2f}%)" if (t04_values and t08_values) else "N/A"

                    md_lines.append(f"| {k} | {t04_str} | {t08_str} | {diff_str} |\n")

            md_lines.append("\n")

    # Model Size Comparison section
    md_lines.append("## Model Size Comparison (1.5B vs 3B)\n")
    md_lines.append("\n")

    for temp in temperatures_report:
        md_lines.append(f"### TEMPERATURE: {temp.upper()}\n")
        md_lines.append("\n")

        for approach in approaches:
            md_lines.append(f"#### Approach: {approach.upper()}\n")
            md_lines.append("\n")

            for method in methods:
                md_lines.append(f"##### Method: {method.upper()}\n")
                md_lines.append("\n")

                # Collect all n values for this approach and method
                all_n_values = set()
                for model_size in model_sizes_report:
                    dataset_name = f"{model_size}-{temp}-{approach}"
                    if dataset_name in all_results:
                        for seed in seeds:
                            if seed in all_results[dataset_name]:
                                all_n_values.update(all_results[dataset_name][seed][method].keys())

                if not all_n_values:
                    md_lines.append(f"*No data available for {method}*\n")
                    md_lines.append("\n")
                    continue

                # Table header
                md_lines.append("| n | 1.5B Acc | 3B Acc | 3B vs 1.5B |\n")
                md_lines.append("|---|----------|--------|------------|\n")

                for n in sorted(all_n_values):
                    b15_values = []
                    b3_values = []

                    # Collect 1.5B values
                    b15_dataset = f"1.5B-{temp}-{approach}"
                    if b15_dataset in all_results:
                        for seed in seeds:
                            if seed in all_results[b15_dataset]:
                                if n in all_results[b15_dataset][seed][method]:
                                    b15_values.append(all_results[b15_dataset][seed][method][n])

                    # Collect 3B values
                    b3_dataset = f"3B-{temp}-{approach}"
                    if b3_dataset in all_results:
                        for seed in seeds:
                            if seed in all_results[b3_dataset]:
                                if n in all_results[b3_dataset][seed][method]:
                                    b3_values.append(all_results[b3_dataset][seed][method][n])

                    if b15_values or b3_values:
                        b15_mean = np.mean(b15_values) if b15_values else 0
                        b3_mean = np.mean(b3_values) if b3_values else 0

                        diff = b3_mean - b15_mean
                        improvement = (diff / b15_mean * 100) if b15_mean > 0 else 0

                        b15_str = f"{b15_mean:.4f}" if b15_values else "N/A"
                        b3_str = f"{b3_mean:.4f}" if b3_values else "N/A"
                        diff_str = f"{diff:+.4f} ({improvement:+.2f}%)" if (b15_values and b3_values) else "N/A"

                        md_lines.append(f"| {n} | {b15_str} | {b3_str} | {diff_str} |\n")

                md_lines.append("\n")

    # Save markdown report
    report_path = os.path.join(output_dir, 'analysis_report.md')
    with open(report_path, 'w') as f:
        f.writelines(md_lines)

    print(f"Analysis report saved to: {report_path}")
    print("="*80)


if __name__ == "__main__":
    main()
