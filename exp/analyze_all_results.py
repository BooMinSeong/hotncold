"""
Analyze experimental results across multiple datasets and seeds.
Computes metrics for different approaches (bon, beam_search, dvts) with
hnc and default temperature strategies, averaging across seeds 0, 42, 64.
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


def plot_configuration_comparison(all_results, approach, method, output_dir, hnc_seeds):
    """
    Compare hnc vs default for same method.
    Lines with error bands. Default shown as dashed baseline.
    """
    strategies = ['hnc', 'default']
    default_seeds = [0, 42, 64]

    # Set style
    sns.set_style("whitegrid")

    # Collect all n values
    all_n_values = set()
    for strategy in strategies:
        dataset_name = f"{strategy}-{approach}"
        strategy_seeds = hnc_seeds if strategy == 'hnc' else default_seeds
        if dataset_name in all_results:
            for seed in strategy_seeds:
                if seed in all_results[dataset_name]:
                    all_n_values.update(all_results[dataset_name][seed][method].keys())

    if not all_n_values:
        return

    n_values = sorted(all_n_values)

    # Prepare data
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'hnc': '#1f77b4', 'default': '#ff7f0e'}

    for strategy in strategies:
        dataset_name = f"{strategy}-{approach}"
        strategy_seeds = hnc_seeds if strategy == 'hnc' else default_seeds

        if dataset_name not in all_results:
            continue

        means = []
        stds = []
        valid_n = []

        for n in n_values:
            values = []
            for seed in strategy_seeds:
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

            # hnc: solid line with markers, default: dashed line without markers
            if strategy == 'hnc':
                ax.plot(valid_n, means, 'o-', label=f'{strategy}',
                       color=colors[strategy], linewidth=2, markersize=8)
            else:
                ax.plot(valid_n, means, '--', label=f'{strategy} (baseline)',
                       color=colors[strategy], linewidth=2)

            ax.fill_between(valid_n, means - stds, means + stds,
                           alpha=0.2, color=colors[strategy])

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title(f'{approach.replace("_", " ").title()} - {method.upper()}\nHNC vs Default Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{approach}-{method}-hnc_vs_default.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scaling_curves(all_results, approach, output_dir, hnc_seeds):
    """
    Scaling curve showing all methods for both strategies.
    X-axis in log scale. Default shown as dashed baseline.
    """
    strategies = ['hnc', 'default']
    methods = ['naive', 'weighted', 'maj']
    default_seeds = [0, 42, 64]

    # Set style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color mapping
    colors = {
        'hnc': {'naive': '#1f77b4', 'weighted': '#aec7e8', 'maj': '#3182bd'},
        'default': {'naive': '#ff7f0e', 'weighted': '#ffbb78', 'maj': '#d62728'}
    }

    for strategy in strategies:
        dataset_name = f"{strategy}-{approach}"
        strategy_seeds = hnc_seeds if strategy == 'hnc' else default_seeds

        if dataset_name not in all_results:
            continue

        for method in methods:
            # Collect all n values
            all_n_values = set()
            for seed in strategy_seeds:
                if seed in all_results[dataset_name]:
                    all_n_values.update(all_results[dataset_name][seed][method].keys())

            if not all_n_values:
                continue

            n_values = sorted(all_n_values)
            means = []
            stds = []

            for n in n_values:
                values = []
                for seed in strategy_seeds:
                    if seed in all_results[dataset_name]:
                        if n in all_results[dataset_name][seed][method]:
                            values.append(all_results[dataset_name][seed][method][n])

                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))

            if means:
                # hnc: solid line with markers, default: dashed line
                if strategy == 'hnc':
                    label = f'{strategy}-{method}'
                    ax.plot(n_values, means, 'o-', label=label,
                           color=colors[strategy][method], linewidth=2, markersize=6)
                else:
                    label = f'{strategy}-{method} (baseline)'
                    ax.plot(n_values, means, '--', label=label,
                           color=colors[strategy][method], linewidth=2)

                means = np.array(means)
                stds = np.array(stds)
                ax.fill_between(n_values, means - stds, means + stds,
                               alpha=0.15, color=colors[strategy][method])

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title(f'{approach.replace("_", " ").title()}\nScaling Curves',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{approach}-scaling_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Dataset configurations
    datasets_config = {
        'hnc-bon': {
            'path': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon',
            'filter_strings': []
            
        },
        'hnc-beam_search': {
            'path': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-beam_search',
            'filter_strings': []
        },
        'hnc-dvts': {
            'path': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-dvts',
            'filter_strings': []
        },
        'default-bon': {
            'path': 'ENSEONG/default-Qwen2.5-1.5B-Instruct-bon',
            'subset_template': 'HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--seed-{seed}--agg_strategy-last',
            'filter_strings': []
        },
        'default-beam_search': {
            'path': 'ENSEONG/default-Qwen2.5-1.5B-Instruct-beam_search',
            'subset_template': 'HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'filter_strings': []
        },
        'default-dvts': {
            'path': 'ENSEONG/default-Qwen2.5-1.5B-Instruct-dvts',
            'subset_template': 'HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last',
            'filter_strings': []
        }
    }

    # Seed configurations
    hnc_seeds = [128, 192, 256]  # temp 0.6~1.2 : [0, 42, 64] # temp 0.4~1.6 [128, 192, 256]
    default_seeds = [0, 42, 64]  # default temp 0.8 only have [0, 42, 64]

    # Store all results: dataset_name -> seed -> method -> n_samples -> accuracy
    all_results = defaultdict(lambda: defaultdict(dict))

    # Process each dataset
    for dataset_name, config in datasets_config.items():
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}")

        # Use appropriate seeds for each dataset type
        dataset_seeds = default_seeds if 'default' in dataset_name else hnc_seeds

        for seed in dataset_seeds:
            print(f"\nLoading seed {seed}...")

            try:
                # Load dataset
                # Legacy: use subset_template if provided
                if 'subset_template' in config:
                    subset_name = config['subset_template'].format(seed=seed)
                    print(f"  Using subset: {subset_name}")
                    dataset = load_dataset(config['path'], subset_name)
                # New: find matching subset
                else:
                    configs = get_dataset_config_names(config['path'])
                    matching = [c for c in configs if f'seed-{seed}' in c]

                    # Apply filter_strings if provided and not empty
                    if 'filter_strings' in config and config['filter_strings']:
                        matching = [c for c in matching if all(f in c for f in config['filter_strings'])]

                    if not matching:
                        raise ValueError(f"No matching subset for seed {seed}")

                    print(f"  Using subset: {matching[0]}")
                    dataset = load_dataset(config['path'], matching[0])

                # Analyze
                results = analyze_single_dataset(dataset, dataset_name, seed)
                all_results[dataset_name][seed] = results

            except Exception as e:
                print(f"  Error processing {dataset_name} seed {seed}: {e}")
                continue

    # Group by approach type
    approaches = ['bon', 'beam_search', 'dvts']
    strategies = ['hnc', 'default']

    # Generate visualizations
    print("\n\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Create output directory with seed information
    seed_str = '-'.join(map(str, hnc_seeds))
    output_dir = f"exp/results_analysis_seeds_{seed_str}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    for approach in approaches:
        print(f"\n  Generating plots for {approach}...")

        # Scaling curves for each approach
        plot_scaling_curves(all_results, approach, output_dir, hnc_seeds)
        print(f"    - Scaling curves: {approach}-scaling_curves.png")

        for method in ['naive', 'weighted', 'maj']:
            # Cross-configuration comparison
            plot_configuration_comparison(all_results, approach, method, output_dir, hnc_seeds)
            print(f"    - Configuration comparison ({method}): {approach}-{method}-hnc_vs_default.png")

    print(f"\nAll visualizations saved to {output_dir}/")
    print("="*80)

    # Generate markdown report
    print("\n\n" + "="*80)
    print("GENERATING ANALYSIS REPORT")
    print("="*80)

    # Build markdown report
    md_lines = ["# Analysis Report\n"]

    # Metadata section
    md_lines.append("## Metadata\n")
    md_lines.append(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append(f"- **HNC Seeds**: {hnc_seeds}\n")
    md_lines.append(f"- **Default Seeds**: {default_seeds}\n")
    md_lines.append(f"- **Approaches**: {', '.join(approaches)}\n")
    md_lines.append("\n")

    # Cross-configuration comparison section
    md_lines.append("## Cross-Configuration Comparison (HNC vs DEFAULT)\n")
    md_lines.append("\n")

    for approach in approaches:
        md_lines.append(f"### APPROACH: {approach.upper()}\n")
        md_lines.append("\n")

        methods = ['naive', 'weighted', 'maj']

        for method in methods:
            md_lines.append(f"#### Method: {method.upper()}\n")
            md_lines.append("\n")

            # Collect all n values for this approach and method
            all_n_values = set()
            for strategy in strategies:
                dataset_name = f"{strategy}-{approach}"
                strategy_seeds = hnc_seeds if strategy == 'hnc' else default_seeds
                if dataset_name in all_results:
                    for seed in strategy_seeds:
                        if seed in all_results[dataset_name]:
                            all_n_values.update(all_results[dataset_name][seed][method].keys())

            if not all_n_values:
                md_lines.append(f"*No data available for {method}*\n")
                md_lines.append("\n")
                continue

            # Table header
            md_lines.append("| n | HNC Acc | Default Acc | Difference | Improvement |\n")
            md_lines.append("|---|---------|-------------|------------|-------------|\n")

            for n in sorted(all_n_values):
                hnc_values = []
                default_values = []

                # Collect HNC values
                hnc_dataset = f"hnc-{approach}"
                if hnc_dataset in all_results:
                    for seed in hnc_seeds:
                        if seed in all_results[hnc_dataset]:
                            if n in all_results[hnc_dataset][seed][method]:
                                hnc_values.append(all_results[hnc_dataset][seed][method][n])

                # Collect Default values
                default_dataset = f"default-{approach}"
                if default_dataset in all_results:
                    for seed in default_seeds:
                        if seed in all_results[default_dataset]:
                            if n in all_results[default_dataset][seed][method]:
                                default_values.append(all_results[default_dataset][seed][method][n])

                if hnc_values or default_values:
                    hnc_mean = np.mean(hnc_values) if hnc_values else 0
                    default_mean = np.mean(default_values) if default_values else 0
                    diff = hnc_mean - default_mean
                    improvement = (diff / default_mean * 100) if default_mean > 0 else 0

                    hnc_str = f"{hnc_mean:.4f}" if hnc_values else "N/A"
                    default_str = f"{default_mean:.4f}" if default_values else "N/A"
                    diff_str = f"{diff:+.4f}" if (hnc_values and default_values) else "N/A"
                    improvement_str = f"{improvement:+.2f}%" if (hnc_values and default_values) else "N/A"

                    md_lines.append(f"| {n} | {hnc_str} | {default_str} | {diff_str} | {improvement_str} |\n")

            md_lines.append("\n")

    # Save markdown report
    report_path = os.path.join(output_dir, 'analysis_report.md')
    with open(report_path, 'w') as f:
        f.writelines(md_lines)

    print(f"Analysis report saved to: {report_path}")
    print("="*80)


if __name__ == "__main__":
    main()
