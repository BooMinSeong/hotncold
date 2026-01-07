"""
Temperature-specific analysis for HNC multi-temperature experiments.

This module analyzes the impact of different temperature values on model performance,
comparing temperatures across BoN and DVTS approaches. Beam Search is excluded as
temperature attribution is ambiguous due to multi-iteration tree structure.
"""

import os
import re
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from math_verify import parse, verify

# Import temperature utilities
import sys
sys.path.insert(0, os.path.dirname(__file__))
from temperature_utils import (
    assign_temperatures_to_completions,
    group_by_temperature,
    supports_temperature_analysis,
    get_approach_from_dataset_name,
)


def evaluate_answer(completion: str, gold_answer: str) -> bool:
    """
    Evaluate if a completion contains the correct answer.

    Args:
        completion: Generated completion text
        gold_answer: Gold standard answer

    Returns:
        True if correct, False otherwise
    """
    try:
        # Parse gold answer
        gold = parse("\\boxed{" + gold_answer + "}")

        # Parse prediction from completion
        # Extract content in \boxed{...}
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, completion)
        if match:
            pred_text = match.group(1)
        else:
            # Fallback: try to extract last number-like string
            # This is a simplified approach; production code should use robust parser
            pred_text = completion.strip().split()[-1] if completion.strip() else ""

        pred = parse("\\boxed{" + pred_text + "}")

        return verify(gold, pred)
    except:
        return False


def analyze_per_temperature_accuracy(
    dataset,
    approach: str,
    temperatures: list[float] = None,
    n: int = 64,
    beam_width: int = 16,
) -> dict:
    """
    Compute accuracy for each temperature independently.

    Args:
        dataset: HuggingFace dataset with completions and answers
        approach: 'bon' or 'dvts'
        temperatures: List of temperature values
        n: Total samples for BoN
        beam_width: Beam width for DVTS

    Returns:
        Dictionary mapping temperature to accuracy:
        {0.4: 0.45, 0.8: 0.52, ...}
    """
    if temperatures is None:
        temperatures = [0.4, 0.8, 1.2, 1.6]

    if 'train' in dataset:
        dataset = dataset['train']

    # Aggregate results by temperature
    temp_results = defaultdict(list)

    for problem in tqdm(dataset, desc=f"  Per-temp accuracy ({approach})", leave=False):
        completions = problem['completions']
        gold_answer = problem['answer']

        # Assign temperatures to completions
        assigned = assign_temperatures_to_completions(
            completions, [0] * len(completions),  # Dummy scores
            approach, n=n, temperatures=temperatures, beam_width=beam_width
        )

        # Evaluate each completion
        for completion, temp in zip(completions, assigned['temperatures']):
            is_correct = evaluate_answer(completion, gold_answer)
            temp_results[temp].append(is_correct)

    # Compute accuracy for each temperature
    accuracies = {}
    for temp in temperatures:
        if temp in temp_results:
            accuracies[temp] = sum(temp_results[temp]) / len(temp_results[temp])
        else:
            accuracies[temp] = 0.0

    return accuracies


def analyze_temperature_prm_distributions(
    dataset,
    approach: str,
    temperatures: list[float] = None,
    n: int = 64,
    beam_width: int = 16,
) -> dict:
    """
    Analyze PRM score distributions per temperature.

    Args:
        dataset: HuggingFace dataset with scores field
        approach: 'bon' or 'dvts'
        temperatures: List of temperature values
        n: Total samples for BoN
        beam_width: Beam width for DVTS

    Returns:
        Dictionary with statistics per temperature:
        {
            0.4: {
                'mean': 0.65,
                'std': 0.15,
                'median': 0.68,
                'scores': [0.5, 0.7, ...]
            },
            ...
        }
    """
    if temperatures is None:
        temperatures = [0.4, 0.8, 1.2, 1.6]

    if 'train' in dataset:
        dataset = dataset['train']

    # Aggregate scores by temperature
    temp_scores = defaultdict(list)

    for problem in tqdm(dataset, desc=f"  PRM distributions ({approach})", leave=False):
        scores = problem['scores']

        # Assign temperatures
        assigned = assign_temperatures_to_completions(
            [0] * len(scores),  # Dummy completions
            scores,
            approach, n=n, temperatures=temperatures, beam_width=beam_width
        )

        # Group scores by temperature
        for score, temp in zip(scores, assigned['temperatures']):
            # scores might be lists of step scores; take aggregate
            if isinstance(score, list):
                score = np.mean(score) if score else 0.0
            temp_scores[temp].append(score)

    # Compute statistics
    stats = {}
    for temp in temperatures:
        if temp in temp_scores and temp_scores[temp]:
            scores = temp_scores[temp]
            stats[temp] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'scores': scores,
            }
        else:
            stats[temp] = {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'scores': [],
            }

    return stats


def temperature_ablation_study(
    dataset,
    approach: str,
    temperatures: list[float] = None,
    n: int = 64,
    beam_width: int = 16,
    method: str = 'weighted',
) -> dict:
    """
    Test performance when removing each temperature (ablation study).

    Evaluates:
    - Baseline: all temperatures
    - Remove each: performance without each individual temperature
    - Single only: performance with only one temperature

    Args:
        dataset: HuggingFace dataset
        approach: 'bon' or 'dvts'
        temperatures: List of temperature values
        n: Total samples for BoN
        beam_width: Beam width for DVTS
        method: Aggregation method ('naive', 'weighted', 'maj')

    Returns:
        Dictionary with ablation results:
        {
            'baseline': 0.65,
            'remove_0.4': 0.63,
            'only_0.4': 0.45,
            ...
        }
    """
    if temperatures is None:
        temperatures = [0.4, 0.8, 1.2, 1.6]

    if 'train' in dataset:
        dataset = dataset['train']

    def weighted_voting(answers: list, scores: list) -> str:
        """Weighted voting based on scores."""
        if not answers:
            return ""
        answer_scores = defaultdict(float)
        for ans, score in zip(answers, scores):
            if isinstance(score, list):
                score = np.mean(score) if score else 0.0
            answer_scores[ans] += score
        return max(answer_scores.items(), key=lambda x: x[1])[0] if answer_scores else ""

    def naive_selection(answers: list, scores: list) -> str:
        """Select answer with highest score."""
        if not answers:
            return ""
        max_idx = max(range(len(scores)), key=lambda i: (
            np.mean(scores[i]) if isinstance(scores[i], list) else scores[i]
        ))
        return answers[max_idx]

    def majority_voting(answers: list) -> str:
        """Majority vote."""
        if not answers:
            return ""
        from collections import Counter
        counts = Counter(answers)
        return counts.most_common(1)[0][0] if counts else ""

    # Choose aggregation function
    if method == 'weighted':
        agg_fn = weighted_voting
    elif method == 'naive':
        agg_fn = naive_selection
    elif method == 'maj':
        agg_fn = lambda ans, scores: majority_voting(ans)
    else:
        raise ValueError(f"Unknown method: {method}")

    results = {}

    # Baseline: all temperatures
    correct_count = 0
    total_count = 0
    for problem in tqdm(dataset, desc=f"  Ablation baseline ({approach})", leave=False):
        completions = problem['completions']
        scores = problem['scores']
        gold_answer = problem['answer']

        # Extract answers from completions
        answers = []
        for completion in completions:
            boxed_pattern = r'\\boxed\{([^}]+)\}'
            match = re.search(boxed_pattern, completion)
            if match:
                answers.append(match.group(1))
            else:
                answers.append(completion.strip().split()[-1] if completion.strip() else "")

        # Aggregate
        if method == 'maj':
            pred_answer = agg_fn(answers, None)
        else:
            pred_answer = agg_fn(answers, scores)

        # Evaluate
        try:
            gold = parse("\\boxed{" + gold_answer + "}")
            pred = parse("\\boxed{" + pred_answer + "}")
            is_correct = verify(gold, pred)
        except:
            is_correct = False

        correct_count += is_correct
        total_count += 1

    results['baseline'] = correct_count / total_count if total_count > 0 else 0.0

    # Remove each temperature
    for temp_to_remove in temperatures:
        correct_count = 0
        total_count = 0
        for problem in dataset:
            completions = problem['completions']
            scores = problem['scores']
            gold_answer = problem['answer']

            # Assign temperatures
            assigned = assign_temperatures_to_completions(
                completions, scores,
                approach, n=n, temperatures=temperatures, beam_width=beam_width
            )

            # Filter out completions with temp_to_remove
            filtered_completions = []
            filtered_scores = []
            for comp, score, temp in zip(completions, scores, assigned['temperatures']):
                if temp != temp_to_remove:
                    filtered_completions.append(comp)
                    filtered_scores.append(score)

            if not filtered_completions:
                continue

            # Extract answers
            answers = []
            for completion in filtered_completions:
                boxed_pattern = r'\\boxed\{([^}]+)\}'
                match = re.search(boxed_pattern, completion)
                if match:
                    answers.append(match.group(1))
                else:
                    answers.append(completion.strip().split()[-1] if completion.strip() else "")

            # Aggregate
            if method == 'maj':
                pred_answer = agg_fn(answers, None)
            else:
                pred_answer = agg_fn(answers, filtered_scores)

            # Evaluate
            try:
                gold = parse("\\boxed{" + gold_answer + "}")
                pred = parse("\\boxed{" + pred_answer + "}")
                is_correct = verify(gold, pred)
            except:
                is_correct = False

            correct_count += is_correct
            total_count += 1

        results[f'remove_{temp_to_remove}'] = correct_count / total_count if total_count > 0 else 0.0

    # Single temperature only
    for temp_only in temperatures:
        correct_count = 0
        total_count = 0
        for problem in dataset:
            completions = problem['completions']
            scores = problem['scores']
            gold_answer = problem['answer']

            # Assign temperatures
            assigned = assign_temperatures_to_completions(
                completions, scores,
                approach, n=n, temperatures=temperatures, beam_width=beam_width
            )

            # Filter to only temp_only
            filtered_completions = []
            filtered_scores = []
            for comp, score, temp in zip(completions, scores, assigned['temperatures']):
                if temp == temp_only:
                    filtered_completions.append(comp)
                    filtered_scores.append(score)

            if not filtered_completions:
                continue

            # Extract answers
            answers = []
            for completion in filtered_completions:
                boxed_pattern = r'\\boxed\{([^}]+)\}'
                match = re.search(boxed_pattern, completion)
                if match:
                    answers.append(match.group(1))
                else:
                    answers.append(completion.strip().split()[-1] if completion.strip() else "")

            # Aggregate
            if method == 'maj':
                pred_answer = agg_fn(answers, None)
            else:
                pred_answer = agg_fn(answers, filtered_scores)

            # Evaluate
            try:
                gold = parse("\\boxed{" + gold_answer + "}")
                pred = parse("\\boxed{" + pred_answer + "}")
                is_correct = verify(gold, pred)
            except:
                is_correct = False

            correct_count += is_correct
            total_count += 1

        results[f'only_{temp_only}'] = correct_count / total_count if total_count > 0 else 0.0

    return results


def plot_temperature_accuracy_bars(per_temp_accuracy: dict, approach: str, output_path: str, seeds: list):
    """
    Bar chart showing accuracy for each temperature.

    Args:
        per_temp_accuracy: Dict mapping seed -> {temp: accuracy}
        approach: 'bon' or 'dvts'
        output_path: Path to save plot
        seeds: List of seeds
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Aggregate across seeds
    temperatures = sorted(set(temp for seed_data in per_temp_accuracy.values() for temp in seed_data.keys()))

    means = []
    stds = []
    for temp in temperatures:
        values = [per_temp_accuracy[seed][temp] for seed in seeds if temp in per_temp_accuracy.get(seed, {})]
        means.append(np.mean(values) if values else 0)
        stds.append(np.std(values) if values else 0)

    # Color gradient from cool to warm
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(temperatures)))

    bars = ax.bar(range(len(temperatures)), means, yerr=stds, capsize=5, color=colors, alpha=0.8)

    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{approach.upper()} - Per-Temperature Accuracy\nAveraged across seeds {seeds}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(temperatures)))
    ax.set_xticklabels([f'{t:.1f}' for t in temperatures])
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(means) * 1.2 if means else 1.0)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_temperature_prm_distributions(prm_stats: dict, approach: str, output_path: str, seeds: list):
    """
    Violin plots showing PRM score distributions per temperature.

    Args:
        prm_stats: Dict mapping seed -> {temp: {'scores': [...]}}
        approach: 'bon' or 'dvts'
        output_path: Path to save plot
        seeds: List of seeds
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect all scores per temperature across seeds
    temperatures = sorted(set(temp for seed_data in prm_stats.values() for temp in seed_data.keys()))

    # Prepare data for violin plot
    data_for_plot = []
    labels = []
    for temp in temperatures:
        all_scores = []
        for seed in seeds:
            if seed in prm_stats and temp in prm_stats[seed]:
                all_scores.extend(prm_stats[seed][temp]['scores'])
        if all_scores:
            data_for_plot.append(all_scores)
            labels.append(f'{temp:.1f}')

    if data_for_plot:
        parts = ax.violinplot(data_for_plot, positions=range(len(data_for_plot)),
                              showmeans=True, showmedians=True)

        # Color the violins
        colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(data_for_plot)))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        ax.set_xlabel('Temperature', fontsize=12)
        ax.set_ylabel('PRM Score', fontsize=12)
        ax.set_title(f'{approach.upper()} - PRM Score Distributions by Temperature\nAveraged across seeds {seeds}',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_temperature_ablation_heatmap(ablation_results: dict, approach: str, output_path: str, seeds: list):
    """
    Heatmap showing ablation study results.

    Args:
        ablation_results: Dict mapping seed -> ablation results dict
        approach: 'bon' or 'dvts'
        output_path: Path to save plot
        seeds: List of seeds
    """
    sns.set_style("white")

    # Aggregate across seeds
    # Get all unique configurations
    all_configs = set()
    for seed_data in ablation_results.values():
        all_configs.update(seed_data.keys())

    configs = sorted(all_configs, key=lambda x: (
        0 if x == 'baseline' else
        1 if x.startswith('only_') else
        2
    ))

    # Compute means
    means = []
    for config in configs:
        values = [ablation_results[seed][config] for seed in seeds if config in ablation_results.get(seed, {})]
        means.append(np.mean(values) if values else 0)

    # Create heatmap data (single column)
    data = np.array(means).reshape(-1, 1)

    fig, ax = plt.subplots(figsize=(6, max(10, len(configs) * 0.5)))

    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=max(means) * 1.1 if means else 1.0,
                yticklabels=configs, xticklabels=['Accuracy'], ax=ax, cbar_kws={'label': 'Accuracy'})

    ax.set_title(f'{approach.upper()} - Temperature Ablation Study\nAveraged across seeds {seeds}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """
    Run comprehensive temperature analysis on HNC datasets.

    Only analyzes approaches that support temperature analysis (bon, dvts).
    Beam Search is excluded.
    """
    # Dataset configurations
    datasets_config = {
        'hnc-bon': {
            'path': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon',
            'filter_strings': []
        },
        'hnc-dvts': {
            'path': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-dvts',
            'filter_strings': []
        },
    }

    # Seed configuration
    hnc_seeds = [128, 192, 256]
    temperatures = [0.4, 0.8, 1.2, 1.6]
    n = 64
    beam_width = 16

    # Store all results
    all_results = defaultdict(lambda: defaultdict(dict))

    # Process each dataset
    for dataset_name, config in datasets_config.items():
        approach = get_approach_from_dataset_name(dataset_name)

        if not supports_temperature_analysis(approach):
            print(f"Skipping {dataset_name}: approach '{approach}' not supported for temperature analysis")
            continue

        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name} (approach: {approach})")
        print(f"{'='*80}")

        for seed in hnc_seeds:
            print(f"\nLoading seed {seed}...")

            try:
                # Find matching subset
                configs = get_dataset_config_names(config['path'])
                matching = [c for c in configs if f'seed-{seed}' in c]

                if config['filter_strings']:
                    matching = [c for c in matching if all(f in c for f in config['filter_strings'])]

                if not matching:
                    raise ValueError(f"No matching subset for seed {seed}")

                print(f"  Using subset: {matching[0]}")
                dataset = load_dataset(config['path'], matching[0])

                # Per-temperature accuracy
                print("  Analyzing per-temperature accuracy...")
                per_temp_acc = analyze_per_temperature_accuracy(
                    dataset, approach, temperatures, n, beam_width
                )
                all_results[dataset_name][seed]['per_temp_accuracy'] = per_temp_acc

                # PRM distributions
                print("  Analyzing PRM score distributions...")
                prm_stats = analyze_temperature_prm_distributions(
                    dataset, approach, temperatures, n, beam_width
                )
                all_results[dataset_name][seed]['prm_distributions'] = prm_stats

                # Ablation study
                print("  Running ablation study (this may take a while)...")
                ablation = temperature_ablation_study(
                    dataset, approach, temperatures, n, beam_width, method='weighted'
                )
                all_results[dataset_name][seed]['ablation'] = ablation

                print(f"  Completed seed {seed}")

            except Exception as e:
                print(f"  Error processing {dataset_name} seed {seed}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Generate visualizations
    print("\n\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    seed_str = '-'.join(map(str, hnc_seeds))
    output_dir = f"exp/temperature_analysis_seeds_{seed_str}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    for dataset_name in datasets_config.keys():
        approach = get_approach_from_dataset_name(dataset_name)

        if dataset_name not in all_results:
            continue

        print(f"\n  Generating plots for {dataset_name}...")

        # Per-temperature accuracy
        per_temp_acc_data = {
            seed: all_results[dataset_name][seed]['per_temp_accuracy']
            for seed in hnc_seeds if seed in all_results[dataset_name]
        }
        if per_temp_acc_data:
            output_path = os.path.join(output_dir, f'{approach}_temp_accuracy_bars.png')
            plot_temperature_accuracy_bars(per_temp_acc_data, approach, output_path, hnc_seeds)
            print(f"    - Temperature accuracy: {approach}_temp_accuracy_bars.png")

        # PRM distributions
        prm_dist_data = {
            seed: all_results[dataset_name][seed]['prm_distributions']
            for seed in hnc_seeds if seed in all_results[dataset_name]
        }
        if prm_dist_data:
            output_path = os.path.join(output_dir, f'{approach}_temp_prm_distributions.png')
            plot_temperature_prm_distributions(prm_dist_data, approach, output_path, hnc_seeds)
            print(f"    - PRM distributions: {approach}_temp_prm_distributions.png")

        # Ablation heatmap
        ablation_data = {
            seed: all_results[dataset_name][seed]['ablation']
            for seed in hnc_seeds if seed in all_results[dataset_name]
        }
        if ablation_data:
            output_path = os.path.join(output_dir, f'{approach}_temp_ablation_heatmap.png')
            plot_temperature_ablation_heatmap(ablation_data, approach, output_path, hnc_seeds)
            print(f"    - Ablation heatmap: {approach}_temp_ablation_heatmap.png")

    print(f"\nAll visualizations saved to {output_dir}/")
    print("="*80)

    # Generate markdown report
    print("\n\n" + "="*80)
    print("GENERATING TEMPERATURE ANALYSIS REPORT")
    print("="*80)

    md_lines = ["# Temperature Analysis Report\n\n"]

    # Metadata
    md_lines.append("## Metadata\n\n")
    md_lines.append(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append(f"- **HNC Seeds**: {hnc_seeds}\n")
    md_lines.append(f"- **Temperatures**: {temperatures}\n")
    md_lines.append(f"- **Analyzed Approaches**: BON, DVTS\n")
    md_lines.append(f"- **Excluded**: BEAM_SEARCH (multi-iteration structure)\n\n")

    # Per-temperature performance
    md_lines.append("## Per-Temperature Performance\n\n")

    for dataset_name in datasets_config.keys():
        approach = get_approach_from_dataset_name(dataset_name)

        if dataset_name not in all_results:
            continue

        md_lines.append(f"### {approach.upper()}\n\n")

        if approach == 'bon':
            md_lines.append("- **Temperature inference**: Sequential (position-based chunks)\n")
        elif approach == 'dvts':
            md_lines.append("- **Temperature inference**: Cyclic per beam (final iteration only)\n")
            md_lines.append("- **Caveat**: Only final iteration temperature known\n")
        md_lines.append("\n")

        # Accuracy table
        md_lines.append("#### Accuracy by Temperature\n\n")
        md_lines.append("| Temperature | Accuracy (mean ± std) |\n")
        md_lines.append("|-------------|----------------------|\n")

        for temp in temperatures:
            values = []
            for seed in hnc_seeds:
                if seed in all_results[dataset_name]:
                    if 'per_temp_accuracy' in all_results[dataset_name][seed]:
                        if temp in all_results[dataset_name][seed]['per_temp_accuracy']:
                            values.append(all_results[dataset_name][seed]['per_temp_accuracy'][temp])

            if values:
                mean_acc = np.mean(values)
                std_acc = np.std(values)
                md_lines.append(f"| {temp:.1f} | {mean_acc:.4f} ± {std_acc:.4f} |\n")

        md_lines.append("\n")

        # PRM statistics table
        md_lines.append("#### PRM Score Statistics\n\n")
        md_lines.append("| Temperature | Mean Score ± std |\n")
        md_lines.append("|-------------|------------------|\n")

        for temp in temperatures:
            mean_scores = []
            for seed in hnc_seeds:
                if seed in all_results[dataset_name]:
                    if 'prm_distributions' in all_results[dataset_name][seed]:
                        if temp in all_results[dataset_name][seed]['prm_distributions']:
                            mean_scores.append(all_results[dataset_name][seed]['prm_distributions'][temp]['mean'])

            if mean_scores:
                mean_score = np.mean(mean_scores)
                std_score = np.std(mean_scores)
                md_lines.append(f"| {temp:.1f} | {mean_score:.4f} ± {std_score:.4f} |\n")

        md_lines.append("\n")

    # Ablation study
    md_lines.append("## Ablation Study\n\n")

    for dataset_name in datasets_config.keys():
        approach = get_approach_from_dataset_name(dataset_name)

        if dataset_name not in all_results:
            continue

        md_lines.append(f"### {approach.upper()}\n\n")

        # Get all configs
        all_configs = set()
        for seed in hnc_seeds:
            if seed in all_results[dataset_name] and 'ablation' in all_results[dataset_name][seed]:
                all_configs.update(all_results[dataset_name][seed]['ablation'].keys())

        configs = sorted(all_configs, key=lambda x: (
            0 if x == 'baseline' else
            1 if x.startswith('only_') else
            2
        ))

        md_lines.append("| Configuration | Accuracy (mean ± std) |\n")
        md_lines.append("|---------------|----------------------|\n")

        for config in configs:
            values = []
            for seed in hnc_seeds:
                if seed in all_results[dataset_name]:
                    if 'ablation' in all_results[dataset_name][seed]:
                        if config in all_results[dataset_name][seed]['ablation']:
                            values.append(all_results[dataset_name][seed]['ablation'][config])

            if values:
                mean_acc = np.mean(values)
                std_acc = np.std(values)
                md_lines.append(f"| {config} | {mean_acc:.4f} ± {std_acc:.4f} |\n")

        md_lines.append("\n")

    # Save report
    report_path = os.path.join(output_dir, 'temperature_analysis_report.md')
    with open(report_path, 'w') as f:
        f.writelines(md_lines)

    print(f"Temperature analysis report saved to: {report_path}")
    print("="*80)


if __name__ == "__main__":
    main()

