"""
Temperature analysis with difficulty stratification for HNC experiments.

This module establishes baseline problem difficulty from default datasets (single temp=0.8)
and analyzes HNC multi-temperature performance stratified by difficulty. Supports all three
approaches: BoN, DVTS, and Beam Search.

Key Features:
- Baseline establishment from default datasets
- Problem difficulty computation (mean accuracy)
- 5-level difficulty stratification (quintiles)
- Temperature-specific analysis per difficulty level
- Comprehensive visualizations and reporting

Usage:
    python exp/temperature_analysis_stratified.py
"""

import os
import re
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass

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
    infer_temperature_from_position,
    supports_temperature_analysis,
    validate_temperature_config,
)


# ======================================================================================
# DATA STRUCTURES
# ======================================================================================

@dataclass
class ProblemBaseline:
    """Baseline statistics for a single problem."""
    unique_id: str
    problem_text: str
    answer: str
    mean_accuracy: float  # Mean across all completions, seeds, approaches
    num_evaluations: int  # Total completions evaluated


@dataclass
class DifficultyLevel:
    """Difficulty level definition."""
    level: int  # 1-5 (1=easiest, 5=hardest)
    min_accuracy: float
    max_accuracy: float
    problem_count: int
    problem_ids: list[str]


@dataclass
class TemperatureResult:
    """Results for a single temperature."""
    temperature: float
    overall_accuracy: float
    accuracy_by_difficulty: dict[int, float]  # level -> accuracy
    sample_count_by_difficulty: dict[int, int]  # level -> count


# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================

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
        # Extract content in \\boxed{...}
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, completion)
        if match:
            pred_text = match.group(1)
        else:
            # Fallback: try to extract last number-like string
            pred_text = completion.strip().split()[-1] if completion.strip() else ""

        pred = parse("\\boxed{" + pred_text + "}")

        return verify(gold, pred)
    except:
        return False


# ======================================================================================
# PHASE 1: BASELINE ESTABLISHMENT
# ======================================================================================

def load_default_datasets(
    approaches: list[str],
    seeds: list[int],
    dataset_paths: dict[str, str]
) -> dict[str, dict[int, any]]:
    """
    Load default datasets for all approaches and seeds.

    Args:
        approaches: ['bon', 'beam_search', 'dvts']
        seeds: [0, 42, 64]
        dataset_paths: {'bon': 'ENSEONG/default-...', ...}

    Returns:
        {approach: {seed: Dataset}}
    """
    datasets = defaultdict(dict)

    for approach in approaches:
        print(f"\n  Loading default {approach} datasets...")
        for seed in seeds:
            try:
                # Find matching subset
                configs = get_dataset_config_names(dataset_paths[approach])
                matching = [c for c in configs if f'seed-{seed}' in c]

                if not matching:
                    print(f"    Warning: No subset found for seed {seed}")
                    continue

                print(f"    Seed {seed}: {matching[0]}")
                dataset = load_dataset(dataset_paths[approach], matching[0])
                datasets[approach][seed] = dataset

            except Exception as e:
                print(f"    Error loading {approach} seed {seed}: {e}")
                continue

    return dict(datasets)


def compute_problem_baselines(
    datasets: dict[str, dict[int, any]],
    aggregate_across_approaches: bool = True
) -> dict[str, ProblemBaseline]:
    """
    Compute baseline difficulty for each problem.

    For each unique_id:
    1. Collect all completions across seeds and approaches
    2. Evaluate each completion with math_verify
    3. Compute mean accuracy

    Args:
        datasets: {approach: {seed: Dataset}}
        aggregate_across_approaches: If True, average across all approaches.
                                     If False, keep separate per approach.

    Returns:
        {unique_id: ProblemBaseline} or {f"{approach}:{unique_id}": ProblemBaseline}
    """
    print("\n  Computing problem baselines...")

    # Collect all evaluations per problem
    problem_evaluations = defaultdict(list)
    problem_metadata = {}  # Store problem text and answer

    for approach, seeds_data in datasets.items():
        for seed, dataset in seeds_data.items():
            if 'train' in dataset:
                dataset = dataset['train']

            for problem in tqdm(dataset, desc=f"    Evaluating {approach} seed {seed}", leave=False):
                unique_id = problem.get('unique_id', problem.get('problem', ''))
                if not unique_id:
                    continue

                # Store metadata (only once per problem)
                if unique_id not in problem_metadata:
                    problem_metadata[unique_id] = {
                        'problem_text': problem.get('problem', ''),
                        'answer': problem['answer']
                    }

                # Evaluate all completions
                completions = problem['completions']
                gold_answer = problem['answer']

                for completion in completions:
                    is_correct = evaluate_answer(completion, gold_answer)
                    key = unique_id if aggregate_across_approaches else f"{approach}:{unique_id}"
                    problem_evaluations[key].append(is_correct)

    # Compute mean accuracy per problem
    baselines = {}
    for problem_key, evaluations in problem_evaluations.items():
        unique_id = problem_key.split(':')[-1] if ':' in problem_key else problem_key
        mean_acc = sum(evaluations) / len(evaluations) if evaluations else 0.0

        baselines[problem_key] = ProblemBaseline(
            unique_id=unique_id,
            problem_text=problem_metadata.get(unique_id, {}).get('problem_text', ''),
            answer=problem_metadata.get(unique_id, {}).get('answer', ''),
            mean_accuracy=mean_acc,
            num_evaluations=len(evaluations)
        )

    print(f"  Computed baselines for {len(baselines)} problems")
    return baselines


def stratify_by_difficulty(
    baselines: dict[str, ProblemBaseline],
    num_levels: int = 5
) -> dict[int, DifficultyLevel]:
    """
    Stratify problems into difficulty quintiles.

    Args:
        baselines: {unique_id: ProblemBaseline}
        num_levels: Number of difficulty levels (default 5)

    Returns:
        {level: DifficultyLevel} where level 1 is easiest, 5 is hardest
    """
    print("\n  Stratifying by difficulty...")

    # Extract accuracies
    accuracies = [b.mean_accuracy for b in baselines.values()]
    problem_ids = list(baselines.keys())

    # Compute quintile boundaries
    quintiles = np.percentile(accuracies, np.linspace(0, 100, num_levels + 1))

    # Assign problems to levels
    levels = {}
    for level_idx in range(num_levels):
        # Level 5 (hardest) = lowest accuracy (< Q1)
        # Level 1 (easiest) = highest accuracy (>= Q4)
        level = num_levels - level_idx  # Reverse: 5, 4, 3, 2, 1

        min_acc = quintiles[level_idx]
        max_acc = quintiles[level_idx + 1]

        # Find problems in this range
        if level_idx == num_levels - 1:
            # Last level: include max boundary
            level_problems = [
                pid for pid, b in baselines.items()
                if min_acc <= b.mean_accuracy <= max_acc
            ]
        else:
            level_problems = [
                pid for pid, b in baselines.items()
                if min_acc <= b.mean_accuracy < max_acc
            ]

        levels[level] = DifficultyLevel(
            level=level,
            min_accuracy=min_acc,
            max_accuracy=max_acc,
            problem_count=len(level_problems),
            problem_ids=level_problems
        )

        print(f"    Level {level}: {len(level_problems)} problems, accuracy [{min_acc:.3f}, {max_acc:.3f}]")

    return levels


# ======================================================================================
# PHASE 2: HNC TEMPERATURE ANALYSIS
# ======================================================================================

def load_hnc_datasets(
    approaches: list[str],
    seeds: list[int],
    dataset_paths: dict[str, str]
) -> dict[str, dict[int, any]]:
    """
    Load HNC multi-temperature datasets.

    Args:
        approaches: ['bon', 'beam_search', 'dvts']
        seeds: [128, 192, 256]
        dataset_paths: {'bon': 'ENSEONG/hnc-...', ...}

    Returns:
        {approach: {seed: Dataset}}
    """
    datasets = defaultdict(dict)

    for approach in approaches:
        print(f"\n  Loading HNC {approach} datasets...")
        for seed in seeds:
            try:
                # Find matching subset
                configs = get_dataset_config_names(dataset_paths[approach])
                matching = [c for c in configs if f'seed-{seed}' in c]

                if not matching:
                    print(f"    Warning: No subset found for seed {seed}")
                    continue

                print(f"    Seed {seed}: {matching[0]}")
                dataset = load_dataset(dataset_paths[approach], matching[0])
                datasets[approach][seed] = dataset

            except Exception as e:
                print(f"    Error loading {approach} seed {seed}: {e}")
                continue

    return dict(datasets)


def analyze_temperature_by_difficulty(
    dataset: any,
    approach: str,
    difficulty_map: dict[str, int],  # unique_id -> difficulty level
    temperatures: list[float],
    n: int = 64,
    beam_width: int = 16
) -> dict[float, TemperatureResult]:
    """
    Analyze each temperature's performance stratified by difficulty.

    For each temperature:
    1. Filter completions assigned to that temperature
    2. Evaluate each completion
    3. Group by difficulty level
    4. Compute accuracy per level

    Args:
        dataset: HNC dataset with multi-temperature completions
        approach: 'bon', 'beam_search', or 'dvts'
        difficulty_map: Maps unique_id to difficulty level (1-5)
        temperatures: [0.4, 0.8, 1.2, 1.6]
        n: Total samples
        beam_width: Beam width for DVTS

    Returns:
        {temperature: TemperatureResult}
    """
    if 'train' in dataset:
        dataset = dataset['train']

    # Validate temperature configuration
    validate_temperature_config(approach, n, temperatures, beam_width if approach == 'dvts' else None)

    # Collect results by (temperature, difficulty_level)
    temp_level_results = defaultdict(lambda: defaultdict(list))

    for problem in tqdm(dataset, desc=f"    Analyzing {approach}", leave=False):
        unique_id = problem.get('unique_id', problem.get('problem', ''))
        if not unique_id or unique_id not in difficulty_map:
            continue

        difficulty_level = difficulty_map[unique_id]
        completions = problem['completions']
        gold_answer = problem['answer']

        # Assign temperatures to each completion
        for position, completion in enumerate(completions):
            # Infer temperature from position
            temp = infer_temperature_from_position(
                position, approach, n=n, temperatures=temperatures, beam_width=beam_width
            )

            # Evaluate completion
            is_correct = evaluate_answer(completion, gold_answer)

            # Record result
            temp_level_results[temp][difficulty_level].append(is_correct)

    # Compute accuracy per temperature per difficulty level
    results = {}
    for temp in temperatures:
        # Overall accuracy (across all difficulty levels)
        all_results = []
        for level_results in temp_level_results[temp].values():
            all_results.extend(level_results)
        overall_acc = sum(all_results) / len(all_results) if all_results else 0.0

        # Accuracy by difficulty level
        acc_by_difficulty = {}
        count_by_difficulty = {}
        for level in range(1, 6):  # Levels 1-5
            level_results = temp_level_results[temp][level]
            acc_by_difficulty[level] = sum(level_results) / len(level_results) if level_results else 0.0
            count_by_difficulty[level] = len(level_results)

        results[temp] = TemperatureResult(
            temperature=temp,
            overall_accuracy=overall_acc,
            accuracy_by_difficulty=acc_by_difficulty,
            sample_count_by_difficulty=count_by_difficulty
        )

    return results


def aggregate_across_seeds(
    results_by_seed: dict[int, dict[float, TemperatureResult]]
) -> dict[float, dict[str, any]]:
    """
    Aggregate temperature results across seeds.

    Computes mean and std for:
    - Overall accuracy
    - Accuracy by difficulty level

    Args:
        results_by_seed: {seed: {temperature: TemperatureResult}}

    Returns:
        {
            temperature: {
                'overall_mean': float,
                'overall_std': float,
                'by_difficulty': {
                    level: {'mean': float, 'std': float}
                }
            }
        }
    """
    # Collect all temperatures
    all_temps = set()
    for seed_results in results_by_seed.values():
        all_temps.update(seed_results.keys())

    aggregated = {}
    for temp in sorted(all_temps):
        # Collect overall accuracies across seeds
        overall_accs = []
        by_difficulty_accs = defaultdict(list)

        for seed, seed_results in results_by_seed.items():
            if temp in seed_results:
                overall_accs.append(seed_results[temp].overall_accuracy)

                for level, acc in seed_results[temp].accuracy_by_difficulty.items():
                    by_difficulty_accs[level].append(acc)

        # Compute statistics
        aggregated[temp] = {
            'overall_mean': np.mean(overall_accs) if overall_accs else 0.0,
            'overall_std': np.std(overall_accs) if overall_accs else 0.0,
            'by_difficulty': {
                level: {
                    'mean': np.mean(accs) if accs else 0.0,
                    'std': np.std(accs) if accs else 0.0,
                    'count': len(accs)
                }
                for level, accs in by_difficulty_accs.items()
            }
        }

    return aggregated


# ======================================================================================
# PHASE 3: VISUALIZATION
# ======================================================================================

def plot_difficulty_distribution(
    difficulty_levels: dict[int, DifficultyLevel],
    output_path: str
):
    """
    Bar chart showing number of problems per difficulty level.
    Also shows mean accuracy range for each level.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    levels = sorted(difficulty_levels.keys(), reverse=True)  # 5, 4, 3, 2, 1
    counts = [difficulty_levels[l].problem_count for l in levels]
    labels = [f"Level {l}\n[{difficulty_levels[l].min_accuracy:.3f}, {difficulty_levels[l].max_accuracy:.3f}]"
              for l in levels]

    # Color gradient from red (hard) to green (easy)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(levels)))

    bars = ax.barh(range(len(levels)), counts, color=colors, alpha=0.8)

    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of Problems', fontsize=12)
    ax.set_title('Problem Difficulty Distribution\n(Based on Default Dataset Mean Accuracy)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {count}', ha='left', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_temperature_by_difficulty_heatmap(
    aggregated_results: dict[float, dict[str, any]],
    approach: str,
    output_path: str
):
    """
    Heatmap: rows=temperatures, cols=difficulty levels, values=accuracy
    Shows mean accuracy with standard deviation annotations.
    """
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for heatmap
    temperatures = sorted(aggregated_results.keys())
    levels = list(range(1, 6))  # 1-5

    # Build matrix: rows=temps, cols=levels
    matrix = []
    annotations = []
    for temp in temperatures:
        row = []
        ann_row = []
        for level in levels:
            if level in aggregated_results[temp]['by_difficulty']:
                mean = aggregated_results[temp]['by_difficulty'][level]['mean']
                std = aggregated_results[temp]['by_difficulty'][level]['std']
                row.append(mean)
                ann_row.append(f'{mean:.3f}\n±{std:.3f}')
            else:
                row.append(0.0)
                ann_row.append('N/A')
        matrix.append(row)
        annotations.append(ann_row)

    matrix = np.array(matrix)

    # Create heatmap
    sns.heatmap(matrix, annot=np.array(annotations), fmt='', cmap='RdYlGn',
                vmin=0, vmax=matrix.max() * 1.1 if matrix.max() > 0 else 1.0,
                xticklabels=[f'Level {l}' for l in levels],
                yticklabels=[f'{t:.1f}' for t in temperatures],
                ax=ax, cbar_kws={'label': 'Accuracy'})

    ax.set_xlabel('Difficulty Level (1=Easy, 5=Hard)', fontsize=12)
    ax.set_ylabel('Temperature', fontsize=12)
    ax.set_title(f'{approach.upper()} - Temperature Accuracy by Difficulty\n(mean ± std)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_temperature_scaling_by_difficulty(
    aggregated_results: dict[float, dict[str, any]],
    approach: str,
    output_path: str
):
    """
    Line plot with one line per difficulty level.
    X-axis: temperature, Y-axis: accuracy
    Shows how each difficulty level responds to temperature.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    temperatures = sorted(aggregated_results.keys())
    levels = list(range(1, 6))  # 1-5
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(levels)))

    for level, color in zip(levels, colors):
        means = []
        stds = []
        for temp in temperatures:
            if level in aggregated_results[temp]['by_difficulty']:
                means.append(aggregated_results[temp]['by_difficulty'][level]['mean'])
                stds.append(aggregated_results[temp]['by_difficulty'][level]['std'])
            else:
                means.append(0.0)
                stds.append(0.0)

        means = np.array(means)
        stds = np.array(stds)

        # Plot line with error band
        ax.plot(temperatures, means, 'o-', label=f'Level {level}',
                color=color, linewidth=2, markersize=8)
        ax.fill_between(temperatures, means - stds, means + stds,
                        alpha=0.2, color=color)

    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{approach.upper()} - Temperature Scaling by Difficulty Level',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', title='Difficulty')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


def plot_difficulty_correlation(
    baselines: dict[str, ProblemBaseline],
    output_path: str
):
    """
    Histogram showing distribution of mean accuracies.
    Overlays quintile boundaries.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    accuracies = [b.mean_accuracy for b in baselines.values()]

    # Histogram
    ax.hist(accuracies, bins=50, color='steelblue', alpha=0.7, edgecolor='black')

    # Quintile lines
    quintiles = np.percentile(accuracies, [20, 40, 60, 80])
    for i, q in enumerate(quintiles, start=1):
        ax.axvline(q, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Q{i}: {q:.3f}' if i == 1 else f'Q{i}: {q:.3f}')

    ax.set_xlabel('Mean Accuracy (Default Dataset)', fontsize=12)
    ax.set_ylabel('Number of Problems', fontsize=12)
    ax.set_title('Distribution of Problem Difficulties\nwith Quintile Boundaries',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_path}")


# ======================================================================================
# PHASE 4: REPORT GENERATION
# ======================================================================================

def generate_report(
    baselines: dict[str, ProblemBaseline],
    difficulty_levels: dict[int, DifficultyLevel],
    all_results: dict[str, dict[float, dict[str, any]]],
    output_path: str,
    hnc_seeds: list[int],
    default_seeds: list[int],
    temperatures: list[float]
):
    """
    Generate comprehensive markdown report.

    Args:
        baselines: Problem baseline statistics
        difficulty_levels: Difficulty level definitions
        all_results: {approach: aggregated_results}
        output_path: Path to save report
        hnc_seeds: HNC seeds used
        default_seeds: Default seeds used
        temperatures: Temperature values analyzed
    """
    md_lines = ["# Temperature Analysis with Difficulty Stratification\n\n"]

    # Metadata
    md_lines.append("## Metadata\n\n")
    md_lines.append(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append(f"- **Baseline**: Default datasets (seeds {default_seeds}, single temp=0.8)\n")
    md_lines.append(f"- **HNC Analysis**: Multi-temp datasets (seeds {hnc_seeds}, temps {temperatures})\n")
    md_lines.append(f"- **Approaches**: {', '.join(all_results.keys())}\n\n")

    # Baseline statistics
    md_lines.append("## Baseline Establishment\n\n")
    md_lines.append(f"- **Total Problems**: {len(baselines)}\n")
    accuracies = [b.mean_accuracy for b in baselines.values()]
    md_lines.append(f"- **Mean Accuracy**: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}\n")
    md_lines.append(f"- **Min Accuracy**: {np.min(accuracies):.4f}\n")
    md_lines.append(f"- **Max Accuracy**: {np.max(accuracies):.4f}\n\n")

    # Difficulty stratification
    md_lines.append("## Difficulty Stratification\n\n")
    md_lines.append("| Level | Description | Accuracy Range | Problem Count |\n")
    md_lines.append("|-------|-------------|----------------|---------------|\n")
    for level in sorted(difficulty_levels.keys(), reverse=True):
        desc = {5: "Hardest", 4: "Hard", 3: "Medium", 2: "Easy", 1: "Easiest"}[level]
        dl = difficulty_levels[level]
        md_lines.append(f"| {level} | {desc} | [{dl.min_accuracy:.3f}, {dl.max_accuracy:.3f}] | {dl.problem_count} |\n")
    md_lines.append("\n")

    # Temperature analysis per approach
    for approach in sorted(all_results.keys()):
        md_lines.append(f"## {approach.upper()} - Temperature Analysis\n\n")

        # Overall performance table
        md_lines.append("### Overall Performance by Temperature\n\n")
        md_lines.append("| Temperature | Accuracy (mean ± std) |\n")
        md_lines.append("|-------------|-----------------------|\n")
        for temp in temperatures:
            if temp in all_results[approach]:
                mean = all_results[approach][temp]['overall_mean']
                std = all_results[approach][temp]['overall_std']
                md_lines.append(f"| {temp:.1f} | {mean:.4f} ± {std:.4f} |\n")
        md_lines.append("\n")

        # Performance by difficulty level
        md_lines.append("### Performance by Difficulty Level\n\n")
        for level in sorted(difficulty_levels.keys(), reverse=True):
            desc = {5: "Hardest", 4: "Hard", 3: "Medium", 2: "Easy", 1: "Easiest"}[level]
            md_lines.append(f"#### Level {level} ({desc})\n\n")
            md_lines.append("| Temperature | Accuracy (mean ± std) |\n")
            md_lines.append("|-------------|-----------------------|\n")

            for temp in temperatures:
                if temp in all_results[approach]:
                    if level in all_results[approach][temp]['by_difficulty']:
                        mean = all_results[approach][temp]['by_difficulty'][level]['mean']
                        std = all_results[approach][temp]['by_difficulty'][level]['std']
                        md_lines.append(f"| {temp:.1f} | {mean:.4f} ± {std:.4f} |\n")
            md_lines.append("\n")

    # Save report
    with open(output_path, 'w') as f:
        f.writelines(md_lines)
    print(f"    Saved: {output_path}")


# ======================================================================================
# MAIN FUNCTION
# ======================================================================================

def main():
    """
    Main analysis flow:

    1. BASELINE ESTABLISHMENT
       - Load default datasets (bon, beam_search, dvts) x (seeds 0, 42, 64)
       - Compute mean accuracy per problem (aggregate across seeds)

    2. DIFFICULTY STRATIFICATION
       - Sort problems by mean accuracy
       - Divide into 5 quintiles
       - Create difficulty_map: {unique_id: level}

    3. HNC TEMPERATURE ANALYSIS
       - Load HNC datasets (bon, beam_search, dvts) x (seeds 128, 192, 256)
       - For each approach:
         - For each seed:
           - Analyze temperature performance by difficulty
         - Aggregate across seeds
         - Generate visualizations

    4. REPORT GENERATION
       - Markdown report with statistics and findings
    """

    # Configuration
    default_seeds = [0, 42, 64]
    hnc_seeds = [128, 192, 256]
    temperatures = [0.4, 0.8, 1.2, 1.6]
    approaches = ['bon', 'beam_search', 'dvts']

    dataset_paths = {
        'default': {
            'bon': 'ENSEONG/default-Qwen2.5-1.5B-Instruct-bon',
            'beam_search': 'ENSEONG/default-Qwen2.5-1.5B-Instruct-beam_search',
            'dvts': 'ENSEONG/default-Qwen2.5-1.5B-Instruct-dvts',
        },
        'hnc': {
            'bon': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon',
            'beam_search': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-beam_search',
            'dvts': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-dvts',
        }
    }

    # Phase 1: Baseline establishment
    print("=" * 80)
    print("PHASE 1: ESTABLISHING BASELINE FROM DEFAULT DATASETS")
    print("=" * 80)

    default_datasets = load_default_datasets(
        approaches, default_seeds, dataset_paths['default']
    )

    baselines = compute_problem_baselines(
        default_datasets, aggregate_across_approaches=True
    )

    # Phase 2: Difficulty stratification
    print("\n" + "=" * 80)
    print("PHASE 2: DIFFICULTY STRATIFICATION")
    print("=" * 80)

    difficulty_levels = stratify_by_difficulty(baselines, num_levels=5)

    # Create difficulty map for lookups
    difficulty_map = {}
    for level, level_info in difficulty_levels.items():
        for problem_id in level_info.problem_ids:
            difficulty_map[problem_id] = level

    # Phase 3: HNC temperature analysis
    print("\n" + "=" * 80)
    print("PHASE 3: HNC TEMPERATURE ANALYSIS")
    print("=" * 80)

    hnc_datasets = load_hnc_datasets(
        approaches, hnc_seeds, dataset_paths['hnc']
    )

    all_results = {}

    for approach in approaches:
        print(f"\n  Analyzing {approach}...")

        results_by_seed = {}
        for seed in hnc_seeds:
            if seed in hnc_datasets[approach]:
                dataset = hnc_datasets[approach][seed]
                results_by_seed[seed] = analyze_temperature_by_difficulty(
                    dataset, approach, difficulty_map, temperatures, n=64, beam_width=16
                )

        # Aggregate across seeds
        if results_by_seed:
            aggregated = aggregate_across_seeds(results_by_seed)
            all_results[approach] = aggregated

    # Phase 4: Visualization and reporting
    print("\n" + "=" * 80)
    print("PHASE 4: VISUALIZATION AND REPORTING")
    print("=" * 80)

    output_dir = "exp/temperature_analysis_stratified"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n  Output directory: {output_dir}")

    print("\n  Generating baseline visualizations...")
    # Baseline visualizations
    plot_difficulty_distribution(
        difficulty_levels,
        os.path.join(output_dir, 'difficulty_distribution.png')
    )
    plot_difficulty_correlation(
        baselines,
        os.path.join(output_dir, 'difficulty_correlation.png')
    )

    # Temperature analysis visualizations per approach
    print("\n  Generating temperature analysis visualizations...")
    for approach in approaches:
        if approach in all_results:
            print(f"    {approach}...")
            plot_temperature_by_difficulty_heatmap(
                all_results[approach],
                approach,
                os.path.join(output_dir, f'{approach}_temp_by_difficulty_heatmap.png')
            )
            plot_temperature_scaling_by_difficulty(
                all_results[approach],
                approach,
                os.path.join(output_dir, f'{approach}_temp_scaling_by_difficulty.png')
            )

    # Generate markdown report
    print("\n  Generating report...")
    generate_report(
        baselines, difficulty_levels, all_results,
        os.path.join(output_dir, 'temperature_analysis_stratified_report.md'),
        hnc_seeds, default_seeds, temperatures
    )

    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE! Results saved to {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

