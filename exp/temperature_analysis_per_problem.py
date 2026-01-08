"""
Per-Problem Temperature Sensitivity Analysis for HNC Experiments.

This module analyzes which specific problems are solved at which temperatures,
revealing per-problem patterns hidden by averaging in difficulty-stratified analysis.

Key Features:
- Per-problem temperature performance tracking
- Problem classification by temperature sensitivity patterns
- Cross-approach comparison (BoN, DVTS, Beam Search)
- Temperature Success Matrix Heatmap showing individual problem patterns
- Comprehensive visualizations and reporting

Usage:
    python exp/temperature_analysis_per_problem.py
"""

import os
import re
import json
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from math_verify import parse, verify

# Import temperature utilities
import sys
sys.path.insert(0, os.path.dirname(__file__))
from temperature_utils import (
    infer_temperature_from_position,
    validate_temperature_config,
)


# ======================================================================================
# DATA STRUCTURES
# ======================================================================================

@dataclass
class ProblemTemperatureProfile:
    """Temperature performance profile for a single problem (aggregated across seeds)."""
    unique_id: str
    problem_text: str
    answer: str
    difficulty_level: int  # From baseline (1-5)

    # Per-temperature results (aggregated across seeds)
    accuracy_by_temp: dict[float, float] = field(default_factory=dict)  # temp → accuracy
    correct_count_by_temp: dict[float, int] = field(default_factory=dict)  # temp → correct
    total_count_by_temp: dict[float, int] = field(default_factory=dict)  # temp → total
    best_prm_score_by_temp: dict[float, float] = field(default_factory=dict)  # temp → score

    # Per-seed details (for variance analysis)
    accuracy_by_temp_by_seed: dict[int, dict[float, float]] = field(default_factory=dict)

    # Classification
    solvable_temps: list[float] = field(default_factory=list)
    best_temp: Optional[float] = None
    best_temp_accuracy: float = 0.0

    # Sensitivity metrics
    temp_sensitivity_score: float = 0.0  # Std dev of accuracies
    temp_robustness: int = 0  # Number of temps where solved

    # Categories
    primary_category: str = ''  # 'unsolvable', 'single-temp', 'multi-temp'
    single_temp_category: Optional[str] = None  # '0.4-only', etc.
    best_temp_category: Optional[str] = None  # '0.4-best', etc.
    robustness_level: Optional[str] = None  # 'robust', 'moderate', 'fragile'

    # Pattern analysis
    monotonicity: str = ''  # 'increasing', 'decreasing', 'non-monotonic'
    is_extreme_dependent: bool = False  # Only solved at 0.4 or 1.6
    is_middle_optimal: bool = False  # Best at 0.8 or 1.2


@dataclass
class TemperatureSensitivityAnalysis:
    """Aggregated analysis across all problems for one approach."""
    approach: str
    seeds: list[int]
    temperatures: list[float]

    # All problem profiles
    problems: list[ProblemTemperatureProfile] = field(default_factory=list)

    # Category counts
    primary_category_counts: dict[str, int] = field(default_factory=dict)
    single_temp_counts: dict[str, int] = field(default_factory=dict)
    best_temp_counts: dict[str, int] = field(default_factory=dict)
    robustness_counts: dict[str, int] = field(default_factory=dict)
    monotonicity_counts: dict[str, int] = field(default_factory=dict)

    # Per-temperature statistics
    unique_solves_per_temp: dict[float, int] = field(default_factory=dict)
    total_solves_per_temp: dict[float, int] = field(default_factory=dict)
    exclusive_solves_per_temp: dict[float, set] = field(default_factory=dict)

    # Difficulty correlations
    difficulty_temp_correlation: float = 0.0

    # Per-difficulty breakdowns
    category_by_difficulty: dict[int, dict[str, int]] = field(default_factory=dict)


@dataclass
class MultiApproachComparison:
    """Comparison across all three approaches."""
    approaches: list[str]
    analyses: dict[str, TemperatureSensitivityAnalysis] = field(default_factory=dict)

    # Cross-approach consistency
    consistent_problems: list[str] = field(default_factory=list)
    inconsistent_problems: list[str] = field(default_factory=list)

    # Approach-specific advantages
    unique_solves_by_approach: dict[str, set] = field(default_factory=dict)


# ======================================================================================
# HELPER FUNCTIONS (reused from temperature_analysis_stratified.py)
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
) -> dict[str, 'ProblemBaseline']:
    """
    Compute baseline difficulty for each problem.

    Args:
        datasets: {approach: {seed: Dataset}}
        aggregate_across_approaches: If True, average across all approaches

    Returns:
        {unique_id: ProblemBaseline}
    """
    from dataclasses import dataclass as dc

    @dc
    class ProblemBaseline:
        unique_id: str
        problem_text: str
        answer: str
        mean_accuracy: float
        num_evaluations: int

    print("\n  Computing problem baselines...")

    problem_evaluations = defaultdict(list)
    problem_metadata = {}

    for approach, seeds_data in datasets.items():
        for seed, dataset in seeds_data.items():
            if 'train' in dataset:
                dataset = dataset['train']

            for problem in tqdm(dataset, desc=f"    Evaluating {approach} seed {seed}", leave=False):
                unique_id = problem.get('unique_id', problem.get('problem', ''))
                if not unique_id:
                    continue

                if unique_id not in problem_metadata:
                    problem_metadata[unique_id] = {
                        'problem_text': problem.get('problem', ''),
                        'answer': problem['answer']
                    }

                completions = problem['completions']
                gold_answer = problem['answer']

                for completion in completions:
                    is_correct = evaluate_answer(completion, gold_answer)
                    key = unique_id if aggregate_across_approaches else f"{approach}:{unique_id}"
                    problem_evaluations[key].append(is_correct)

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
    baselines: dict,
    num_levels: int = 5
) -> dict[int, 'DifficultyLevel']:
    """
    Stratify problems into difficulty quintiles.

    Args:
        baselines: {unique_id: ProblemBaseline}
        num_levels: Number of difficulty levels (default 5)

    Returns:
        {level: DifficultyLevel}
    """
    from dataclasses import dataclass as dc

    @dc
    class DifficultyLevel:
        level: int
        min_accuracy: float
        max_accuracy: float
        problem_count: int
        problem_ids: list[str]

    print("\n  Stratifying by difficulty...")

    accuracies = [b.mean_accuracy for b in baselines.values()]
    problem_ids = list(baselines.keys())

    quintiles = np.percentile(accuracies, np.linspace(0, 100, num_levels + 1))

    levels = {}
    for level_idx in range(num_levels):
        level = num_levels - level_idx

        min_acc = quintiles[level_idx]
        max_acc = quintiles[level_idx + 1]

        if level_idx == num_levels - 1:
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


# ======================================================================================
# CORE ANALYSIS FUNCTIONS
# ======================================================================================

def analyze_problem_temperature_profile_single_seed(
    problem: dict,
    approach: str,
    temperatures: list[float],
    n: int = 64,
    beam_width: int = 16
) -> dict[float, dict]:
    """
    Analyze temperature performance for a single problem from one seed.

    Args:
        problem: Problem dict with completions and answer
        approach: 'bon', 'beam_search', or 'dvts'
        temperatures: [0.4, 0.8, 1.2, 1.6]
        n: Total samples
        beam_width: Beam width for DVTS

    Returns:
        {temp: {'correct': count, 'total': count, 'scores': [...]}}
    """
    completions = problem['completions']
    gold_answer = problem['answer']

    # Track results by temperature
    temp_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'scores': []})

    for position, completion in enumerate(completions):
        # Infer temperature from position
        temp = infer_temperature_from_position(
            position, approach, n=n, temperatures=temperatures, beam_width=beam_width
        )

        # Evaluate completion
        is_correct = evaluate_answer(completion, gold_answer)

        # Record results
        temp_results[temp]['correct'] += int(is_correct)
        temp_results[temp]['total'] += 1

        # Store PRM score if available
        if 'scores' in problem and position < len(problem['scores']):
            score = problem['scores'][position]
            if isinstance(score, list) and len(score) > 0:
                temp_results[temp]['scores'].append(score[-1])  # Last step score

    return dict(temp_results)


def aggregate_problem_across_seeds(
    problem_id: str,
    results_by_seed: dict[int, dict[float, dict]],
    temperatures: list[float]
) -> tuple[dict, dict, dict, dict]:
    """
    Aggregate problem results across seeds.

    Args:
        problem_id: Problem unique ID
        results_by_seed: {seed: {temp: {'correct': ..., 'total': ...}}}
        temperatures: [0.4, 0.8, 1.2, 1.6]

    Returns:
        (accuracy_by_temp, correct_by_temp, total_by_temp, accuracy_by_temp_by_seed)
    """
    accuracy_by_temp = {}
    correct_by_temp = defaultdict(int)
    total_by_temp = defaultdict(int)
    accuracy_by_temp_by_seed = {}

    # Aggregate across seeds
    for seed, temp_results in results_by_seed.items():
        accuracy_by_temp_by_seed[seed] = {}

        for temp in temperatures:
            if temp in temp_results:
                correct = temp_results[temp]['correct']
                total = temp_results[temp]['total']

                correct_by_temp[temp] += correct
                total_by_temp[temp] += total

                accuracy_by_temp_by_seed[seed][temp] = correct / total if total > 0 else 0.0

    # Compute mean accuracy
    for temp in temperatures:
        total = total_by_temp[temp]
        accuracy_by_temp[temp] = correct_by_temp[temp] / total if total > 0 else 0.0

    return accuracy_by_temp, dict(correct_by_temp), dict(total_by_temp), accuracy_by_temp_by_seed


# ======================================================================================
# CLASSIFICATION FUNCTIONS
# ======================================================================================

def classify_primary_category(solvable_temps: list[float]) -> str:
    """Classify into unsolvable, single-temp, or multi-temp."""
    if len(solvable_temps) == 0:
        return 'unsolvable'
    elif len(solvable_temps) == 1:
        return 'single-temp'
    else:
        return 'multi-temp'


def classify_single_temp_category(solvable_temps: list[float]) -> Optional[str]:
    """Classify single-temp problem."""
    if len(solvable_temps) == 1:
        temp = solvable_temps[0]
        return f'{temp:.1f}-only'
    return None


def classify_best_temp_category(best_temp: Optional[float]) -> Optional[str]:
    """Classify by best temperature."""
    if best_temp is not None:
        return f'{best_temp:.1f}-best'
    return None


def classify_robustness_level(
    accuracy_by_temp: dict[float, float],
    solvable_temps: list[float]
) -> Optional[str]:
    """Classify robustness level."""
    if len(solvable_temps) < 2:
        return None

    solvable_accs = [accuracy_by_temp[t] for t in solvable_temps]
    std = np.std(solvable_accs) if len(solvable_accs) > 1 else 0.0

    if len(solvable_temps) >= 3 and std < 0.1:
        return 'robust'
    elif len(solvable_temps) == 2 and abs(solvable_accs[0] - solvable_accs[1]) > 0.3:
        return 'fragile'
    else:
        return 'moderate'


def classify_monotonicity(
    accuracy_by_temp: dict[float, float],
    temperatures: list[float]
) -> str:
    """Classify monotonicity pattern."""
    accs = [accuracy_by_temp.get(t, 0.0) for t in sorted(temperatures)]

    # Check increasing
    is_increasing = all(accs[i] <= accs[i+1] for i in range(len(accs)-1))
    if is_increasing:
        return 'increasing'

    # Check decreasing
    is_decreasing = all(accs[i] >= accs[i+1] for i in range(len(accs)-1))
    if is_decreasing:
        return 'decreasing'

    return 'non-monotonic'


def build_problem_temperature_profile(
    unique_id: str,
    problem_text: str,
    answer: str,
    difficulty_level: int,
    accuracy_by_temp: dict[float, float],
    correct_by_temp: dict[float, int],
    total_by_temp: dict[float, int],
    accuracy_by_temp_by_seed: dict[int, dict[float, float]],
    temperatures: list[float]
) -> ProblemTemperatureProfile:
    """
    Build complete problem temperature profile with all classifications.

    Args:
        unique_id: Problem ID
        problem_text: Problem text
        answer: Gold answer
        difficulty_level: Difficulty level (1-5)
        accuracy_by_temp: {temp: accuracy}
        correct_by_temp: {temp: correct_count}
        total_by_temp: {temp: total_count}
        accuracy_by_temp_by_seed: {seed: {temp: accuracy}}
        temperatures: [0.4, 0.8, 1.2, 1.6]

    Returns:
        ProblemTemperatureProfile
    """
    # Identify solvable temperatures
    solvable_temps = [t for t in temperatures if accuracy_by_temp.get(t, 0.0) > 0]

    # Find best temperature
    if solvable_temps:
        best_temp = max(solvable_temps, key=lambda t: accuracy_by_temp[t])
        best_temp_accuracy = accuracy_by_temp[best_temp]
    else:
        best_temp = None
        best_temp_accuracy = 0.0

    # Compute sensitivity metrics
    accs = [accuracy_by_temp.get(t, 0.0) for t in temperatures]
    temp_sensitivity_score = float(np.std(accs))
    temp_robustness = len(solvable_temps)

    # Classify
    primary_category = classify_primary_category(solvable_temps)
    single_temp_category = classify_single_temp_category(solvable_temps)
    best_temp_category = classify_best_temp_category(best_temp) if primary_category == 'multi-temp' else None
    robustness_level = classify_robustness_level(accuracy_by_temp, solvable_temps)
    monotonicity = classify_monotonicity(accuracy_by_temp, temperatures)

    # Pattern flags
    is_extreme_dependent = (
        set(solvable_temps) <= {0.4, 1.6} and len(solvable_temps) > 0
    )
    is_middle_optimal = best_temp in [0.8, 1.2] if best_temp else False

    return ProblemTemperatureProfile(
        unique_id=unique_id,
        problem_text=problem_text,
        answer=answer,
        difficulty_level=difficulty_level,
        accuracy_by_temp=accuracy_by_temp,
        correct_count_by_temp=correct_by_temp,
        total_count_by_temp=total_by_temp,
        accuracy_by_temp_by_seed=accuracy_by_temp_by_seed,
        solvable_temps=solvable_temps,
        best_temp=best_temp,
        best_temp_accuracy=best_temp_accuracy,
        temp_sensitivity_score=temp_sensitivity_score,
        temp_robustness=temp_robustness,
        primary_category=primary_category,
        single_temp_category=single_temp_category,
        best_temp_category=best_temp_category,
        robustness_level=robustness_level,
        monotonicity=monotonicity,
        is_extreme_dependent=is_extreme_dependent,
        is_middle_optimal=is_middle_optimal
    )


# ======================================================================================
# AGGREGATION AND COMPARISON FUNCTIONS
# ======================================================================================

def aggregate_sensitivity_analysis(
    problems: list[ProblemTemperatureProfile],
    approach: str,
    seeds: list[int],
    temperatures: list[float]
) -> TemperatureSensitivityAnalysis:
    """
    Aggregate analysis across all problems for one approach.

    Args:
        problems: List of problem profiles
        approach: 'bon', 'dvts', or 'beam_search'
        seeds: Seeds analyzed
        temperatures: Temperature values

    Returns:
        TemperatureSensitivityAnalysis
    """
    # Count categories
    primary_category_counts = defaultdict(int)
    single_temp_counts = defaultdict(int)
    best_temp_counts = defaultdict(int)
    robustness_counts = defaultdict(int)
    monotonicity_counts = defaultdict(int)

    for p in problems:
        primary_category_counts[p.primary_category] += 1

        if p.single_temp_category:
            single_temp_counts[p.single_temp_category] += 1

        if p.best_temp_category:
            best_temp_counts[p.best_temp_category] += 1

        if p.robustness_level:
            robustness_counts[p.robustness_level] += 1

        monotonicity_counts[p.monotonicity] += 1

    # Per-temperature statistics
    total_solves_per_temp = defaultdict(int)
    exclusive_solves_per_temp = defaultdict(set)

    for p in problems:
        for temp in p.solvable_temps:
            total_solves_per_temp[temp] += 1

        # Exclusive: solved ONLY at this temperature
        if len(p.solvable_temps) == 1:
            exclusive_solves_per_temp[p.solvable_temps[0]].add(p.unique_id)

    unique_solves_per_temp = {t: len(ids) for t, ids in exclusive_solves_per_temp.items()}

    # Difficulty-temperature correlation
    valid_problems = [p for p in problems if p.best_temp is not None]
    if len(valid_problems) > 1:
        difficulties = [p.difficulty_level for p in valid_problems]
        best_temps = [p.best_temp for p in valid_problems]
        correlation, _ = pearsonr(difficulties, best_temps)
    else:
        correlation = 0.0

    # Per-difficulty breakdowns
    category_by_difficulty = defaultdict(lambda: defaultdict(int))
    for p in problems:
        category_by_difficulty[p.difficulty_level][p.primary_category] += 1

    return TemperatureSensitivityAnalysis(
        approach=approach,
        seeds=seeds,
        temperatures=temperatures,
        problems=problems,
        primary_category_counts=dict(primary_category_counts),
        single_temp_counts=dict(single_temp_counts),
        best_temp_counts=dict(best_temp_counts),
        robustness_counts=dict(robustness_counts),
        monotonicity_counts=dict(monotonicity_counts),
        unique_solves_per_temp=unique_solves_per_temp,
        total_solves_per_temp=dict(total_solves_per_temp),
        exclusive_solves_per_temp=dict(exclusive_solves_per_temp),
        difficulty_temp_correlation=correlation,
        category_by_difficulty=dict(category_by_difficulty)
    )


def compare_approaches(
    analyses: dict[str, TemperatureSensitivityAnalysis]
) -> MultiApproachComparison:
    """
    Compare temperature sensitivity across approaches.

    Args:
        analyses: {approach: TemperatureSensitivityAnalysis}

    Returns:
        MultiApproachComparison
    """
    approaches = list(analyses.keys())

    # Build problem maps: {unique_id: {approach: ProblemTemperatureProfile}}
    problem_map = defaultdict(dict)
    for approach, analysis in analyses.items():
        for problem in analysis.problems:
            problem_map[problem.unique_id][approach] = problem

    # Find problems common to all approaches
    common_problem_ids = [
        pid for pid, probs in problem_map.items()
        if len(probs) == len(approaches)
    ]

    # Check consistency
    consistent_problems = []
    inconsistent_problems = []

    for pid in common_problem_ids:
        best_temps = [problem_map[pid][app].best_temp for app in approaches]
        best_temps_set = set(bt for bt in best_temps if bt is not None)

        if len(best_temps_set) == 1:
            consistent_problems.append(pid)
        else:
            inconsistent_problems.append(pid)

    # Unique solves by approach
    unique_solves_by_approach = {}
    for approach, analysis in analyses.items():
        solved_by_this = {p.unique_id for p in analysis.problems if p.primary_category != 'unsolvable'}

        # Find problems solved ONLY by this approach
        solved_by_others = set()
        for other_approach, other_analysis in analyses.items():
            if other_approach != approach:
                solved_by_others.update({
                    p.unique_id for p in other_analysis.problems
                    if p.primary_category != 'unsolvable'
                })

        unique_to_this = solved_by_this - solved_by_others
        unique_solves_by_approach[approach] = unique_to_this

    return MultiApproachComparison(
        approaches=approaches,
        analyses=analyses,
        consistent_problems=consistent_problems,
        inconsistent_problems=inconsistent_problems,
        unique_solves_by_approach=unique_solves_by_approach
    )


# ======================================================================================
# VISUALIZATION FUNCTIONS
# ======================================================================================

def plot_temperature_success_matrix_heatmap(
    problems: list[ProblemTemperatureProfile],
    temperatures: list[float],
    approach: str,
    output_path: str
):
    """
    Critical visualization: Per-problem temperature success matrix.
    Rows=problems, Columns=temperatures, Values=accuracy.
    """
    print(f"      Generating temperature success matrix heatmap...")

    # Sort problems by best_temp, then by difficulty
    problems_sorted = sorted(
        problems,
        key=lambda p: (
            p.best_temp if p.best_temp is not None else -1,
            p.difficulty_level
        )
    )

    # Build matrix
    matrix = []
    best_temps = []
    for p in problems_sorted:
        row = [p.accuracy_by_temp.get(t, 0.0) for t in temperatures]
        matrix.append(row)
        best_temps.append(p.best_temp)

    matrix = np.array(matrix)

    # Create larger figure with better visibility
    fig_height = max(16, min(60, len(problems_sorted) * 0.08))
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Use better colormap: white (0) -> red (1) for better visibility
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#ffffff', '#ffe6e6', '#ffcccc', '#ff9999', '#ff6666', '#ff3333', '#ff0000']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom_red', colors, N=n_bins)

    # Heatmap with cell borders for better separation
    sns.heatmap(
        matrix,
        cmap=cmap,
        vmin=0, vmax=1,
        xticklabels=[f'T={t:.1f}' for t in temperatures],
        yticklabels=False,  # Too many problems
        ax=ax,
        cbar_kws={'label': 'Accuracy', 'shrink': 0.8},
        linewidths=0.5,
        linecolor='lightgray',
        square=False
    )

    # Add temperature group separators
    # Find boundaries where best_temp changes
    boundaries = []
    prev_temp = None
    for i, temp in enumerate(best_temps):
        if temp != prev_temp and i > 0:
            boundaries.append(i)
        prev_temp = temp

    # Draw horizontal lines at boundaries
    for boundary in boundaries:
        ax.axhline(y=boundary, color='black', linewidth=2, alpha=0.7)

    # Highlight each temperature column with subtle background
    for i, temp in enumerate(temperatures):
        ax.add_patch(plt.Rectangle((i, 0), 1, len(problems_sorted),
                                   fill=False, edgecolor='blue', linewidth=2, alpha=0.3))

    ax.set_xlabel('Temperature', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Problems (n={len(problems_sorted)}, sorted by best temperature)', fontsize=12)
    ax.set_title(
        f'{approach.upper()} - Per-Problem Temperature Success Matrix\n'
        f'White=0% accuracy, Red=100% accuracy | Black lines separate temperature preference groups',
        fontsize=14, fontweight='bold', pad=20
    )

    # Improve x-axis tick labels
    ax.tick_params(axis='x', labelsize=12, rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close()


def plot_category_distribution_by_approach(
    analyses: dict[str, TemperatureSensitivityAnalysis],
    output_path: str
):
    """Stacked bar chart showing category distribution per approach."""
    print(f"      Generating category distribution plot...")

    approaches = list(analyses.keys())
    categories = ['unsolvable', 'single-temp', 'multi-temp']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Build data
    data = {cat: [] for cat in categories}
    for approach in approaches:
        counts = analyses[approach].primary_category_counts
        total = len(analyses[approach].problems)
        for cat in categories:
            pct = (counts.get(cat, 0) / total * 100) if total > 0 else 0
            data[cat].append(pct)

    # Stacked bars
    x = np.arange(len(approaches))
    bottom = np.zeros(len(approaches))

    colors = {'unsolvable': '#d62728', 'single-temp': '#ff7f0e', 'multi-temp': '#2ca02c'}

    for cat in categories:
        ax.bar(x, data[cat], label=cat, bottom=bottom, color=colors.get(cat, 'gray'))
        # Add percentage labels
        for i, val in enumerate(data[cat]):
            if val > 3:  # Only show if > 3%
                ax.text(i, bottom[i] + val/2, f'{val:.1f}%',
                       ha='center', va='center', fontsize=10, color='white', fontweight='bold')
        bottom += data[cat]

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in approaches])
    ax.set_ylabel('Percentage of Problems', fontsize=12)
    ax.set_title('Problem Category Distribution by Approach', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_temperature_preference_distribution(
    analyses: dict[str, TemperatureSensitivityAnalysis],
    temperatures: list[float],
    output_path: str
):
    """Bar chart showing which temperature is best for multi-temp problems."""
    print(f"      Generating temperature preference plot...")

    fig, axes = plt.subplots(1, len(analyses), figsize=(15, 5))
    if len(analyses) == 1:
        axes = [axes]

    for ax, (approach, analysis) in zip(axes, analyses.items()):
        counts = analysis.best_temp_counts

        temps_str = [f'{t:.1f}' for t in temperatures]
        values = [counts.get(f'{t:.1f}-best', 0) for t in temperatures]

        bars = ax.bar(temps_str, values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(temperatures))))

        ax.set_xlabel('Best Temperature', fontsize=11)
        ax.set_ylabel('Count of Problems', fontsize=11)
        ax.set_title(f'{approach.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height,
                       f'{int(height)}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Temperature Preference Distribution (Multi-Temp Problems)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_difficulty_vs_optimal_temperature(
    analyses: dict[str, TemperatureSensitivityAnalysis],
    output_path: str
):
    """Scatter plot: difficulty vs optimal temperature."""
    print(f"      Generating difficulty vs temperature scatter...")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'bon': '#1f77b4', 'dvts': '#ff7f0e', 'beam_search': '#2ca02c'}

    for approach, analysis in analyses.items():
        # Extract data
        difficulties = []
        best_temps = []
        robustness = []

        for p in analysis.problems:
            if p.best_temp is not None:
                difficulties.append(p.difficulty_level)
                best_temps.append(p.best_temp)
                robustness.append(p.temp_robustness * 20)  # Scale for size

        # Add jitter for visibility
        jittered_temps = np.array(best_temps) + np.random.normal(0, 0.02, len(best_temps))

        ax.scatter(
            difficulties, jittered_temps, s=robustness, alpha=0.5,
            label=f'{approach.upper()} (r={analysis.difficulty_temp_correlation:.3f})',
            color=colors.get(approach, 'gray')
        )

    ax.set_xlabel('Difficulty Level (1=Easy, 5=Hard)', fontsize=12)
    ax.set_ylabel('Best Temperature', fontsize=12)
    ax.set_title('Problem Difficulty vs Optimal Temperature\n(Point size = Robustness)', fontsize=14, fontweight='bold')
    ax.set_yticks([0.4, 0.8, 1.2, 1.6])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_temperature_exclusive_counts(
    analyses: dict[str, TemperatureSensitivityAnalysis],
    temperatures: list[float],
    output_path: str
):
    """Bar chart showing unique problems solved per temperature."""
    print(f"      Generating temperature exclusive counts plot...")

    fig, axes = plt.subplots(1, len(analyses), figsize=(15, 5))
    if len(analyses) == 1:
        axes = [axes]

    for ax, (approach, analysis) in zip(axes, analyses.items()):
        temps_str = [f'{t:.1f}' for t in temperatures]
        unique_counts = [analysis.unique_solves_per_temp.get(t, 0) for t in temperatures]
        total_counts = [analysis.total_solves_per_temp.get(t, 0) for t in temperatures]

        x = np.arange(len(temps_str))
        width = 0.35

        ax.bar(x - width/2, total_counts, width, label='Total Solves', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, unique_counts, width, label='Unique Solves', color='darkgreen', alpha=0.7)

        ax.set_xlabel('Temperature', fontsize=11)
        ax.set_ylabel('Count of Problems', fontsize=11)
        ax.set_title(f'{approach.upper()}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(temps_str)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Temperature Problem Coverage\n(Unique = solved ONLY at this temp)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_distribution_by_temperature(
    analysis: TemperatureSensitivityAnalysis,
    temperatures: list[float],
    output_path: str
):
    """Box plot showing accuracy distribution per temperature."""
    print(f"      Generating accuracy distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect accuracies
    data = []
    labels = []

    for temp in temperatures:
        accs = [
            p.accuracy_by_temp.get(temp, 0.0)
            for p in analysis.problems
            if p.accuracy_by_temp.get(temp, 0.0) > 0  # Only problems solved at this temp
        ]
        if accs:
            data.append(accs)
            labels.append(f'{temp:.1f}\n(n={len(accs)})')

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax.set_xlabel('Temperature', fontsize=12)
    ax.set_ylabel('Accuracy (for solved problems)', fontsize=12)
    ax.set_title(f'{analysis.approach.upper()} - Accuracy Distribution by Temperature', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_monotonicity_analysis(
    analyses: dict[str, TemperatureSensitivityAnalysis],
    output_path: str
):
    """Stacked bar showing monotonicity patterns."""
    print(f"      Generating monotonicity analysis plot...")

    fig, ax = plt.subplots(figsize=(10, 6))

    approaches = list(analyses.keys())
    patterns = ['increasing', 'decreasing', 'non-monotonic']

    data = {pattern: [] for pattern in patterns}
    for approach in approaches:
        counts = analyses[approach].monotonicity_counts
        total = len(analyses[approach].problems)
        for pattern in patterns:
            pct = (counts.get(pattern, 0) / total * 100) if total > 0 else 0
            data[pattern].append(pct)

    x = np.arange(len(approaches))
    bottom = np.zeros(len(approaches))

    colors = {'increasing': '#2ca02c', 'decreasing': '#d62728', 'non-monotonic': '#ff7f0e'}

    for pattern in patterns:
        ax.bar(x, data[pattern], label=pattern, bottom=bottom, color=colors.get(pattern, 'gray'))
        bottom += data[pattern]

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in approaches])
    ax.set_ylabel('Percentage of Problems', fontsize=12)
    ax.set_title('Temperature-Accuracy Monotonicity Patterns', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# ======================================================================================
# REPORT GENERATION
# ======================================================================================

def generate_markdown_report(
    analyses: dict[str, TemperatureSensitivityAnalysis],
    comparison: MultiApproachComparison,
    hnc_seeds: list[int],
    default_seeds: list[int],
    temperatures: list[float],
    output_path: str
):
    """
    Generate comprehensive markdown report.

    Args:
        analyses: {approach: TemperatureSensitivityAnalysis}
        comparison: MultiApproachComparison
        hnc_seeds: HNC seeds used
        default_seeds: Default seeds used
        temperatures: Temperature values
        output_path: Path to save report
    """
    print("      Generating markdown report...")

    md_lines = ["# Per-Problem Temperature Sensitivity Analysis\n\n"]

    # Metadata
    md_lines.append("## Metadata\n\n")
    md_lines.append(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append(f"- **Baseline**: Default datasets (seeds {default_seeds}, single temp=0.8)\n")
    md_lines.append(f"- **HNC Analysis**: Multi-temp datasets (seeds {hnc_seeds}, temps {temperatures})\n")
    md_lines.append(f"- **Approaches**: {', '.join(analyses.keys())}\n")
    md_lines.append(f"- **Total Problems**: {len(analyses[list(analyses.keys())[0]].problems)}\n\n")

    # Summary statistics
    md_lines.append("## Summary Statistics\n\n")
    for approach in sorted(analyses.keys()):
        analysis = analyses[approach]
        md_lines.append(f"### {approach.upper()}\n\n")

        # Primary categories
        md_lines.append("**Category Distribution:**\n\n")
        md_lines.append("| Category | Count | Percentage |\n")
        md_lines.append("|----------|-------|------------|\n")
        total = len(analysis.problems)
        for cat in ['unsolvable', 'single-temp', 'multi-temp']:
            count = analysis.primary_category_counts.get(cat, 0)
            pct = (count / total * 100) if total > 0 else 0
            md_lines.append(f"| {cat} | {count} | {pct:.1f}% |\n")
        md_lines.append("\n")

        # Temperature preferences
        if analysis.best_temp_counts:
            md_lines.append("**Temperature Preferences (Multi-Temp Problems):**\n\n")
            md_lines.append("| Best Temperature | Count |\n")
            md_lines.append("|------------------|-------|\n")
            for temp in temperatures:
                cat = f'{temp:.1f}-best'
                count = analysis.best_temp_counts.get(cat, 0)
                md_lines.append(f"| {temp:.1f} | {count} |\n")
            md_lines.append("\n")

        # Temperature coverage
        md_lines.append("**Temperature Coverage:**\n\n")
        md_lines.append("| Temperature | Total Solves | Unique Solves |\n")
        md_lines.append("|-------------|--------------|---------------|\n")
        for temp in temperatures:
            total_solves = analysis.total_solves_per_temp.get(temp, 0)
            unique_solves = analysis.unique_solves_per_temp.get(temp, 0)
            md_lines.append(f"| {temp:.1f} | {total_solves} | {unique_solves} |\n")
        md_lines.append("\n")

        # Difficulty correlation
        md_lines.append(f"**Difficulty-Temperature Correlation:** {analysis.difficulty_temp_correlation:.4f}\n\n")

    # Cross-approach comparison
    md_lines.append("## Cross-Approach Comparison\n\n")
    md_lines.append(f"- **Consistent Problems**: {len(comparison.consistent_problems)} problems have same best temperature across approaches\n")
    md_lines.append(f"- **Inconsistent Problems**: {len(comparison.inconsistent_problems)} problems have different best temperatures\n\n")

    md_lines.append("**Approach-Specific Advantages:**\n\n")
    for approach in sorted(comparison.approaches):
        unique_count = len(comparison.unique_solves_by_approach.get(approach, set()))
        md_lines.append(f"- **{approach.upper()}**: {unique_count} problems solved uniquely by this approach\n")
    md_lines.append("\n")

    # Example problems
    md_lines.append("## Example Problems\n\n")
    for approach in sorted(analyses.keys()):
        analysis = analyses[approach]
        md_lines.append(f"### {approach.upper()}\n\n")

        # Get examples for each category
        categories_to_show = {
            'single-temp': [p for p in analysis.problems if p.primary_category == 'single-temp'][:3],
            'multi-temp-robust': [p for p in analysis.problems if p.robustness_level == 'robust'][:3],
            'unsolvable': [p for p in analysis.problems if p.primary_category == 'unsolvable'][:2]
        }

        for cat_name, examples in categories_to_show.items():
            if examples:
                md_lines.append(f"**{cat_name}:**\n\n")
                for p in examples:
                    problem_text = p.problem_text[:100] + "..." if len(p.problem_text) > 100 else p.problem_text
                    md_lines.append(f"- Problem: {problem_text}\n")
                    md_lines.append(f"  - Accuracies: {', '.join([f'{t:.1f}={p.accuracy_by_temp.get(t, 0.0):.2f}' for t in temperatures])}\n")
                    best_temp_str = f"{p.best_temp:.1f}" if p.best_temp else "N/A"
                    category_str = p.single_temp_category or p.best_temp_category or "N/A"
                    md_lines.append(f"  - Best: {best_temp_str}, Category: {category_str}\n")
                md_lines.append("\n")

    # Key findings
    md_lines.append("## Key Findings\n\n")
    md_lines.append("### Temperature Allocation Efficiency\n\n")
    for approach in sorted(analyses.keys()):
        analysis = analyses[approach]
        md_lines.append(f"**{approach.upper()}:**\n")
        for temp in temperatures:
            unique = analysis.unique_solves_per_temp.get(temp, 0)
            total = analysis.total_solves_per_temp.get(temp, 0)
            if total > 0:
                md_lines.append(f"- Temperature {temp:.1f}: {unique} unique solves / {total} total solves ({unique/total*100:.1f}% unique)\n")
        md_lines.append("\n")

    # Recommendations
    md_lines.append("## Recommendations\n\n")
    md_lines.append("Based on the analysis:\n\n")
    md_lines.append("1. **Temperature Diversity**: ")
    avg_unique = np.mean([len(s) for s in comparison.unique_solves_by_approach.values()])
    if avg_unique > 10:
        md_lines.append("High approach-specific advantage suggests maintaining all three approaches.\n")
    else:
        md_lines.append("Low approach-specific advantage suggests approaches may be redundant.\n")

    md_lines.append("2. **Temperature Range**: ")
    for approach in analyses.keys():
        extreme_count = sum([analyses[approach].unique_solves_per_temp.get(0.4, 0),
                            analyses[approach].unique_solves_per_temp.get(1.6, 0)])
        if extreme_count > 5:
            md_lines.append(f"{approach.upper()} shows significant extreme temperature value ({extreme_count} unique problems). ")
    md_lines.append("\n")

    md_lines.append("3. **Difficulty-Aware Selection**: ")
    correlations = [a.difficulty_temp_correlation for a in analyses.values()]
    avg_corr = np.mean(correlations)
    if avg_corr > 0.1:
        md_lines.append(f"Positive correlation ({avg_corr:.3f}) suggests harder problems benefit from higher temperatures.\n")
    elif avg_corr < -0.1:
        md_lines.append(f"Negative correlation ({avg_corr:.3f}) suggests harder problems benefit from lower temperatures.\n")
    else:
        md_lines.append(f"Weak correlation ({avg_corr:.3f}) suggests difficulty does not strongly predict optimal temperature.\n")

    md_lines.append("\n")

    # Save report
    with open(output_path, 'w') as f:
        f.writelines(md_lines)


# ======================================================================================
# MAIN FUNCTION
# ======================================================================================

def main():
    """
    Main analysis flow.

    1. Load baseline datasets and establish difficulty levels
    2. Load HNC multi-temperature datasets
    3. For each approach and seed: analyze per-problem temperature performance
    4. Aggregate across seeds to build problem profiles
    5. Generate visualizations
    6. Generate comprehensive report
    """

    # Configuration
    default_seeds = [0, 42, 64]
    hnc_seeds = [128, 192, 256]
    temperatures = [0.4, 0.8, 1.2, 1.6]
    approaches = ['bon', 'dvts', 'beam_search']
    n = 64
    beam_width = 16

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

    output_dir = "exp/temperature_analysis_per_problem"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("PER-PROBLEM TEMPERATURE SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Phase 1: Baseline establishment
    print("\n" + "=" * 80)
    print("PHASE 1: ESTABLISHING BASELINE FROM DEFAULT DATASETS")
    print("=" * 80)

    default_datasets = load_default_datasets(
        approaches, default_seeds, dataset_paths['default']
    )

    baselines = compute_problem_baselines(
        default_datasets, aggregate_across_approaches=True
    )

    difficulty_levels = stratify_by_difficulty(baselines, num_levels=5)

    # Create difficulty map
    difficulty_map = {}
    for level, level_info in difficulty_levels.items():
        for problem_id in level_info.problem_ids:
            difficulty_map[problem_id] = level

    # Phase 2: Load HNC datasets
    print("\n" + "=" * 80)
    print("PHASE 2: LOADING HNC MULTI-TEMPERATURE DATASETS")
    print("=" * 80)

    hnc_datasets = load_hnc_datasets(
        approaches, hnc_seeds, dataset_paths['hnc']
    )

    # Phase 3: Per-problem analysis
    print("\n" + "=" * 80)
    print("PHASE 3: PER-PROBLEM TEMPERATURE ANALYSIS")
    print("=" * 80)

    all_analyses = {}

    for approach in approaches:
        print(f"\n  Analyzing {approach}...")

        # Track results by problem and seed
        problem_results_by_seed = defaultdict(lambda: defaultdict(dict))
        problem_metadata = {}

        # Analyze each seed
        for seed in hnc_seeds:
            if seed not in hnc_datasets[approach]:
                continue

            dataset = hnc_datasets[approach][seed]
            if 'train' in dataset:
                dataset = dataset['train']

            print(f"    Processing seed {seed}...")
            for problem in tqdm(dataset, desc=f"      Problems", leave=False):
                unique_id = problem.get('unique_id', problem.get('problem', ''))
                if not unique_id:
                    continue

                # Store metadata
                if unique_id not in problem_metadata:
                    problem_metadata[unique_id] = {
                        'problem_text': problem.get('problem', ''),
                        'answer': problem['answer'],
                        'difficulty_level': difficulty_map.get(unique_id, 3)  # Default to medium
                    }

                # Analyze this problem for this seed
                temp_stats = analyze_problem_temperature_profile_single_seed(
                    problem, approach, temperatures, n, beam_width
                )

                problem_results_by_seed[unique_id][seed] = temp_stats

        # Aggregate across seeds and build profiles
        print(f"    Building problem profiles...")
        problem_profiles = []

        for unique_id, results_by_seed in problem_results_by_seed.items():
            metadata = problem_metadata[unique_id]

            # Aggregate across seeds
            accuracy_by_temp, correct_by_temp, total_by_temp, accuracy_by_temp_by_seed = \
                aggregate_problem_across_seeds(unique_id, results_by_seed, temperatures)

            # Build profile
            profile = build_problem_temperature_profile(
                unique_id=unique_id,
                problem_text=metadata['problem_text'],
                answer=metadata['answer'],
                difficulty_level=metadata['difficulty_level'],
                accuracy_by_temp=accuracy_by_temp,
                correct_by_temp=correct_by_temp,
                total_by_temp=total_by_temp,
                accuracy_by_temp_by_seed=accuracy_by_temp_by_seed,
                temperatures=temperatures
            )

            problem_profiles.append(profile)

        # Aggregate analysis
        analysis = aggregate_sensitivity_analysis(
            problem_profiles, approach, hnc_seeds, temperatures
        )

        all_analyses[approach] = analysis

        print(f"    Analyzed {len(problem_profiles)} problems")

    # Phase 4: Cross-approach comparison
    print("\n" + "=" * 80)
    print("PHASE 4: CROSS-APPROACH COMPARISON")
    print("=" * 80)

    comparison = compare_approaches(all_analyses)

    print(f"  Consistent problems: {len(comparison.consistent_problems)}")
    print(f"  Inconsistent problems: {len(comparison.inconsistent_problems)}")

    # Phase 5: Visualization
    print("\n" + "=" * 80)
    print("PHASE 5: GENERATING VISUALIZATIONS")
    print("=" * 80)

    print("\n  Critical visualizations:")
    for approach, analysis in all_analyses.items():
        plot_temperature_success_matrix_heatmap(
            analysis.problems, temperatures, approach,
            os.path.join(output_dir, f'{approach}_temperature_success_matrix.png')
        )

    print("\n  Category and preference visualizations:")
    plot_category_distribution_by_approach(
        all_analyses,
        os.path.join(output_dir, 'category_distribution_by_approach.png')
    )

    plot_temperature_preference_distribution(
        all_analyses, temperatures,
        os.path.join(output_dir, 'temperature_preference_distribution.png')
    )

    print("\n  Analysis visualizations:")
    plot_difficulty_vs_optimal_temperature(
        all_analyses,
        os.path.join(output_dir, 'difficulty_vs_optimal_temperature.png')
    )

    plot_temperature_exclusive_counts(
        all_analyses, temperatures,
        os.path.join(output_dir, 'temperature_exclusive_counts.png')
    )

    plot_monotonicity_analysis(
        all_analyses,
        os.path.join(output_dir, 'monotonicity_analysis.png')
    )

    # Phase 6: Report generation
    print("\n" + "=" * 80)
    print("PHASE 6: GENERATING REPORT")
    print("=" * 80)

    generate_markdown_report(
        all_analyses, comparison, hnc_seeds, default_seeds, temperatures,
        os.path.join(output_dir, 'per_problem_analysis_report.md')
    )

    # Export problem profiles to JSON
    print("      Exporting problem profiles to JSON...")
    for approach, analysis in all_analyses.items():
        profiles_serializable = []
        for p in analysis.problems:
            profile_dict = asdict(p)
            # Convert sets to lists for JSON serialization
            for key, val in profile_dict.items():
                if isinstance(val, set):
                    profile_dict[key] = list(val)
            profiles_serializable.append(profile_dict)

        with open(os.path.join(output_dir, f'problem_profiles_{approach}.json'), 'w') as f:
            json.dump(profiles_serializable, f, indent=2)

    # Export comparison
    print("      Exporting cross-approach comparison...")
    comparison_dict = {
        'approaches': comparison.approaches,
        'consistent_problems': comparison.consistent_problems,
        'inconsistent_problems': comparison.inconsistent_problems,
        'unique_solves_by_approach': {k: list(v) for k, v in comparison.unique_solves_by_approach.items()}
    }
    with open(os.path.join(output_dir, 'multi_approach_comparison.json'), 'w') as f:
        json.dump(comparison_dict, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE! Results saved to {output_dir}/")
    print("=" * 80)

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    for approach in sorted(all_analyses.keys()):
        analysis = all_analyses[approach]
        print(f"\n{approach.upper()}:")
        print(f"  Total problems: {len(analysis.problems)}")
        print(f"  Unsolvable: {analysis.primary_category_counts.get('unsolvable', 0)}")
        print(f"  Single-temp: {analysis.primary_category_counts.get('single-temp', 0)}")
        print(f"  Multi-temp: {analysis.primary_category_counts.get('multi-temp', 0)}")
        print(f"  Difficulty-temperature correlation: {analysis.difficulty_temp_correlation:.4f}")


if __name__ == "__main__":
    main()

