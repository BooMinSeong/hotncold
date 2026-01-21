"""Difficulty stratification utilities for experiment analysis.

This module provides functions to compute problem difficulty baselines
and stratify problems into difficulty levels.

Extracted from:
- exp/temperature_analysis_per_problem.py (compute_problem_baselines, stratify_by_difficulty)
- exp/temperature_analysis_stratified.py (compute_problem_baselines, stratify_by_difficulty)
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from .core import evaluate_answer


@dataclass
class ProblemBaseline:
    """Baseline statistics for a single problem.

    Attributes:
        unique_id: Problem identifier
        problem_text: Original problem text
        answer: Gold answer
        mean_accuracy: Mean accuracy across all evaluations
        num_evaluations: Total number of completions evaluated
    """

    unique_id: str
    problem_text: str
    answer: str
    mean_accuracy: float
    num_evaluations: int


@dataclass
class DifficultyLevel:
    """Difficulty level definition.

    Attributes:
        level: Level number (1=easiest, 5=hardest by default)
        min_accuracy: Minimum accuracy threshold for this level
        max_accuracy: Maximum accuracy threshold for this level
        problem_count: Number of problems in this level
        problem_ids: List of problem identifiers in this level
    """

    level: int
    min_accuracy: float
    max_accuracy: float
    problem_count: int
    problem_ids: list[str]


def compute_problem_baselines(
    datasets: dict[str, dict[int, Any]],
    aggregate_across_approaches: bool = True,
    verbose: bool = True,
) -> dict[str, ProblemBaseline]:
    """Compute baseline difficulty for each problem.

    For each unique problem:
    1. Collect all completions across seeds and approaches
    2. Evaluate each completion with math_verify
    3. Compute mean accuracy

    Args:
        datasets: Nested dict {approach: {seed: Dataset}}
        aggregate_across_approaches: If True, average across all approaches.
                                     If False, keep separate per approach.
        verbose: Print progress messages

    Returns:
        Dict mapping problem key to ProblemBaseline.
        Key is unique_id if aggregating, or "{approach}:{unique_id}" otherwise.

    Example:
        >>> baselines = compute_problem_baselines(datasets)
        >>> for pid, baseline in baselines.items():
        ...     print(f"{pid}: {baseline.mean_accuracy:.3f} ({baseline.num_evaluations} evals)")
    """
    if verbose:
        print("\n  Computing problem baselines...")

    # Collect all evaluations per problem
    problem_evaluations = defaultdict(list)
    problem_metadata = {}  # Store problem text and answer

    for approach, seeds_data in datasets.items():
        for seed, dataset in seeds_data.items():
            if "train" in dataset:
                dataset = dataset["train"]

            iterator = tqdm(
                dataset,
                desc=f"    Evaluating {approach} seed {seed}",
                leave=False,
                disable=not verbose,
            )

            for problem in iterator:
                unique_id = problem.get("unique_id", problem.get("problem", ""))
                if not unique_id:
                    continue

                # Store metadata (only once per problem)
                if unique_id not in problem_metadata:
                    problem_metadata[unique_id] = {
                        "problem_text": problem.get("problem", ""),
                        "answer": problem["answer"],
                    }

                # Evaluate all completions
                completions = problem.get("completions", [])
                gold_answer = problem["answer"]

                for completion in completions:
                    is_correct = evaluate_answer(completion, gold_answer)
                    key = unique_id if aggregate_across_approaches else f"{approach}:{unique_id}"
                    problem_evaluations[key].append(is_correct)

    # Compute mean accuracy per problem
    baselines = {}
    for problem_key, evaluations in problem_evaluations.items():
        unique_id = problem_key.split(":")[-1] if ":" in problem_key else problem_key
        mean_acc = sum(evaluations) / len(evaluations) if evaluations else 0.0

        baselines[problem_key] = ProblemBaseline(
            unique_id=unique_id,
            problem_text=problem_metadata.get(unique_id, {}).get("problem_text", ""),
            answer=problem_metadata.get(unique_id, {}).get("answer", ""),
            mean_accuracy=mean_acc,
            num_evaluations=len(evaluations),
        )

    if verbose:
        print(f"  Computed baselines for {len(baselines)} problems")

    return baselines


def compute_problem_baselines_from_preds(
    datasets: dict[str, dict[int, Any]],
    aggregate_across_approaches: bool = True,
    verbose: bool = True,
) -> dict[str, ProblemBaseline]:
    """Compute baseline difficulty for each problem using preds field.

    Similar to compute_problem_baselines, but uses the 'preds' field directly
    instead of parsing completions. This is faster when predictions are pre-parsed.

    For each unique problem:
    1. Collect all predictions across seeds and approaches
    2. Evaluate each prediction with math_verify
    3. Compute mean accuracy

    Args:
        datasets: Nested dict {approach: {seed: Dataset}}
        aggregate_across_approaches: If True, average across all approaches.
                                     If False, keep separate per approach.
        verbose: Print progress messages

    Returns:
        Dict mapping problem key to ProblemBaseline.
        Key is unique_id if aggregating, or "{approach}:{unique_id}" otherwise.

    Example:
        >>> baselines = compute_problem_baselines_from_preds(datasets)
        >>> for pid, baseline in baselines.items():
        ...     print(f"{pid}: {baseline.mean_accuracy:.3f} ({baseline.num_evaluations} evals)")
    """
    from math_verify import parse, verify

    if verbose:
        print("\n  Computing problem baselines from preds...")

    # Collect all evaluations per problem
    problem_evaluations = defaultdict(list)
    problem_metadata = {}  # Store problem text and answer

    for approach, seeds_data in datasets.items():
        for seed, dataset in seeds_data.items():
            if "train" in dataset:
                dataset = dataset["train"]

            iterator = tqdm(
                dataset,
                desc=f"    Evaluating {approach} seed {seed}",
                leave=False,
                disable=not verbose,
            )

            for problem in iterator:
                unique_id = problem.get("unique_id", problem.get("problem", ""))
                if not unique_id:
                    continue

                # Store metadata (only once per problem)
                if unique_id not in problem_metadata:
                    problem_metadata[unique_id] = {
                        "problem_text": problem.get("problem", ""),
                        "answer": problem["answer"],
                    }

                # Parse gold answer once
                gold_answer = problem["answer"]
                try:
                    gold = parse("\\boxed{" + gold_answer + "}")
                except Exception:
                    # Skip problems with unparseable gold answers
                    continue

                # Evaluate all predictions
                preds = problem.get("preds", [])
                for pred in preds:
                    try:
                        pred_parsed = parse("\\boxed{" + pred + "}")
                        is_correct = verify(gold, pred_parsed)
                    except Exception:
                        is_correct = False

                    key = unique_id if aggregate_across_approaches else f"{approach}:{unique_id}"
                    problem_evaluations[key].append(is_correct)

    # Compute mean accuracy per problem
    baselines = {}
    for problem_key, evaluations in problem_evaluations.items():
        unique_id = problem_key.split(":")[-1] if ":" in problem_key else problem_key
        mean_acc = sum(evaluations) / len(evaluations) if evaluations else 0.0

        baselines[problem_key] = ProblemBaseline(
            unique_id=unique_id,
            problem_text=problem_metadata.get(unique_id, {}).get("problem_text", ""),
            answer=problem_metadata.get(unique_id, {}).get("answer", ""),
            mean_accuracy=mean_acc,
            num_evaluations=len(evaluations),
        )

    if verbose:
        print(f"  Computed baselines for {len(baselines)} problems")

    return baselines


def stratify_by_difficulty(
    baselines: dict[str, ProblemBaseline],
    num_levels: int = 5,
    verbose: bool = True,
) -> dict[int, DifficultyLevel]:
    """Stratify problems into difficulty levels based on accuracy percentiles.

    Problems are divided into quintiles (or specified number of levels) based
    on their baseline accuracy. Level 1 is easiest (highest accuracy),
    level N is hardest (lowest accuracy).

    Args:
        baselines: Dict mapping problem key to ProblemBaseline
        num_levels: Number of difficulty levels (default: 5 for quintiles)
        verbose: Print progress messages

    Returns:
        Dict mapping level number to DifficultyLevel

    Example:
        >>> levels = stratify_by_difficulty(baselines, num_levels=5)
        >>> for level, info in sorted(levels.items()):
        ...     print(f"Level {level}: {info.problem_count} problems")
    """
    if verbose:
        print("\n  Stratifying by difficulty...")

    # Extract accuracies
    accuracies = [b.mean_accuracy for b in baselines.values()]

    if not accuracies:
        if verbose:
            print("  Warning: No baselines to stratify")
        return {}

    # Compute percentile boundaries
    percentiles = np.linspace(0, 100, num_levels + 1)
    boundaries = np.percentile(accuracies, percentiles)

    # Assign problems to levels
    levels = {}
    for level_idx in range(num_levels):
        # Level 5 (hardest) = lowest accuracy (< Q1)
        # Level 1 (easiest) = highest accuracy (>= Q4)
        level = num_levels - level_idx  # Reverse: 5, 4, 3, 2, 1

        min_acc = boundaries[level_idx]
        max_acc = boundaries[level_idx + 1]

        # Find problems in this range
        if level_idx == num_levels - 1:
            # Last level: include max boundary
            level_problems = [
                pid
                for pid, b in baselines.items()
                if min_acc <= b.mean_accuracy <= max_acc
            ]
        else:
            level_problems = [
                pid
                for pid, b in baselines.items()
                if min_acc <= b.mean_accuracy < max_acc
            ]

        levels[level] = DifficultyLevel(
            level=level,
            min_accuracy=min_acc,
            max_accuracy=max_acc,
            problem_count=len(level_problems),
            problem_ids=level_problems,
        )

        if verbose:
            print(
                f"    Level {level}: {len(level_problems)} problems, "
                f"accuracy [{min_acc:.3f}, {max_acc:.3f}]"
            )

    return levels


def stratify_by_absolute_difficulty(
    baselines: dict[str, ProblemBaseline],
    thresholds: dict[int, tuple[float, float]],
    verbose: bool = True,
) -> dict[int, DifficultyLevel]:
    """Stratify problems into difficulty levels based on absolute accuracy thresholds.

    Unlike stratify_by_difficulty which uses percentiles, this function uses
    fixed accuracy ranges. This allows comparing difficulty across different
    datasets or experiments.

    Args:
        baselines: Dict mapping problem key to ProblemBaseline
        thresholds: Dict mapping level number to (min_accuracy, max_accuracy) tuple.
                   Example: {1: (0.8, 1.0), 2: (0.6, 0.8), ...}
                   Level 1 should be easiest (highest accuracy),
                   highest level number should be hardest (lowest accuracy).
        verbose: Print progress messages

    Returns:
        Dict mapping level number to DifficultyLevel

    Example:
        >>> thresholds = {
        ...     1: (0.8, 1.0),  # Easiest: 80-100% accuracy
        ...     2: (0.6, 0.8),  # 60-80% accuracy
        ...     3: (0.4, 0.6),  # 40-60% accuracy
        ...     4: (0.2, 0.4),  # 20-40% accuracy
        ...     5: (0.0, 0.2),  # Hardest: 0-20% accuracy
        ... }
        >>> levels = stratify_by_absolute_difficulty(baselines, thresholds)
        >>> for level, info in sorted(levels.items()):
        ...     print(f"Level {level}: {info.problem_count} problems")
    """
    if verbose:
        print("\n  Stratifying by absolute difficulty thresholds...")

    if not baselines:
        if verbose:
            print("  Warning: No baselines to stratify")
        return {}

    # Assign problems to levels
    levels = {}
    for level, (min_acc, max_acc) in sorted(thresholds.items()):
        # Find problems in this range
        # Use inclusive bounds on both ends for each level
        level_problems = [
            pid
            for pid, b in baselines.items()
            if min_acc <= b.mean_accuracy <= max_acc
        ]

        levels[level] = DifficultyLevel(
            level=level,
            min_accuracy=min_acc,
            max_accuracy=max_acc,
            problem_count=len(level_problems),
            problem_ids=level_problems,
        )

        if verbose:
            print(
                f"    Level {level}: {len(level_problems)} problems, "
                f"accuracy [{min_acc:.3f}, {max_acc:.3f}]"
            )

    return levels


def get_problems_by_difficulty(
    baselines: dict[str, ProblemBaseline],
    min_accuracy: float = 0.0,
    max_accuracy: float = 1.0,
) -> list[str]:
    """Get problem IDs within an accuracy range.

    Args:
        baselines: Dict mapping problem key to ProblemBaseline
        min_accuracy: Minimum accuracy threshold (inclusive)
        max_accuracy: Maximum accuracy threshold (inclusive)

    Returns:
        List of problem IDs in the specified range
    """
    return [
        pid
        for pid, b in baselines.items()
        if min_accuracy <= b.mean_accuracy <= max_accuracy
    ]


def compute_difficulty_statistics(
    baselines: dict[str, ProblemBaseline]
) -> dict[str, float]:
    """Compute summary statistics for problem difficulties.

    Args:
        baselines: Dict mapping problem key to ProblemBaseline

    Returns:
        Dict with 'mean', 'std', 'min', 'max', 'median' accuracy stats
    """
    if not baselines:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}

    accuracies = [b.mean_accuracy for b in baselines.values()]

    return {
        "mean": float(np.mean(accuracies)),
        "std": float(np.std(accuracies)),
        "min": float(np.min(accuracies)),
        "max": float(np.max(accuracies)),
        "median": float(np.median(accuracies)),
    }
