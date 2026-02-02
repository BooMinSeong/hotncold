"""Difficulty-based temperature analysis utilities.

This module provides functions to analyze how different temperatures perform
across problems of varying difficulty levels.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datasets import Dataset
from tqdm import tqdm

from .difficulty import (
    DifficultyLevel,
    ProblemBaseline,
    compute_problem_baselines_from_completions,
    stratify_by_absolute_difficulty,
)
from .metrics import analyze_single_dataset, aggregate_across_seeds
from .visualization import plot_scaling_curve, save_figure, setup_style


# Default difficulty thresholds (5 levels based on absolute accuracy)
DEFAULT_DIFFICULTY_THRESHOLDS = {
    1: (0.8, 1.0),  # Easiest: 80-100% accuracy
    2: (0.6, 0.8),  # 60-80% accuracy
    3: (0.4, 0.6),  # 40-60% accuracy
    4: (0.2, 0.4),  # 20-40% accuracy
    5: (0.0, 0.2),  # Hardest: 0-20% accuracy
}


def compute_universal_difficulty_baselines(
    datasets_by_temp: dict[float, dict[int, Any]],
    reference_temp: float | None = None,
    verbose: bool = True,
) -> dict[str, ProblemBaseline]:
    """Compute difficulty baselines using completions from a reference temperature.

    This creates a temperature-independent difficulty metric by using
    a single reference temperature (typically the lowest) as the baseline.
    This allows us to identify problems that are inherently difficult
    (low accuracy at reference temp) vs easy (high accuracy at reference temp).

    The baseline is computed using is_correct_preds field (list of booleans,
    one per completion). For each seed, accuracy is computed as the fraction
    of correct completions, then averaged across seeds. This provides much
    finer-grained difficulty measurement than using majority vote results.

    Args:
        datasets_by_temp: Nested dict {temperature: {seed: Dataset}}
        reference_temp: Reference temperature to use for baseline.
                       If None, uses the lowest temperature.
        verbose: Print progress messages

    Returns:
        Dict mapping problem unique_id to ProblemBaseline

    Raises:
        ValueError: If datasets don't have is_correct_preds field
    """
    # Select reference temperature
    if reference_temp is None:
        reference_temp = min(datasets_by_temp.keys())

    if reference_temp not in datasets_by_temp:
        available_temps = sorted(datasets_by_temp.keys())
        raise ValueError(
            f"Reference temperature {reference_temp} not found. "
            f"Available temperatures: {available_temps}"
        )

    if verbose:
        print("\nComputing difficulty baselines from reference temperature...")
        print(f"  Reference temperature: {reference_temp}")
        print(f"  Using {len(datasets_by_temp[reference_temp])} seeds")
        print(f"  Baseline method: completions (is_correct_preds)")

    # Use only the reference temperature data
    # Format as {approach: {seed: Dataset}} for baseline computation
    reference_data = {f"T{reference_temp}": datasets_by_temp[reference_temp]}

    # Compute baselines using completions (is_correct_preds field)
    baselines = compute_problem_baselines_from_completions(
        reference_data,
        aggregate_across_approaches=True,  # Aggregate across seeds
        verbose=verbose,
    )

    return baselines


def filter_dataset_by_problems(
    dataset: Any,
    problem_ids: list[str],
) -> Dataset:
    """Filter dataset to only include specified problems.

    Args:
        dataset: HuggingFace dataset
        problem_ids: List of problem unique_ids to include

    Returns:
        Filtered Dataset object
    """
    if "train" in dataset:
        dataset = dataset["train"]

    problem_ids_set = set(problem_ids)
    filtered = []

    for problem in dataset:
        unique_id = problem.get("unique_id", problem.get("problem", ""))
        if unique_id in problem_ids_set:
            filtered.append(problem)

    # Convert list to Dataset
    if filtered:
        return Dataset.from_dict({k: [d[k] for d in filtered] for k in filtered[0].keys()})
    else:
        return Dataset.from_dict({})


def analyze_temperature_by_difficulty(
    datasets_by_temp: dict[float, dict[int, Any]],
    difficulty_levels: dict[int, DifficultyLevel],
    verbose: bool = True,
) -> dict[int, dict[float, dict[str, dict[int, dict[str, float]]]]]:
    """Analyze temperature performance within each difficulty level.

    For each difficulty level, compute accuracy metrics (naive, weighted, maj)
    for each temperature across different sample counts.

    Args:
        datasets_by_temp: Nested dict {temperature: {seed: Dataset}}
        difficulty_levels: Dict mapping level to DifficultyLevel info
        verbose: Print progress messages

    Returns:
        Nested dict structure:
        {
            level: {
                temperature: {
                    method: {
                        n_samples: {
                            'mean': float,
                            'std': float,
                            'values': list,
                            'seeds': list
                        }
                    }
                }
            }
        }
    """
    if verbose:
        print("\nAnalyzing temperature performance by difficulty level...")

    results = {}

    for level, level_info in sorted(difficulty_levels.items()):
        if level_info.problem_count == 0:
            if verbose:
                print(f"\n  Level {level}: Skipping (0 problems)")
            continue

        if verbose:
            print(
                f"\n  Level {level}: {level_info.problem_count} problems "
                f"(accuracy {level_info.min_accuracy:.2f}-{level_info.max_accuracy:.2f})"
            )

        results[level] = {}

        for temp in sorted(datasets_by_temp.keys()):
            if verbose:
                print(f"    Temperature {temp}...")

            results[level][temp] = {}

            # Filter datasets for this difficulty level
            filtered_datasets = {}
            for seed, dataset in datasets_by_temp[temp].items():
                filtered = filter_dataset_by_problems(dataset, level_info.problem_ids)
                if len(filtered) > 0:
                    filtered_datasets[seed] = filtered

            if not filtered_datasets:
                if verbose:
                    print(f"      No data for temperature {temp}")
                continue

            # Analyze each seed separately
            all_results = {}
            for seed, filtered_dataset in filtered_datasets.items():
                seed_results = analyze_single_dataset(
                    filtered_dataset,
                    f"Level{level}-T{temp}",
                    seed,
                    verbose=False,
                )
                all_results[seed] = seed_results

            # Aggregate across seeds for each method and n_samples
            for method in ["naive", "weighted", "maj"]:
                results[level][temp][method] = {}

                # Get all n_samples values
                all_n_samples = set()
                for seed_results in all_results.values():
                    all_n_samples.update(seed_results.get(method, {}).keys())

                for n in sorted(all_n_samples):
                    values_by_seed = {}
                    for seed, seed_results in all_results.items():
                        if n in seed_results.get(method, {}):
                            values_by_seed[seed] = seed_results[method][n]

                    if values_by_seed:
                        results[level][temp][method][n] = aggregate_across_seeds(
                            values_by_seed
                        )

            if verbose:
                # Print summary for this temperature
                if "naive" in results[level][temp]:
                    n_samples = list(results[level][temp]["naive"].keys())
                    if n_samples:
                        max_n = max(n_samples)
                        max_acc = results[level][temp]["naive"][max_n]["mean"]
                        print(f"      Naive@{max_n}: {max_acc:.3f}")

    return results


def generate_difficulty_temperature_plots(
    results: dict[int, dict[float, dict[str, dict[int, dict[str, float]]]]],
    difficulty_levels: dict[int, DifficultyLevel],
    output_dir: str,
    verbose: bool = True,
) -> None:
    """Generate visualization plots for temperature-difficulty analysis.

    Creates:
    1. Difficulty distribution bar chart
    2. Per-level temperature scaling curves
    3. Temperature × Difficulty heatmap

    Args:
        results: Analysis results from analyze_temperature_by_difficulty
        difficulty_levels: Difficulty level definitions
        output_dir: Output directory for plots
        verbose: Print progress messages
    """
    setup_style("whitegrid")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\nGenerating plots...")

    # 1. Difficulty distribution
    if verbose:
        print("  Creating difficulty distribution plot...")

    fig, ax = plt.subplots(figsize=(10, 6))
    levels = sorted(difficulty_levels.keys())
    counts = [difficulty_levels[lvl].problem_count for lvl in levels]
    acc_ranges = [
        f"{difficulty_levels[lvl].min_accuracy:.1f}-{difficulty_levels[lvl].max_accuracy:.1f}"
        for lvl in levels
    ]

    bars = ax.bar(
        [f"Level {lvl}\n({acc_ranges[i]})" for i, lvl in enumerate(levels)],
        counts,
        color=sns.color_palette("viridis", len(levels)),
    )

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Difficulty Level (Accuracy Range)", fontsize=12)
    ax.set_ylabel("Number of Problems", fontsize=12)
    ax.set_title("Problem Distribution by Difficulty Level", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    save_figure(fig, output_path / "difficulty_distribution.png")

    # 2. Per-level temperature scaling curves
    if verbose:
        print("  Creating per-level temperature comparison plots...")

    temp_colors = {
        0.1: "#1f77b4",
        0.2: "#ff7f0e",
        0.4: "#2ca02c",
        0.8: "#d62728",
    }

    for level in sorted(results.keys()):
        if level not in difficulty_levels:
            continue

        level_info = difficulty_levels[level]
        level_dir = output_path / f"level_{level}"
        level_dir.mkdir(parents=True, exist_ok=True)

        # All methods on one plot
        fig, ax = plt.subplots(figsize=(12, 7))

        for method in ["naive", "weighted", "maj"]:
            for temp in sorted(results[level].keys()):
                if method not in results[level][temp]:
                    continue

                method_data = results[level][temp][method]
                if not method_data:
                    continue

                n_values = sorted(method_data.keys())
                means = [method_data[n]["mean"] for n in n_values]
                stds = [method_data[n]["std"] for n in n_values]

                color = temp_colors.get(temp, "#888888")
                linestyle = {"naive": "-", "weighted": "--", "maj": "-."}[method]

                plot_scaling_curve(
                    n_values,
                    means,
                    stds,
                    ax=ax,
                    label=f"{method.capitalize()} T{temp}",
                    color=color,
                    linestyle=linestyle,
                    marker="o",
                )

        ax.set_xlabel("Number of Samples", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(
            f"Level {level}: Temperature Comparison\n"
            f"({level_info.problem_count} problems, "
            f"accuracy {level_info.min_accuracy:.2f}-{level_info.max_accuracy:.2f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(alpha=0.3)

        save_figure(fig, level_dir / "temperature_scaling_curves.png")

        # Individual method plots
        for method in ["naive", "weighted", "maj"]:
            fig, ax = plt.subplots(figsize=(10, 6))

            for temp in sorted(results[level].keys()):
                if method not in results[level][temp]:
                    continue

                method_data = results[level][temp][method]
                if not method_data:
                    continue

                n_values = sorted(method_data.keys())
                means = [method_data[n]["mean"] for n in n_values]
                stds = [method_data[n]["std"] for n in n_values]

                color = temp_colors.get(temp, "#888888")

                plot_scaling_curve(
                    n_values,
                    means,
                    stds,
                    ax=ax,
                    label=f"T{temp}",
                    color=color,
                    marker="o",
                )

            ax.set_xlabel("Number of Samples", fontsize=12)
            ax.set_ylabel("Accuracy", fontsize=12)
            ax.set_title(
                f"Level {level}: {method.capitalize()} Temperature Comparison\n"
                f"({level_info.problem_count} problems, "
                f"accuracy {level_info.min_accuracy:.2f}-{level_info.max_accuracy:.2f})",
                fontsize=14,
                fontweight="bold",
            )
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

            save_figure(fig, level_dir / f"temperature_comparison_{method}.png")

    # 3. Temperature × Difficulty heatmap (using max n for each method)
    if verbose:
        print("  Creating temperature-difficulty heatmaps...")

    # Get all temperatures and levels
    temps = sorted(set(t for level_results in results.values() for t in level_results.keys()))
    levels = sorted(results.keys())

    # Create heatmap for each method
    for method in ["naive", "weighted", "maj"]:
        if verbose:
            print(f"    Creating heatmap for {method} method...")

        # Extract max performance for each (level, temp) combination
        heatmap_data = []

        for level in levels:
            row = []
            for temp in temps:
                if temp in results[level] and method in results[level][temp]:
                    method_data = results[level][temp][method]
                    if method_data:
                        max_n = max(method_data.keys())
                        row.append(method_data[max_n]["mean"])
                    else:
                        row.append(0.0)
                else:
                    row.append(0.0)
            heatmap_data.append(row)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            xticklabels=[f"T{t}" for t in temps],
            yticklabels=[
                f"L{lvl}\n{difficulty_levels[lvl].min_accuracy:.1f}-{difficulty_levels[lvl].max_accuracy:.1f}"
                for lvl in levels
            ],
            cbar_kws={"label": "Accuracy"},
            ax=ax,
        )
        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel("Difficulty Level (Accuracy Range)", fontsize=12)
        ax.set_title(
            f"Temperature Performance Across Difficulty Levels\n({method.capitalize()} method at max samples)",
            fontsize=14,
            fontweight="bold",
        )

        save_figure(fig, output_path / f"temperature_difficulty_heatmap_{method}.png")

    if verbose:
        print(f"  Saved plots to {output_dir}/")


def generate_difficulty_temperature_report(
    results: dict[int, dict[float, dict[str, dict[int, dict[str, float]]]]],
    difficulty_levels: dict[int, DifficultyLevel],
    output_path: str,
    reference_temp: float | None = None,
    verbose: bool = True,
) -> None:
    """Generate markdown report for temperature-difficulty analysis.

    Args:
        results: Analysis results from analyze_temperature_by_difficulty
        difficulty_levels: Difficulty level definitions
        output_path: Output path for markdown file
        reference_temp: Reference temperature used for difficulty baseline
        verbose: Print progress messages
    """
    if verbose:
        print("\nGenerating markdown report...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Temperature Performance by Difficulty Level\n\n")

        # Overview
        f.write("## Overview\n\n")
        f.write(
            "This report analyzes how different temperatures perform across "
            "problems of varying difficulty.\n\n"
        )
        if reference_temp is not None:
            f.write(
                f"**Difficulty Baseline:** Problems are categorized based on their "
                f"accuracy using `is_correct_preds` (all completions) at the reference temperature **T={reference_temp}**. "
                f"This provides fine-grained difficulty measurement by evaluating each completion individually, "
                f"then averaging across seeds. This allows us to identify which temperatures improve performance "
                f"on problems that are inherently difficult (low accuracy at T={reference_temp}).\n\n"
            )

        # Difficulty distribution
        f.write("## Difficulty Distribution\n\n")
        f.write("| Level | Accuracy Range | Problem Count |\n")
        f.write("|-------|----------------|---------------|\n")
        for level in sorted(difficulty_levels.keys()):
            info = difficulty_levels[level]
            f.write(
                f"| {level} | {info.min_accuracy:.2f}-{info.max_accuracy:.2f} | "
                f"{info.problem_count} |\n"
            )
        f.write("\n")

        # Per-level analysis
        for level in sorted(results.keys()):
            level_info = difficulty_levels[level]
            f.write(f"## Level {level}: ")
            f.write(f"{level_info.min_accuracy:.2f}-{level_info.max_accuracy:.2f} Accuracy\n\n")
            f.write(f"**{level_info.problem_count} problems**\n\n")

            if level_info.problem_count == 0:
                f.write("*No problems in this difficulty range.*\n\n")
                continue

            # Get all sample counts and temperatures
            all_n = set()
            all_temps = sorted(results[level].keys())

            for temp_results in results[level].values():
                for method_results in temp_results.values():
                    all_n.update(method_results.keys())

            # Table for each method
            for method in ["naive", "weighted", "maj"]:
                f.write(f"### {method.capitalize()} Method\n\n")
                f.write("| N | " + " | ".join(f"T{t}" for t in all_temps) + " |\n")
                f.write("|---|" + "|".join("---" for _ in all_temps) + "|\n")

                for n in sorted(all_n):
                    row = [f"{n}"]
                    for temp in all_temps:
                        if (
                            method in results[level][temp]
                            and n in results[level][temp][method]
                        ):
                            data = results[level][temp][method][n]
                            row.append(f"{data['mean']:.3f} ± {data['std']:.3f}")
                        else:
                            row.append("-")
                    f.write("| " + " | ".join(row) + " |\n")

                f.write("\n")

            # Best temperature analysis
            f.write("### Best Temperature\n\n")
            # Find best temperature at max n for naive method
            max_n = max(all_n) if all_n else 0
            if max_n > 0:
                best_temp = None
                best_acc = 0.0

                for temp in all_temps:
                    if (
                        "naive" in results[level][temp]
                        and max_n in results[level][temp]["naive"]
                    ):
                        acc = results[level][temp]["naive"][max_n]["mean"]
                        if acc > best_acc:
                            best_acc = acc
                            best_temp = temp

                if best_temp is not None:
                    f.write(
                        f"At {max_n} samples (naive method), **T{best_temp}** "
                        f"performs best with {best_acc:.3f} accuracy.\n\n"
                    )
            else:
                f.write("*Insufficient data to determine best temperature.*\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write("### Temperature Recommendations by Difficulty\n\n")
        f.write("| Difficulty Level | Best Temperature | Accuracy |\n")
        f.write("|------------------|------------------|----------|\n")

        for level in sorted(results.keys()):
            level_info = difficulty_levels[level]

            # Find max n
            all_n = set()
            for temp_results in results[level].values():
                for method_results in temp_results.values():
                    all_n.update(method_results.keys())

            max_n = max(all_n) if all_n else 0

            if max_n > 0:
                best_temp = None
                best_acc = 0.0

                for temp in results[level].keys():
                    if (
                        "naive" in results[level][temp]
                        and max_n in results[level][temp]["naive"]
                    ):
                        acc = results[level][temp]["naive"][max_n]["mean"]
                        if acc > best_acc:
                            best_acc = acc
                            best_temp = temp

                if best_temp is not None:
                    f.write(
                        f"| Level {level} ({level_info.min_accuracy:.2f}-"
                        f"{level_info.max_accuracy:.2f}) | T{best_temp} | "
                        f"{best_acc:.3f} |\n"
                    )
                else:
                    f.write(
                        f"| Level {level} ({level_info.min_accuracy:.2f}-"
                        f"{level_info.max_accuracy:.2f}) | - | - |\n"
                    )
            else:
                f.write(
                    f"| Level {level} ({level_info.min_accuracy:.2f}-"
                    f"{level_info.max_accuracy:.2f}) | - | - |\n"
                )

    if verbose:
        print(f"  Saved report to {output_path}")
