#!/usr/bin/env python
"""Unified analysis script for experiment results (Auto-Discovery Version).

This script uses the auto-discovery system to analyze experiment results.
All metadata (seeds, temperatures, etc.) is automatically discovered from Hub.

Usage:
    # Analyze a single experiment
    python exp/scripts/analyze_results.py ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon

    # Analyze a category from registry
    python exp/scripts/analyze_results.py --category math500_hnc

    # Compare HNC vs Default for MATH-500
    python exp/scripts/analyze_results.py --category math500_hnc,math500_default --analysis-type hnc_comparison

    # List available experiments
    python exp/scripts/analyze_results.py --list
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add exp directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_hub_registry, HubRegistry
from analysis import (
    discover_experiment,
    ExperimentConfig,
    load_experiment_data,
    load_experiment_data_by_temperature,
    analyze_single_dataset,
    analyze_pass_at_k,
    setup_style,
    save_figure,
)


def generate_output_dir_from_category(category: str) -> str:
    """Generate output directory from category name.

    Examples:
        math500_Qwen2.5-1.5B -> exp/analysis_output-MATH500-Qwen2.5-1.5B
        math500_Qwen2.5-1.5B_hnc -> exp/analysis_output-MATH500-Qwen2.5-1.5B_hnc
        aime25_Qwen2.5-3B -> exp/analysis_output-AIME25-Qwen2.5-3B

    Returns:
        Output directory path based on category
    """
    parts = category.split("_", 1)  # Split only on first underscore
    if len(parts) < 2:
        return "exp/analysis_output"

    dataset = parts[0].upper()  # math500 -> MATH500, aime25 -> AIME25
    model_and_strategy = parts[1]  # Qwen2.5-1.5B or Qwen2.5-1.5B_hnc

    return f"exp/analysis_output-{dataset}-{model_and_strategy}"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze experiment results with auto-discovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Main arguments
    parser.add_argument(
        "hub_paths",
        nargs="*",
        help="Hub dataset paths to analyze (e.g., ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon)",
    )

    # Registry options
    parser.add_argument(
        "--registry",
        type=str,
        default="exp/configs/registry.yaml",
        help="Path to registry YAML file",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Category from registry (comma-separated for multiple)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available categories and experiments",
    )

    # Analysis options
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["summary", "hnc_comparison", "temperature_comparison", "model_comparison"],
        default="summary",
        help="Type of analysis to run",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Filter by specific temperature",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/analysis_output",
        help="Output directory for results (auto-generated from category if not specified)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip report generation",
    )

    # Verbose
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def list_registry(registry: HubRegistry):
    """List all categories and experiments in the registry."""
    print("\n" + "=" * 60)
    print("Available Experiments")
    print("=" * 60)

    for category in registry.get_categories():
        print(f"\n[{category}]")
        for path in registry.get_category(category):
            print(f"  - {path}")

    print(f"\nTotal: {len(registry.all_paths)} experiments in {len(registry.get_categories())} categories")


def discover_and_load(
    hub_paths: list[str],
    temperature: Optional[float] = None,
    verbose: bool = True,
) -> dict[str, tuple[ExperimentConfig, dict[float | tuple, dict[int, Any]]]]:
    """Discover and load datasets from hub paths, organized by temperature.

    Returns:
        Dict mapping hub_path -> (config, {temperature: {seed: dataset}})
    """
    results = {}

    for path in hub_paths:
        if verbose:
            print(f"\nDiscovering: {path}")

        try:
            config = discover_experiment(path)
            if verbose:
                print(f"  Model: {config.model}")
                print(f"  Approach: {config.approach}")
                print(f"  Strategy: {config.strategy}")
                print(f"  Seeds: {config.seeds}")
                print(f"  Temperatures: {config.temperatures}")

            # Filter temperatures if specified
            temps_to_load = None
            if temperature is not None:
                if config.strategy == "hnc":
                    # For HNC, find matching temperature tuple
                    for t in config.temperatures:
                        if isinstance(t, tuple) and temperature in t:
                            temps_to_load = [t]
                            break
                else:
                    temps_to_load = [temperature]

            # Load datasets organized by temperature
            datasets_by_temp = load_experiment_data_by_temperature(
                config, temperatures=temps_to_load, verbose=verbose
            )
            results[path] = (config, datasets_by_temp)

        except Exception as e:
            if verbose:
                print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    return results


def analyze_all(
    loaded_data: dict[str, tuple[ExperimentConfig, dict[float | tuple, dict[int, Any]]]],
    verbose: bool = True,
) -> dict[str, dict[float | tuple, dict[int, dict]]]:
    """Analyze all loaded datasets, organized by temperature.

    Returns:
        Dict mapping hub_path -> temperature -> seed -> {method -> n -> accuracy, 'pass@k' -> k -> value}
    """
    all_results = {}

    for hub_path, (config, datasets_by_temp) in loaded_data.items():
        if verbose:
            print(f"\nAnalyzing: {hub_path}")

        path_results = {}

        for temp, datasets in datasets_by_temp.items():
            if verbose:
                temp_str = f"temps_{temp}" if isinstance(temp, tuple) else f"T={temp}"
                print(f"  Temperature: {temp_str}")

            temp_results = {}

            for seed, dataset in datasets.items():
                # Analyze pred_* methods (naive, weighted, maj)
                metrics = analyze_single_dataset(dataset, hub_path, seed, verbose=verbose)
                temp_results[seed] = metrics

                # Analyze pass@k
                pass_at_k = analyze_pass_at_k(dataset, hub_path, seed, verbose=verbose)
                temp_results[seed]['pass@k'] = pass_at_k

            path_results[temp] = temp_results

        all_results[hub_path] = path_results

    return all_results


def format_temperature(temp: float | tuple) -> str:
    """Format temperature for display."""
    if isinstance(temp, tuple):
        return f"temps_{'-'.join(str(t) for t in temp)}"
    return f"T={temp}"


def format_temperature_short(temp: float | tuple) -> str:
    """Format temperature for short display (filenames, etc.)."""
    if isinstance(temp, tuple):
        return f"temps_{'_'.join(str(t) for t in temp)}"
    return f"T{temp}"


# =============================================================================
# Summary Analysis
# =============================================================================

def run_summary_analysis(
    loaded_data: dict[str, tuple[ExperimentConfig, dict[float | tuple, dict[int, Any]]]],
    all_results: dict[str, dict[float | tuple, dict[int, dict]]],
    output_dir: str,
    generate_plots: bool = True,
    generate_report: bool = True,
    verbose: bool = True,
):
    """Run summary analysis: basic stats for each experiment by temperature."""
    print("\n" + "=" * 60)
    print("Summary Analysis")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    summary_data = []

    for hub_path, (config, datasets_by_temp) in loaded_data.items():
        if hub_path not in all_results:
            continue

        path_results = all_results[hub_path]

        for temp, temp_results in path_results.items():
            # Collect best accuracy (naive@max_n) across seeds for this temperature
            best_accuracies = []
            for seed, metrics in temp_results.items():
                naive = metrics.get("naive", {})
                if naive:
                    max_n = max(naive.keys())
                    best_accuracies.append(naive[max_n])

            if best_accuracies:
                mean_acc = np.mean(best_accuracies)
                std_acc = np.std(best_accuracies)

                summary_data.append({
                    "hub_path": hub_path,
                    "model": config.model,
                    "approach": config.approach,
                    "strategy": config.strategy,
                    "temperature": temp,
                    "seeds": list(temp_results.keys()),
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                })

                print(f"\n{hub_path} ({format_temperature(temp)})")
                print(f"  Model: {config.model}")
                print(f"  Approach: {config.approach}, Strategy: {config.strategy}")
                print(f"  Seeds: {list(temp_results.keys())}")
                print(f"  Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    if generate_report and summary_data:
        report_path = os.path.join(output_dir, "summary_report.md")
        generate_summary_report(summary_data, report_path)
        print(f"\nReport saved to: {report_path}")


def generate_summary_report(summary_data: list[dict], output_path: str):
    """Generate markdown summary report."""
    md_lines = [
        "# Experiment Summary Report\n\n",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Results\n\n",
        "| Experiment | Model | Approach | Strategy | Temperature | Accuracy | Std |\n",
        "|------------|-------|----------|----------|-------------|----------|-----|\n",
    ]

    for data in sorted(summary_data, key=lambda x: x["mean_acc"], reverse=True):
        exp_name = data["hub_path"].split("/")[-1]
        model_short = data["model"].replace("Qwen2.5-", "").replace("-Instruct", "")
        temp_str = format_temperature(data["temperature"])
        md_lines.append(
            f"| {exp_name} | {model_short} | {data['approach']} | "
            f"{data['strategy']} | {temp_str} | {data['mean_acc']:.4f} | {data['std_acc']:.4f} |\n"
        )

    with open(output_path, "w") as f:
        f.writelines(md_lines)


# =============================================================================
# HNC vs Default Comparison
# =============================================================================

def run_hnc_comparison(
    loaded_data: dict[str, tuple[ExperimentConfig, dict[float | tuple, dict[int, Any]]]],
    all_results: dict[str, dict[float | tuple, dict[int, dict]]],
    output_dir: str,
    generate_plots: bool = True,
    generate_report: bool = True,
    verbose: bool = True,
):
    """Run HNC vs Default comparison analysis."""
    print("\n" + "=" * 60)
    print("HNC vs Default Comparison")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Group by approach and strategy
    approach_groups = defaultdict(lambda: {"hnc": [], "default": []})

    for hub_path, (config, _) in loaded_data.items():
        if hub_path not in all_results:
            continue
        approach_groups[config.approach][config.strategy].append((hub_path, config))

    methods = ["naive", "weighted", "maj"]

    if generate_plots:
        for approach, strategies in approach_groups.items():
            print(f"\nApproach: {approach}")

            # Scaling curves
            plot_hnc_scaling_curves(
                strategies, all_results, approach,
                os.path.join(output_dir, f"{approach}-scaling_curves.png")
            )

            # Method comparisons
            for method in methods:
                plot_hnc_method_comparison(
                    strategies, all_results, approach, method,
                    os.path.join(output_dir, f"{approach}-{method}-hnc_vs_default.png")
                )

            # Pass@k curves
            plot_hnc_pass_at_k(
                strategies, all_results, approach,
                os.path.join(output_dir, f"{approach}-pass_at_k.png")
            )

    if generate_report:
        report_path = os.path.join(output_dir, "hnc_comparison_report.md")
        generate_hnc_comparison_report(approach_groups, all_results, report_path)
        print(f"\nReport saved to: {report_path}")


def plot_hnc_scaling_curves(
    strategies: dict[str, list],
    all_results: dict,
    approach: str,
    output_path: str,
):
    """Plot scaling curves for HNC vs Default."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ["naive", "weighted", "maj"]
    colors = {
        "hnc": {"naive": "#1f77b4", "weighted": "#aec7e8", "maj": "#3182bd"},
        "default": {"naive": "#ff7f0e", "weighted": "#ffbb78", "maj": "#d62728"},
    }

    for strategy, experiments in strategies.items():
        if not experiments:
            continue

        for method in methods:
            # Collect data across all seeds and temperatures
            n_data = defaultdict(list)

            for hub_path, config in experiments:
                if hub_path not in all_results:
                    continue
                # Iterate over temperatures
                for temp, temp_results in all_results[hub_path].items():
                    for seed, metrics in temp_results.items():
                        if method in metrics:
                            for n, acc in metrics[method].items():
                                n_data[n].append(acc)

            if not n_data:
                continue

            n_values = sorted(n_data.keys())
            means = [np.mean(n_data[n]) for n in n_values]
            stds = [np.std(n_data[n]) for n in n_values]

            label = f"{strategy}-{method}"
            linestyle = "o-" if strategy == "hnc" else "--"
            ax.plot(n_values, means, linestyle, label=label,
                   color=colors[strategy][method], linewidth=2, markersize=6)
            ax.fill_between(n_values, np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.15, color=colors[strategy][method])

    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{approach.replace('_', ' ').title()}\nHNC vs Default Scaling Curves",
                fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.legend(loc="best", ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_hnc_method_comparison(
    strategies: dict[str, list],
    all_results: dict,
    approach: str,
    method: str,
    output_path: str,
):
    """Plot HNC vs Default comparison for a specific method."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"hnc": "#1f77b4", "default": "#ff7f0e"}

    for strategy, experiments in strategies.items():
        if not experiments:
            continue

        n_data = defaultdict(list)

        for hub_path, config in experiments:
            if hub_path not in all_results:
                continue
            # Iterate over temperatures
            for temp, temp_results in all_results[hub_path].items():
                for seed, metrics in temp_results.items():
                    if method in metrics:
                        for n, acc in metrics[method].items():
                            n_data[n].append(acc)

        if not n_data:
            continue

        n_values = sorted(n_data.keys())
        means = [np.mean(n_data[n]) for n in n_values]
        stds = [np.std(n_data[n]) for n in n_values]

        linestyle = "o-" if strategy == "hnc" else "--"
        label = strategy if strategy == "hnc" else f"{strategy} (baseline)"
        ax.plot(n_values, means, linestyle, label=label,
               color=colors[strategy], linewidth=2, markersize=8)
        ax.fill_between(n_values, np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.2, color=colors[strategy])

    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{approach.replace('_', ' ').title()} - {method.upper()}\nHNC vs Default",
                fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_hnc_pass_at_k(
    strategies: dict[str, list],
    all_results: dict,
    approach: str,
    output_path: str,
):
    """Plot pass@k curves for HNC vs Default."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"hnc": "#1f77b4", "default": "#ff7f0e"}

    for strategy, experiments in strategies.items():
        if not experiments:
            continue

        k_data = defaultdict(list)

        for hub_path, config in experiments:
            if hub_path not in all_results:
                continue
            # Iterate over temperatures
            for temp, temp_results in all_results[hub_path].items():
                for seed, metrics in temp_results.items():
                    if "pass@k" in metrics:
                        for k, val in metrics["pass@k"].items():
                            k_data[k].append(val)

        if not k_data:
            continue

        k_values = sorted(k_data.keys())
        means = [np.mean(k_data[k]) for k in k_values]
        stds = [np.std(k_data[k]) for k in k_values]

        linestyle = "o-" if strategy == "hnc" else "--"
        label = strategy if strategy == "hnc" else f"{strategy} (baseline)"
        ax.plot(k_values, means, linestyle, label=label,
               color=colors[strategy], linewidth=2, markersize=8)
        ax.fill_between(k_values, np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.2, color=colors[strategy])

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Pass@k", fontsize=12)
    ax.set_title(f"{approach.replace('_', ' ').title()}\nPass@k Comparison",
                fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved: {output_path}")


def generate_hnc_comparison_report(
    approach_groups: dict,
    all_results: dict,
    output_path: str,
):
    """Generate markdown report for HNC comparison."""
    md_lines = [
        "# HNC vs Default Comparison Report\n\n",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
    ]

    methods = ["naive", "weighted", "maj"]

    for approach, strategies in approach_groups.items():
        md_lines.append(f"## {approach.upper()}\n\n")

        for method in methods:
            md_lines.append(f"### {method.upper()}\n\n")

            # Collect data
            strategy_data = {"hnc": defaultdict(list), "default": defaultdict(list)}

            for strategy, experiments in strategies.items():
                for hub_path, config in experiments:
                    if hub_path not in all_results:
                        continue
                    # Iterate over temperatures
                    for temp, temp_results in all_results[hub_path].items():
                        for seed, metrics in temp_results.items():
                            if method in metrics:
                                for n, acc in metrics[method].items():
                                    strategy_data[strategy][n].append(acc)

            all_n = set()
            for data in strategy_data.values():
                all_n.update(data.keys())

            if not all_n:
                md_lines.append("*No data available*\n\n")
                continue

            md_lines.append("| n | HNC | Default | Diff |\n")
            md_lines.append("|---|-----|---------|------|\n")

            for n in sorted(all_n):
                hnc_vals = strategy_data["hnc"].get(n, [])
                def_vals = strategy_data["default"].get(n, [])

                hnc_mean = np.mean(hnc_vals) if hnc_vals else 0
                def_mean = np.mean(def_vals) if def_vals else 0

                hnc_str = f"{hnc_mean:.4f}" if hnc_vals else "N/A"
                def_str = f"{def_mean:.4f}" if def_vals else "N/A"
                diff = hnc_mean - def_mean if hnc_vals and def_vals else 0
                diff_str = f"{diff:+.4f}" if hnc_vals and def_vals else "N/A"

                md_lines.append(f"| {n} | {hnc_str} | {def_str} | {diff_str} |\n")

            md_lines.append("\n")

    with open(output_path, "w") as f:
        f.writelines(md_lines)


# =============================================================================
# Temperature Comparison
# =============================================================================

def run_temperature_comparison(
    loaded_data: dict[str, tuple[ExperimentConfig, dict[float | tuple, dict[int, Any]]]],
    all_results: dict[str, dict[float | tuple, dict[int, dict]]],
    output_dir: str,
    generate_plots: bool = True,
    generate_report: bool = True,
    verbose: bool = True,
):
    """Run temperature comparison analysis (like legacy analyze_aime25_results.py)."""
    print("\n" + "=" * 60)
    print("Temperature Comparison")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Group experiments by (model, approach)
    experiment_groups = defaultdict(list)
    for hub_path, (config, datasets_by_temp) in loaded_data.items():
        if hub_path not in all_results:
            continue
        key = (config.model, config.approach)
        experiment_groups[key].append((hub_path, config, all_results[hub_path]))

    methods = ["naive", "weighted", "maj"]

    if generate_plots:
        for (model, approach), experiments in experiment_groups.items():
            model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
            print(f"\n{model_short} - {approach}")

            # Collect all temperatures across experiments
            all_temps = set()
            for hub_path, config, path_results in experiments:
                all_temps.update(path_results.keys())

            if len(all_temps) < 2:
                print(f"  Skipping: Only {len(all_temps)} temperature(s) found")
                continue

            # Plot scaling curves with all temperatures
            plot_temperature_scaling_curves(
                experiments, all_temps, model_short, approach,
                os.path.join(output_dir, f"{model_short}-{approach}-temp_scaling.png")
            )

            # Plot temperature comparison for each method
            for method in methods:
                plot_temperature_method_comparison(
                    experiments, all_temps, model_short, approach, method,
                    os.path.join(output_dir, f"{model_short}-{approach}-{method}-temp_comparison.png")
                )

            # Plot pass@k comparison by temperature
            plot_temperature_pass_at_k(
                experiments, all_temps, model_short, approach,
                os.path.join(output_dir, f"{model_short}-{approach}-pass_at_k_by_temp.png")
            )

    if generate_report:
        report_path = os.path.join(output_dir, "temperature_comparison_report.md")
        generate_temperature_comparison_report(experiment_groups, report_path)
        print(f"\nReport saved to: {report_path}")


def plot_temperature_scaling_curves(
    experiments: list,
    temperatures: set,
    model_short: str,
    approach: str,
    output_path: str,
):
    """Plot scaling curves comparing different temperatures."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette for temperatures
    temp_colors = {}
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    sorted_temps = sorted(temperatures, key=lambda x: x if isinstance(x, (int, float)) else x[0])
    for idx, temp in enumerate(sorted_temps):
        temp_colors[temp] = color_palette[idx % len(color_palette)]

    methods = ["naive", "weighted", "maj"]
    linestyles = {"naive": "-", "weighted": "--", "maj": ":"}

    for temp in sorted_temps:
        temp_str = format_temperature_short(temp)

        for method in methods:
            n_data = defaultdict(list)

            for hub_path, config, path_results in experiments:
                if temp not in path_results:
                    continue
                for seed, metrics in path_results[temp].items():
                    if method in metrics:
                        for n, acc in metrics[method].items():
                            n_data[n].append(acc)

            if not n_data:
                continue

            n_values = sorted(n_data.keys())
            means = [np.mean(n_data[n]) for n in n_values]
            stds = [np.std(n_data[n]) for n in n_values]

            label = f"{temp_str}-{method}"
            ax.plot(n_values, means, linestyles[method], label=label,
                   color=temp_colors[temp], linewidth=2, markersize=4, marker='o')
            ax.fill_between(n_values, np.array(means) - np.array(stds),
                           np.array(means) + np.array(stds),
                           alpha=0.1, color=temp_colors[temp])

    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{model_short} - {approach.replace('_', ' ').title()}\nTemperature Comparison",
                fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.legend(loc="best", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_temperature_method_comparison(
    experiments: list,
    temperatures: set,
    model_short: str,
    approach: str,
    method: str,
    output_path: str,
):
    """Plot single method comparison across temperatures."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette for temperatures
    temp_colors = {}
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    sorted_temps = sorted(temperatures, key=lambda x: x if isinstance(x, (int, float)) else x[0])
    for idx, temp in enumerate(sorted_temps):
        temp_colors[temp] = color_palette[idx % len(color_palette)]

    for temp in sorted_temps:
        temp_str = format_temperature(temp)

        n_data = defaultdict(list)

        for hub_path, config, path_results in experiments:
            if temp not in path_results:
                continue
            for seed, metrics in path_results[temp].items():
                if method in metrics:
                    for n, acc in metrics[method].items():
                        n_data[n].append(acc)

        if not n_data:
            continue

        n_values = sorted(n_data.keys())
        means = [np.mean(n_data[n]) for n in n_values]
        stds = [np.std(n_data[n]) for n in n_values]

        ax.plot(n_values, means, 'o-', label=temp_str,
               color=temp_colors[temp], linewidth=2, markersize=6)
        ax.fill_between(n_values, np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.2, color=temp_colors[temp])

    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{model_short} - {approach.replace('_', ' ').title()} - {method.upper()}\nTemperature Comparison",
                fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved: {output_path}")


def plot_temperature_pass_at_k(
    experiments: list,
    temperatures: set,
    model_short: str,
    approach: str,
    output_path: str,
):
    """Plot pass@k curves comparing different temperatures."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette for temperatures
    temp_colors = {}
    color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    sorted_temps = sorted(temperatures, key=lambda x: x if isinstance(x, (int, float)) else x[0])
    for idx, temp in enumerate(sorted_temps):
        temp_colors[temp] = color_palette[idx % len(color_palette)]

    for temp in sorted_temps:
        temp_str = format_temperature(temp)

        k_data = defaultdict(list)

        for hub_path, config, path_results in experiments:
            if temp not in path_results:
                continue
            for seed, metrics in path_results[temp].items():
                if "pass@k" in metrics:
                    for k, val in metrics["pass@k"].items():
                        k_data[k].append(val)

        if not k_data:
            continue

        k_values = sorted(k_data.keys())
        means = [np.mean(k_data[k]) for k in k_values]
        stds = [np.std(k_data[k]) for k in k_values]

        ax.plot(k_values, means, 'o-', label=temp_str,
               color=temp_colors[temp], linewidth=2, markersize=6)
        ax.fill_between(k_values, np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.2, color=temp_colors[temp])

    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Pass@k", fontsize=12)
    ax.set_title(f"{model_short} - {approach.replace('_', ' ').title()}\nPass@k by Temperature",
                fontsize=14, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved: {output_path}")


def generate_temperature_comparison_report(
    experiment_groups: dict,
    output_path: str,
):
    """Generate markdown report for temperature comparison."""
    md_lines = [
        "# Temperature Comparison Report\n\n",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
    ]

    methods = ["naive", "weighted", "maj"]

    for (model, approach), experiments in experiment_groups.items():
        model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
        md_lines.append(f"## {model_short} - {approach.upper()}\n\n")

        # Collect all temperatures
        all_temps = set()
        for hub_path, config, path_results in experiments:
            all_temps.update(path_results.keys())

        sorted_temps = sorted(all_temps, key=lambda x: x if isinstance(x, (int, float)) else x[0])

        if len(sorted_temps) < 2:
            md_lines.append(f"*Only {len(sorted_temps)} temperature(s) found*\n\n")
            continue

        for method in methods:
            md_lines.append(f"### {method.upper()}\n\n")

            # Collect data by temperature
            temp_data = {temp: defaultdict(list) for temp in sorted_temps}

            for hub_path, config, path_results in experiments:
                for temp, temp_results in path_results.items():
                    for seed, metrics in temp_results.items():
                        if method in metrics:
                            for n, acc in metrics[method].items():
                                temp_data[temp][n].append(acc)

            # Get all n values
            all_n = set()
            for data in temp_data.values():
                all_n.update(data.keys())

            if not all_n:
                md_lines.append("*No data available*\n\n")
                continue

            # Create table header
            temp_headers = [format_temperature_short(t) for t in sorted_temps]
            md_lines.append("| n | " + " | ".join(temp_headers) + " |\n")
            md_lines.append("|---| " + " | ".join(["---"] * len(sorted_temps)) + " |\n")

            for n in sorted(all_n):
                row = [str(n)]
                for temp in sorted_temps:
                    vals = temp_data[temp].get(n, [])
                    if vals:
                        mean = np.mean(vals)
                        std = np.std(vals)
                        row.append(f"{mean:.4f}±{std:.3f}")
                    else:
                        row.append("N/A")
                md_lines.append("| " + " | ".join(row) + " |\n")

            md_lines.append("\n")

        # Pass@k section
        md_lines.append("### Pass@k\n\n")

        temp_passk_data = {temp: defaultdict(list) for temp in sorted_temps}
        for hub_path, config, path_results in experiments:
            for temp, temp_results in path_results.items():
                for seed, metrics in temp_results.items():
                    if "pass@k" in metrics:
                        for k, val in metrics["pass@k"].items():
                            temp_passk_data[temp][k].append(val)

        all_k = set()
        for data in temp_passk_data.values():
            all_k.update(data.keys())

        if all_k:
            md_lines.append("| k | " + " | ".join(temp_headers) + " |\n")
            md_lines.append("|---| " + " | ".join(["---"] * len(sorted_temps)) + " |\n")

            for k in sorted(all_k):
                row = [str(k)]
                for temp in sorted_temps:
                    vals = temp_passk_data[temp].get(k, [])
                    if vals:
                        mean = np.mean(vals)
                        row.append(f"{mean:.4f}")
                    else:
                        row.append("N/A")
                md_lines.append("| " + " | ".join(row) + " |\n")

            md_lines.append("\n")

    with open(output_path, "w") as f:
        f.writelines(md_lines)


# =============================================================================
# Model Comparison
# =============================================================================

def run_model_comparison(
    loaded_data: dict[str, tuple[ExperimentConfig, dict[float | tuple, dict[int, Any]]]],
    all_results: dict[str, dict[float | tuple, dict[int, dict]]],
    output_dir: str,
    generate_plots: bool = True,
    generate_report: bool = True,
    verbose: bool = True,
):
    """Run model comparison analysis."""
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Group by approach
    approach_groups = defaultdict(list)

    for hub_path, (config, _) in loaded_data.items():
        if hub_path not in all_results:
            continue
        approach_groups[config.approach].append((hub_path, config, all_results[hub_path]))

    if generate_plots:
        for approach, experiments in approach_groups.items():
            print(f"\nApproach: {approach}")

            plot_model_scaling_curves(
                experiments, approach,
                os.path.join(output_dir, f"{approach}-model_comparison.png")
            )

    if generate_report:
        report_path = os.path.join(output_dir, "model_comparison_report.md")
        generate_model_comparison_report(approach_groups, report_path)
        print(f"\nReport saved to: {report_path}")


def plot_model_scaling_curves(
    experiments: list[tuple],
    approach: str,
    output_path: str,
):
    """Plot scaling curves comparing different models."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model
    model_groups = defaultdict(list)
    for hub_path, config, path_results in experiments:
        model_groups[config.model].append((hub_path, config, path_results))

    color_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    model_colors = {}
    for idx, model in enumerate(sorted(model_groups.keys())):
        model_colors[model] = color_list[idx % len(color_list)]

    for model, model_exps in model_groups.items():
        n_data = defaultdict(list)

        for hub_path, config, path_results in model_exps:
            # Iterate over temperatures
            for temp, temp_results in path_results.items():
                for seed, metrics in temp_results.items():
                    if "naive" in metrics:
                        for n, acc in metrics["naive"].items():
                            n_data[n].append(acc)

        if not n_data:
            continue

        n_values = sorted(n_data.keys())
        means = [np.mean(n_data[n]) for n in n_values]
        stds = [np.std(n_data[n]) for n in n_values]

        model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
        ax.plot(n_values, means, "o-", label=model_short,
               color=model_colors[model], linewidth=2, markersize=8)
        ax.fill_between(n_values, np.array(means) - np.array(stds),
                       np.array(means) + np.array(stds),
                       alpha=0.2, color=model_colors[model])

    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{approach.replace('_', ' ').title()}\nModel Comparison",
                fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"  Saved: {output_path}")


def generate_model_comparison_report(
    approach_groups: dict,
    output_path: str,
):
    """Generate markdown report for model comparison."""
    md_lines = [
        "# Model Comparison Report\n\n",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
    ]

    for approach, experiments in approach_groups.items():
        md_lines.append(f"## {approach.upper()}\n\n")

        # Group by model
        model_data = defaultdict(lambda: defaultdict(list))

        for hub_path, config, path_results in experiments:
            # Iterate over temperatures
            for temp, temp_results in path_results.items():
                for seed, metrics in temp_results.items():
                    if "naive" in metrics:
                        for n, acc in metrics["naive"].items():
                            model_data[config.model][n].append(acc)

        all_n = set()
        for data in model_data.values():
            all_n.update(data.keys())

        if not all_n:
            md_lines.append("*No data available*\n\n")
            continue

        models = sorted(model_data.keys())
        model_shorts = [m.replace("Qwen2.5-", "").replace("-Instruct", "") for m in models]

        md_lines.append("| n | " + " | ".join(model_shorts) + " |\n")
        md_lines.append("|---| " + " | ".join(["---"] * len(models)) + " |\n")

        for n in sorted(all_n):
            row = [str(n)]
            for model in models:
                vals = model_data[model].get(n, [])
                if vals:
                    row.append(f"{np.mean(vals):.4f}")
                else:
                    row.append("N/A")
            md_lines.append("| " + " | ".join(row) + " |\n")

        md_lines.append("\n")

    with open(output_path, "w") as f:
        f.writelines(md_lines)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Load registry
    registry = load_hub_registry(args.registry)

    # List mode
    if args.list:
        list_registry(registry)
        return

    # Collect hub paths to analyze
    hub_paths = list(args.hub_paths)

    if args.category:
        for cat in args.category.split(","):
            cat = cat.strip()
            paths = registry.get_category(cat)
            if paths:
                hub_paths.extend(paths)
            else:
                print(f"Warning: Category '{cat}' not found")

    if not hub_paths:
        print("No experiments specified. Use --list to see available experiments.")
        return

    # Remove duplicates while preserving order
    hub_paths = list(dict.fromkeys(hub_paths))

    # Auto-generate output directory from category if not explicitly specified
    if args.category and args.output_dir == "exp/analysis_output":
        first_category = args.category.split(",")[0].strip()
        args.output_dir = generate_output_dir_from_category(first_category)
        print(f"Auto-generated output directory: {args.output_dir}")

    print(f"\nAnalyzing {len(hub_paths)} experiment(s)")

    # Discover and load
    loaded_data = discover_and_load(hub_paths, args.temperature, verbose=args.verbose)

    if not loaded_data:
        print("No data loaded!")
        return

    # Analyze
    all_results = analyze_all(loaded_data, verbose=args.verbose)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis based on type
    if args.analysis_type == "summary":
        run_summary_analysis(
            loaded_data, all_results, args.output_dir,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            verbose=args.verbose,
        )
    elif args.analysis_type == "hnc_comparison":
        run_hnc_comparison(
            loaded_data, all_results, args.output_dir,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            verbose=args.verbose,
        )
    elif args.analysis_type == "temperature_comparison":
        run_temperature_comparison(
            loaded_data, all_results, args.output_dir,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            verbose=args.verbose,
        )
    elif args.analysis_type == "model_comparison":
        run_model_comparison(
            loaded_data, all_results, args.output_dir,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
            verbose=args.verbose,
        )

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
