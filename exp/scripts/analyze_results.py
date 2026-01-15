#!/usr/bin/env python
"""Unified analysis script for experiment results.

This script provides a unified interface for analyzing experiment results
using the registry-based configuration system.

Analysis Types:
    - hnc_comparison: Compare HNC vs Default strategies (MATH-500)
    - temperature_comparison: Compare T=0.4 vs T=0.8 (AIME25)
    - model_comparison: Compare model sizes (1.5B vs 3B)
    - scaling: Generate scaling curves

Usage:
    # MATH-500: HNC vs Default comparison
    python exp/scripts/analyze_results.py \\
        --filter-dataset="MATH-500" \\
        --analysis-type="hnc_comparison"

    # AIME25: Temperature comparison
    python exp/scripts/analyze_results.py \\
        --filter-dataset="aime25" \\
        --filter-model="Qwen2.5-1.5B-Instruct" \\
        --analysis-type="temperature_comparison"

    # AIME25: Model size comparison
    python exp/scripts/analyze_results.py \\
        --filter-dataset="aime25" \\
        --analysis-type="model_comparison"
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add exp directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_registry, Registry, ResultEntry
from analysis import (
    load_from_registry,
    analyze_single_dataset,
    analyze_pass_at_k,
    setup_style,
    save_figure,
    STRATEGY_COLORS,
    METHOD_COLORS,
    APPROACH_COLORS,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified analysis script for experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Registry path
    parser.add_argument(
        "--registry",
        type=str,
        default="exp/configs/registry.yaml",
        help="Path to registry YAML file",
    )

    # Filters
    parser.add_argument(
        "--filter-model",
        type=str,
        default=None,
        help="Filter by model name(s), comma-separated",
    )
    parser.add_argument(
        "--filter-dataset",
        type=str,
        default=None,
        help="Filter by dataset name(s), comma-separated",
    )
    parser.add_argument(
        "--filter-approach",
        type=str,
        default=None,
        help="Filter by approach(es), comma-separated (bon, beam_search, dvts)",
    )
    parser.add_argument(
        "--filter-strategy",
        type=str,
        default=None,
        help="Filter by strategy(ies), comma-separated (hnc, default)",
    )
    parser.add_argument(
        "--filter-name",
        type=str,
        default=None,
        help="Filter by result name(s), comma-separated",
    )

    # Analysis options
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["hnc_comparison", "temperature_comparison", "model_comparison", "scaling", "default"],
        default="default",
        help="Type of analysis to run",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp/analysis_output",
        help="Output directory for results",
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
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def print_registry_summary(registry: Registry, filtered: list[ResultEntry]):
    """Print summary of registry and filtered results."""
    print(f"\n{'='*60}")
    print("Registry Summary")
    print(f"{'='*60}")
    print(f"Total results in registry: {len(registry.results)}")
    print(f"Filtered results: {len(filtered)}")

    if filtered:
        print("\nFiltered results:")
        for result in filtered:
            print(f"  - {result.name}")
            print(f"    Model: {result.model}, Dataset: {result.dataset}")
            print(f"    Approach: {result.approach}, Strategy: {result.strategy}")


# =============================================================================
# Data Loading and Processing
# =============================================================================

def load_all_results(
    results: list[ResultEntry],
    verbose: bool = True,
) -> dict:
    """
    Load all results and analyze them.

    Returns:
        Dict mapping result_name -> seed -> {method -> n -> accuracy, 'pass@k' -> k -> value}
    """
    all_results = defaultdict(lambda: defaultdict(dict))

    for result in results:
        print(f"\nLoading: {result.name}")

        try:
            datasets = load_from_registry(result, verbose=verbose)

            if not datasets:
                print(f"  Warning: No datasets loaded for {result.name}")
                continue

            for seed, dataset in datasets.items():
                # Analyze pred_* methods (naive, weighted, maj)
                metrics = analyze_single_dataset(dataset, result.name, seed, verbose=verbose)
                all_results[result.name][seed] = metrics

                # Analyze pass@k
                pass_at_k = analyze_pass_at_k(dataset, result.name, seed, verbose=verbose)
                all_results[result.name][seed]['pass@k'] = pass_at_k

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return dict(all_results)


def group_results_by_key(
    results: list[ResultEntry],
    key: str,
) -> dict[str, list[ResultEntry]]:
    """Group results by a specific key (model, approach, strategy, dataset)."""
    grouped = defaultdict(list)
    for result in results:
        value = getattr(result, key)
        grouped[value].append(result)
    return dict(grouped)


# =============================================================================
# HNC vs Default Comparison (MATH-500)
# =============================================================================

def plot_hnc_comparison(
    all_data: dict,
    results: list[ResultEntry],
    approach: str,
    method: str,
    output_path: str,
):
    """
    Compare HNC vs Default-T0.8 vs Default-T0.4 for same approach and method.
    Lines with error bands.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by strategy and temperature
    strategy_groups = {
        'hnc': [],
        'default-T0.8': [],
        'default-T0.4': [],
    }

    for result in results:
        if result.approach != approach:
            continue
        if result.strategy == 'hnc':
            strategy_groups['hnc'].append(result)
        elif result.strategy == 'default':
            if 0.8 in result.temperatures:
                strategy_groups['default-T0.8'].append(result)
            elif 0.4 in result.temperatures:
                strategy_groups['default-T0.4'].append(result)

    colors = {'hnc': '#1f77b4', 'default-T0.8': '#ff7f0e', 'default-T0.4': '#2ca02c'}

    for strategy, strategy_results in strategy_groups.items():
        if not strategy_results:
            continue

        # Collect all n values and accuracies across seeds
        all_n_values = set()
        seed_data = defaultdict(lambda: defaultdict(list))

        for result in strategy_results:
            if result.name not in all_data:
                continue
            for seed, metrics in all_data[result.name].items():
                if method in metrics:
                    for n, acc in metrics[method].items():
                        all_n_values.add(n)
                        seed_data[n][seed].append(acc)

        if not all_n_values:
            continue

        n_values = sorted(all_n_values)
        means = []
        stds = []
        valid_n = []

        for n in n_values:
            values = []
            for seed_values in seed_data[n].values():
                values.extend(seed_values)

            if values:
                valid_n.append(n)
                means.append(np.mean(values))
                stds.append(np.std(values))

        if valid_n:
            means = np.array(means)
            stds = np.array(stds)

            # HNC: solid line with markers, Default: dashed line
            if strategy == 'hnc':
                ax.plot(valid_n, means, 'o-', label=strategy,
                       color=colors[strategy], linewidth=2, markersize=8)
            else:
                ax.plot(valid_n, means, '--', label=f'{strategy} (baseline)',
                       color=colors[strategy], linewidth=2)

            ax.fill_between(valid_n, means - stds, means + stds,
                           alpha=0.2, color=colors[strategy])

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{approach.replace("_", " ").title()} - {method.upper()}\nHNC vs Default Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def plot_scaling_curves(
    all_data: dict,
    results: list[ResultEntry],
    approach: str,
    output_path: str,
):
    """
    Scaling curves showing all methods for all strategies.
    X-axis in log scale.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['naive', 'weighted', 'maj']

    # Group by strategy
    strategy_groups = {
        'hnc': [],
        'default-T0.8': [],
        'default-T0.4': [],
    }

    for result in results:
        if result.approach != approach:
            continue
        if result.strategy == 'hnc':
            strategy_groups['hnc'].append(result)
        elif result.strategy == 'default':
            if 0.8 in result.temperatures:
                strategy_groups['default-T0.8'].append(result)
            elif 0.4 in result.temperatures:
                strategy_groups['default-T0.4'].append(result)

    # Color mapping
    colors = {
        'hnc': {'naive': '#1f77b4', 'weighted': '#aec7e8', 'maj': '#3182bd'},
        'default-T0.8': {'naive': '#ff7f0e', 'weighted': '#ffbb78', 'maj': '#d62728'},
        'default-T0.4': {'naive': '#2ca02c', 'weighted': '#98df8a', 'maj': '#ff9896'}
    }

    for strategy, strategy_results in strategy_groups.items():
        if not strategy_results:
            continue

        for method in methods:
            # Collect data
            all_n_values = set()
            seed_data = defaultdict(list)

            for result in strategy_results:
                if result.name not in all_data:
                    continue
                for seed, metrics in all_data[result.name].items():
                    if method in metrics:
                        for n, acc in metrics[method].items():
                            all_n_values.add(n)
                            seed_data[n].append(acc)

            if not all_n_values:
                continue

            n_values = sorted(all_n_values)
            means = []
            stds = []

            for n in n_values:
                if seed_data[n]:
                    means.append(np.mean(seed_data[n]))
                    stds.append(np.std(seed_data[n]))

            if means:
                means = np.array(means)
                stds = np.array(stds)

                # HNC: solid line with markers, Default: dashed line
                if strategy == 'hnc':
                    label = f'{strategy}-{method}'
                    ax.plot(n_values, means, 'o-', label=label,
                           color=colors[strategy][method], linewidth=2, markersize=6)
                else:
                    label = f'{strategy}-{method}'
                    ax.plot(n_values, means, '--', label=label,
                           color=colors[strategy][method], linewidth=2)

                ax.fill_between(n_values, means - stds, means + stds,
                               alpha=0.15, color=colors[strategy][method])

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{approach.replace("_", " ").title()}\nScaling Curves',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def plot_pass_at_k_curves(
    all_data: dict,
    results: list[ResultEntry],
    approach: str,
    output_path: str,
):
    """
    Pass@k scaling curves comparing strategies.
    X-axis: k values (log scale)
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by strategy
    strategy_groups = {
        'hnc': [],
        'default-T0.8': [],
        'default-T0.4': [],
    }

    for result in results:
        if result.approach != approach:
            continue
        if result.strategy == 'hnc':
            strategy_groups['hnc'].append(result)
        elif result.strategy == 'default':
            if 0.8 in result.temperatures:
                strategy_groups['default-T0.8'].append(result)
            elif 0.4 in result.temperatures:
                strategy_groups['default-T0.4'].append(result)

    colors = {'hnc': '#1f77b4', 'default-T0.8': '#ff7f0e', 'default-T0.4': '#2ca02c'}

    for strategy, strategy_results in strategy_groups.items():
        if not strategy_results:
            continue

        # Collect all k values
        all_k_values = set()
        seed_data = defaultdict(list)

        for result in strategy_results:
            if result.name not in all_data:
                continue
            for seed, metrics in all_data[result.name].items():
                if 'pass@k' in metrics:
                    for k, val in metrics['pass@k'].items():
                        all_k_values.add(k)
                        seed_data[k].append(val)

        if not all_k_values:
            continue

        k_values = sorted(all_k_values)
        means = []
        stds = []
        valid_k = []

        for k in k_values:
            if seed_data[k]:
                valid_k.append(k)
                means.append(np.mean(seed_data[k]))
                stds.append(np.std(seed_data[k]))

        if valid_k:
            means = np.array(means)
            stds = np.array(stds)

            if strategy == 'hnc':
                ax.plot(valid_k, means, 'o-', label=strategy,
                       color=colors[strategy], linewidth=2, markersize=8)
            else:
                ax.plot(valid_k, means, '--', label=f'{strategy} (baseline)',
                       color=colors[strategy], linewidth=2)

            ax.fill_between(valid_k, means - stds, means + stds,
                           alpha=0.2, color=colors[strategy])

    ax.set_xlabel('k (number of samples)', fontsize=12)
    ax.set_ylabel('Pass@k', fontsize=12)
    ax.set_title(f'{approach.replace("_", " ").title()}\nPass@k Scaling Curves',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def plot_pass_at_k_bar_comparison(
    all_data: dict,
    results: list[ResultEntry],
    approach: str,
    output_path: str,
):
    """
    Bar chart comparing pass@k at specific k values.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    selected_k = [1, 8, 32, 64]

    # Group by strategy
    strategy_groups = {
        'hnc': [],
        'default-T0.8': [],
        'default-T0.4': [],
    }

    for result in results:
        if result.approach != approach:
            continue
        if result.strategy == 'hnc':
            strategy_groups['hnc'].append(result)
        elif result.strategy == 'default':
            if 0.8 in result.temperatures:
                strategy_groups['default-T0.8'].append(result)
            elif 0.4 in result.temperatures:
                strategy_groups['default-T0.4'].append(result)

    colors = {'hnc': '#1f77b4', 'default-T0.8': '#ff7f0e', 'default-T0.4': '#2ca02c'}
    bar_width = 0.25
    x_positions = np.arange(len(selected_k))

    for idx, (strategy, strategy_results) in enumerate(strategy_groups.items()):
        if not strategy_results:
            continue

        # Collect data
        seed_data = defaultdict(list)

        for result in strategy_results:
            if result.name not in all_data:
                continue
            for seed, metrics in all_data[result.name].items():
                if 'pass@k' in metrics:
                    for k, val in metrics['pass@k'].items():
                        seed_data[k].append(val)

        means = []
        stds = []

        for k in selected_k:
            if seed_data[k]:
                means.append(np.mean(seed_data[k]))
                stds.append(np.std(seed_data[k]))
            else:
                means.append(0)
                stds.append(0)

        if any(m > 0 for m in means):
            offset = (idx - 1) * bar_width
            ax.bar(x_positions + offset, means, bar_width,
                   yerr=stds, label=strategy, color=colors[strategy],
                   capsize=5, alpha=0.8)

    ax.set_xlabel('k (number of samples)', fontsize=12)
    ax.set_ylabel('Pass@k', fontsize=12)
    ax.set_title(f'{approach.replace("_", " ").title()}\nPass@k Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(selected_k)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def run_hnc_comparison(
    results: list[ResultEntry],
    output_dir: str,
    verbose: bool = True,
    generate_plots: bool = True,
    generate_report: bool = True,
):
    """Run HNC vs Default comparison analysis for MATH-500."""
    print(f"\n{'='*60}")
    print("Running HNC vs Default Comparison Analysis")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    all_data = load_all_results(results, verbose=verbose)

    if not all_data:
        print("No data loaded!")
        return

    # Get unique approaches
    approaches = list(set(r.approach for r in results))
    methods = ['naive', 'weighted', 'maj']

    if generate_plots:
        print("\n  Generating plots...")

        for approach in approaches:
            print(f"\n  Approach: {approach}")

            # Scaling curves
            plot_scaling_curves(
                all_data, results, approach,
                os.path.join(output_dir, f'{approach}-scaling_curves.png')
            )

            # HNC vs Default comparison for each method
            for method in methods:
                plot_hnc_comparison(
                    all_data, results, approach, method,
                    os.path.join(output_dir, f'{approach}-{method}-hnc_vs_default.png')
                )

            # Pass@k curves
            plot_pass_at_k_curves(
                all_data, results, approach,
                os.path.join(output_dir, f'{approach}-pass_at_k_curves.png')
            )

            # Pass@k bar comparison
            plot_pass_at_k_bar_comparison(
                all_data, results, approach,
                os.path.join(output_dir, f'{approach}-pass_at_k_comparison.png')
            )

    if generate_report:
        report_path = os.path.join(output_dir, 'analysis_report.md')
        generate_hnc_comparison_report(all_data, results, report_path)
        print(f"\nReport saved to: {report_path}")

    return all_data


def generate_hnc_comparison_report(
    all_data: dict,
    results: list[ResultEntry],
    output_path: str,
):
    """Generate markdown report for HNC vs Default comparison."""
    md_lines = ["# HNC vs Default Comparison Report\n\n"]

    # Metadata
    md_lines.append("## Metadata\n\n")
    md_lines.append(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append(f"- **Dataset**: MATH-500\n")
    md_lines.append("\n")

    approaches = list(set(r.approach for r in results))
    methods = ['naive', 'weighted', 'maj']

    for approach in approaches:
        md_lines.append(f"## {approach.upper()}\n\n")

        for method in methods:
            md_lines.append(f"### {method.upper()}\n\n")

            # Collect data by strategy
            strategy_data = {
                'hnc': defaultdict(list),
                'default-T0.8': defaultdict(list),
                'default-T0.4': defaultdict(list),
            }

            for result in results:
                if result.approach != approach:
                    continue
                if result.name not in all_data:
                    continue

                strategy_key = None
                if result.strategy == 'hnc':
                    strategy_key = 'hnc'
                elif result.strategy == 'default':
                    if 0.8 in result.temperatures:
                        strategy_key = 'default-T0.8'
                    elif 0.4 in result.temperatures:
                        strategy_key = 'default-T0.4'

                if strategy_key:
                    for seed, metrics in all_data[result.name].items():
                        if method in metrics:
                            for n, acc in metrics[method].items():
                                strategy_data[strategy_key][n].append(acc)

            # Get all n values
            all_n = set()
            for data in strategy_data.values():
                all_n.update(data.keys())

            if not all_n:
                md_lines.append("*No data available*\n\n")
                continue

            # Table header
            md_lines.append("| n | HNC | Default-T0.8 | Default-T0.4 | HNC vs T0.8 | HNC vs T0.4 |\n")
            md_lines.append("|---|-----|--------------|--------------|-------------|-------------|\n")

            for n in sorted(all_n):
                hnc_vals = strategy_data['hnc'].get(n, [])
                t08_vals = strategy_data['default-T0.8'].get(n, [])
                t04_vals = strategy_data['default-T0.4'].get(n, [])

                hnc_mean = np.mean(hnc_vals) if hnc_vals else 0
                t08_mean = np.mean(t08_vals) if t08_vals else 0
                t04_mean = np.mean(t04_vals) if t04_vals else 0

                hnc_str = f"{hnc_mean:.4f}" if hnc_vals else "N/A"
                t08_str = f"{t08_mean:.4f}" if t08_vals else "N/A"
                t04_str = f"{t04_mean:.4f}" if t04_vals else "N/A"

                diff_t08 = hnc_mean - t08_mean if hnc_vals and t08_vals else 0
                diff_t04 = hnc_mean - t04_mean if hnc_vals and t04_vals else 0

                diff_t08_str = f"{diff_t08:+.4f}" if hnc_vals and t08_vals else "N/A"
                diff_t04_str = f"{diff_t04:+.4f}" if hnc_vals and t04_vals else "N/A"

                md_lines.append(f"| {n} | {hnc_str} | {t08_str} | {t04_str} | {diff_t08_str} | {diff_t04_str} |\n")

            md_lines.append("\n")

    with open(output_path, 'w') as f:
        f.writelines(md_lines)


# =============================================================================
# Temperature Comparison (AIME25)
# =============================================================================

def plot_temperature_comparison(
    all_data: dict,
    results: list[ResultEntry],
    model: str,
    approach: str,
    method: str,
    output_path: str,
):
    """
    Compare T=0.4 vs T=0.8 for same model and approach.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by temperature
    temp_groups = {
        'T0.4': [],
        'T0.8': [],
    }

    for result in results:
        if result.model != model or result.approach != approach:
            continue
        if 0.4 in result.temperatures:
            temp_groups['T0.4'].append(result)
        if 0.8 in result.temperatures:
            temp_groups['T0.8'].append(result)

    colors = {'T0.4': '#2ca02c', 'T0.8': '#ff7f0e'}

    for temp, temp_results in temp_groups.items():
        if not temp_results:
            continue

        # Collect data
        all_n_values = set()
        seed_data = defaultdict(list)

        for result in temp_results:
            if result.name not in all_data:
                continue
            for seed, metrics in all_data[result.name].items():
                if method in metrics:
                    for n, acc in metrics[method].items():
                        all_n_values.add(n)
                        seed_data[n].append(acc)

        if not all_n_values:
            continue

        n_values = sorted(all_n_values)
        means = []
        stds = []
        valid_n = []

        for n in n_values:
            if seed_data[n]:
                valid_n.append(n)
                means.append(np.mean(seed_data[n]))
                stds.append(np.std(seed_data[n]))

        if valid_n:
            means = np.array(means)
            stds = np.array(stds)

            if temp == 'T0.4':
                ax.plot(valid_n, means, 'o-', label=temp,
                       color=colors[temp], linewidth=2, markersize=8)
            else:
                ax.plot(valid_n, means, '--', label=f'{temp} (baseline)',
                       color=colors[temp], linewidth=2)

            ax.fill_between(valid_n, means - stds, means + stds,
                           alpha=0.2, color=colors[temp])

    model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'AIME25 - {model_short} - {approach.replace("_", " ").title()} - {method.upper()}\nTemperature Comparison',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def plot_temperature_scaling_curves(
    all_data: dict,
    results: list[ResultEntry],
    model: str,
    approach: str,
    output_path: str,
):
    """
    Scaling curves showing all methods for T=0.4 vs T=0.8.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['naive', 'weighted', 'maj']

    # Group by temperature
    temp_groups = {
        'T0.4': [],
        'T0.8': [],
    }

    for result in results:
        if result.model != model or result.approach != approach:
            continue
        if 0.4 in result.temperatures:
            temp_groups['T0.4'].append(result)
        if 0.8 in result.temperatures:
            temp_groups['T0.8'].append(result)

    # Color mapping
    colors = {
        'T0.4': {'naive': '#2ca02c', 'weighted': '#98df8a', 'maj': '#d4f1d4'},
        'T0.8': {'naive': '#ff7f0e', 'weighted': '#ffbb78', 'maj': '#ffd9b3'}
    }

    for temp, temp_results in temp_groups.items():
        if not temp_results:
            continue

        for method in methods:
            all_n_values = set()
            seed_data = defaultdict(list)

            for result in temp_results:
                if result.name not in all_data:
                    continue
                for seed, metrics in all_data[result.name].items():
                    if method in metrics:
                        for n, acc in metrics[method].items():
                            all_n_values.add(n)
                            seed_data[n].append(acc)

            if not all_n_values:
                continue

            n_values = sorted(all_n_values)
            means = []
            stds = []

            for n in n_values:
                if seed_data[n]:
                    means.append(np.mean(seed_data[n]))
                    stds.append(np.std(seed_data[n]))

            if means:
                means = np.array(means)
                stds = np.array(stds)

                label = f'{temp}-{method}'
                if temp == 'T0.4':
                    ax.plot(n_values, means, 'o-', label=label,
                           color=colors[temp][method], linewidth=2, markersize=6)
                else:
                    ax.plot(n_values, means, '--', label=label,
                           color=colors[temp][method], linewidth=2)

                ax.fill_between(n_values, means - stds, means + stds,
                               alpha=0.15, color=colors[temp][method])

    model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'AIME25 - {model_short} - {approach.replace("_", " ").title()}\nScaling Curves (T=0.4 vs T=0.8)',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def plot_temperature_pass_at_k(
    all_data: dict,
    results: list[ResultEntry],
    model: str,
    approach: str,
    output_path: str,
):
    """
    Pass@k curves comparing T=0.4 vs T=0.8.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by temperature
    temp_groups = {
        'T0.4': [],
        'T0.8': [],
    }

    for result in results:
        if result.model != model or result.approach != approach:
            continue
        if 0.4 in result.temperatures:
            temp_groups['T0.4'].append(result)
        if 0.8 in result.temperatures:
            temp_groups['T0.8'].append(result)

    colors = {'T0.4': '#2ca02c', 'T0.8': '#ff7f0e'}

    for temp, temp_results in temp_groups.items():
        if not temp_results:
            continue

        all_k_values = set()
        seed_data = defaultdict(list)

        for result in temp_results:
            if result.name not in all_data:
                continue
            for seed, metrics in all_data[result.name].items():
                if 'pass@k' in metrics:
                    for k, val in metrics['pass@k'].items():
                        all_k_values.add(k)
                        seed_data[k].append(val)

        if not all_k_values:
            continue

        k_values = sorted(all_k_values)
        means = []
        stds = []
        valid_k = []

        for k in k_values:
            if seed_data[k]:
                valid_k.append(k)
                means.append(np.mean(seed_data[k]))
                stds.append(np.std(seed_data[k]))

        if valid_k:
            means = np.array(means)
            stds = np.array(stds)

            if temp == 'T0.4':
                ax.plot(valid_k, means, 'o-', label=temp,
                       color=colors[temp], linewidth=2, markersize=8)
            else:
                ax.plot(valid_k, means, '--', label=f'{temp} (baseline)',
                       color=colors[temp], linewidth=2)

            ax.fill_between(valid_k, means - stds, means + stds,
                           alpha=0.2, color=colors[temp])

    model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
    ax.set_xlabel('k (number of samples)', fontsize=12)
    ax.set_ylabel('Pass@k', fontsize=12)
    ax.set_title(f'AIME25 - {model_short} - {approach.replace("_", " ").title()}\nPass@k (T=0.4 vs T=0.8)',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def run_temperature_comparison(
    results: list[ResultEntry],
    output_dir: str,
    verbose: bool = True,
    generate_plots: bool = True,
    generate_report: bool = True,
):
    """Run temperature comparison analysis for AIME25."""
    print(f"\n{'='*60}")
    print("Running Temperature Comparison Analysis")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    all_data = load_all_results(results, verbose=verbose)

    if not all_data:
        print("No data loaded!")
        return

    # Get unique models and approaches
    models = list(set(r.model for r in results))
    approaches = list(set(r.approach for r in results))
    methods = ['naive', 'weighted', 'maj']

    if generate_plots:
        print("\n  Generating plots...")

        for model in models:
            model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
            print(f"\n  Model: {model_short}")

            for approach in approaches:
                print(f"    Approach: {approach}")

                # Scaling curves
                plot_temperature_scaling_curves(
                    all_data, results, model, approach,
                    os.path.join(output_dir, f'aime25-{model_short}-{approach}-scaling_curves.png')
                )

                # Temperature comparison for each method
                for method in methods:
                    plot_temperature_comparison(
                        all_data, results, model, approach, method,
                        os.path.join(output_dir, f'aime25-{model_short}-{approach}-{method}-temp_comparison.png')
                    )

                # Pass@k curves
                plot_temperature_pass_at_k(
                    all_data, results, model, approach,
                    os.path.join(output_dir, f'aime25-{model_short}-{approach}-pass_at_k_curves.png')
                )

    if generate_report:
        report_path = os.path.join(output_dir, 'analysis_report.md')
        generate_temperature_comparison_report(all_data, results, report_path)
        print(f"\nReport saved to: {report_path}")

    return all_data


def generate_temperature_comparison_report(
    all_data: dict,
    results: list[ResultEntry],
    output_path: str,
):
    """Generate markdown report for temperature comparison."""
    md_lines = ["# Temperature Comparison Report (AIME25)\n\n"]

    md_lines.append("## Metadata\n\n")
    md_lines.append(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append(f"- **Dataset**: AIME25\n")
    md_lines.append("\n")

    models = list(set(r.model for r in results))
    approaches = list(set(r.approach for r in results))
    methods = ['naive', 'weighted', 'maj']

    for model in models:
        model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
        md_lines.append(f"## Model: {model_short}\n\n")

        for approach in approaches:
            md_lines.append(f"### {approach.upper()}\n\n")

            for method in methods:
                md_lines.append(f"#### {method.upper()}\n\n")

                # Collect data by temperature
                temp_data = {
                    'T0.4': defaultdict(list),
                    'T0.8': defaultdict(list),
                }

                for result in results:
                    if result.model != model or result.approach != approach:
                        continue
                    if result.name not in all_data:
                        continue

                    temp_key = None
                    if 0.4 in result.temperatures:
                        temp_key = 'T0.4'
                    elif 0.8 in result.temperatures:
                        temp_key = 'T0.8'

                    if temp_key:
                        for seed, metrics in all_data[result.name].items():
                            if method in metrics:
                                for n, acc in metrics[method].items():
                                    temp_data[temp_key][n].append(acc)

                all_n = set()
                for data in temp_data.values():
                    all_n.update(data.keys())

                if not all_n:
                    md_lines.append("*No data available*\n\n")
                    continue

                md_lines.append("| n | T=0.4 | T=0.8 | T=0.4 vs T=0.8 |\n")
                md_lines.append("|---|-------|-------|----------------|\n")

                for n in sorted(all_n):
                    t04_vals = temp_data['T0.4'].get(n, [])
                    t08_vals = temp_data['T0.8'].get(n, [])

                    t04_mean = np.mean(t04_vals) if t04_vals else 0
                    t08_mean = np.mean(t08_vals) if t08_vals else 0

                    t04_str = f"{t04_mean:.4f}" if t04_vals else "N/A"
                    t08_str = f"{t08_mean:.4f}" if t08_vals else "N/A"

                    diff = t04_mean - t08_mean if t04_vals and t08_vals else 0
                    diff_str = f"{diff:+.4f}" if t04_vals and t08_vals else "N/A"

                    md_lines.append(f"| {n} | {t04_str} | {t08_str} | {diff_str} |\n")

                md_lines.append("\n")

    with open(output_path, 'w') as f:
        f.writelines(md_lines)


# =============================================================================
# Model Size Comparison
# =============================================================================

def plot_model_comparison(
    all_data: dict,
    results: list[ResultEntry],
    approach: str,
    method: str,
    temperature: float,
    output_path: str,
):
    """
    Compare model sizes (e.g., 1.5B vs 3B) for same approach and method.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model
    model_groups = defaultdict(list)
    for result in results:
        if result.approach != approach:
            continue
        if temperature not in result.temperatures:
            continue
        model_groups[result.model].append(result)

    # Define colors for models
    model_colors = {}
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, model in enumerate(sorted(model_groups.keys())):
        model_colors[model] = color_list[idx % len(color_list)]

    for model, model_results in model_groups.items():
        if not model_results:
            continue

        all_n_values = set()
        seed_data = defaultdict(list)

        for result in model_results:
            if result.name not in all_data:
                continue
            for seed, metrics in all_data[result.name].items():
                if method in metrics:
                    for n, acc in metrics[method].items():
                        all_n_values.add(n)
                        seed_data[n].append(acc)

        if not all_n_values:
            continue

        n_values = sorted(all_n_values)
        means = []
        stds = []
        valid_n = []

        for n in n_values:
            if seed_data[n]:
                valid_n.append(n)
                means.append(np.mean(seed_data[n]))
                stds.append(np.std(seed_data[n]))

        if valid_n:
            means = np.array(means)
            stds = np.array(stds)

            model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
            ax.plot(valid_n, means, 'o-', label=model_short,
                   color=model_colors[model], linewidth=2, markersize=8)
            ax.fill_between(valid_n, means - stds, means + stds,
                           alpha=0.2, color=model_colors[model])

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'AIME25 - {approach.replace("_", " ").title()} - {method.upper()}\nModel Size Comparison (T={temperature})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def plot_model_scaling_curves(
    all_data: dict,
    results: list[ResultEntry],
    approach: str,
    temperature: float,
    output_path: str,
):
    """
    Scaling curves comparing models for all methods.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = ['naive', 'weighted', 'maj']

    # Group by model
    model_groups = defaultdict(list)
    for result in results:
        if result.approach != approach:
            continue
        if temperature not in result.temperatures:
            continue
        model_groups[result.model].append(result)

    # Color mapping
    base_colors = {
        'Qwen2.5-1.5B-Instruct': '#1f77b4',
        'Qwen2.5-3B-Instruct': '#ff7f0e',
    }

    method_alphas = {'naive': 1.0, 'weighted': 0.7, 'maj': 0.5}

    for model, model_results in model_groups.items():
        if not model_results:
            continue

        base_color = base_colors.get(model, '#333333')
        model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")

        for method in methods:
            all_n_values = set()
            seed_data = defaultdict(list)

            for result in model_results:
                if result.name not in all_data:
                    continue
                for seed, metrics in all_data[result.name].items():
                    if method in metrics:
                        for n, acc in metrics[method].items():
                            all_n_values.add(n)
                            seed_data[n].append(acc)

            if not all_n_values:
                continue

            n_values = sorted(all_n_values)
            means = []
            stds = []

            for n in n_values:
                if seed_data[n]:
                    means.append(np.mean(seed_data[n]))
                    stds.append(np.std(seed_data[n]))

            if means:
                means = np.array(means)
                stds = np.array(stds)

                label = f'{model_short}-{method}'
                ax.plot(n_values, means, 'o-', label=label,
                       color=base_color, linewidth=2, markersize=6,
                       alpha=method_alphas[method])
                ax.fill_between(n_values, means - stds, means + stds,
                               alpha=0.15 * method_alphas[method], color=base_color)

    ax.set_xlabel('Number of Samples (n)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'AIME25 - {approach.replace("_", " ").title()}\nModel Size Comparison (T={temperature})',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def plot_model_pass_at_k(
    all_data: dict,
    results: list[ResultEntry],
    approach: str,
    temperature: float,
    output_path: str,
):
    """
    Pass@k curves comparing models.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by model
    model_groups = defaultdict(list)
    for result in results:
        if result.approach != approach:
            continue
        if temperature not in result.temperatures:
            continue
        model_groups[result.model].append(result)

    model_colors = {
        'Qwen2.5-1.5B-Instruct': '#1f77b4',
        'Qwen2.5-3B-Instruct': '#ff7f0e',
    }

    for model, model_results in model_groups.items():
        if not model_results:
            continue

        all_k_values = set()
        seed_data = defaultdict(list)

        for result in model_results:
            if result.name not in all_data:
                continue
            for seed, metrics in all_data[result.name].items():
                if 'pass@k' in metrics:
                    for k, val in metrics['pass@k'].items():
                        all_k_values.add(k)
                        seed_data[k].append(val)

        if not all_k_values:
            continue

        k_values = sorted(all_k_values)
        means = []
        stds = []
        valid_k = []

        for k in k_values:
            if seed_data[k]:
                valid_k.append(k)
                means.append(np.mean(seed_data[k]))
                stds.append(np.std(seed_data[k]))

        if valid_k:
            means = np.array(means)
            stds = np.array(stds)

            model_short = model.replace("Qwen2.5-", "").replace("-Instruct", "")
            color = model_colors.get(model, '#333333')
            ax.plot(valid_k, means, 'o-', label=model_short,
                   color=color, linewidth=2, markersize=8)
            ax.fill_between(valid_k, means - stds, means + stds,
                           alpha=0.2, color=color)

    ax.set_xlabel('k (number of samples)', fontsize=12)
    ax.set_ylabel('Pass@k', fontsize=12)
    ax.set_title(f'AIME25 - {approach.replace("_", " ").title()}\nModel Size Pass@k Comparison (T={temperature})',
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"    Saved: {output_path}")


def run_model_comparison(
    results: list[ResultEntry],
    output_dir: str,
    verbose: bool = True,
    generate_plots: bool = True,
    generate_report: bool = True,
):
    """Run model size comparison analysis."""
    print(f"\n{'='*60}")
    print("Running Model Size Comparison Analysis")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    all_data = load_all_results(results, verbose=verbose)

    if not all_data:
        print("No data loaded!")
        return

    # Get unique approaches and temperatures
    approaches = list(set(r.approach for r in results))
    all_temps = set()
    for r in results:
        all_temps.update(r.temperatures)
    temperatures = sorted(all_temps)
    methods = ['naive', 'weighted', 'maj']

    if generate_plots:
        print("\n  Generating plots...")

        for approach in approaches:
            for temp in temperatures:
                print(f"\n  Approach: {approach}, T={temp}")

                # Model scaling curves
                plot_model_scaling_curves(
                    all_data, results, approach, temp,
                    os.path.join(output_dir, f'aime25-{approach}-T{temp}-model_scaling.png')
                )

                # Model comparison for each method
                for method in methods:
                    plot_model_comparison(
                        all_data, results, approach, method, temp,
                        os.path.join(output_dir, f'aime25-{approach}-{method}-T{temp}-model_comparison.png')
                    )

                # Model pass@k comparison
                plot_model_pass_at_k(
                    all_data, results, approach, temp,
                    os.path.join(output_dir, f'aime25-{approach}-T{temp}-model_pass_at_k.png')
                )

    if generate_report:
        report_path = os.path.join(output_dir, 'analysis_report.md')
        generate_model_comparison_report(all_data, results, report_path)
        print(f"\nReport saved to: {report_path}")

    return all_data


def generate_model_comparison_report(
    all_data: dict,
    results: list[ResultEntry],
    output_path: str,
):
    """Generate markdown report for model comparison."""
    md_lines = ["# Model Size Comparison Report\n\n"]

    md_lines.append("## Metadata\n\n")
    md_lines.append(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    md_lines.append(f"- **Dataset**: AIME25\n")
    md_lines.append("\n")

    approaches = list(set(r.approach for r in results))
    all_temps = set()
    for r in results:
        all_temps.update(r.temperatures)
    temperatures = sorted(all_temps)
    methods = ['naive', 'weighted', 'maj']
    models = sorted(set(r.model for r in results))

    for temp in temperatures:
        md_lines.append(f"## Temperature: {temp}\n\n")

        for approach in approaches:
            md_lines.append(f"### {approach.upper()}\n\n")

            for method in methods:
                md_lines.append(f"#### {method.upper()}\n\n")

                # Collect data by model
                model_data = {m: defaultdict(list) for m in models}

                for result in results:
                    if result.approach != approach:
                        continue
                    if temp not in result.temperatures:
                        continue
                    if result.name not in all_data:
                        continue

                    for seed, metrics in all_data[result.name].items():
                        if method in metrics:
                            for n, acc in metrics[method].items():
                                model_data[result.model][n].append(acc)

                all_n = set()
                for data in model_data.values():
                    all_n.update(data.keys())

                if not all_n:
                    md_lines.append("*No data available*\n\n")
                    continue

                # Table header
                model_shorts = [m.replace("Qwen2.5-", "").replace("-Instruct", "") for m in models]
                header = "| n | " + " | ".join(model_shorts) + " | Improvement |\n"
                sep = "|---| " + " | ".join(["---"] * len(models)) + " | --- |\n"
                md_lines.append(header)
                md_lines.append(sep)

                for n in sorted(all_n):
                    row = [str(n)]
                    vals = []
                    for model in models:
                        model_vals = model_data[model].get(n, [])
                        if model_vals:
                            mean = np.mean(model_vals)
                            row.append(f"{mean:.4f}")
                            vals.append(mean)
                        else:
                            row.append("N/A")
                            vals.append(0)

                    if len(vals) >= 2 and all(v > 0 for v in vals):
                        diff = vals[-1] - vals[0]
                        row.append(f"{diff:+.4f}")
                    else:
                        row.append("N/A")

                    md_lines.append("| " + " | ".join(row) + " |\n")

                md_lines.append("\n")

    with open(output_path, 'w') as f:
        f.writelines(md_lines)


# =============================================================================
# Default Analysis
# =============================================================================

def run_default_analysis(
    results: list[ResultEntry],
    output_dir: str,
    verbose: bool = True,
    generate_plots: bool = True,
    generate_report: bool = True,
):
    """Run default analysis: compute accuracy for each result."""
    print(f"\n{'='*60}")
    print("Running Default Analysis")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    all_data = load_all_results(results, verbose=verbose)

    if not all_data:
        print("No data loaded!")
        return

    # Summarize results
    all_accuracies = {}

    for result in results:
        if result.name not in all_data:
            continue

        best_accuracies = []
        for seed, metrics in all_data[result.name].items():
            naive = metrics.get("naive", {})
            if naive:
                max_n = max(naive.keys())
                best_accuracies.append(naive[max_n])

        if best_accuracies:
            all_accuracies[result.name] = {
                "mean": np.mean(best_accuracies),
                "std": np.std(best_accuracies),
                "seeds": list(all_data[result.name].keys()),
            }
            print(f"  {result.name}: {all_accuracies[result.name]['mean']:.3f} ± {all_accuracies[result.name]['std']:.3f}")

    if generate_report and all_accuracies:
        report_path = os.path.join(output_dir, "analysis_report.md")

        md_lines = [
            "# Analysis Report\n\n",
            "## Summary\n\n",
            "| Result | Accuracy | Std | Seeds |\n",
            "|--------|----------|-----|-------|\n",
        ]

        for name, data in sorted(all_accuracies.items()):
            seeds_str = ", ".join(map(str, data.get("seeds", [])))
            md_lines.append(
                f"| {name} | {data['mean']:.3f} | {data['std']:.3f} | {seeds_str} |\n"
            )

        with open(report_path, "w") as f:
            f.writelines(md_lines)

        print(f"\nReport saved to: {report_path}")

    return all_accuracies


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Load registry
    print(f"Loading registry from: {args.registry}")
    registry = load_registry(args.registry)

    # Apply filters
    filtered = registry.filter(
        model=args.filter_model,
        dataset=args.filter_dataset,
        approach=args.filter_approach,
        strategy=args.filter_strategy,
        name=args.filter_name,
    )

    # Print summary
    print_registry_summary(registry, filtered)

    if not filtered:
        print("\nNo results match the specified filters.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run analysis based on type
    if args.analysis_type == "hnc_comparison":
        run_hnc_comparison(
            filtered,
            args.output_dir,
            verbose=args.verbose,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
        )
    elif args.analysis_type == "temperature_comparison":
        run_temperature_comparison(
            filtered,
            args.output_dir,
            verbose=args.verbose,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
        )
    elif args.analysis_type == "model_comparison":
        run_model_comparison(
            filtered,
            args.output_dir,
            verbose=args.verbose,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
        )
    elif args.analysis_type == "scaling":
        # Same as hnc_comparison but only scaling curves
        run_hnc_comparison(
            filtered,
            args.output_dir,
            verbose=args.verbose,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
        )
    elif args.analysis_type == "default":
        run_default_analysis(
            filtered,
            args.output_dir,
            verbose=args.verbose,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report,
        )

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
