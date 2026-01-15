#!/usr/bin/env python
"""Unified analysis script for experiment results.

This script provides a unified interface for analyzing experiment results
using the registry-based configuration system.

Usage:
    # Basic usage with filters
    python exp/scripts/analyze_results.py \\
        --registry=exp/configs/registry.yaml \\
        --filter-model="Qwen2.5-1.5B-Instruct" \\
        --filter-dataset="MATH-500" \\
        --filter-strategy="hnc"

    # AIME25 model comparison
    python exp/scripts/analyze_results.py \\
        --registry=exp/configs/registry.yaml \\
        --filter-dataset="aime25" \\
        --analysis-type="model_comparison"

    # Temperature analysis
    python exp/scripts/analyze_results.py \\
        --registry=exp/configs/registry.yaml \\
        --filter-strategy="hnc" \\
        --analysis-type="temperature"
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add exp directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_registry, Registry, ResultEntry
from analysis import (
    load_from_registry,
    analyze_single_dataset,
    analyze_pass_at_k,
    aggregate_across_seeds,
    compute_accuracy_by_method,
    setup_style,
    plot_comparison,
    plot_bar_comparison,
    create_results_table,
    STRATEGY_COLORS,
    METHOD_COLORS,
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
        choices=["default", "model_comparison", "temperature", "scaling"],
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
        "--output-format",
        type=str,
        choices=["human", "oneline", "json"],
        default="human",
        help="Output format",
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
            print(f"    Hub: {result.hub_path}")
            print(f"    Model: {result.model}, Dataset: {result.dataset}")
            print(f"    Approach: {result.approach}, Strategy: {result.strategy}")
            print(f"    Seeds: {result.seeds}, Temps: {result.temperatures}")


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

    all_accuracies = {}

    for result in results:
        print(f"\nAnalyzing: {result.name}")

        try:
            # Load datasets
            datasets = load_from_registry(result, verbose=verbose)

            if not datasets:
                print(f"  Warning: No datasets loaded for {result.name}")
                continue

            # Analyze each seed
            seed_results = {}
            for seed, dataset in datasets.items():
                metrics = analyze_single_dataset(
                    dataset, result.name, seed, verbose=verbose
                )
                seed_results[seed] = metrics

            # Aggregate across seeds
            if seed_results:
                # Get best accuracy (naive@64 or highest n)
                best_accuracies = []
                for seed, metrics in seed_results.items():
                    naive = metrics.get("naive", {})
                    if naive:
                        max_n = max(naive.keys())
                        best_accuracies.append(naive[max_n])

                if best_accuracies:
                    import numpy as np

                    all_accuracies[result.name] = {
                        "mean": np.mean(best_accuracies),
                        "std": np.std(best_accuracies),
                        "seeds": list(seed_results.keys()),
                    }
                    print(
                        f"  Accuracy: {all_accuracies[result.name]['mean']:.3f} "
                        f"± {all_accuracies[result.name]['std']:.3f}"
                    )

        except Exception as e:
            print(f"  Error analyzing {result.name}: {e}")
            continue

    # Generate report
    if generate_report and all_accuracies:
        report_path = os.path.join(output_dir, "analysis_report.md")
        generate_analysis_report(all_accuracies, report_path)
        print(f"\nReport saved to: {report_path}")

    return all_accuracies


def run_model_comparison(
    results: list[ResultEntry],
    output_dir: str,
    verbose: bool = True,
    generate_plots: bool = True,
    generate_report: bool = True,
):
    """Run model comparison analysis."""
    print(f"\n{'='*60}")
    print("Running Model Comparison Analysis")
    print(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)

    # Group results by model
    by_model = defaultdict(list)
    for result in results:
        by_model[result.model].append(result)

    print(f"\nModels found: {list(by_model.keys())}")

    model_accuracies = {}

    for model, model_results in by_model.items():
        print(f"\n--- {model} ---")
        model_accuracies[model] = {}

        for result in model_results:
            try:
                datasets = load_from_registry(result, verbose=verbose)
                if not datasets:
                    continue

                accuracies = []
                for seed, dataset in datasets.items():
                    metrics = analyze_single_dataset(
                        dataset, result.name, seed, verbose=False
                    )
                    naive = metrics.get("naive", {})
                    if naive:
                        max_n = max(naive.keys())
                        accuracies.append(naive[max_n])

                if accuracies:
                    import numpy as np

                    key = f"{result.approach}-{result.strategy}"
                    model_accuracies[model][key] = {
                        "mean": np.mean(accuracies),
                        "std": np.std(accuracies),
                    }
                    print(f"  {key}: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")

            except Exception as e:
                print(f"  Error: {e}")

    # Generate comparison plot
    if generate_plots and model_accuracies:
        plot_path = os.path.join(output_dir, "model_comparison.png")
        # Prepare data for bar chart
        categories = list(next(iter(model_accuracies.values())).keys())
        values = {model: [data.get(cat, {}).get("mean", 0) for cat in categories]
                  for model, data in model_accuracies.items()}
        errors = {model: [data.get(cat, {}).get("std", 0) for cat in categories]
                  for model, data in model_accuracies.items()}

        plot_bar_comparison(
            categories=categories,
            values=values,
            errors=errors,
            output_path=plot_path,
            title="Model Comparison",
            ylabel="Accuracy",
        )
        print(f"\nPlot saved to: {plot_path}")

    return model_accuracies


def generate_analysis_report(accuracies: dict, output_path: str):
    """Generate markdown analysis report."""
    lines = [
        "# Analysis Report\n\n",
        "## Summary\n\n",
        "| Result | Accuracy | Std | Seeds |\n",
        "|--------|----------|-----|-------|\n",
    ]

    for name, data in sorted(accuracies.items()):
        seeds_str = ", ".join(map(str, data.get("seeds", [])))
        lines.append(
            f"| {name} | {data['mean']:.3f} | {data['std']:.3f} | {seeds_str} |\n"
        )

    with open(output_path, "w") as f:
        f.writelines(lines)


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
    if args.analysis_type == "default":
        run_default_analysis(
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
    elif args.analysis_type == "temperature":
        print("Temperature analysis not yet implemented")
    elif args.analysis_type == "scaling":
        print("Scaling analysis not yet implemented")

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
