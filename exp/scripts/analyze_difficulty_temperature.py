#!/usr/bin/env python
"""Difficulty-based temperature performance analysis script.

This script analyzes how different temperatures perform across problems of
varying difficulty levels, using absolute accuracy thresholds to define difficulty.

Usage:
    # Analyze from category
    python exp/scripts/analyze_difficulty_temperature.py \
        --category math500_Qwen2.5-1.5B \
        --approach bon \
        --output-dir exp/analysis_output-MATH500-Qwen2.5-1.5B-difficulty

    # Analyze from hub path
    python exp/scripts/analyze_difficulty_temperature.py \
        --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
        --output-dir exp/analysis_output-MATH500-Qwen2.5-1.5B-difficulty

    # Customize difficulty thresholds
    python exp/scripts/analyze_difficulty_temperature.py \
        --category math500_Qwen2.5-1.5B \
        --approach bon \
        --thresholds "1:0.9,1.0" "2:0.7,0.9" "3:0.5,0.7" "4:0.3,0.5" "5:0.0,0.3"
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add exp directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_hub_registry, HubRegistry
from analysis import (
    discover_experiment,
    ExperimentConfig,
    load_experiment_data_by_temperature,
)
from analysis.difficulty_temperature import (
    DEFAULT_DIFFICULTY_THRESHOLDS,
    compute_universal_difficulty_baselines,
    analyze_temperature_by_difficulty,
    generate_difficulty_temperature_plots,
    generate_difficulty_temperature_report,
)
from analysis.difficulty import stratify_by_absolute_difficulty


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze temperature performance by difficulty level",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--hub-path",
        type=str,
        help="Hub dataset path (e.g., ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon)",
    )
    input_group.add_argument(
        "--category",
        type=str,
        help="Category from registry (requires --approach)",
    )

    # Registry options
    parser.add_argument(
        "--registry",
        type=str,
        default="exp/configs/registry.yaml",
        help="Path to registry YAML file",
    )
    parser.add_argument(
        "--approach",
        type=str,
        choices=["bon", "beam_search", "dvts"],
        help="Approach to use when using --category",
    )

    # Difficulty configuration
    parser.add_argument(
        "--thresholds",
        nargs="+",
        help='Custom difficulty thresholds as "level:min,max" (e.g., "1:0.8,1.0" "2:0.6,0.8")',
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
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
    parser.add_argument(
        "--save-baselines",
        action="store_true",
        help="Save difficulty baselines to JSON",
    )

    # Verbose
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def parse_thresholds(threshold_strings: list[str]) -> dict[int, tuple[float, float]]:
    """Parse threshold strings into difficulty thresholds dict.

    Args:
        threshold_strings: List of strings like "1:0.8,1.0"

    Returns:
        Dict mapping level -> (min_accuracy, max_accuracy)
    """
    thresholds = {}
    for s in threshold_strings:
        try:
            level_str, range_str = s.split(":")
            min_str, max_str = range_str.split(",")
            level = int(level_str)
            min_acc = float(min_str)
            max_acc = float(max_str)
            thresholds[level] = (min_acc, max_acc)
        except Exception as e:
            raise ValueError(f"Invalid threshold format '{s}': {e}")

    return thresholds


def get_hub_path_from_category(
    registry: HubRegistry,
    category: str,
    approach: str,
    verbose: bool = True,
) -> Optional[str]:
    """Get hub path from category and approach.

    Args:
        registry: Hub registry
        category: Category name (e.g., "math500_Qwen2.5-1.5B")
        approach: Approach name (bon, beam_search, dvts)
        verbose: Print messages

    Returns:
        Hub path if found, None otherwise
    """
    paths = registry.get_category(category)
    if not paths:
        if verbose:
            print(f"Error: Category '{category}' not found in registry")
        return None

    # Filter by approach
    for path in paths:
        if approach.lower() in path.lower():
            return path

    if verbose:
        print(f"Error: No path found for category '{category}' with approach '{approach}'")
        print(f"Available paths for category: {paths}")

    return None


def main():
    args = parse_args()

    # Determine hub path
    hub_path = None

    if args.hub_path:
        hub_path = args.hub_path
    elif args.category:
        if not args.approach:
            print("Error: --approach is required when using --category")
            sys.exit(1)

        registry = load_hub_registry(args.registry)
        hub_path = get_hub_path_from_category(
            registry, args.category, args.approach, verbose=args.verbose
        )

        if not hub_path:
            sys.exit(1)

    if not hub_path:
        print("Error: No hub path specified")
        sys.exit(1)

    print("=" * 70)
    print("Difficulty-Based Temperature Performance Analysis")
    print("=" * 70)
    print(f"\nHub path: {hub_path}")

    # Parse difficulty thresholds
    if args.thresholds:
        try:
            difficulty_thresholds = parse_thresholds(args.thresholds)
            if args.verbose:
                print(f"\nUsing custom difficulty thresholds:")
                for level, (min_acc, max_acc) in sorted(difficulty_thresholds.items()):
                    print(f"  Level {level}: {min_acc:.2f} - {max_acc:.2f}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        difficulty_thresholds = DEFAULT_DIFFICULTY_THRESHOLDS
        if args.verbose:
            print(f"\nUsing default difficulty thresholds:")
            for level, (min_acc, max_acc) in sorted(difficulty_thresholds.items()):
                print(f"  Level {level}: {min_acc:.2f} - {max_acc:.2f}")

    # Discover experiment configuration
    if args.verbose:
        print(f"\nDiscovering experiment configuration...")

    try:
        config = discover_experiment(hub_path)
    except Exception as e:
        print(f"Error discovering experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.verbose:
        print(f"  Model: {config.model}")
        print(f"  Approach: {config.approach}")
        print(f"  Strategy: {config.strategy}")
        print(f"  Seeds: {config.seeds}")
        print(f"  Temperatures: {config.temperatures}")

    # Filter to single temperatures only (not HNC)
    single_temps = [t for t in config.temperatures if isinstance(t, (int, float))]
    if not single_temps:
        print("Error: No single-temperature experiments found (HNC not supported)")
        sys.exit(1)

    if len(single_temps) < 2:
        print(f"Warning: Only {len(single_temps)} temperature(s) found")

    # Load datasets organized by temperature
    if args.verbose:
        print(f"\nLoading datasets for {len(single_temps)} temperatures...")

    try:
        datasets_by_temp = load_experiment_data_by_temperature(
            config,
            temperatures=single_temps,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not datasets_by_temp:
        print("Error: No datasets loaded")
        sys.exit(1)

    # Compute universal difficulty baselines
    try:
        baselines = compute_universal_difficulty_baselines(
            datasets_by_temp,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error computing difficulty baselines: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.verbose:
        print(f"\nComputed baselines for {len(baselines)} problems")

    # Save baselines if requested
    if args.save_baselines:
        baselines_path = os.path.join(args.output_dir, "difficulty_baselines.json")
        os.makedirs(args.output_dir, exist_ok=True)

        baselines_data = {
            pid: {
                "unique_id": b.unique_id,
                "mean_accuracy": b.mean_accuracy,
                "num_evaluations": b.num_evaluations,
                "answer": b.answer,
            }
            for pid, b in baselines.items()
        }

        with open(baselines_path, "w") as f:
            json.dump(baselines_data, f, indent=2)

        if args.verbose:
            print(f"Saved baselines to: {baselines_path}")

    # Stratify by absolute difficulty
    try:
        difficulty_levels = stratify_by_absolute_difficulty(
            baselines,
            thresholds=difficulty_thresholds,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error stratifying by difficulty: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Analyze temperature performance by difficulty level
    try:
        results = analyze_temperature_by_difficulty(
            datasets_by_temp,
            difficulty_levels,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error analyzing temperature by difficulty: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate plots
    if not args.no_plots:
        try:
            generate_difficulty_temperature_plots(
                results,
                difficulty_levels,
                args.output_dir,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()

    # Generate report
    if not args.no_report:
        try:
            report_path = os.path.join(args.output_dir, "difficulty_temperature_report.md")
            generate_difficulty_temperature_report(
                results,
                difficulty_levels,
                report_path,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"Error generating report: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
