#!/usr/bin/env python
"""Preprocess experiment datasets by pre-computing evaluation results.

This script loads raw experiment datasets, evaluates all predictions using
math_verify, and saves the results as boolean fields (is_correct_*).

This preprocessing step is done once per experiment, and then analysis scripts
can use the preprocessed datasets for 20-30x faster analysis.

Usage:
    # Preprocess a single hub path
    python exp/scripts/preprocess_dataset.py \
        --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
        --push-to-hub

    # Preprocess from category (all experiments in category)
    python exp/scripts/preprocess_dataset.py \
        --category math500_Qwen2.5-1.5B \
        --push-to-hub

    # Preprocess specific subsets only
    python exp/scripts/preprocess_dataset.py \
        --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
        --subsets "HuggingFaceH4_MATH-500--T-0.4--top_p-1.0--n-64--seed-42--agg_strategy-last" \
        --push-to-hub

    # Save locally instead of pushing to hub
    python exp/scripts/preprocess_dataset.py \
        --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
        --output-dir /tmp/preprocessed

    # Custom output hub path
    python exp/scripts/preprocess_dataset.py \
        --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
        --output-hub-path ENSEONG/my-custom-preprocessed-dataset \
        --push-to-hub
"""

import argparse
import os
import sys
from pathlib import Path

from datasets import load_dataset

# Add exp directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_hub_registry
from analysis import discover_experiment
from analysis.preprocessing import (
    preprocess_single_subset,
    validate_preprocessing,
    get_preprocessing_stats,
)


def get_default_output_hub_path(input_hub_path: str) -> str:
    """Generate default output hub path by adding 'preprocessed-' prefix.

    Args:
        input_hub_path: Original hub path (e.g., "ENSEONG/default-MATH-500-...")

    Returns:
        Output hub path (e.g., "ENSEONG/preprocessed-default-MATH-500-...")
    """
    parts = input_hub_path.split("/")
    if len(parts) == 2:
        org, repo = parts
        return f"{org}/preprocessed-{repo}"
    else:
        return f"preprocessed-{input_hub_path}"


def preprocess_experiment(
    hub_path: str,
    output_hub_path: str = None,
    output_dir: str = None,
    push_to_hub: bool = False,
    subset_filter: list[str] = None,
    validate: bool = False,
    force: bool = False,
    verbose: bool = True,
):
    """Preprocess entire experiment (all subsets).

    Args:
        hub_path: Source hub dataset path
        output_hub_path: Target hub path (default: add 'preprocessed-' prefix)
        output_dir: Local directory to save (if not pushing to hub)
        push_to_hub: Whether to push to Hub
        subset_filter: Only process specific subsets (default: all)
        validate: Run validation after preprocessing
        force: Force reprocessing even if already preprocessed
        verbose: Print progress messages
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Preprocessing Experiment")
        print(f"{'='*70}")
        print(f"  Source: {hub_path}")

    # Determine output path
    if push_to_hub:
        if output_hub_path is None:
            output_hub_path = get_default_output_hub_path(hub_path)
        if verbose:
            print(f"  Target: {output_hub_path}")
    elif output_dir:
        if verbose:
            print(f"  Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    else:
        print("Error: Must specify either --push-to-hub or --output-dir")
        sys.exit(1)

    # Discover all subsets
    if verbose:
        print(f"\n  Discovering subsets...")

    try:
        config = discover_experiment(hub_path)
    except Exception as e:
        print(f"Error discovering experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if verbose:
        print(f"    Found {len(config.subsets)} subsets")
        print(f"    Seeds: {config.seeds}")
        print(f"    Temperatures: {config.temperatures}")
        print(f"    Approach: {config.approach}")
        print(f"    Strategy: {config.strategy}")

    # Filter subsets if specified
    subsets_to_process = config.subsets
    if subset_filter:
        subsets_to_process = [s for s in config.subsets if s.raw_name in subset_filter]
        if verbose:
            print(f"    Filtering to {len(subsets_to_process)} subsets")

    if not subsets_to_process:
        print("No subsets to process")
        sys.exit(1)

    # Process each subset
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, subset in enumerate(subsets_to_process, 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Subset {idx}/{len(subsets_to_process)}: {subset.raw_name}")
            print(f"{'='*70}")

        try:
            # Load original dataset
            if verbose:
                print(f"  Loading from {hub_path}...")

            dataset = load_dataset(hub_path, subset.raw_name)

            if "train" not in dataset:
                print(f"  Warning: No 'train' split found. Skipping.")
                skip_count += 1
                continue

            original_dataset = dataset["train"]

            # Check if already preprocessed
            if not force:
                stats = get_preprocessing_stats(original_dataset)
                if stats["is_preprocessed"]:
                    if verbose:
                        print(f"  Already preprocessed ({stats['num_is_correct_fields']} fields)")
                        print(f"  Use --force to reprocess")
                    skip_count += 1
                    continue

            # Preprocess
            preprocessed_dataset = preprocess_single_subset(
                original_dataset,
                subset.raw_name,
                add_metadata=True,
                verbose=verbose,
            )

            # Validate if requested
            if validate:
                if verbose:
                    print(f"\n  Validating preprocessing...")
                is_valid = validate_preprocessing(
                    original_dataset,
                    preprocessed_dataset,
                    num_samples=100,
                    verbose=verbose,
                )
                if not is_valid:
                    print(f"  Warning: Validation failed for {subset.raw_name}")

            # Save results
            if push_to_hub:
                if verbose:
                    print(f"\n  Pushing to Hub: {output_hub_path}")
                    print(f"    Config: {subset.raw_name}")

                preprocessed_dataset.push_to_hub(
                    output_hub_path,
                    config_name=subset.raw_name,
                    private=False,
                )

                if verbose:
                    print(f"  ✓ Successfully pushed to Hub")
            else:
                # Save locally
                subset_dir = os.path.join(output_dir, subset.raw_name)
                if verbose:
                    print(f"\n  Saving to {subset_dir}")

                preprocessed_dataset.save_to_disk(subset_dir)

                if verbose:
                    print(f"  ✓ Successfully saved locally")

            success_count += 1

        except Exception as e:
            print(f"\n  ✗ Error processing {subset.raw_name}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"Preprocessing Summary")
        print(f"{'='*70}")
        print(f"  Total subsets: {len(subsets_to_process)}")
        print(f"  Successful: {success_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Errors: {error_count}")
        print(f"{'='*70}")

    if error_count > 0:
        sys.exit(1)


def preprocess_category(
    category: str,
    registry_path: str,
    output_prefix: str = "preprocessed-",
    push_to_hub: bool = False,
    validate: bool = False,
    force: bool = False,
    verbose: bool = True,
):
    """Preprocess all experiments in a category.

    Args:
        category: Category name from registry
        registry_path: Path to registry YAML file
        output_prefix: Prefix for output hub paths
        push_to_hub: Whether to push to Hub
        validate: Run validation after preprocessing
        force: Force reprocessing even if already preprocessed
        verbose: Print progress messages
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Preprocessing Category: {category}")
        print(f"{'='*70}")

    # Load registry
    registry = load_hub_registry(registry_path)
    hub_paths = registry.get_category(category)

    if not hub_paths:
        print(f"Error: Category '{category}' not found in registry")
        sys.exit(1)

    if verbose:
        print(f"  Found {len(hub_paths)} experiments in category")
        for path in hub_paths:
            print(f"    - {path}")

    # Process each experiment
    for idx, hub_path in enumerate(hub_paths, 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Experiment {idx}/{len(hub_paths)}")
            print(f"{'='*70}")

        # Generate output path
        output_hub_path = get_default_output_hub_path(hub_path)

        preprocess_experiment(
            hub_path=hub_path,
            output_hub_path=output_hub_path,
            push_to_hub=push_to_hub,
            validate=validate,
            force=force,
            verbose=verbose,
        )

    if verbose:
        print(f"\n{'='*70}")
        print(f"Category preprocessing complete!")
        print(f"{'='*70}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess experiment datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--hub-path",
        type=str,
        help="Hub dataset path to preprocess",
    )
    input_group.add_argument(
        "--category",
        type=str,
        help="Category from registry to preprocess",
    )

    # Output options
    parser.add_argument(
        "--output-hub-path",
        type=str,
        help="Output hub path (default: add 'preprocessed-' prefix to input)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Local output directory (if not pushing to hub)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push preprocessed datasets to Hub",
    )

    # Processing options
    parser.add_argument(
        "--subsets",
        nargs="+",
        help="Only process specific subsets (by name)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if already preprocessed",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after preprocessing",
    )

    # Registry options (for category mode)
    parser.add_argument(
        "--registry",
        type=str,
        default="exp/configs/registry.yaml",
        help="Path to registry YAML file",
    )

    # Verbose
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=True,
        help="Verbose output (default: True)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode (disable verbose)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Handle verbose flag
    verbose = args.verbose and not args.quiet

    # Validate arguments
    if not args.push_to_hub and not args.output_dir:
        print("Error: Must specify either --push-to-hub or --output-dir")
        sys.exit(1)

    # Process based on input mode
    if args.hub_path:
        preprocess_experiment(
            hub_path=args.hub_path,
            output_hub_path=args.output_hub_path,
            output_dir=args.output_dir,
            push_to_hub=args.push_to_hub,
            subset_filter=args.subsets,
            validate=args.validate,
            force=args.force,
            verbose=verbose,
        )
    elif args.category:
        if args.subsets:
            print("Warning: --subsets is ignored when using --category")

        preprocess_category(
            category=args.category,
            registry_path=args.registry,
            push_to_hub=args.push_to_hub,
            validate=args.validate,
            force=args.force,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
