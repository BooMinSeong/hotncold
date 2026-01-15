"""Dataset loading utilities for experiment analysis.

This module provides functions to load datasets from Hugging Face Hub,
with support for registry-based configuration and seed/temperature filtering.

Extracted from:
- exp/temperature_analysis_per_problem.py (load_default_datasets, load_hnc_datasets)
- exp/temperature_analysis_stratified.py (load_default_datasets, load_hnc_datasets)
"""

from collections import defaultdict
from typing import Any, Optional

from datasets import get_dataset_config_names, load_dataset

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import ResultEntry


def load_datasets_by_seed(
    dataset_path: str,
    seeds: list[int],
    subset_template: Optional[str] = None,
    filter_strings: Optional[list[str]] = None,
    temperature: Optional[float] = None,
    verbose: bool = True,
) -> dict[int, Any]:
    """Load datasets for multiple seeds.

    This unified function handles both template-based and filter-based
    subset discovery.

    Args:
        dataset_path: Hugging Face Hub dataset path
        seeds: List of seed values to load
        subset_template: Template string with {seed} and optional {temperature} placeholders
        filter_strings: List of strings that must all appear in subset name
        temperature: Temperature value for template substitution
        verbose: Whether to print progress messages

    Returns:
        Dictionary mapping seed -> Dataset

    Examples:
        >>> # Template-based loading
        >>> datasets = load_datasets_by_seed(
        ...     "ENSEONG/default-Qwen2.5-1.5B-Instruct-bon",
        ...     seeds=[0, 42, 64],
        ...     subset_template="HuggingFaceH4_MATH-500--T-{temperature}--...-seed-{seed}--...",
        ...     temperature=0.4
        ... )

        >>> # Filter-based loading
        >>> datasets = load_datasets_by_seed(
        ...     "ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon",
        ...     seeds=[128, 192, 256],
        ...     filter_strings=["T-0.8"]
        ... )
    """
    datasets = {}

    for seed in seeds:
        try:
            if subset_template:
                # Template-based subset discovery
                subset_name = subset_template.replace("{seed}", str(seed))
                if temperature is not None:
                    subset_name = subset_name.replace("{temperature}", str(temperature))
                if verbose:
                    print(f"  Seed {seed}: {subset_name}")
                dataset = load_dataset(dataset_path, subset_name)
            else:
                # Filter-based subset discovery
                configs = get_dataset_config_names(dataset_path)
                matching = [c for c in configs if f"seed-{seed}" in c]

                if filter_strings:
                    for filter_str in filter_strings:
                        matching = [c for c in matching if filter_str in c]

                if not matching:
                    if verbose:
                        print(f"  Warning: No subset found for seed {seed}")
                    continue

                if verbose:
                    print(f"  Seed {seed}: {matching[0]}")
                dataset = load_dataset(dataset_path, matching[0])

            datasets[seed] = dataset

        except Exception as e:
            if verbose:
                print(f"  Error loading seed {seed}: {e}")
            continue

    return datasets


def load_multi_approach_datasets(
    approaches: list[str],
    seeds: list[int],
    dataset_paths: dict[str, str],
    subset_templates: Optional[dict[str, str]] = None,
    filter_strings: Optional[list[str]] = None,
    temperature: Optional[float] = None,
    verbose: bool = True,
) -> dict[str, dict[int, Any]]:
    """Load datasets for multiple approaches and seeds.

    Args:
        approaches: List of approaches (e.g., ['bon', 'beam_search', 'dvts'])
        seeds: List of seed values
        dataset_paths: Dict mapping approach -> Hub path
        subset_templates: Dict mapping approach -> subset template (optional)
        filter_strings: List of filter strings for subset discovery
        temperature: Temperature value for template substitution
        verbose: Print progress messages

    Returns:
        Nested dict: {approach: {seed: Dataset}}
    """
    datasets = defaultdict(dict)

    for approach in approaches:
        if approach not in dataset_paths:
            if verbose:
                print(f"  Warning: No dataset path for approach {approach}")
            continue

        if verbose:
            print(f"\n  Loading {approach} datasets...")

        template = subset_templates.get(approach) if subset_templates else None

        datasets[approach] = load_datasets_by_seed(
            dataset_path=dataset_paths[approach],
            seeds=seeds,
            subset_template=template,
            filter_strings=filter_strings,
            temperature=temperature,
            verbose=verbose,
        )

    return dict(datasets)


def load_from_registry(
    result: ResultEntry,
    seeds: Optional[list[int]] = None,
    temperature: Optional[float] = None,
    filter_strings: Optional[list[str]] = None,
    verbose: bool = True,
) -> dict[int, Any]:
    """Load datasets using a registry entry.

    This function provides a convenient way to load datasets using
    configuration from the registry.

    Args:
        result: ResultEntry from registry
        seeds: Override seeds (default: use result.seeds)
        temperature: Temperature for template substitution
        filter_strings: Additional filter strings
        verbose: Print progress messages

    Returns:
        Dictionary mapping seed -> Dataset
    """
    seeds = seeds or result.seeds
    if not seeds:
        raise ValueError(f"No seeds specified for {result.name}")

    if verbose:
        print(f"Loading {result.name} from {result.hub_path}")

    return load_datasets_by_seed(
        dataset_path=result.hub_path,
        seeds=seeds,
        subset_template=result.subset_template,
        filter_strings=filter_strings,
        temperature=temperature,
        verbose=verbose,
    )


def get_available_subsets(
    dataset_path: str,
    seed: Optional[int] = None,
    temperature: Optional[float] = None,
) -> list[str]:
    """Get available subset names for a dataset.

    Useful for discovering what configurations are available.

    Args:
        dataset_path: Hugging Face Hub dataset path
        seed: Filter by seed value (optional)
        temperature: Filter by temperature value (optional)

    Returns:
        List of matching subset names
    """
    configs = get_dataset_config_names(dataset_path)

    if seed is not None:
        configs = [c for c in configs if f"seed-{seed}" in c]

    if temperature is not None:
        configs = [c for c in configs if f"T-{temperature}" in c]

    return sorted(configs)
