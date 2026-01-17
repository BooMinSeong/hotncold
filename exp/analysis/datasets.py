"""Dataset loading utilities for experiment analysis (Auto-Discovery Version).

This module provides functions to load datasets from Hugging Face Hub,
using the auto-discovery system for configuration.

The key principle: Hub data is the Single Source of Truth.
Instead of manually specifying seeds/temperatures, we discover them automatically.

Example usage:
    from exp.analysis.discovery import discover_experiment
    from exp.analysis.datasets import load_experiment_data

    # Discover configuration from Hub
    config = discover_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
    print(f"Seeds: {config.seeds}")  # Auto-discovered!

    # Load datasets for all discovered seeds
    datasets = load_experiment_data(config)
    for seed, dataset in datasets.items():
        print(f"Seed {seed}: {len(dataset['train'])} samples")
"""

from typing import Any, Optional

from datasets import load_dataset

from .discovery import ExperimentConfig, discover_experiment


def load_experiment_data(
    config: ExperimentConfig,
    seeds: Optional[list[int]] = None,
    temperature: Optional[float | tuple] = None,
    verbose: bool = True,
) -> dict[int, Any]:
    """Load datasets for an experiment configuration.

    Args:
        config: ExperimentConfig from discover_experiment()
        seeds: Specific seeds to load (default: all discovered seeds)
        temperature: Specific temperature to filter by
        verbose: Print progress messages

    Returns:
        Dictionary mapping seed -> Dataset
    """
    seeds = seeds or config.seeds
    datasets = {}

    for seed in seeds:
        subset_name = config.get_subset_name(seed, temperature)
        if not subset_name:
            if verbose:
                temp_str = f", temp {temperature}" if temperature else ""
                print(f"  Warning: No subset found for seed {seed}{temp_str}")
            continue

        try:
            if verbose:
                short_name = subset_name[:60] + "..." if len(subset_name) > 60 else subset_name
                print(f"  Loading seed {seed}: {short_name}")
            dataset = load_dataset(config.hub_path, subset_name)
            datasets[seed] = dataset
        except Exception as e:
            if verbose:
                print(f"  Error loading seed {seed}: {e}")

    return datasets


def load_from_hub_path(
    hub_path: str,
    seeds: Optional[list[int]] = None,
    temperature: Optional[float | tuple] = None,
    verbose: bool = True,
) -> dict[int, Any]:
    """Load datasets directly from a hub path.

    Convenience function that combines discovery and loading.

    Args:
        hub_path: HuggingFace Hub dataset path
        seeds: Specific seeds to load (default: all discovered)
        temperature: Temperature filter
        verbose: Print progress

    Returns:
        Dictionary mapping seed -> Dataset
    """
    if verbose:
        print(f"Discovering {hub_path}...")

    config = discover_experiment(hub_path)

    if verbose:
        print(f"  Found {len(config.subsets)} subsets")
        print(f"  Seeds: {config.seeds}")
        print(f"  Temperatures: {config.temperatures}")

    return load_experiment_data(config, seeds, temperature, verbose)


def load_multiple_experiments(
    hub_paths: list[str],
    seeds: Optional[list[int]] = None,
    temperature: Optional[float | tuple] = None,
    verbose: bool = True,
) -> dict[str, dict[int, Any]]:
    """Load datasets from multiple hub paths.

    Args:
        hub_paths: List of HuggingFace Hub dataset paths
        seeds: Specific seeds to load (default: all discovered per experiment)
        temperature: Temperature filter
        verbose: Print progress

    Returns:
        Nested dict: {hub_path: {seed: Dataset}}
    """
    results = {}

    for hub_path in hub_paths:
        if verbose:
            print(f"\n=== {hub_path} ===")
        results[hub_path] = load_from_hub_path(hub_path, seeds, temperature, verbose)

    return results


def get_available_configs(hub_path: str) -> list[str]:
    """Get all available subset names for a hub dataset.

    Args:
        hub_path: HuggingFace Hub dataset path

    Returns:
        List of subset names (sorted)
    """
    config = discover_experiment(hub_path)
    return sorted(s.raw_name for s in config.subsets)


def summarize_experiment(hub_path: str) -> dict[str, Any]:
    """Get a summary of an experiment's available configurations.

    Args:
        hub_path: HuggingFace Hub dataset path

    Returns:
        Dictionary with experiment summary
    """
    config = discover_experiment(hub_path)

    return {
        "hub_path": hub_path,
        "approach": config.approach,
        "model": config.model,
        "strategy": config.strategy,
        "seeds": config.seeds,
        "temperatures": config.temperatures,
        "datasets": config.datasets,
        "total_subsets": len(config.subsets),
        "subsets_by_seed": {
            seed: len(subsets)
            for seed, subsets in config.group_by_seed().items()
        },
        "subsets_by_temperature": {
            str(temp): len(subsets)
            for temp, subsets in config.group_by_temperature().items()
        },
    }


def load_experiment_data_by_temperature(
    config: ExperimentConfig,
    seeds: Optional[list[int]] = None,
    temperatures: Optional[list[float | tuple]] = None,
    verbose: bool = True,
) -> dict[float | tuple, dict[int, Any]]:
    """Load datasets organized by temperature, then by seed.

    This is the recommended function for temperature comparison analysis.
    Returns data in a structure that makes it easy to compare across temperatures.

    Args:
        config: ExperimentConfig from discover_experiment()
        seeds: Specific seeds to load (default: all discovered seeds)
        temperatures: Specific temperatures to load (default: all discovered)
        verbose: Print progress messages

    Returns:
        Nested dict: {temperature: {seed: Dataset}}
        Temperature is float for default strategy, tuple for hnc

    Example:
        >>> config = discover_experiment("ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-bon")
        >>> data = load_experiment_data_by_temperature(config)
        >>> # data[0.4][42] = Dataset for T=0.4, seed=42
        >>> # data[0.8][42] = Dataset for T=0.8, seed=42
    """
    seeds = seeds or config.seeds
    temperatures = temperatures or config.temperatures
    datasets: dict[float | tuple, dict[int, Any]] = {}

    for temp in temperatures:
        if verbose:
            temp_str = f"temps_{temp}" if isinstance(temp, tuple) else f"T={temp}"
            print(f"\nLoading temperature {temp_str}...")

        datasets[temp] = {}

        for seed in seeds:
            subset_name = config.get_subset_name(seed, temp)
            if not subset_name:
                if verbose:
                    print(f"  Warning: No subset found for seed {seed}, temp {temp}")
                continue

            try:
                if verbose:
                    short_name = subset_name[:50] + "..." if len(subset_name) > 50 else subset_name
                    print(f"  Loading seed {seed}: {short_name}")
                dataset = load_dataset(config.hub_path, subset_name)
                datasets[temp][seed] = dataset
            except Exception as e:
                if verbose:
                    print(f"  Error loading seed {seed}: {e}")

    return datasets


def load_all_experiment_data(
    config: ExperimentConfig,
    verbose: bool = True,
) -> dict[tuple[float | tuple, int], Any]:
    """Load all datasets for all temperature/seed combinations.

    Returns a flat dictionary with (temperature, seed) tuples as keys.

    Args:
        config: ExperimentConfig from discover_experiment()
        verbose: Print progress messages

    Returns:
        Dict mapping (temperature, seed) -> Dataset

    Example:
        >>> config = discover_experiment("ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-bon")
        >>> data = load_all_experiment_data(config)
        >>> dataset = data[(0.4, 42)]  # Dataset for T=0.4, seed=42
    """
    datasets = {}

    all_configs = config.iter_all_configs()
    if verbose:
        print(f"Loading {len(all_configs)} configurations...")

    for seed, temp, subset_name in all_configs:
        try:
            if verbose:
                temp_str = f"temps_{temp}" if isinstance(temp, tuple) else f"T={temp}"
                print(f"  Loading {temp_str}, seed={seed}...")
            dataset = load_dataset(config.hub_path, subset_name)
            datasets[(temp, seed)] = dataset
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")

    return datasets
