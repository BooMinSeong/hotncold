"""Metrics calculation functions for experiment analysis.

This module provides functions to compute accuracy metrics, pass@k scores,
and aggregate results across seeds.

Extracted from:
- exp/analyze_all_results.py (analyze_single_dataset, analyze_pass_at_k)
- exp/analyze_aime25_results.py (analyze_single_dataset, analyze_pass_at_k)
"""

import re
from typing import Any, Callable, Optional

import numpy as np
from tqdm import tqdm

from .core import evaluate_result


def analyze_single_dataset(
    dataset: Any,
    dataset_name: str,
    seed: int,
    verbose: bool = True,
) -> dict[str, dict[int, float]]:
    """Analyze a single dataset and return accuracy by method and sample count.

    Evaluates all prediction keys (pred_naive@N, pred_weighted@N, pred_maj@N)
    and computes accuracy for each.

    Args:
        dataset: HuggingFace dataset (or dict with 'train' split)
        dataset_name: Name for logging
        seed: Seed value for logging
        verbose: Whether to print progress

    Returns:
        Nested dict: {method: {n_samples: accuracy}}
        where method is 'naive', 'weighted', or 'maj'

    Example:
        >>> results = analyze_single_dataset(dataset, "default-1.5B-bon", 42)
        >>> print(results['naive'][64])  # naive accuracy with 64 samples
        0.756
    """
    results_by_method = {"naive": {}, "weighted": {}, "maj": {}}

    if "train" in dataset:
        dataset = dataset["train"]

    # Find all prediction keys
    pred_keys = [key for key in dataset.features.keys() if key.startswith("pred_")]
    if verbose:
        print(f"  Found {len(pred_keys)} prediction keys to evaluate")

    # Accumulate results for each key
    results_accumulator = {key: [] for key in pred_keys}

    # Evaluate all predictions
    if verbose:
        print(f"  Evaluating predictions...")
    for data in tqdm(
        dataset, desc=f"  {dataset_name} (seed {seed})", leave=False, disable=not verbose
    ):
        for key in pred_keys:
            result = evaluate_result(data, key)
            results_accumulator[key].append(result)

    # Calculate accuracy for each key
    for key in pred_keys:
        results = results_accumulator[key]
        accuracy = sum(results) / len(results) if results else 0.0

        # Parse key: pred_method@number format
        match = re.match(r"pred_(naive|weighted|maj)@(\d+)", key)
        if match:
            method = match.group(1)
            n_samples = int(match.group(2))
            results_by_method[method][n_samples] = accuracy

    return results_by_method


def analyze_pass_at_k(
    dataset: Any,
    dataset_name: str,
    seed: int,
    verbose: bool = True,
) -> dict[int, float]:
    """Extract pass@k metrics from dataset.

    The dataset should already contain pass@{k} fields computed during generation.
    This function aggregates them across problems.

    Args:
        dataset: HuggingFace dataset with pass@k fields
        dataset_name: Name of dataset (for logging)
        seed: Seed value (for logging)
        verbose: Whether to print progress

    Returns:
        Dictionary mapping k values to mean pass@k probabilities:
        {1: 0.45, 2: 0.62, 4: 0.75, ...}
    """
    if "train" in dataset:
        dataset = dataset["train"]

    # Find all pass@k fields
    pass_k_fields = {}
    for key in dataset.features.keys():
        if key.startswith("pass@"):
            # Extract k value from field name (e.g., 'pass@1' -> 1)
            try:
                k = int(key.split("@")[1])
                pass_k_fields[k] = key
            except (IndexError, ValueError):
                continue

    if not pass_k_fields:
        if verbose:
            print(f"  Warning: No pass@k fields found in {dataset_name} (seed {seed})")
        return {}

    if verbose:
        print(f"  Found {len(pass_k_fields)} pass@k fields: {sorted(pass_k_fields.keys())}")

    # Aggregate pass@k values across problems
    results = {}
    for k, field_name in pass_k_fields.items():
        pass_k_values = [row[field_name] for row in dataset if field_name in row]
        if pass_k_values:
            results[k] = sum(pass_k_values) / len(pass_k_values)

    return results


def aggregate_across_seeds(
    results_by_seed: dict[int, Any],
    metric_extractor: Optional[Callable] = None,
) -> dict[str, Any]:
    """Aggregate results across seeds with mean and standard deviation.

    Args:
        results_by_seed: Dictionary mapping seed -> result
        metric_extractor: Optional function to extract numeric value from result
                         If None, results are assumed to be numeric

    Returns:
        Dictionary with 'mean', 'std', 'values', and 'seeds' keys

    Example:
        >>> results = {0: 0.75, 42: 0.78, 64: 0.76}
        >>> agg = aggregate_across_seeds(results)
        >>> print(f"{agg['mean']:.3f} ± {agg['std']:.3f}")
        0.763 ± 0.013
    """
    if metric_extractor is None:
        metric_extractor = lambda x: x

    values = [metric_extractor(v) for v in results_by_seed.values()]
    seeds = list(results_by_seed.keys())

    return {
        "mean": np.mean(values) if values else 0.0,
        "std": np.std(values) if values else 0.0,
        "values": values,
        "seeds": seeds,
    }


def compute_accuracy_by_method(
    datasets: dict[int, Any],
    dataset_name: str,
    verbose: bool = True,
) -> dict[str, dict[int, dict[str, float]]]:
    """Compute accuracy aggregated across seeds for each method and sample count.

    Args:
        datasets: Dictionary mapping seed -> Dataset
        dataset_name: Name for logging
        verbose: Print progress

    Returns:
        {method: {n_samples: {'mean': float, 'std': float, 'values': list}}}
    """
    # Collect results per seed
    all_results = {}
    for seed, dataset in datasets.items():
        all_results[seed] = analyze_single_dataset(dataset, dataset_name, seed, verbose)

    # Aggregate across seeds
    aggregated = {"naive": {}, "weighted": {}, "maj": {}}

    # Get all n_samples values
    all_n_samples = set()
    for seed_results in all_results.values():
        for method in ["naive", "weighted", "maj"]:
            all_n_samples.update(seed_results.get(method, {}).keys())

    for method in ["naive", "weighted", "maj"]:
        for n in sorted(all_n_samples):
            values_by_seed = {}
            for seed, seed_results in all_results.items():
                if n in seed_results.get(method, {}):
                    values_by_seed[seed] = seed_results[method][n]

            if values_by_seed:
                aggregated[method][n] = aggregate_across_seeds(values_by_seed)

    return aggregated


def compute_pass_at_k_aggregated(
    datasets: dict[int, Any],
    dataset_name: str,
    verbose: bool = True,
) -> dict[int, dict[str, float]]:
    """Compute pass@k aggregated across seeds.

    Args:
        datasets: Dictionary mapping seed -> Dataset
        dataset_name: Name for logging
        verbose: Print progress

    Returns:
        {k: {'mean': float, 'std': float, 'values': list}}
    """
    # Collect results per seed
    all_results = {}
    for seed, dataset in datasets.items():
        all_results[seed] = analyze_pass_at_k(dataset, dataset_name, seed, verbose)

    # Get all k values
    all_k = set()
    for seed_results in all_results.values():
        all_k.update(seed_results.keys())

    # Aggregate across seeds
    aggregated = {}
    for k in sorted(all_k):
        values_by_seed = {}
        for seed, seed_results in all_results.items():
            if k in seed_results:
                values_by_seed[seed] = seed_results[k]

        if values_by_seed:
            aggregated[k] = aggregate_across_seeds(values_by_seed)

    return aggregated
