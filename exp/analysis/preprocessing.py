"""Preprocessing utilities for experiment datasets.

This module provides functions to preprocess experiment datasets by
pre-computing all prediction evaluations and storing them as boolean fields.

This eliminates the need for expensive math_verify calls during analysis,
providing 20-30x speedup.
"""

import re
from datetime import datetime
from typing import Any, Optional

from datasets import Dataset
from tqdm import tqdm

from .core import evaluate_result


def preprocess_single_subset(
    dataset: Dataset,
    subset_name: str,
    add_metadata: bool = True,
    verbose: bool = True,
) -> Dataset:
    """Preprocess a single dataset subset by adding is_correct_* fields.

    For each pred_* field in the dataset, this function:
    1. Evaluates the prediction using math_verify
    2. Adds a corresponding is_correct_* boolean field

    Args:
        dataset: HuggingFace Dataset to preprocess
        subset_name: Name of the subset (for logging)
        add_metadata: Whether to add preprocessing metadata field
        verbose: Print progress messages

    Returns:
        Dataset with added is_correct_* fields

    Example:
        >>> dataset = load_dataset("ENSEONG/default-MATH-500-...", "subset1")
        >>> preprocessed = preprocess_single_subset(dataset["train"], "subset1")
        >>> # Now has fields: is_correct_naive@1, is_correct_naive@2, ...
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Preprocessing: {subset_name}")
        print(f"{'='*70}")

    # Extract all pred_* fields
    pred_fields = [k for k in dataset.features.keys() if k.startswith("pred_")]

    if not pred_fields:
        if verbose:
            print("  Warning: No pred_* fields found. Skipping.")
        return dataset

    if verbose:
        print(f"  Dataset size: {len(dataset)} problems")
        print(f"  Prediction fields: {len(pred_fields)}")
        print(f"  Total evaluations: {len(dataset) * len(pred_fields)}")

    # Prepare new columns
    new_columns = {f"is_correct_{field[5:]}": [] for field in pred_fields}

    # Evaluate each row
    if verbose:
        print(f"\n  Evaluating predictions...")

    for row in tqdm(dataset, desc=f"  Evaluating {subset_name}", disable=not verbose):
        for pred_field in pred_fields:
            try:
                is_correct = evaluate_result(row, pred_field)
            except Exception:
                # If evaluation fails, mark as incorrect
                is_correct = False

            new_columns[f"is_correct_{pred_field[5:]}"].append(is_correct)

    # Add new columns to dataset
    if verbose:
        print(f"\n  Adding {len(new_columns)} new columns to dataset...")

    for col_name, col_data in new_columns.items():
        dataset = dataset.add_column(col_name, col_data)

    # Add preprocessing metadata
    if add_metadata:
        metadata = {
            "preprocessing_version": "1.0",
            "preprocessed_at": datetime.now().isoformat(),
            "num_pred_fields": len(pred_fields),
            "total_evaluations": len(dataset) * len(pred_fields),
        }

        dataset = dataset.add_column(
            "_preprocessing_metadata",
            [metadata] * len(dataset),
        )

        if verbose:
            print(f"  Added preprocessing metadata")

    if verbose:
        print(f"\n  ✓ Preprocessing complete")
        print(f"{'='*70}")

    return dataset


def validate_preprocessing(
    original_dataset: Dataset,
    preprocessed_dataset: Dataset,
    num_samples: int = 100,
    verbose: bool = True,
) -> bool:
    """Validate preprocessing by comparing samples with original evaluation.

    Args:
        original_dataset: Original dataset with pred_* fields
        preprocessed_dataset: Preprocessed dataset with is_correct_* fields
        num_samples: Number of samples to validate
        verbose: Print validation results

    Returns:
        True if validation passes, False otherwise
    """
    import random

    if verbose:
        print(f"\n{'='*70}")
        print(f"Validating Preprocessing")
        print(f"{'='*70}")
        print(f"  Sampling {num_samples} predictions for validation...")

    # Extract pred fields
    pred_fields = [k for k in original_dataset.features.keys() if k.startswith("pred_")]

    if not pred_fields:
        if verbose:
            print("  No pred_* fields to validate")
        return True

    # Sample random (row, field) pairs
    num_rows = len(original_dataset)
    samples = []
    for _ in range(num_samples):
        row_idx = random.randint(0, num_rows - 1)
        pred_field = random.choice(pred_fields)
        samples.append((row_idx, pred_field))

    # Validate each sample
    mismatches = 0
    errors = 0

    for row_idx, pred_field in tqdm(samples, desc="  Validating", disable=not verbose):
        try:
            # Evaluate from original
            original_row = original_dataset[row_idx]
            expected = evaluate_result(original_row, pred_field)

            # Check preprocessed
            preprocessed_row = preprocessed_dataset[row_idx]
            is_correct_field = f"is_correct_{pred_field[5:]}"
            actual = preprocessed_row[is_correct_field]

            if expected != actual:
                mismatches += 1
        except Exception as e:
            errors += 1
            if verbose:
                print(f"  Error validating row {row_idx}, field {pred_field}: {e}")

    if verbose:
        print(f"\n  Validation Results:")
        print(f"    Samples checked: {num_samples}")
        print(f"    Mismatches: {mismatches}")
        print(f"    Errors: {errors}")

        if mismatches == 0 and errors == 0:
            print(f"    ✓ All validations passed!")
        elif mismatches > 0:
            print(f"    ✗ Validation failed: {mismatches} mismatches")
        else:
            print(f"    ⚠ Validation completed with {errors} errors")

        print(f"{'='*70}")

    return mismatches == 0 and errors == 0


def get_preprocessing_stats(dataset: Dataset) -> dict[str, Any]:
    """Get statistics about a preprocessed dataset.

    Args:
        dataset: Preprocessed dataset

    Returns:
        Dictionary with preprocessing statistics
    """
    stats = {
        "is_preprocessed": False,
        "num_problems": len(dataset),
        "num_is_correct_fields": 0,
        "preprocessing_metadata": None,
    }

    # Check for is_correct_* fields
    is_correct_fields = [k for k in dataset.features.keys() if k.startswith("is_correct_")]
    stats["num_is_correct_fields"] = len(is_correct_fields)
    stats["is_preprocessed"] = len(is_correct_fields) > 0

    # Extract metadata if available
    if "_preprocessing_metadata" in dataset.features:
        try:
            stats["preprocessing_metadata"] = dataset[0]["_preprocessing_metadata"]
        except Exception:
            pass

    return stats


def check_if_preprocessed(dataset: Dataset) -> bool:
    """Check if a dataset has been preprocessed.

    Args:
        dataset: Dataset to check

    Returns:
        True if dataset contains is_correct_* fields, False otherwise
    """
    return any(k.startswith("is_correct_") for k in dataset.features.keys())
