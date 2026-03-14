#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

from huggingface_hub import list_repo_refs, repo_exists

logger = logging.getLogger(__name__)


@dataclass
class DatasetCheckResult:
    """Result of checking a dataset for missing ranges.

    Attributes:
        dataset_name: The HuggingFace Hub dataset ID (e.g., "org/dataset-name").
        filters: List of filter strings applied (e.g., ["seed-0", "T-0.8"]).
        coverage_pct: Percentage of the dataset covered (0-100).
        missing_ranges: List of (start, end) tuples representing gaps in coverage.
        status: One of "OK", "MISSING", "NOT_FOUND", "NO_BRANCHES".
    """

    dataset_name: str
    filters: List[str]
    coverage_pct: float
    missing_ranges: List[Tuple[int, int]]
    status: str


def parse_chunk_range(branch_name: str) -> Tuple[int, int] | None:
    """Parse chunk range from branch name.

    Extracts (start, end) from branch names like "chunk-0_100", "seed-0_T-0.8_chunk-100_200".

    Args:
        branch_name: The branch/revision name to parse.

    Returns:
        Tuple of (start, end) if pattern matches, None otherwise.
    """
    match = re.search(r"chunk-(\d+)_(\d+)", branch_name)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        return (start, end)
    return None


def find_missing_ranges(
    ranges: List[Tuple[int, int]], total: int
) -> List[Tuple[int, int]]:
    """Find missing ranges in [0, total) given a list of covered ranges.

    Merges overlapping ranges and identifies gaps.

    Args:
        ranges: List of (start, end) tuples representing covered ranges.
        total: Total range size (exclusive upper bound).

    Returns:
        List of (start, end) tuples representing missing ranges.
    """
    if not ranges:
        return [(0, total)]

    # Sort ranges by start position
    sorted_ranges = sorted(ranges)

    # Merge overlapping/adjacent ranges
    merged = []
    current_start, current_end = sorted_ranges[0]

    for start, end in sorted_ranges[1:]:
        if start <= current_end:  # Overlapping or adjacent
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    # Find gaps
    missing = []

    # Check for gap at the beginning
    if merged[0][0] > 0:
        missing.append((0, merged[0][0]))

    # Check for gaps between merged ranges
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]
        if gap_start < gap_end:
            missing.append((gap_start, gap_end))

    # Check for gap at the end
    if merged[-1][1] < total:
        missing.append((merged[-1][1], total))

    return missing


def check_dataset(
    dataset_name: str,
    filters: List[str] | None = None,
    total: int = 500,
) -> DatasetCheckResult:
    """Check HuggingFace Hub dataset for missing ranges.

    Queries the dataset repository for branches/revisions, filters them by the
    provided filter strings, parses chunk ranges from branch names, and identifies
    gaps in coverage.

    Args:
        dataset_name: HuggingFace Hub dataset ID (e.g., "org/dataset-name").
        filters: Optional list of strings that must all appear in branch names
            (e.g., ["seed-0", "T-0.8"]). If None, all branches are considered.
        total: Total expected range size (default: 500).

    Returns:
        DatasetCheckResult object with coverage information and missing ranges.

    Examples:
        >>> result = check_dataset("org/my-dataset", filters=["seed-0", "T-0.8"], total=500)
        >>> if result.status == "MISSING":
        ...     print(f"Missing ranges: {result.missing_ranges}")
        >>> elif result.status == "OK":
        ...     print("Dataset is complete!")
    """
    filters = filters or []

    # Check if dataset exists
    if not repo_exists(dataset_name, repo_type="dataset"):
        logger.warning(f"Dataset {dataset_name} not found on Hub")
        return DatasetCheckResult(
            dataset_name=dataset_name,
            filters=filters,
            coverage_pct=0.0,
            missing_ranges=[(0, total)],
            status="NOT_FOUND",
        )

    # Get all branches/revisions
    try:
        refs = list_repo_refs(dataset_name, repo_type="dataset")
        branches = [ref.name for ref in refs.branches if ref.name != "main"]
    except Exception as e:
        logger.error(f"Error listing branches for {dataset_name}: {e}")
        return DatasetCheckResult(
            dataset_name=dataset_name,
            filters=filters,
            coverage_pct=0.0,
            missing_ranges=[(0, total)],
            status="NOT_FOUND",
        )

    if not branches:
        logger.warning(f"No branches found for {dataset_name}")
        return DatasetCheckResult(
            dataset_name=dataset_name,
            filters=filters,
            coverage_pct=0.0,
            missing_ranges=[(0, total)],
            status="NO_BRANCHES",
        )

    # Filter branches by all filter strings
    filtered_branches = branches
    for filter_str in filters:
        filtered_branches = [b for b in filtered_branches if filter_str in b]

    if not filtered_branches:
        logger.info(f"No branches match filters {filters} in {dataset_name}")
        return DatasetCheckResult(
            dataset_name=dataset_name,
            filters=filters,
            coverage_pct=0.0,
            missing_ranges=[(0, total)],
            status="MISSING",
        )

    # Parse chunk ranges from filtered branches
    ranges = []
    for branch in filtered_branches:
        chunk_range = parse_chunk_range(branch)
        if chunk_range:
            ranges.append(chunk_range)

    if not ranges:
        logger.warning(f"No valid chunk ranges found in branches for {dataset_name}")
        return DatasetCheckResult(
            dataset_name=dataset_name,
            filters=filters,
            coverage_pct=0.0,
            missing_ranges=[(0, total)],
            status="MISSING",
        )

    # Find missing ranges
    missing_ranges = find_missing_ranges(ranges, total)

    # Calculate coverage
    covered = sum(end - start for start, end in ranges)
    coverage_pct = (covered / total) * 100.0 if total > 0 else 0.0

    # Determine status
    status = "OK" if not missing_ranges else "MISSING"

    logger.info(
        f"Dataset {dataset_name} (filters={filters}): "
        f"{coverage_pct:.1f}% coverage, "
        f"{len(missing_ranges)} missing ranges"
    )

    return DatasetCheckResult(
        dataset_name=dataset_name,
        filters=filters,
        coverage_pct=coverage_pct,
        missing_ranges=missing_ranges,
        status=status,
    )
