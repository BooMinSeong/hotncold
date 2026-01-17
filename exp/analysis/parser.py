"""Subset name parser for auto-discovery.

This module parses subset names from HuggingFace Hub datasets
to extract experiment configuration parameters.

Naming patterns:
1. Default strategy (single temperature):
   - BON: {source}_{dataset}--T-{temp}--top_p-{value}--n-{n}--seed-{seed}--agg_strategy-last
   - Beam/DVTS: {source}_{dataset}--T-{temp}--top_p-{value}--n-{n}--m-{m}--iters-{iters}--look-{look}--seed-{seed}--agg_strategy--last

2. HNC strategy (multi-temperature):
   - BON: {source}_{dataset}--temps_{t1}_{t2}...--top_p-{value}--n-{n}--seed-{seed}--agg_strategy-last
   - Beam/DVTS: {source}_{dataset}--temps_{t1}_{t2}...--top_p-{value}--n-{n}--m-{m}--iters-{iters}--look-{look}--seed-{seed}--agg_strategy--last
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SubsetInfo:
    """Parsed information from a subset name.

    Attributes:
        raw_name: Original subset name
        dataset_source: Dataset source (e.g., "HuggingFaceH4", "math-ai")
        dataset_name: Dataset name (e.g., "MATH-500", "aime25")
        strategy: Temperature strategy ("default" or "hnc")
        temperature: Single temperature for default strategy
        temperatures: List of temperatures for HNC strategy
        temperature_ratios: List of ratios for HNC strategy
        top_p: Top-p sampling parameter
        n: Number of samples
        seed: Random seed
        m: Beam width (for beam_search/dvts)
        iters: Number of iterations (for beam_search/dvts)
        lookahead: Lookahead value (for beam_search/dvts)
        agg_strategy: Aggregation strategy
        is_beam_search_type: Whether this is beam_search or dvts (has m, iters, look)
    """
    raw_name: str
    dataset_source: str = ""
    dataset_name: str = ""
    strategy: str = "default"
    temperature: Optional[float] = None
    temperatures: list[float] = field(default_factory=list)
    temperature_ratios: list[float] = field(default_factory=list)
    top_p: float = 1.0
    n: int = 64
    seed: int = 0
    m: Optional[int] = None
    iters: Optional[int] = None
    lookahead: Optional[int] = None
    agg_strategy: str = "last"

    @property
    def is_beam_search_type(self) -> bool:
        """Check if this config is for beam_search or dvts."""
        return self.m is not None

    @property
    def is_hnc(self) -> bool:
        """Check if this is HNC strategy."""
        return self.strategy == "hnc"

    @property
    def full_dataset_name(self) -> str:
        """Get full dataset identifier."""
        return f"{self.dataset_source}/{self.dataset_name}"

    def get_temperature_key(self) -> str:
        """Get a key representing the temperature configuration.

        For default strategy: "T-0.8"
        For HNC strategy: "temps_0.4_0.8_1.2_1.6"
        """
        if self.is_hnc:
            temps = "_".join(str(t) for t in self.temperatures)
            return f"temps_{temps}"
        else:
            return f"T-{self.temperature}"


def parse_subset_name(name: str) -> SubsetInfo:
    """Parse a subset name into structured information.

    Args:
        name: Subset name from HuggingFace Hub

    Returns:
        SubsetInfo with parsed parameters

    Examples:
        >>> info = parse_subset_name("HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--seed-42--agg_strategy-last")
        >>> info.dataset_name
        'MATH-500'
        >>> info.temperature
        0.8
        >>> info.seed
        42

        >>> info = parse_subset_name("HuggingFaceH4_MATH-500--temps_0.4_0.8_1.2_1.6__r_0.25_0.25_0.25_0.25--top_p-1.0--n-64--seed-128--agg_strategy-last")
        >>> info.is_hnc
        True
        >>> info.temperatures
        [0.4, 0.8, 1.2, 1.6]
    """
    info = SubsetInfo(raw_name=name)

    # Split by '--' to get segments
    segments = name.split("--")

    if not segments:
        return info

    # First segment: dataset source and name
    first_segment = segments[0]
    if "_" in first_segment:
        parts = first_segment.split("_", 1)
        info.dataset_source = parts[0]
        info.dataset_name = parts[1] if len(parts) > 1 else ""
    else:
        info.dataset_name = first_segment

    # Parse remaining segments
    for segment in segments[1:]:
        # Single temperature: T-{value}
        if segment.startswith("T-"):
            match = re.match(r"T-(\d+\.?\d*)", segment)
            if match:
                info.temperature = float(match.group(1))
                info.strategy = "default"

        # HNC temperatures: temps_{t1}_{t2}_...__r_{r1}_{r2}_...
        elif segment.startswith("temps_"):
            info.strategy = "hnc"
            # Split temps and ratios
            if "__r_" in segment:
                temps_part, ratios_part = segment.split("__r_")
            else:
                temps_part = segment
                ratios_part = None

            # Parse temperatures
            temps_str = temps_part.replace("temps_", "")
            info.temperatures = [float(t) for t in temps_str.split("_") if t]

            # Parse ratios if present
            if ratios_part:
                info.temperature_ratios = [float(r) for r in ratios_part.split("_") if r]

        # Top-p
        elif segment.startswith("top_p-"):
            match = re.match(r"top_p-(\d+\.?\d*)", segment)
            if match:
                info.top_p = float(match.group(1))

        # Number of samples
        elif segment.startswith("n-"):
            match = re.match(r"n-(\d+)", segment)
            if match:
                info.n = int(match.group(1))

        # Seed
        elif segment.startswith("seed-"):
            match = re.match(r"seed-(\d+)", segment)
            if match:
                info.seed = int(match.group(1))

        # Beam width (beam_search/dvts)
        elif segment.startswith("m-"):
            match = re.match(r"m-(\d+)", segment)
            if match:
                info.m = int(match.group(1))

        # Iterations (beam_search/dvts)
        elif segment.startswith("iters-"):
            match = re.match(r"iters-(\d+)", segment)
            if match:
                info.iters = int(match.group(1))

        # Lookahead (beam_search/dvts)
        elif segment.startswith("look-"):
            match = re.match(r"look-(\d+)", segment)
            if match:
                info.lookahead = int(match.group(1))

        # Aggregation strategy
        elif segment.startswith("agg_strategy"):
            # Handle both "agg_strategy-last" and "agg_strategy" followed by "--last"
            if "-" in segment:
                info.agg_strategy = segment.split("-", 1)[1]
            else:
                # The value might be in the next segment (agg_strategy--last pattern)
                pass

        # Handle standalone "last" after "agg_strategy--"
        elif segment == "last":
            info.agg_strategy = "last"

    return info


def infer_approach_from_subset(info: SubsetInfo) -> str:
    """Infer the approach (bon, beam_search, dvts) from parsed subset info.

    Note: beam_search and dvts have identical config patterns,
    so we can only distinguish them from the hub_path.

    Args:
        info: Parsed SubsetInfo

    Returns:
        "bon" if no beam parameters, "beam_search_or_dvts" otherwise
    """
    if info.is_beam_search_type:
        return "beam_search_or_dvts"
    return "bon"


def infer_approach_from_hub_path(hub_path: str) -> str:
    """Infer approach from hub path.

    Args:
        hub_path: Hub dataset path (e.g., "ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")

    Returns:
        Approach name ("bon", "beam_search", "dvts")
    """
    path_lower = hub_path.lower()
    if "dvts" in path_lower:
        return "dvts"
    elif "beam_search" in path_lower or "beam-search" in path_lower:
        return "beam_search"
    else:
        return "bon"


def infer_model_from_hub_path(hub_path: str) -> str:
    """Infer model name from hub path.

    Args:
        hub_path: Hub dataset path (e.g., "ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")

    Returns:
        Model name (e.g., "Qwen2.5-1.5B-Instruct")
    """
    # Pattern: strategy-{model}-approach or strategy-dataset-{model}-approach
    # Examples:
    #   ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon
    #   ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-bon

    repo_name = hub_path.split("/")[-1]

    # Common model patterns
    model_patterns = [
        r"(Qwen2\.5-\d+\.?\d*B-Instruct)",
        r"(Llama-3\.\d+-\d+B-Instruct)",
        r"(Mistral-\d+B-Instruct)",
    ]

    for pattern in model_patterns:
        match = re.search(pattern, repo_name)
        if match:
            return match.group(1)

    return "unknown"


def infer_strategy_from_hub_path(hub_path: str) -> str:
    """Infer strategy from hub path.

    Args:
        hub_path: Hub dataset path

    Returns:
        Strategy name ("hnc" or "default")
    """
    repo_name = hub_path.split("/")[-1].lower()
    if repo_name.startswith("hnc"):
        return "hnc"
    return "default"
