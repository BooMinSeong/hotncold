"""Configuration loading utilities for experiment analysis.

This module provides functions to load and parse the experiment registry from YAML files.
"""

from pathlib import Path
from typing import Union

import yaml

from .schemas import (
    AnalysisConfig,
    DefaultParams,
    OutputConfig,
    Registry,
    ResultEntry,
)

__all__ = [
    "load_registry",
    "load_analysis_config",
    "Registry",
    "ResultEntry",
    "AnalysisConfig",
    "DefaultParams",
    "OutputConfig",
]


def load_registry(path: Union[str, Path]) -> Registry:
    """Load experiment registry from YAML file.

    Args:
        path: Path to registry YAML file

    Returns:
        Registry object with all experiment results

    Raises:
        FileNotFoundError: If registry file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Parse results
    results = []
    for entry in data.get("results", []):
        results.append(
            ResultEntry(
                name=entry["name"],
                hub_path=entry["hub_path"],
                model=entry["model"],
                dataset=entry["dataset"],
                approach=entry["approach"],
                strategy=entry["strategy"],
                seeds=entry.get("seeds", []),
                temperatures=entry.get("temperatures", []),
                subset_template=entry.get("subset_template"),
            )
        )

    # Parse defaults
    defaults_data = data.get("defaults", {})
    defaults = DefaultParams(
        n=defaults_data.get("n", 64),
        beam_width=defaults_data.get("beam_width", 16),
    )

    # Parse output config
    output_data = data.get("output", {})
    output = OutputConfig(
        format=output_data.get("format", "human"),
        generate_plots=output_data.get("generate_plots", True),
        generate_report=output_data.get("generate_report", True),
    )

    return Registry(results=results, defaults=defaults, output=output)


def load_analysis_config(
    registry_path: str = "exp/configs/registry.yaml",
    filter_model: str = None,
    filter_dataset: str = None,
    filter_approach: str = None,
    filter_strategy: str = None,
    analysis_type: str = "default",
    output_dir: str = "exp/analysis_output",
    seeds_override: list[int] = None,
    temperatures_override: list[float] = None,
) -> AnalysisConfig:
    """Create analysis configuration from parameters.

    This is a convenience function to create AnalysisConfig from keyword arguments,
    useful for CLI argument parsing.

    Args:
        registry_path: Path to registry YAML file
        filter_model: Model filter (comma-separated)
        filter_dataset: Dataset filter (comma-separated)
        filter_approach: Approach filter (comma-separated)
        filter_strategy: Strategy filter (comma-separated)
        analysis_type: Type of analysis (default, model_comparison, temperature)
        output_dir: Output directory for results
        seeds_override: Override seeds from registry
        temperatures_override: Override temperatures from registry

    Returns:
        AnalysisConfig object
    """
    return AnalysisConfig(
        registry_path=registry_path,
        filter_model=filter_model,
        filter_dataset=filter_dataset,
        filter_approach=filter_approach,
        filter_strategy=filter_strategy,
        analysis_type=analysis_type,
        output_dir=output_dir,
        seeds_override=seeds_override,
        temperatures_override=temperatures_override,
    )
