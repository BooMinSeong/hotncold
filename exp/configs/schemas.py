"""Configuration schemas for experiment analysis (Auto-Discovery Version).

This module defines dataclasses for the auto-discovery based registry system.
The key principle: Hub data is the Single Source of Truth.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import yaml


@dataclass
class OutputConfig:
    """Output configuration for analysis scripts.

    Attributes:
        format: Output format ("human", "oneline", "json")
        generate_plots: Whether to generate visualization plots
        generate_report: Whether to generate markdown reports
    """
    format: str = "human"
    generate_plots: bool = True
    generate_report: bool = True


@dataclass
class HubRegistry:
    """Registry of Hub dataset paths organized by category.

    This is a minimal registry that only stores hub_paths.
    All metadata (seeds, temperatures, etc.) is auto-discovered.

    Attributes:
        hub_paths: Dictionary mapping category -> list of hub paths
        output: Output configuration
    """
    hub_paths: dict[str, list[str]] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)

    @property
    def all_paths(self) -> list[str]:
        """Get all hub paths as a flat list."""
        paths = []
        for category_paths in self.hub_paths.values():
            paths.extend(category_paths)
        return paths

    def get_category(self, category: str) -> list[str]:
        """Get paths for a specific category."""
        return self.hub_paths.get(category, [])

    def get_categories(self) -> list[str]:
        """Get all category names."""
        return list(self.hub_paths.keys())

    def filter_by_pattern(self, pattern: str) -> list[str]:
        """Filter paths containing a pattern (case-insensitive).

        Args:
            pattern: Pattern to search for in path names

        Returns:
            List of matching paths
        """
        pattern_lower = pattern.lower()
        return [p for p in self.all_paths if pattern_lower in p.lower()]

    def filter_by_approach(self, approach: str) -> list[str]:
        """Filter paths by approach (bon, beam_search, dvts).

        Args:
            approach: Approach name to filter by

        Returns:
            List of matching paths
        """
        return self.filter_by_pattern(approach)

    def filter_by_model(self, model_pattern: str) -> list[str]:
        """Filter paths by model pattern.

        Args:
            model_pattern: Model pattern to filter by (e.g., "1.5B", "3B")

        Returns:
            List of matching paths
        """
        return self.filter_by_pattern(model_pattern)


def load_hub_registry(path: Union[str, Path]) -> HubRegistry:
    """Load hub registry from YAML file.

    Args:
        path: Path to registry YAML file

    Returns:
        HubRegistry object

    Raises:
        FileNotFoundError: If registry file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Parse hub_paths
    hub_paths = data.get("hub_paths", {})

    # Parse output config
    output_data = data.get("output", {})
    output = OutputConfig(
        format=output_data.get("format", "human"),
        generate_plots=output_data.get("generate_plots", True),
        generate_report=output_data.get("generate_report", True),
    )

    return HubRegistry(hub_paths=hub_paths, output=output)


@dataclass
class AnalysisConfig:
    """Configuration for running analysis.

    Attributes:
        registry_path: Path to registry YAML file
        filter_approach: Approach filter (bon, beam_search, dvts)
        filter_strategy: Strategy filter (hnc, default)
        filter_model: Model filter pattern
        filter_category: Category filter
        output_dir: Output directory for results
    """
    registry_path: str = "exp/configs/registry.yaml"
    filter_approach: Optional[str] = None
    filter_strategy: Optional[str] = None
    filter_model: Optional[str] = None
    filter_category: Optional[str] = None
    output_dir: str = "exp/analysis_output"


# =============================================================================
# Legacy compatibility (deprecated, will be removed)
# =============================================================================

@dataclass
class ResultEntry:
    """DEPRECATED: Legacy result entry for backward compatibility.

    Use ExperimentConfig from exp.analysis.discovery instead.
    """
    name: str
    hub_path: str
    model: str = ""
    dataset: str = ""
    approach: str = ""
    strategy: str = ""
    seeds: list[int] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    subset_template: Optional[str] = None

    def get_subset_name(self, seed: int, temperature: Optional[float] = None) -> Optional[str]:
        """DEPRECATED: Use ExperimentConfig.get_subset_name() instead."""
        if not self.subset_template:
            return None
        result = self.subset_template.replace("{seed}", str(seed))
        if temperature is not None:
            result = result.replace("{temperature}", str(temperature))
        return result


@dataclass
class DefaultParams:
    """DEPRECATED: Default parameters for experiments."""
    n: int = 64
    beam_width: int = 16


@dataclass
class Registry:
    """DEPRECATED: Legacy registry class.

    Use HubRegistry and auto-discovery instead.
    """
    results: list[ResultEntry] = field(default_factory=list)
    defaults: DefaultParams = field(default_factory=DefaultParams)
    output: OutputConfig = field(default_factory=OutputConfig)

    def filter(self, **kwargs) -> list[ResultEntry]:
        """DEPRECATED: Use HubRegistry.filter_by_* methods instead."""
        return self.results

    def get_by_name(self, name: str) -> Optional[ResultEntry]:
        """DEPRECATED"""
        for result in self.results:
            if result.name == name:
                return result
        return None
