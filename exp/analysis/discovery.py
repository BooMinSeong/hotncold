"""Auto-discovery of experiment configurations from HuggingFace Hub.

This module provides functions to automatically discover and classify
experiment configurations by parsing subset names from Hub datasets.

The key principle is: Hub data is the Single Source of Truth.
Instead of manually specifying seeds, temperatures, etc. in a registry,
we discover them automatically from the actual data.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Optional

from datasets import get_dataset_config_names

from .parser import (
    SubsetInfo,
    parse_subset_name,
    infer_approach_from_hub_path,
    infer_model_from_hub_path,
    infer_strategy_from_hub_path,
)


@dataclass
class ExperimentConfig:
    """Discovered experiment configuration from a Hub dataset.

    This represents a complete experiment setup discovered from Hub,
    with all subsets automatically classified.

    Attributes:
        hub_path: HuggingFace Hub dataset path
        approach: Search approach (bon, beam_search, dvts)
        model: Model name
        strategy: Temperature strategy inferred from hub_path (hnc or default)
        subsets: List of all parsed SubsetInfo objects
    """
    hub_path: str
    approach: str = ""
    model: str = ""
    strategy: str = ""
    subsets: list[SubsetInfo] = field(default_factory=list)

    def __post_init__(self):
        if not self.approach:
            self.approach = infer_approach_from_hub_path(self.hub_path)
        if not self.model:
            self.model = infer_model_from_hub_path(self.hub_path)
        if not self.strategy:
            self.strategy = infer_strategy_from_hub_path(self.hub_path)

    @property
    def seeds(self) -> list[int]:
        """Get unique seeds from all subsets."""
        return sorted(set(s.seed for s in self.subsets))

    @property
    def temperatures(self) -> list[float]:
        """Get unique temperatures from all subsets.

        For HNC strategy, returns the temperature lists.
        For default strategy, returns single temperatures.
        """
        temps = set()
        for s in self.subsets:
            if s.is_hnc:
                temps.add(tuple(s.temperatures))
            elif s.temperature is not None:
                temps.add(s.temperature)
        return sorted(temps, key=lambda x: x if isinstance(x, float) else x[0])

    @property
    def datasets(self) -> list[str]:
        """Get unique dataset names from all subsets."""
        return sorted(set(s.dataset_name for s in self.subsets))

    def group_by_seed(self) -> dict[int, list[SubsetInfo]]:
        """Group subsets by seed."""
        result = defaultdict(list)
        for s in self.subsets:
            result[s.seed].append(s)
        return dict(result)

    def group_by_temperature(self) -> dict[Any, list[SubsetInfo]]:
        """Group subsets by temperature configuration.

        Keys are either float (default) or tuple of floats (hnc).
        """
        result = defaultdict(list)
        for s in self.subsets:
            if s.is_hnc:
                key = tuple(s.temperatures)
            else:
                key = s.temperature
            result[key].append(s)
        return dict(result)

    def group_by_dataset(self) -> dict[str, list[SubsetInfo]]:
        """Group subsets by dataset name."""
        result = defaultdict(list)
        for s in self.subsets:
            result[s.dataset_name].append(s)
        return dict(result)

    def filter_by_seed(self, seed: int) -> list[SubsetInfo]:
        """Get subsets matching a specific seed."""
        return [s for s in self.subsets if s.seed == seed]

    def filter_by_temperature(self, temperature: float | tuple[float, ...]) -> list[SubsetInfo]:
        """Get subsets matching a specific temperature configuration."""
        result = []
        for s in self.subsets:
            if s.is_hnc and tuple(s.temperatures) == temperature:
                result.append(s)
            elif not s.is_hnc and s.temperature == temperature:
                result.append(s)
        return result

    def get_subset_name(self, seed: int, temperature: Optional[float | tuple] = None) -> Optional[str]:
        """Find subset name matching seed and optional temperature.

        Args:
            seed: Seed value to match
            temperature: Temperature to match (float for default, tuple for hnc)

        Returns:
            Subset name if found, None otherwise
        """
        for s in self.subsets:
            if s.seed != seed:
                continue
            if temperature is None:
                return s.raw_name
            if s.is_hnc and tuple(s.temperatures) == temperature:
                return s.raw_name
            if not s.is_hnc and s.temperature == temperature:
                return s.raw_name
        return None

    def get_all_subset_names_for_seed(self, seed: int) -> dict[float | tuple, str]:
        """Get all subset names for a seed, organized by temperature.

        Args:
            seed: Seed value to match

        Returns:
            Dictionary mapping temperature -> subset_name
            Temperature is float for default strategy, tuple for hnc
        """
        result = {}
        for s in self.subsets:
            if s.seed != seed:
                continue
            if s.is_hnc:
                key = tuple(s.temperatures)
            else:
                key = s.temperature
            if key is not None:
                result[key] = s.raw_name
        return result

    def iter_all_configs(self) -> list[tuple[int, float | tuple, str]]:
        """Iterate over all (seed, temperature, subset_name) combinations.

        Returns:
            List of (seed, temperature, subset_name) tuples
        """
        result = []
        for s in self.subsets:
            if s.is_hnc:
                temp = tuple(s.temperatures)
            else:
                temp = s.temperature
            if temp is not None:
                result.append((s.seed, temp, s.raw_name))
        return result


@lru_cache(maxsize=32)
def _get_config_names_cached(hub_path: str) -> tuple[str, ...]:
    """Cached version of get_dataset_config_names."""
    return tuple(get_dataset_config_names(hub_path))


def discover_experiment(hub_path: str, use_cache: bool = True) -> ExperimentConfig:
    """Discover experiment configuration from a Hub dataset.

    This function queries the Hub for all available configs (subsets),
    parses each one, and returns a structured ExperimentConfig.

    Args:
        hub_path: HuggingFace Hub dataset path
        use_cache: Whether to cache Hub API calls

    Returns:
        ExperimentConfig with all discovered subsets

    Example:
        >>> config = discover_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
        >>> config.seeds
        [128, 192, 256]
        >>> config.strategy
        'hnc'
        >>> len(config.subsets)
        3
    """
    if use_cache:
        config_names = list(_get_config_names_cached(hub_path))
    else:
        config_names = get_dataset_config_names(hub_path)

    subsets = [parse_subset_name(name) for name in config_names]

    return ExperimentConfig(
        hub_path=hub_path,
        subsets=subsets,
    )


@dataclass
class DiscoveredRegistry:
    """A registry of discovered experiments.

    This replaces the manual registry.yaml with auto-discovered configurations.
    """
    experiments: dict[str, ExperimentConfig] = field(default_factory=dict)

    def add(self, hub_path: str, name: Optional[str] = None) -> ExperimentConfig:
        """Add an experiment by hub path.

        Args:
            hub_path: HuggingFace Hub dataset path
            name: Optional custom name (default: derived from hub_path)

        Returns:
            Discovered ExperimentConfig
        """
        config = discover_experiment(hub_path)

        if name is None:
            # Generate name from hub_path
            # ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon -> hnc-1.5B-bon
            repo_name = hub_path.split("/")[-1]
            name = repo_name

        self.experiments[name] = config
        return config

    def get(self, name: str) -> Optional[ExperimentConfig]:
        """Get experiment by name."""
        return self.experiments.get(name)

    def filter(
        self,
        approach: Optional[str] = None,
        strategy: Optional[str] = None,
        model: Optional[str] = None,
    ) -> list[ExperimentConfig]:
        """Filter experiments by criteria."""
        results = list(self.experiments.values())

        if approach:
            results = [e for e in results if e.approach == approach]
        if strategy:
            results = [e for e in results if e.strategy == strategy]
        if model:
            results = [e for e in results if model in e.model]

        return results

    @property
    def all_hub_paths(self) -> list[str]:
        """Get all hub paths in registry."""
        return [e.hub_path for e in self.experiments.values()]


def create_registry_from_hub_paths(hub_paths: list[str], verbose: bool = True) -> DiscoveredRegistry:
    """Create a registry from a list of Hub paths.

    This is the main entry point for the auto-discovery system.

    Args:
        hub_paths: List of HuggingFace Hub dataset paths
        verbose: Print progress

    Returns:
        DiscoveredRegistry with all experiments

    Example:
        >>> paths = [
        ...     "ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon",
        ...     "ENSEONG/default-Qwen2.5-1.5B-Instruct-bon",
        ... ]
        >>> registry = create_registry_from_hub_paths(paths)
        >>> registry.filter(strategy="hnc")
        [ExperimentConfig(...)]
    """
    registry = DiscoveredRegistry()

    for path in hub_paths:
        if verbose:
            print(f"Discovering {path}...")
        try:
            registry.add(path)
            config = registry.experiments[path.split("/")[-1]]
            if verbose:
                print(f"  Found {len(config.subsets)} subsets")
                print(f"  Seeds: {config.seeds}")
                print(f"  Temperatures: {config.temperatures}")
        except Exception as e:
            if verbose:
                print(f"  Error: {e}")

    return registry
