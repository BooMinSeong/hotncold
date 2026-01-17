"""Configuration loading utilities for experiment analysis.

This module provides functions to load the experiment registry from YAML files.
The new auto-discovery system uses hub_paths only; all metadata is discovered from Hub.

Example usage (new way):
    from exp.configs import load_hub_registry
    from exp.analysis.discovery import discover_experiment

    registry = load_hub_registry("exp/configs/registry.yaml")
    for path in registry.get_category("math500_hnc"):
        config = discover_experiment(path)
        print(f"{path}: seeds={config.seeds}, temps={config.temperatures}")
"""

from pathlib import Path
from typing import Union

from .schemas import (
    AnalysisConfig,
    HubRegistry,
    OutputConfig,
    load_hub_registry,
    # Legacy exports (deprecated)
    DefaultParams,
    Registry,
    ResultEntry,
)

__all__ = [
    # New API
    "load_hub_registry",
    "HubRegistry",
    "OutputConfig",
    "AnalysisConfig",
    # Legacy (deprecated)
    "load_registry",
    "Registry",
    "ResultEntry",
    "DefaultParams",
]


def load_registry(path: Union[str, Path]) -> Registry:
    """DEPRECATED: Load experiment registry from YAML file.

    This function is deprecated. Use load_hub_registry() and auto-discovery instead.

    The new registry format only contains hub_paths. Use:
        from exp.configs import load_hub_registry
        from exp.analysis.discovery import discover_experiment

        registry = load_hub_registry(path)
        for hub_path in registry.all_paths:
            config = discover_experiment(hub_path)
            # config.seeds, config.temperatures, etc. are auto-discovered

    Args:
        path: Path to registry YAML file

    Returns:
        Empty Registry object (for backward compatibility)
    """
    import warnings
    warnings.warn(
        "load_registry() is deprecated. Use load_hub_registry() and "
        "exp.analysis.discovery.discover_experiment() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Return empty registry for backward compatibility
    hub_registry = load_hub_registry(path)
    return Registry(
        results=[],
        output=hub_registry.output,
    )
