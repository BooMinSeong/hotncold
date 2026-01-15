"""Configuration schemas for experiment analysis.

This module defines dataclasses for experiment result registry and analysis configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ResultEntry:
    """A single experiment result entry in the registry.

    Attributes:
        name: Unique identifier for this result (e.g., "default-1.5B-bon")
        hub_path: Hugging Face Hub path (e.g., "ENSEONG/default-Qwen2.5-1.5B-Instruct-bon")
        model: Model name (e.g., "Qwen2.5-1.5B-Instruct")
        dataset: Dataset name (e.g., "MATH-500", "aime25")
        approach: Search approach ("bon", "beam_search", "dvts")
        strategy: Temperature strategy ("default", "hnc")
        seeds: List of seed values used
        temperatures: List of temperatures used
        subset_template: Optional template for subset names with {seed}, {temperature} placeholders
    """

    name: str
    hub_path: str
    model: str
    dataset: str
    approach: str
    strategy: str
    seeds: list[int] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    subset_template: Optional[str] = None

    def get_subset_name(self, seed: int, temperature: Optional[float] = None) -> Optional[str]:
        """Generate subset name from template.

        Args:
            seed: Seed value to substitute
            temperature: Temperature value to substitute (optional)

        Returns:
            Formatted subset name or None if no template
        """
        if not self.subset_template:
            return None
        result = self.subset_template.replace("{seed}", str(seed))
        if temperature is not None:
            result = result.replace("{temperature}", str(temperature))
        return result


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
class DefaultParams:
    """Default parameters for experiments.

    Attributes:
        n: Number of samples (default 64)
        beam_width: Beam width for beam search/DVTS (default 16)
    """

    n: int = 64
    beam_width: int = 16


@dataclass
class Registry:
    """Registry of all experiment results with filtering capabilities.

    Attributes:
        results: List of ResultEntry objects
        defaults: Default parameters
        output: Output configuration
    """

    results: list[ResultEntry] = field(default_factory=list)
    defaults: DefaultParams = field(default_factory=DefaultParams)
    output: OutputConfig = field(default_factory=OutputConfig)

    def filter(
        self,
        model: Optional[str] = None,
        dataset: Optional[str] = None,
        approach: Optional[str] = None,
        strategy: Optional[str] = None,
        name: Optional[str] = None,
    ) -> list[ResultEntry]:
        """Filter results by criteria.

        Multiple values can be provided as comma-separated strings.
        All specified criteria must match (AND condition).

        Args:
            model: Filter by model name(s)
            dataset: Filter by dataset name(s)
            approach: Filter by approach(es)
            strategy: Filter by strategy(ies)
            name: Filter by result name(s)

        Returns:
            List of matching ResultEntry objects
        """
        filtered = self.results

        if model:
            models = [m.strip() for m in model.split(",")]
            filtered = [r for r in filtered if r.model in models]

        if dataset:
            datasets = [d.strip() for d in dataset.split(",")]
            filtered = [r for r in filtered if r.dataset in datasets]

        if approach:
            approaches = [a.strip() for a in approach.split(",")]
            filtered = [r for r in filtered if r.approach in approaches]

        if strategy:
            strategies = [s.strip() for s in strategy.split(",")]
            filtered = [r for r in filtered if r.strategy in strategies]

        if name:
            names = [n.strip() for n in name.split(",")]
            filtered = [r for r in filtered if r.name in names]

        return filtered

    def get_unique_values(self, field_name: str) -> list[str]:
        """Get unique values for a specific field.

        Args:
            field_name: Name of the field (model, dataset, approach, strategy)

        Returns:
            Sorted list of unique values
        """
        values = set()
        for result in self.results:
            if hasattr(result, field_name):
                values.add(getattr(result, field_name))
        return sorted(values)

    def get_by_name(self, name: str) -> Optional[ResultEntry]:
        """Get a single result by name.

        Args:
            name: Result name to find

        Returns:
            ResultEntry if found, None otherwise
        """
        for result in self.results:
            if result.name == name:
                return result
        return None


@dataclass
class AnalysisConfig:
    """Configuration for running analysis.

    Attributes:
        registry_path: Path to registry YAML file
        filter_model: Model filter
        filter_dataset: Dataset filter
        filter_approach: Approach filter
        filter_strategy: Strategy filter
        analysis_type: Type of analysis to run
        output_dir: Output directory for results
        seeds_override: Override seeds from registry
        temperatures_override: Override temperatures from registry
    """

    registry_path: str = "exp/configs/registry.yaml"
    filter_model: Optional[str] = None
    filter_dataset: Optional[str] = None
    filter_approach: Optional[str] = None
    filter_strategy: Optional[str] = None
    analysis_type: str = "default"  # default, model_comparison, temperature
    output_dir: str = "exp/analysis_output"
    seeds_override: Optional[list[int]] = None
    temperatures_override: Optional[list[float]] = None
