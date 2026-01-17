"""Analysis module for experiment result processing.

This module provides utilities for:
- Auto-discovery of experiment configurations from Hub
- Subset name parsing
- Dataset loading
- Core evaluation functions
- Metrics calculation
- Difficulty stratification
- Visualization

Example usage (new auto-discovery way):
    from exp.analysis import discover_experiment, load_experiment_data

    config = discover_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
    print(f"Auto-discovered seeds: {config.seeds}")
    print(f"Auto-discovered temperatures: {config.temperatures}")

    datasets = load_experiment_data(config)
    for seed, dataset in datasets.items():
        print(f"Seed {seed}: {len(dataset['train'])} samples")
"""

# Auto-discovery (new API)
from .discovery import (
    DiscoveredRegistry,
    ExperimentConfig,
    create_registry_from_hub_paths,
    discover_experiment,
)

# Parser
from .parser import (
    SubsetInfo,
    infer_approach_from_hub_path,
    infer_model_from_hub_path,
    infer_strategy_from_hub_path,
    parse_subset_name,
)

# Dataset loading
from .datasets import (
    get_available_configs,
    load_all_experiment_data,
    load_experiment_data,
    load_experiment_data_by_temperature,
    load_from_hub_path,
    load_multiple_experiments,
    summarize_experiment,
)

# Core evaluation
from .core import (
    evaluate_answer,
    evaluate_result,
    extract_boxed_answer,
)

# Metrics
from .metrics import (
    aggregate_across_seeds,
    analyze_pass_at_k,
    analyze_single_dataset,
    compute_accuracy_by_method,
    compute_pass_at_k_aggregated,
)

# Difficulty analysis
from .difficulty import (
    DifficultyLevel,
    ProblemBaseline,
    compute_difficulty_statistics,
    compute_problem_baselines,
    get_problems_by_difficulty,
    stratify_by_difficulty,
)

# Visualization
from .visualization import (
    APPROACH_COLORS,
    METHOD_COLORS,
    STRATEGY_COLORS,
    create_results_table,
    plot_bar_comparison,
    plot_comparison,
    plot_heatmap,
    plot_scaling_curve,
    save_figure,
    setup_style,
)

__all__ = [
    # Auto-discovery (new)
    "discover_experiment",
    "ExperimentConfig",
    "DiscoveredRegistry",
    "create_registry_from_hub_paths",
    # Parser (new)
    "parse_subset_name",
    "SubsetInfo",
    "infer_approach_from_hub_path",
    "infer_model_from_hub_path",
    "infer_strategy_from_hub_path",
    # Dataset loading (new)
    "load_experiment_data",
    "load_experiment_data_by_temperature",
    "load_all_experiment_data",
    "load_from_hub_path",
    "load_multiple_experiments",
    "get_available_configs",
    "summarize_experiment",
    # Core
    "evaluate_answer",
    "evaluate_result",
    "extract_boxed_answer",
    # Metrics
    "analyze_single_dataset",
    "analyze_pass_at_k",
    "aggregate_across_seeds",
    "compute_accuracy_by_method",
    "compute_pass_at_k_aggregated",
    # Difficulty
    "ProblemBaseline",
    "DifficultyLevel",
    "compute_problem_baselines",
    "stratify_by_difficulty",
    "get_problems_by_difficulty",
    "compute_difficulty_statistics",
    # Visualization
    "setup_style",
    "save_figure",
    "plot_scaling_curve",
    "plot_comparison",
    "plot_bar_comparison",
    "plot_heatmap",
    "create_results_table",
    "STRATEGY_COLORS",
    "METHOD_COLORS",
    "APPROACH_COLORS",
]
