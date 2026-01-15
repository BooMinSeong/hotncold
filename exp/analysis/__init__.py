"""Analysis module for experiment result processing.

This module provides utilities for:
- Core evaluation functions (evaluate_answer, evaluate_result)
- Dataset loading (load_datasets_by_seed, load_from_registry)
- Metrics calculation (analyze_single_dataset, analyze_pass_at_k)
- Difficulty stratification (compute_problem_baselines, stratify_by_difficulty)
- Visualization (plotting functions)
"""

from .core import (
    evaluate_answer,
    evaluate_result,
    extract_boxed_answer,
)

from .datasets import (
    get_available_subsets,
    load_datasets_by_seed,
    load_from_registry,
    load_multi_approach_datasets,
)

from .metrics import (
    aggregate_across_seeds,
    analyze_pass_at_k,
    analyze_single_dataset,
    compute_accuracy_by_method,
    compute_pass_at_k_aggregated,
)

from .difficulty import (
    DifficultyLevel,
    ProblemBaseline,
    compute_difficulty_statistics,
    compute_problem_baselines,
    get_problems_by_difficulty,
    stratify_by_difficulty,
)

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
    # Core
    "evaluate_answer",
    "evaluate_result",
    "extract_boxed_answer",
    # Datasets
    "load_datasets_by_seed",
    "load_multi_approach_datasets",
    "load_from_registry",
    "get_available_subsets",
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
