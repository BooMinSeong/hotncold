"""Visualization utilities for experiment analysis.

This module provides common plotting functions and styles used across analysis scripts.
"""

import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Default color palettes
STRATEGY_COLORS = {
    "hnc": "#1f77b4",
    "default": "#ff7f0e",
    "default-T0.4": "#2ca02c",
    "default-T0.8": "#ff7f0e",
}

METHOD_COLORS = {
    "naive": "#1f77b4",
    "weighted": "#ff7f0e",
    "maj": "#2ca02c",
}

APPROACH_COLORS = {
    "bon": "#1f77b4",
    "beam_search": "#ff7f0e",
    "dvts": "#2ca02c",
}


def setup_style(style: str = "whitegrid") -> None:
    """Set up seaborn plotting style.

    Args:
        style: Seaborn style name (default: "whitegrid")
    """
    sns.set_style(style)


def save_figure(
    fig: plt.Figure,
    output_path: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    close: bool = True,
) -> None:
    """Save figure to file and optionally close it.

    Args:
        fig: Matplotlib figure
        output_path: Output file path
        dpi: Resolution (default: 300)
        bbox_inches: Bounding box setting (default: "tight")
        close: Whether to close the figure after saving
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    if close:
        plt.close(fig)


def plot_scaling_curve(
    n_values: list[int],
    means: list[float],
    stds: Optional[list[float]] = None,
    ax: Optional[plt.Axes] = None,
    label: str = "",
    color: str = "#1f77b4",
    linestyle: str = "-",
    marker: str = "o",
    linewidth: float = 2,
    markersize: float = 8,
    fill_alpha: float = 0.2,
    show_error_band: bool = True,
) -> plt.Axes:
    """Plot a scaling curve with optional error bands.

    Args:
        n_values: X-axis values (number of samples)
        means: Y-axis values (mean accuracy)
        stds: Standard deviations for error bands
        ax: Matplotlib axes (creates new if None)
        label: Legend label
        color: Line color
        linestyle: Line style ("-", "--", etc.)
        marker: Marker style ("o", "s", etc.)
        linewidth: Line width
        markersize: Marker size
        fill_alpha: Transparency for error bands
        show_error_band: Whether to show error bands

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    means = np.array(means)
    ax.plot(
        n_values,
        means,
        marker=marker,
        linestyle=linestyle,
        label=label,
        color=color,
        linewidth=linewidth,
        markersize=markersize,
    )

    if show_error_band and stds is not None:
        stds = np.array(stds)
        ax.fill_between(
            n_values,
            means - stds,
            means + stds,
            alpha=fill_alpha,
            color=color,
        )

    return ax


def plot_comparison(
    results: dict[str, dict[str, Any]],
    output_path: str,
    title: str = "Comparison",
    xlabel: str = "Number of Samples (n)",
    ylabel: str = "Accuracy",
    figsize: tuple[int, int] = (10, 6),
    log_scale: bool = False,
    colors: Optional[dict[str, str]] = None,
) -> None:
    """Plot comparison of multiple configurations.

    Args:
        results: Dict mapping config_name -> {'n_values': [...], 'means': [...], 'stds': [...]}
        output_path: Output file path
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        log_scale: Use log scale for x-axis
        colors: Optional dict mapping config_name -> color
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    default_colors = list(plt.cm.tab10.colors)

    for idx, (config_name, data) in enumerate(results.items()):
        color = colors.get(config_name) if colors else default_colors[idx % len(default_colors)]

        # Determine line style based on name
        linestyle = "--" if "default" in config_name.lower() else "-"
        marker = "o" if "hnc" in config_name.lower() else ""

        plot_scaling_curve(
            n_values=data["n_values"],
            means=data["means"],
            stds=data.get("stds"),
            ax=ax,
            label=config_name,
            color=color,
            linestyle=linestyle,
            marker=marker,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_xscale("log", base=2)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_bar_comparison(
    categories: list[str],
    values: dict[str, list[float]],
    errors: Optional[dict[str, list[float]]] = None,
    output_path: str = "",
    title: str = "Comparison",
    xlabel: str = "",
    ylabel: str = "Accuracy",
    figsize: tuple[int, int] = (10, 6),
    colors: Optional[dict[str, str]] = None,
) -> Optional[plt.Figure]:
    """Plot grouped bar chart comparison.

    Args:
        categories: X-axis category labels
        values: Dict mapping group_name -> list of values
        errors: Dict mapping group_name -> list of error values
        output_path: Output file path (if empty, returns figure)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        colors: Optional dict mapping group_name -> color

    Returns:
        Figure if output_path is empty, None otherwise
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(categories))
    width = 0.8 / len(values)
    default_colors = list(plt.cm.tab10.colors)

    for idx, (group_name, group_values) in enumerate(values.items()):
        color = colors.get(group_name) if colors else default_colors[idx % len(default_colors)]
        offset = (idx - len(values) / 2 + 0.5) * width

        error = errors.get(group_name) if errors else None
        ax.bar(
            x + offset,
            group_values,
            width,
            label=group_name,
            color=color,
            yerr=error,
            capsize=3,
        )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)
        return None
    return fig


def plot_heatmap(
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    output_path: str = "",
    title: str = "Heatmap",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "RdYlGn",
    annot: bool = True,
    fmt: str = ".2f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Optional[plt.Figure]:
    """Plot heatmap.

    Args:
        data: 2D numpy array of values
        row_labels: Row labels
        col_labels: Column labels
        output_path: Output file path
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        cmap: Colormap name
        annot: Show annotations
        fmt: Annotation format
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Figure if output_path is empty, None otherwise
    """
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)
        return None
    return fig


def create_results_table(
    data: dict[str, dict[str, float]],
    row_key: str = "config",
    value_key: str = "accuracy",
    std_key: Optional[str] = "std",
) -> str:
    """Create markdown table from results.

    Args:
        data: Dict mapping config -> {metric: value, ...}
        row_key: Name for row header
        value_key: Key for main value
        std_key: Key for standard deviation (optional)

    Returns:
        Markdown table string
    """
    lines = []

    # Header
    headers = [row_key.capitalize()]
    for config in data:
        sample_metrics = data[config]
        break
    headers.extend(
        [k.capitalize() for k in sample_metrics.keys() if k != std_key]
    )
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    # Rows
    for config, metrics in data.items():
        row = [config]
        for key, value in metrics.items():
            if key == std_key:
                continue
            if isinstance(value, float):
                if std_key and std_key in metrics:
                    row.append(f"{value:.3f} ± {metrics[std_key]:.3f}")
                else:
                    row.append(f"{value:.3f}")
            else:
                row.append(str(value))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)
