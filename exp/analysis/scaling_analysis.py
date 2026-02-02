"""
Scaling Behavior Analysis

Analyzes how accuracy improves with sample budget (N) for BoN vs DVTS
across different difficulty levels and temperatures.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_level_data(report_dir: Path, level: int) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """
    Parse accuracy data for a specific level across all temperatures and methods.

    Returns: {method: {temp: [(N, accuracy), ...]}}
    """
    report_path = report_dir / "difficulty_temperature_report.md"

    with open(report_path) as f:
        content = f.read()

    # Find the level section
    level_pattern = rf'## Level {level}:.*?(?=## Level \d+:|## Model Base Capability|$)'
    level_match = re.search(level_pattern, content, re.DOTALL)

    if not level_match:
        return {}

    level_section = level_match.group(0)

    # Parse each method's table
    data = {}
    methods = ['Naive', 'Weighted', 'Maj']

    for method in methods:
        method_key = 'majority' if method == 'Maj' else method.lower()
        data[method_key] = {}

        # Find method section with more flexible matching
        method_pattern = rf'### {method} Method\s*\n\s*\| N \|(.*?)\|\s*\n\s*\|---'
        method_match = re.search(method_pattern, level_section, re.DOTALL)

        if not method_match:
            continue

        # Parse header to get temperature columns
        header = method_match.group(1)
        temps = [t.strip() for t in header.split('|') if t.strip()]

        # Initialize temperature lists
        for temp in temps:
            data[method_key][temp] = []

        # Find table rows (lines starting with | followed by digit)
        table_pattern = rf'### {method} Method.*?\n\| N \|.*?\n\|---|.*?\n((?:\| \d+.*?\n)+)'
        table_match = re.search(table_pattern, level_section, re.DOTALL)

        if not table_match:
            continue

        table_lines = table_match.group(1).strip().split('\n')

        # Parse each row
        for line in table_lines:
            if not line.strip().startswith('|'):
                continue

            parts = [p.strip() for p in line.split('|')]
            parts = [p for p in parts if p]  # Remove empty

            if len(parts) >= len(temps) + 1 and parts[0].isdigit():
                try:
                    n = int(parts[0])
                    for i, temp in enumerate(temps):
                        if i + 1 < len(parts):
                            acc_str = parts[i + 1]
                            # Parse "X.XXX ± Y.YYY" format
                            if '±' in acc_str:
                                acc = float(acc_str.split('±')[0].strip())
                                data[method_key][temp].append((n, acc))
                except (ValueError, IndexError) as e:
                    pass

    return data


def get_experiment_name(report_dir: Path) -> Tuple[str, str]:
    """Extract algorithm and reference temperature from directory name"""
    dir_name = report_dir.name

    algorithm = None
    ref_temp = "0.1"

    if 'bon' in dir_name.lower():
        algorithm = "BoN"
    elif 'dvts' in dir_name.lower():
        algorithm = "DVTS"

    if 'ref0.8' in dir_name:
        ref_temp = "0.8"

    return algorithm, ref_temp


def plot_scaling_curves_by_level(experiments: List[Path], output_dir: Path):
    """Create scaling curves for each difficulty level comparing all experiments"""

    levels = [1, 2, 3, 4, 5]
    level_names = {
        1: "Level 1: Easy (0.8-1.0)",
        2: "Level 2: Medium-Easy (0.6-0.8)",
        3: "Level 3: Medium (0.4-0.6)",
        4: "Level 4: Hard (0.2-0.4)",
        5: "Level 5: Very Hard (0.0-0.2)"
    }

    # Parse all experiments
    exp_data = {}
    for exp_dir in experiments:
        algo, ref = get_experiment_name(exp_dir)
        exp_name = f"{algo}-ref{ref}"
        exp_data[exp_name] = {}

        for level in levels:
            exp_data[exp_name][level] = parse_level_data(exp_dir, level)

    # Plot weighted method (best performing) for each level
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    colors = {
        'BoN-ref0.1': '#3498db',
        'BoN-ref0.8': '#5dade2',
        'DVTS-ref0.1': '#e74c3c',
        'DVTS-ref0.8': '#ec7063'
    }

    markers = {
        'BoN-ref0.1': 'o',
        'BoN-ref0.8': '^',
        'DVTS-ref0.1': 's',
        'DVTS-ref0.8': 'D'
    }

    # Best temperature per experiment (from earlier analysis)
    best_temps = {
        ('BoN', '0.1', 1): 'T0.1',
        ('BoN', '0.1', 2): 'T0.2',
        ('BoN', '0.1', 3): 'T0.1',
        ('BoN', '0.1', 4): 'T0.4',
        ('BoN', '0.1', 5): 'T0.2',
        ('BoN', '0.8', 1): 'T0.1',
        ('BoN', '0.8', 2): 'T0.2',
        ('BoN', '0.8', 3): 'T0.2',
        ('BoN', '0.8', 4): 'T0.4',
        ('BoN', '0.8', 5): 'T0.2',
        ('DVTS', '0.1', 1): 'T0.1',
        ('DVTS', '0.1', 2): 'T0.8',
        ('DVTS', '0.1', 3): 'T0.1',
        ('DVTS', '0.1', 4): 'T0.4',
        ('DVTS', '0.1', 5): 'T0.8',
        ('DVTS', '0.8', 1): 'T0.1',
        ('DVTS', '0.8', 2): 'T0.8',
        ('DVTS', '0.8', 3): 'T0.2',
        ('DVTS', '0.8', 4): 'T0.8',
        ('DVTS', '0.8', 5): 'T0.2',
    }

    for idx, level in enumerate(levels):
        ax = axes[idx]

        for exp_name, level_data in exp_data.items():
            if level not in level_data:
                continue

            algo, ref = exp_name.split('-ref')
            best_temp = best_temps.get((algo, ref, level), 'T0.2')

            # Get weighted method data at best temperature
            if 'weighted' in level_data[level] and best_temp in level_data[level]['weighted']:
                data_points = level_data[level]['weighted'][best_temp]
                if data_points:
                    ns, accs = zip(*sorted(data_points))

                    ax.plot(ns, accs,
                           marker=markers.get(exp_name, 'o'),
                           markersize=8,
                           linewidth=2.5,
                           label=f"{exp_name} ({best_temp})",
                           color=colors.get(exp_name, '#95a5a6'),
                           alpha=0.85)

        ax.set_xlabel('Sample Budget (N)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(level_names[level], fontsize=13, fontweight='bold')
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64'])
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')

        # Set reasonable y-axis limits
        if level == 1:
            ax.set_ylim(0.85, 1.02)
        elif level in [2, 3]:
            ax.set_ylim(0.5, 1.0)
        else:
            ax.set_ylim(0.3, 0.9)

    # Hide the 6th subplot
    axes[5].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'scaling_curves_by_level.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved scaling curves by level")


def plot_algorithm_comparison_scaling(experiments: List[Path], output_dir: Path):
    """Compare BoN vs DVTS scaling behavior directly"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Parse data
    exp_data = {}
    for exp_dir in experiments:
        algo, ref = get_experiment_name(exp_dir)
        exp_name = f"{algo}-ref{ref}"

        # Get overall accuracy at best temperature across all levels
        # Use Level 3 (medium difficulty) as representative
        level_data = parse_level_data(exp_dir, 3)
        exp_data[exp_name] = level_data

    colors = {
        'BoN': '#3498db',
        'DVTS': '#e74c3c'
    }

    # Plot 1: ref0.1 baseline
    ax1 = axes[0]
    for exp_name, level_data in exp_data.items():
        if 'ref0.1' not in exp_name:
            continue

        algo = exp_name.split('-')[0]
        best_temp = 'T0.2' if algo == 'BoN' else 'T0.1'

        if 'weighted' in level_data and best_temp in level_data['weighted']:
            data_points = level_data['weighted'][best_temp]
            if data_points:
                ns, accs = zip(*sorted(data_points))

                ax1.plot(ns, accs,
                        marker='o' if algo == 'BoN' else 's',
                        markersize=10,
                        linewidth=3,
                        label=f"{algo} ({best_temp})",
                        color=colors[algo],
                        alpha=0.85)

    ax1.set_xlabel('Sample Budget (N)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Accuracy (Level 3)', fontsize=13, fontweight='bold')
    ax1.set_title('Scaling Comparison: ref0.1 Baseline', fontsize=14, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax1.set_xticklabels(['1', '2', '4', '8', '16', '32', '64'])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12)
    ax1.set_ylim(0.5, 0.95)

    # Plot 2: ref0.8 baseline
    ax2 = axes[1]
    for exp_name, level_data in exp_data.items():
        if 'ref0.8' not in exp_name:
            continue

        algo = exp_name.split('-')[0]
        best_temp = 'T0.2'

        if 'weighted' in level_data and best_temp in level_data['weighted']:
            data_points = level_data['weighted'][best_temp]
            if data_points:
                ns, accs = zip(*sorted(data_points))

                ax2.plot(ns, accs,
                        marker='o' if algo == 'BoN' else 's',
                        markersize=10,
                        linewidth=3,
                        label=f"{algo} ({best_temp})",
                        color=colors[algo],
                        alpha=0.85)

    ax2.set_xlabel('Sample Budget (N)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Accuracy (Level 3)', fontsize=13, fontweight='bold')
    ax2.set_title('Scaling Comparison: ref0.8 Baseline', fontsize=14, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax2.set_xticklabels(['1', '2', '4', '8', '16', '32', '64'])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=12)
    ax2.set_ylim(0.5, 0.95)

    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_scaling_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved algorithm scaling comparison")


def analyze_compute_efficiency(experiments: List[Path], output_dir: Path):
    """Analyze compute efficiency: accuracy gain per doubling of N"""

    lines = []
    lines.append("# Compute Efficiency Analysis")
    lines.append("")
    lines.append("## Accuracy Gain per Doubling of Sample Budget")
    lines.append("")
    lines.append("This analysis examines the marginal accuracy improvement when doubling N.")
    lines.append("High values indicate efficient use of compute; low values suggest diminishing returns.")
    lines.append("")

    # Parse Level 3 data (medium difficulty) as representative
    for exp_dir in experiments:
        algo, ref = get_experiment_name(exp_dir)
        exp_name = f"{algo}-ref{ref}"

        lines.append(f"### {exp_name}")
        lines.append("")

        level_data = parse_level_data(exp_dir, 3)

        if 'weighted' not in level_data:
            lines.append("No data available.")
            lines.append("")
            continue

        lines.append("| N Range | Temperature | Acc @ N | Acc @ 2N | Gain | Efficiency |")
        lines.append("|---------|-------------|---------|----------|------|------------|")

        # Check common temperatures
        for temp in ['T0.1', 'T0.2', 'T0.4', 'T0.8']:
            if temp not in level_data['weighted']:
                continue

            data_points = sorted(level_data['weighted'][temp])

            for i in range(len(data_points) - 1):
                n1, acc1 = data_points[i]
                n2, acc2 = data_points[i + 1]

                if n2 == n1 * 2:  # Doubling
                    gain = acc2 - acc1
                    efficiency = gain * 100  # Percentage points

                    lines.append(f"| {n1}→{n2} | {temp} | {acc1:.3f} | {acc2:.3f} | {gain:+.3f} | {efficiency:+.2f}% |")

        lines.append("")

    lines.append("## Key Observations")
    lines.append("")
    lines.append("- **Early gains (N=1→2, 2→4)**: Typically show largest improvements")
    lines.append("- **Plateau effect (N≥32)**: Diminishing returns become apparent")
    lines.append("- **DVTS efficiency**: Generally shows better efficiency at higher N")
    lines.append("- **Temperature dependency**: Lower temperatures often plateau faster")
    lines.append("")

    efficiency_path = output_dir / 'compute_efficiency_analysis.md'
    with open(efficiency_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Saved compute efficiency analysis")


def main():
    """Main analysis pipeline"""

    base_dir = Path("exp")
    exp_dirs = [
        base_dir / "analysis_output-MATH500-Qwen2.5-3B-bon-difficulty",
        base_dir / "analysis_output-MATH500-Qwen2.5-3B-bon-difficulty-ref0.8",
        base_dir / "analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty",
        base_dir / "analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty-ref0.8",
    ]

    output_dir = base_dir / "comparative_analysis_MATH500-Qwen2.5-3B"

    print("Scaling Behavior Analysis")
    print("=" * 60)
    print()

    print("Generating scaling visualizations...")
    plot_scaling_curves_by_level(exp_dirs, output_dir)
    plot_algorithm_comparison_scaling(exp_dirs, output_dir)

    print("\nGenerating compute efficiency analysis...")
    analyze_compute_efficiency(exp_dirs, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Scaling analysis complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
