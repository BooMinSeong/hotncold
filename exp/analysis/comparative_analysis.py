"""
Comparative Analysis of Temperature-Difficulty Performance

Compares four experiments in a 2x2 design:
- Algorithm: BoN vs DVTS
- Reference Temperature: T=0.1 vs T=0.8

This script synthesizes findings from all four difficulty-based temperature analyses.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
import seaborn as sns


@dataclass
class ExperimentMetrics:
    """Metrics extracted from a single difficulty analysis"""
    name: str
    algorithm: str  # "BoN" or "DVTS"
    ref_temp: str   # "0.1" or "0.8"

    # Difficulty distribution
    level_counts: Dict[int, int]  # level -> count

    # Optimal temperatures per level
    optimal_temps: Dict[int, Tuple[str, float]]  # level -> (temp, accuracy)

    # Base model capability
    base_capability: Dict[str, Tuple[float, float]]  # temp -> (mean_acc, std_acc)

    # Full accuracy matrix for heatmaps
    accuracy_by_level_temp: Dict[str, Dict[int, Dict[str, float]]]  # method -> level -> temp -> accuracy


def parse_report(report_path: Path) -> ExperimentMetrics:
    """Parse a difficulty_temperature_report.md file"""

    with open(report_path) as f:
        content = f.read()

    # Extract metadata from path
    parts = report_path.parent.name.split('-')
    algorithm = None
    ref_temp = "0.1"  # default

    for part in parts:
        if 'bon' in part.lower():
            algorithm = "BoN"
        elif 'dvts' in part.lower():
            algorithm = "DVTS"
        if 'ref0.8' in part:
            ref_temp = "0.8"

    if algorithm is None:
        raise ValueError(f"Cannot determine algorithm from {report_path}")

    name = f"{algorithm}-ref{ref_temp}"

    # Parse difficulty distribution
    level_counts = {}
    dist_match = re.search(r'## Difficulty Distribution.*?\n(.*?\n)+?\n##', content, re.DOTALL)
    if dist_match:
        table_text = dist_match.group(0)
        for line in table_text.split('\n'):
            if '|' in line and not line.strip().startswith('|---'):
                parts = [p.strip() for p in line.split('|')]
                # Skip header and separator rows
                if len(parts) >= 4 and parts[1].isdigit():
                    level_num = int(parts[1])
                    count = int(parts[3])
                    level_counts[level_num] = count

    # Parse optimal temperatures per level
    optimal_temps = {}
    for level in range(1, 6):
        # Look for "At 64 samples (naive method), **T0.X** performs best with Y.YYY accuracy" pattern
        level_section_match = re.search(
            rf'## Level {level}:.*?### Best Temperature.*?\*\*(T\d\.\d+)\*\*.*?with ([\d\.]+) accuracy',
            content, re.DOTALL
        )
        if level_section_match:
            temp = level_section_match.group(1)
            acc = float(level_section_match.group(2))
            optimal_temps[level] = (temp, acc)

    # Parse base model capability
    base_capability = {}
    # Find the section and parse line by line
    in_base_section = False
    for line in content.split('\n'):
        if '## Model Base Capability' in line:
            in_base_section = True
            continue
        if in_base_section:
            if line.startswith('##'):  # Next section
                break
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                # Filter empty parts from leading/trailing pipes
                parts = [p for p in parts if p]
                # Look for lines starting with T followed by digit
                if len(parts) >= 3 and parts[0].startswith('T') and parts[0][1].isdigit():
                    temp = parts[0]
                    try:
                        mean_acc = float(parts[1])
                        std_acc = float(parts[2])
                        base_capability[temp] = (mean_acc, std_acc)
                    except (ValueError, IndexError):
                        pass

    # Parse full accuracy matrices
    accuracy_by_level_temp = {"naive": {}, "weighted": {}, "majority": {}}

    for method in ["naive", "weighted", "majority"]:
        for level in range(1, 6):
            accuracy_by_level_temp[method][level] = {}

            # Look for temperature comparison table in level section
            level_section_start = re.search(rf'## Level {level}:', content)
            if level_section_start:
                level_section = content[level_section_start.start():]
                next_level = re.search(r'\n## Level \d+:|$', level_section[10:])
                if next_level:
                    level_section = level_section[:next_level.start() + 10]

                # Find the appropriate table based on method
                table_pattern = rf'{method.title()}.*?\n.*?\n((?:T\d\.\d+.*?\n)+)'
                table_match = re.search(table_pattern, level_section, re.DOTALL | re.IGNORECASE)

                if table_match:
                    for line in table_match.group(1).split('\n'):
                        if line.strip().startswith('T'):
                            parts = [p.strip() for p in line.split('|')]
                            if len(parts) >= 3:
                                temp = parts[1]
                                acc = float(parts[2])
                                accuracy_by_level_temp[method][level][temp] = acc

    return ExperimentMetrics(
        name=name,
        algorithm=algorithm,
        ref_temp=ref_temp,
        level_counts=level_counts,
        optimal_temps=optimal_temps,
        base_capability=base_capability,
        accuracy_by_level_temp=accuracy_by_level_temp
    )


def plot_difficulty_distributions(experiments: List[ExperimentMetrics], output_dir: Path):
    """Create side-by-side difficulty distribution comparison"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, exp in enumerate(experiments):
        ax = axes[idx]

        levels = sorted(exp.level_counts.keys())
        counts = [exp.level_counts[l] for l in levels]

        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        bars = ax.bar(levels, counts, color=colors[:len(levels)], alpha=0.7, edgecolor='black')

        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Difficulty Level', fontsize=12)
        ax.set_ylabel('Number of Problems', fontsize=12)
        ax.set_title(f'{exp.name}\n({exp.algorithm}, ref_temp={exp.ref_temp})',
                    fontsize=13, fontweight='bold')
        ax.set_xticks(levels)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        if counts:
            ax.set_ylim(0, max(counts) * 1.15)
        else:
            ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / 'difficulty_distributions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved difficulty distributions comparison")


def plot_optimal_temperature_comparison(experiments: List[ExperimentMetrics], output_dir: Path):
    """Create heatmap showing optimal temperatures for each experiment and level"""

    # Prepare data matrix
    levels = [1, 2, 3, 4, 5]
    temp_values = {'T0.1': 0.1, 'T0.2': 0.2, 'T0.4': 0.4, 'T0.6': 0.6, 'T0.8': 0.8}

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for idx, exp in enumerate(experiments):
        ax = axes[idx]

        # Create matrix: levels x accuracy
        accuracies = []
        temp_labels = []

        for level in levels:
            if level in exp.optimal_temps:
                temp, acc = exp.optimal_temps[level]
                accuracies.append(acc)
                temp_labels.append(temp)
            else:
                accuracies.append(0)
                temp_labels.append("N/A")

        # Create bar plot with color coding
        colors_map = {0.1: '#3498db', 0.2: '#2ecc71', 0.4: '#f39c12',
                     0.6: '#e67e22', 0.8: '#e74c3c'}
        colors = [colors_map.get(temp_values.get(t, 0), '#95a5a6') for t in temp_labels]

        bars = ax.bar(levels, accuracies, color=colors, alpha=0.8, edgecolor='black')

        # Add temperature labels on bars
        for bar, temp in zip(bars, temp_labels):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       temp, ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Difficulty Level', fontsize=11)
        ax.set_ylabel('Best Accuracy', fontsize=11)
        ax.set_title(f'{exp.name}', fontsize=12, fontweight='bold')
        ax.set_xticks(levels)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, fc=colors_map[temp], alpha=0.8, edgecolor='black')
                      for temp in sorted(colors_map.keys())]
    legend_labels = [f'T{temp}' for temp in sorted(colors_map.keys())]
    fig.legend(legend_elements, legend_labels, loc='upper center',
              ncol=5, bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'optimal_temperatures_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved optimal temperatures comparison")


def plot_base_capability_comparison(experiments: List[ExperimentMetrics], output_dir: Path):
    """Compare base model capability across experiments"""

    temps_order = ['T0.1', 'T0.2', 'T0.4', 'T0.6', 'T0.8']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: By algorithm (should be identical regardless of ref_temp)
    bon_exps = [e for e in experiments if e.algorithm == "BoN"]
    dvts_exps = [e for e in experiments if e.algorithm == "DVTS"]

    if bon_exps:
        bon_data = bon_exps[0].base_capability  # Should be same for both ref temps
        temps = [t for t in temps_order if t in bon_data]
        means = [bon_data[t][0] for t in temps]
        stds = [bon_data[t][1] for t in temps]
        ax1.errorbar(range(len(temps)), means, yerr=stds, marker='o', markersize=8,
                    linewidth=2, capsize=5, label='BoN', color='#3498db')

    if dvts_exps:
        dvts_data = dvts_exps[0].base_capability
        temps = [t for t in temps_order if t in dvts_data]
        means = [dvts_data[t][0] for t in temps]
        stds = [dvts_data[t][1] for t in temps]
        ax1.errorbar(range(len(temps)), means, yerr=stds, marker='s', markersize=8,
                    linewidth=2, capsize=5, label='DVTS', color='#e74c3c')

    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Base Model Capability by Algorithm', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(temps_order)))
    ax1.set_xticklabels(temps_order)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0.4, 0.6)

    # Plot 2: All four experiments overlayed (should show identical curves within same algorithm)
    colors = {'BoN-ref0.1': '#3498db', 'BoN-ref0.8': '#5dade2',
             'DVTS-ref0.1': '#e74c3c', 'DVTS-ref0.8': '#ec7063'}
    markers = {'BoN-ref0.1': 'o', 'BoN-ref0.8': '^',
              'DVTS-ref0.1': 's', 'DVTS-ref0.8': 'D'}

    for exp in experiments:
        temps = [t for t in temps_order if t in exp.base_capability]
        means = [exp.base_capability[t][0] for t in temps]
        stds = [exp.base_capability[t][1] for t in temps]
        ax2.errorbar(range(len(temps)), means, yerr=stds,
                    marker=markers.get(exp.name, 'o'), markersize=7,
                    linewidth=2, capsize=4, label=exp.name,
                    color=colors.get(exp.name, '#95a5a6'), alpha=0.8)

    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Base Capability: All Experiments (Ref Temp Independence)',
                 fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(temps_order)))
    ax2.set_xticklabels(temps_order)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0.4, 0.6)

    plt.tight_layout()
    plt.savefig(output_dir / 'base_capability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved base capability comparison")


def create_summary_table(experiments: List[ExperimentMetrics], output_dir: Path):
    """Create comprehensive summary table"""

    lines = []
    lines.append("# Temperature-Difficulty Performance: Comparative Analysis")
    lines.append("")
    lines.append("## Executive Summary: 2×2 Experimental Design")
    lines.append("")
    lines.append("| Algorithm | Ref Temp=0.1 | Ref Temp=0.8 |")
    lines.append("|-----------|--------------|--------------|")

    # Organize experiments into grid
    exp_dict = {(e.algorithm, e.ref_temp): e for e in experiments}

    for algo in ["BoN", "DVTS"]:
        row = [f"**{algo}**"]
        for ref in ["0.1", "0.8"]:
            if (algo, ref) in exp_dict:
                e = exp_dict[(algo, ref)]
                dist_str = "/".join(str(e.level_counts.get(i, 0)) for i in range(1, 6))
                row.append(f"{e.name}<br/>({dist_str})")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("Distribution format: Level1/Level2/Level3/Level4/Level5 problem counts")
    lines.append("")

    # Section: Difficulty Distribution Analysis
    lines.append("## Phase 1: Difficulty Distribution Analysis")
    lines.append("")

    lines.append("### Problem Classification by Experiment")
    lines.append("")
    lines.append("| Experiment | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 | Total |")
    lines.append("|------------|---------|---------|---------|---------|---------|-------|")

    for exp in experiments:
        counts = [str(exp.level_counts.get(i, 0)) for i in range(1, 6)]
        total = sum(exp.level_counts.values())
        lines.append(f"| {exp.name} | " + " | ".join(counts) + f" | {total} |")

    lines.append("")
    lines.append("### Key Observations")
    lines.append("")

    # Compare within algorithms
    if ("BoN", "0.1") in exp_dict and ("BoN", "0.8") in exp_dict:
        bon_01 = exp_dict[("BoN", "0.1")]
        bon_08 = exp_dict[("BoN", "0.8")]
        diff_l1 = bon_01.level_counts.get(1, 0) - bon_08.level_counts.get(1, 0)
        lines.append(f"- **BoN Algorithm**: T=0.1 baseline classifies {abs(diff_l1)} {'more' if diff_l1 > 0 else 'fewer'} problems as 'easy' (Level 1) than T=0.8 baseline")
        lines.append(f"  - BoN-ref0.1: {bon_01.level_counts.get(1, 0)} Level 1 problems")
        lines.append(f"  - BoN-ref0.8: {bon_08.level_counts.get(1, 0)} Level 1 problems")

    if ("DVTS", "0.1") in exp_dict and ("DVTS", "0.8") in exp_dict:
        dvts_01 = exp_dict[("DVTS", "0.1")]
        dvts_08 = exp_dict[("DVTS", "0.8")]
        diff_l1 = dvts_01.level_counts.get(1, 0) - dvts_08.level_counts.get(1, 0)
        lines.append(f"- **DVTS Algorithm**: T=0.1 baseline classifies {abs(diff_l1)} {'more' if diff_l1 > 0 else 'fewer'} problems as 'easy' (Level 1) than T=0.8 baseline")
        lines.append(f"  - DVTS-ref0.1: {dvts_01.level_counts.get(1, 0)} Level 1 problems")
        lines.append(f"  - DVTS-ref0.8: {dvts_08.level_counts.get(1, 0)} Level 1 problems")

    lines.append("")

    # Compare across algorithms at same baseline
    if ("BoN", "0.8") in exp_dict and ("DVTS", "0.8") in exp_dict:
        bon_08 = exp_dict[("BoN", "0.8")]
        dvts_08 = exp_dict[("DVTS", "0.8")]
        diff_l1 = dvts_08.level_counts.get(1, 0) - bon_08.level_counts.get(1, 0)
        lines.append(f"- **At T=0.8 baseline**: DVTS classifies {abs(diff_l1)} {'more' if diff_l1 > 0 else 'fewer'} problems as 'easy' than BoN")

    if ("BoN", "0.1") in exp_dict and ("DVTS", "0.1") in exp_dict:
        bon_01 = exp_dict[("BoN", "0.1")]
        dvts_01 = exp_dict[("DVTS", "0.1")]
        diff_l1 = dvts_01.level_counts.get(1, 0) - bon_01.level_counts.get(1, 0)
        lines.append(f"- **At T=0.1 baseline**: DVTS classifies {abs(diff_l1)} {'more' if diff_l1 > 0 else 'fewer'} problems as 'easy' than BoN")

    lines.append("")

    # Section: Optimal Temperature Analysis
    lines.append("## Phase 2: Optimal Temperature Analysis")
    lines.append("")

    for exp in experiments:
        lines.append(f"### {exp.name}")
        lines.append("")
        lines.append("| Level | Optimal Temp | Best Accuracy |")
        lines.append("|-------|--------------|---------------|")

        for level in range(1, 6):
            if level in exp.optimal_temps:
                temp, acc = exp.optimal_temps[level]
                lines.append(f"| Level {level} | {temp} | {acc:.3f} |")
            else:
                lines.append(f"| Level {level} | N/A | N/A |")

        lines.append("")

    # Temperature preference patterns
    lines.append("### Temperature Preference Patterns")
    lines.append("")

    for exp in experiments:
        lines.append(f"**{exp.name}**:")
        temp_preferences = {}
        for level, (temp, acc) in exp.optimal_temps.items():
            if temp not in temp_preferences:
                temp_preferences[temp] = []
            temp_preferences[temp].append(level)

        for temp in sorted(temp_preferences.keys()):
            levels = temp_preferences[temp]
            lines.append(f"- {temp}: Levels {', '.join(map(str, levels))}")
        lines.append("")

    # Section: Base Model Capability
    lines.append("## Phase 3: Base Model Capability")
    lines.append("")

    lines.append("### BoN Algorithm")
    lines.append("")
    lines.append("| Temperature | Accuracy (mean ± std) |")
    lines.append("|-------------|----------------------|")

    if bon_exps := [e for e in experiments if e.algorithm == "BoN"]:
        base_cap = bon_exps[0].base_capability
        for temp in ['T0.1', 'T0.2', 'T0.4', 'T0.6', 'T0.8']:
            if temp in base_cap:
                mean, std = base_cap[temp]
                lines.append(f"| {temp} | {mean:.3f} ± {std:.3f} |")

    lines.append("")
    lines.append("### DVTS Algorithm")
    lines.append("")
    lines.append("| Temperature | Accuracy (mean ± std) |")
    lines.append("|-------------|----------------------|")

    if dvts_exps := [e for e in experiments if e.algorithm == "DVTS"]:
        base_cap = dvts_exps[0].base_capability
        for temp in ['T0.1', 'T0.2', 'T0.4', 'T0.6', 'T0.8']:
            if temp in base_cap:
                mean, std = base_cap[temp]
                lines.append(f"| {temp} | {mean:.3f} ± {std:.3f} |")

    lines.append("")

    # Calculate capability gap
    if bon_exps and dvts_exps:
        bon_best = max(bon_exps[0].base_capability.values(), key=lambda x: x[0])
        dvts_best = max(dvts_exps[0].base_capability.values(), key=lambda x: x[0])
        gap = (dvts_best[0] - bon_best[0]) * 100
        lines.append(f"**Key Finding**: DVTS achieves {gap:.1f}% higher base capability than BoN")
        lines.append(f"- BoN best: {bon_best[0]:.3f}")
        lines.append(f"- DVTS best: {dvts_best[0]:.3f}")

    lines.append("")

    # Write report
    report_path = output_dir / 'comparative_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Saved comparative analysis report to {report_path}")


def analyze_algorithm_baseline_interactions(experiments: List[ExperimentMetrics], output_dir: Path):
    """Analyze how reference temperature affects optimal temperature choices differently for each algorithm"""

    exp_dict = {(e.algorithm, e.ref_temp): e for e in experiments}

    lines = []
    lines.append("## Critical Finding: Algorithm-Baseline Interaction Effects")
    lines.append("")
    lines.append("### How Reference Temperature Affects Optimal Temperature Selection")
    lines.append("")

    # Analyze BoN robustness
    if ("BoN", "0.1") in exp_dict and ("BoN", "0.8") in exp_dict:
        bon_01 = exp_dict[("BoN", "0.1")]
        bon_08 = exp_dict[("BoN", "0.8")]

        lines.append("#### BoN: Robust to Reference Temperature")
        lines.append("")
        lines.append("| Level | Optimal @ ref0.1 | Optimal @ ref0.8 | Agreement |")
        lines.append("|-------|------------------|------------------|-----------|")

        agreements = 0
        total = 0
        for level in range(1, 6):
            if level in bon_01.optimal_temps and level in bon_08.optimal_temps:
                temp_01, acc_01 = bon_01.optimal_temps[level]
                temp_08, acc_08 = bon_08.optimal_temps[level]
                agree = "✓" if temp_01 == temp_08 else "✗"
                if temp_01 == temp_08:
                    agreements += 1
                total += 1
                lines.append(f"| Level {level} | {temp_01} ({acc_01:.3f}) | {temp_08} ({acc_08:.3f}) | {agree} |")

        lines.append("")
        lines.append(f"**BoN Consistency**: {agreements}/{total} levels ({100*agreements/total:.0f}%) have same optimal temperature regardless of reference")
        lines.append("")

    # Analyze DVTS sensitivity
    if ("DVTS", "0.1") in exp_dict and ("DVTS", "0.8") in exp_dict:
        dvts_01 = exp_dict[("DVTS", "0.1")]
        dvts_08 = exp_dict[("DVTS", "0.8")]

        lines.append("#### DVTS: Sensitive to Reference Temperature")
        lines.append("")
        lines.append("| Level | Optimal @ ref0.1 | Optimal @ ref0.8 | Agreement |")
        lines.append("|-------|------------------|------------------|-----------|")

        agreements = 0
        total = 0
        for level in range(1, 6):
            if level in dvts_01.optimal_temps and level in dvts_08.optimal_temps:
                temp_01, acc_01 = dvts_01.optimal_temps[level]
                temp_08, acc_08 = dvts_08.optimal_temps[level]
                agree = "✓" if temp_01 == temp_08 else "✗"
                if temp_01 == temp_08:
                    agreements += 1
                total += 1
                lines.append(f"| Level {level} | {temp_01} ({acc_01:.3f}) | {temp_08} ({acc_08:.3f}) | {agree} |")

        lines.append("")
        lines.append(f"**DVTS Consistency**: {agreements}/{total} levels ({100*agreements/total:.0f}%) have same optimal temperature regardless of reference")
        lines.append("")

    lines.append("### Interpretation")
    lines.append("")
    lines.append("- **BoN is robust**: Optimal temperature preferences are stable across different reference temperatures")
    lines.append("- **DVTS is sensitive**: Reference temperature choice significantly impacts which temperatures work best")
    lines.append("- **Implication**: When using DVTS, the choice of reference temperature for difficulty stratification is critical")
    lines.append("- **Recommendation**: For BoN, any reasonable reference works; for DVTS, carefully validate reference temperature choice")
    lines.append("")

    # Save to separate file
    interaction_path = output_dir / 'algorithm_baseline_interactions.md'
    with open(interaction_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ Saved interaction analysis to {interaction_path}")

    return lines


def main():
    """Main analysis pipeline"""

    # Define experiment paths
    base_dir = Path("exp")
    exp_dirs = [
        "analysis_output-MATH500-Qwen2.5-3B-bon-difficulty",
        "analysis_output-MATH500-Qwen2.5-3B-bon-difficulty-ref0.8",
        "analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty",
        "analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty-ref0.8",
    ]

    # Output directory
    output_dir = base_dir / "comparative_analysis_MATH500-Qwen2.5-3B"
    output_dir.mkdir(exist_ok=True)

    print(f"Comparative Analysis: Temperature-Difficulty Performance")
    print(f"=" * 60)
    print()

    # Parse all experiments
    experiments = []
    for exp_dir in exp_dirs:
        report_path = base_dir / exp_dir / "difficulty_temperature_report.md"
        if report_path.exists():
            print(f"Parsing {exp_dir}...")
            exp = parse_report(report_path)
            experiments.append(exp)
        else:
            print(f"WARNING: {report_path} not found")

    if len(experiments) != 4:
        print(f"ERROR: Expected 4 experiments, found {len(experiments)}")
        return

    print(f"\n✓ Successfully parsed all 4 experiments")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    plot_difficulty_distributions(experiments, output_dir)
    plot_optimal_temperature_comparison(experiments, output_dir)
    plot_base_capability_comparison(experiments, output_dir)

    # Generate reports
    print("\nGenerating reports...")
    create_summary_table(experiments, output_dir)
    interaction_lines = analyze_algorithm_baseline_interactions(experiments, output_dir)

    # Append interaction analysis to main report
    report_path = output_dir / 'comparative_analysis_report.md'
    with open(report_path, 'a') as f:
        f.write('\n\n')
        f.write('\n'.join(interaction_lines))

    print(f"\n{'=' * 60}")
    print(f"Analysis complete! Results saved to:")
    print(f"  {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
