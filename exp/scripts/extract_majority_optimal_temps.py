"""
Extract optimal temperatures from Majority method for all 4 experiments
"""

import re
from pathlib import Path

def extract_majority_optimal_temps(report_path: Path):
    """Extract optimal temperatures from majority method at N=64"""

    with open(report_path) as f:
        content = f.read()

    # Get experiment info
    dir_name = report_path.parent.name
    if 'bon' in dir_name.lower():
        algo = "BoN"
    elif 'dvts' in dir_name.lower():
        algo = "DVTS"
    else:
        algo = "Unknown"

    ref_temp = "0.8" if 'ref0.8' in dir_name else "0.1"

    print(f"\n{algo}-ref{ref_temp}:")
    print("=" * 50)

    optimal_temps = {}

    for level in range(1, 6):
        # Find Level section
        level_pattern = rf'## Level {level}:.*?(?=## Level \d+:|## Model Base Capability|$)'
        level_match = re.search(level_pattern, content, re.DOTALL)

        if not level_match:
            continue

        level_section = level_match.group(0)

        # Find Maj Method table
        maj_pattern = r'### Maj Method\s*\n\s*\| N \|(.*?)\|\s*\n\s*\|---.*?\n((?:\| \d+.*?\n)+)'
        maj_match = re.search(maj_pattern, level_section, re.DOTALL)

        if maj_match:
            # Parse header
            header = maj_match.group(1)
            temps = [t.strip() for t in header.split('|') if t.strip()]

            # Parse N=64 row
            table_lines = maj_match.group(2).strip().split('\n')
            n64_accs = {}

            for line in table_lines:
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= len(temps) + 1 and parts[0] == '64':
                    for i, temp in enumerate(temps):
                        if i + 1 < len(parts):
                            acc_str = parts[i + 1]
                            if '±' in acc_str:
                                acc = float(acc_str.split('±')[0].strip())
                                n64_accs[temp] = acc
                    break

            # Find best
            if n64_accs:
                best_temp = max(n64_accs.items(), key=lambda x: x[1])
                optimal_temps[level] = best_temp
                print(f"  Level {level}: {best_temp[0]} → {best_temp[1]:.3f}")

    return algo, ref_temp, optimal_temps


def main():
    base_dir = Path("exp")
    exp_dirs = [
        base_dir / "analysis_output-MATH500-Qwen2.5-3B-bon-difficulty",
        base_dir / "analysis_output-MATH500-Qwen2.5-3B-bon-difficulty-ref0.8",
        base_dir / "analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty",
        base_dir / "analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty-ref0.8",
    ]

    print("\n" + "=" * 70)
    print("MAJORITY METHOD OPTIMAL TEMPERATURES (N=64)")
    print("=" * 70)

    all_results = {}

    for exp_dir in exp_dirs:
        report_path = exp_dir / "difficulty_temperature_report.md"
        if report_path.exists():
            algo, ref, optimal_temps = extract_majority_optimal_temps(report_path)
            all_results[(algo, ref)] = optimal_temps

    # Generate Python dictionary format for scaling_analysis.py
    print("\n" + "=" * 70)
    print("PYTHON DICTIONARY FOR scaling_analysis.py:")
    print("=" * 70)
    print("\nbest_temps_majority = {")

    for (algo, ref), temps in sorted(all_results.items()):
        for level, (temp, acc) in temps.items():
            print(f"    ('{algo}', '{ref}', {level}): '{temp}',")

    print("}")


if __name__ == "__main__":
    main()
