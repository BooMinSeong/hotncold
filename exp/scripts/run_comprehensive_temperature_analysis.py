#!/usr/bin/env python3
"""Run comprehensive temperature-reference baseline analysis.

This script performs the analysis outlined in the revised plan:
- Part 1: Overall temperature effects (Goal 1)
- Part 2: Reference temperature impact on stratification (Goal 2)
- Part 3: Within-baseline algorithm comparisons
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from exp.analysis.temperature_reference_analysis import run_comprehensive_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive temperature-reference baseline analysis"
    )

    parser.add_argument(
        "--bon-ref01-dir",
        type=Path,
        default=Path("exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty"),
        help="BoN T=0.1 baseline analysis directory",
    )
    parser.add_argument(
        "--bon-ref08-dir",
        type=Path,
        default=Path("exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty-ref0.8"),
        help="BoN T=0.8 baseline analysis directory",
    )
    parser.add_argument(
        "--dvts-ref01-dir",
        type=Path,
        default=Path("exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty"),
        help="DVTS T=0.1 baseline analysis directory",
    )
    parser.add_argument(
        "--dvts-ref08-dir",
        type=Path,
        default=Path("exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty-ref0.8"),
        help="DVTS T=0.8 baseline analysis directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("exp/comprehensive_temperature_analysis-MATH500-Qwen2.5-3B"),
        help="Output directory for combined analysis",
    )

    args = parser.parse_args()

    # Verify all directories exist
    for dir_arg in [args.bon_ref01_dir, args.bon_ref08_dir, args.dvts_ref01_dir, args.dvts_ref08_dir]:
        if not dir_arg.exists():
            raise FileNotFoundError(f"Analysis directory not found: {dir_arg}")

    run_comprehensive_analysis(
        bon_ref01_dir=args.bon_ref01_dir,
        bon_ref08_dir=args.bon_ref08_dir,
        dvts_ref01_dir=args.dvts_ref01_dir,
        dvts_ref08_dir=args.dvts_ref08_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
