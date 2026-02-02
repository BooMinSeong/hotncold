# Comparative Analysis: Temperature-Difficulty Performance (Qwen2.5-3B)

## Overview

This directory contains a comprehensive comparative analysis of four temperature-difficulty experiments in a complete 2×2 factorial design:

- **Algorithms**: Best-of-N (BoN) vs Diverse Verifier Tree Search (DVTS)
- **Reference Temperatures**: T=0.1 vs T=0.8 (used for difficulty stratification)
- **Model**: Qwen2.5-3B-Instruct
- **Dataset**: MATH-500

## Quick Summary

**Key Finding**: DVTS achieves **3.2% higher base capability** than BoN (0.566 vs 0.534), but is **significantly more sensitive** to reference temperature choice. BoN shows **robust temperature preferences** across baselines (80% consistency), while DVTS preferences vary dramatically (40% consistency).

### Algorithm Comparison

| Algorithm | Base Accuracy | Temperature Consistency | Best For |
|-----------|---------------|------------------------|----------|
| **BoN** | 0.534 | 80% (4/5 levels) | Robust, predictable deployment |
| **DVTS** | 0.566 (+3.2%) | 40% (2/5 levels) | Maximum accuracy (requires tuning) |

## Files in This Directory

### Reports

1. **`synthesis_report.md`** (⭐ **Start here!**)
   - Executive summary with key findings
   - Critical insights and surprising discoveries
   - Actionable recommendations for practitioners
   - Open research questions
   - **Recommended reading order: 1st**

2. **`comparative_analysis_report.md`**
   - Detailed comparative analysis across all 4 experiments
   - Difficulty distribution tables
   - Optimal temperature analysis per level
   - Base model capability comparison
   - **Recommended reading order: 2nd**

3. **`algorithm_baseline_interactions.md`**
   - Deep dive into how reference temperature affects optimal temperature choices
   - BoN robustness vs DVTS sensitivity analysis
   - Consistency metrics and interpretation
   - **Recommended reading order: 3rd**

### Visualizations

1. **`difficulty_distributions_comparison.png`**
   - Side-by-side bar charts showing problem counts per difficulty level
   - Highlights how T=0.1 vs T=0.8 baseline affects classification
   - Shows DVTS classifies more problems as "easy" than BoN

2. **`optimal_temperatures_comparison.png`**
   - Bar charts showing best accuracy achieved at each difficulty level
   - Color-coded by optimal temperature (blue=T0.1, green=T0.2, orange=T0.4, red=T0.8)
   - Reveals algorithm-specific temperature preferences

3. **`base_capability_comparison.png`**
   - Two plots: by algorithm (left) and all experiments overlayed (right)
   - Shows DVTS's consistent 3.2% advantage over BoN
   - Demonstrates that reference temperature doesn't affect base capability

## Key Findings

### 1. Algorithm Performance Gap

**DVTS outperforms BoN by 3.2%** in base capability:
- BoN best: 0.534 ± 0.001 (at T0.1)
- DVTS best: 0.566 ± 0.002 (at T0.2)

**Why?** DVTS's tree search structure enables better exploration of solution paths through diversity.

### 2. Reference Temperature Matters

**T=0.1 baseline classifies more problems as "easy"**:
- BoN: 225 (ref0.1) vs 198 (ref0.8) = +27 problems (+14%)
- DVTS: 241 (ref0.1) vs 219 (ref0.8) = +22 problems (+11%)

**Interpretation**: T=0.1 produces more consistent outputs, making it easier to identify truly easy problems.

### 3. Temperature Preferences

**BoN is conservative**:
- Prefers T0.1-T0.2 for most difficulty levels
- Only uses T0.4 for hard problems (Level 4)
- 80% consistency across reference temperatures

**DVTS is adventurous**:
- Uses T0.8 for medium-easy and hard problems
- Leverages diversity through high temperature
- 40% consistency (varies by reference temperature)

### 4. Robustness vs Sensitivity Trade-off

**BoN**: Robust but lower performance
- ✓ Predictable behavior
- ✓ Easy to deploy
- ✗ Lower accuracy

**DVTS**: High performance but sensitive
- ✓ Superior accuracy (+3.2%)
- ✓ Better diversity utilization
- ✗ Requires careful tuning

### 5. Surprising Finding

**Very hard problems don't always need high temperature!**

Level 5 (0.0-0.2 baseline accuracy) optimal temperatures:
- BoN-ref0.1: **T0.2** (0.536) ← Not T0.8!
- BoN-ref0.8: **T0.2** (0.527) ← Not T0.8!
- DVTS-ref0.1: T0.8 (0.537) ✓
- DVTS-ref0.8: **T0.2** (0.500) ← Not T0.8!

**Why?** High temperature introduces noise that can hurt extremely difficult problems. A "sweet spot" exists between exploration and quality.

## Recommendations

### For Practitioners

#### Algorithm Selection

**Use DVTS when:**
- Maximum accuracy is critical
- You can invest in reference temperature tuning
- The 3.2% gain justifies complexity

**Use BoN when:**
- Need predictable, robust behavior
- Quick deployment without tuning
- Lower maintenance overhead

#### Reference Temperature

**Recommended: T=0.1**
- Better identifies truly easy problems
- More consistent baseline measurements
- Works well for both algorithms

#### Temperature Strategy

**For BoN** (simple & robust):
```python
if baseline_acc >= 0.8:   temp = 0.1  # Easy
elif baseline_acc >= 0.4: temp = 0.2  # Medium
else:                     temp = 0.4  # Hard
```

**For DVTS** (using ref0.1 baseline):
```python
if baseline_acc >= 0.8:   temp = 0.1  # Easy
elif baseline_acc >= 0.6: temp = 0.8  # Medium-Easy (leverage diversity!)
elif baseline_acc >= 0.4: temp = 0.1  # Medium
elif baseline_acc >= 0.2: temp = 0.4  # Hard
else:                     temp = 0.8  # Very Hard (max exploration)
```

### For Researchers

**Open Questions:**
1. Why does DVTS show algorithm-baseline interaction?
2. What causes the 3.2% capability gap?
3. Why does BoN prefer low-moderate temperatures?
4. Can we develop adaptive temperature schedulers?
5. Do findings generalize across model scales?

See `synthesis_report.md` for detailed research directions.

## Experimental Setup

- **Algorithms**: BoN (Best-of-N), DVTS (Diverse Verifier Tree Search)
- **Model**: `Qwen/Qwen2.5-Math-3B-Instruct`
- **Dataset**: MATH-500
- **Sample budgets**: N ∈ {1, 2, 4, 8, 16, 32, 64}
- **Temperatures**: T ∈ {0.1, 0.2, 0.4, 0.8}
- **Aggregation methods**: naive, weighted, majority vote
- **Difficulty levels**: 5 levels (0.8-1.0, 0.6-0.8, 0.4-0.6, 0.2-0.4, 0.0-0.2)

## Source Experiments

Results synthesized from:
1. `exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty/` (BoN, ref=T0.1)
2. `exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty-ref0.8/` (BoN, ref=T0.8)
3. `exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty/` (DVTS, ref=T0.1)
4. `exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty-ref0.8/` (DVTS, ref=T0.8)

## Analysis Scripts

- `exp/analysis/comparative_analysis.py`: Main comparison script
- `exp/analysis/synthesis_report.py`: Generates executive summary

## Citation

If you use these findings in your research, please cite:

```bibtex
@techreport{qwen25-3b-temp-difficulty-2026,
  title={Temperature-Difficulty Performance Analysis for Test-Time Compute with Qwen2.5-3B},
  author={Search and Learn (SAL) Team},
  year={2026},
  institution={MATH-500 Benchmark Study}
}
```

## License

This analysis is part of the Search and Learn (SAL) research project.

---

**Generated**: 2026-02-03
**Contact**: See main repository for contact information
