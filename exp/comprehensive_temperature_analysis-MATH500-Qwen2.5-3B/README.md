# Comprehensive Temperature-Difficulty Analysis

## Overview

This directory contains a comprehensive analysis of how temperature and difficulty interact for the Qwen2.5-3B model on MATH-500 problems, using both Best-of-N (BoN) and Diverse Verifier Tree Search (DVTS) algorithms.

## Research Questions

1. **Goal 1 (Temperature Effects):** Do different temperatures solve problems differently? Which temperatures work best for which problem types?

2. **Goal 2 (Baseline Impact):** How does the choice of reference temperature (T=0.1 vs T=0.8) affect difficulty stratification and optimal temperature findings?

## Key Files

### Executive Summary
📄 **`EXECUTIVE_SUMMARY.md`** - Start here!
- Answers both research questions
- Practical recommendations for practitioners
- Key findings and algorithmic insights
- ~10 minute read

### Technical Deep-Dive
📄 **`TECHNICAL_ANALYSIS.md`**
- Detailed experimental analysis
- Statistical considerations
- Methodological discussion
- Future research directions
- ~30 minute read

### Comparative Data
📊 **`stratification_comparison.md`**
- Problem redistribution across baselines
- BoN vs DVTS at same baseline
- Delta analysis

📊 **`optimal_temperature_comparison.md`**
- Consistency analysis across baselines
- BoN: 80% consistent (4/5 levels)
- DVTS: 40% consistent (2/5 levels)

📊 **`base_capability_verification.md`**
- Confirms base capability is baseline-independent
- DVTS shows 3-10% advantage over BoN

### Visualizations
🖼️ **`difficulty_distributions_2x2_comparison.png`**
- Shows how 500 problems are stratified differently
- 2×2 grid: BoN vs DVTS × ref0.1 vs ref0.8

🖼️ **`optimal_temperature_heatmap_comparison.png`**
- Optimal temperature by algorithm, baseline, and difficulty level
- Visualizes robustness differences

🖼️ **`base_capability_comparison_2x2.png`**
- Model performance at each temperature
- Confirms consistency across baselines

### Data Tables
📁 **`stratification_comparison.csv`**
- Machine-readable problem distribution data

## Experimental Design

### 2×2 Full Factorial Design

|           | T=0.1 Baseline | T=0.8 Baseline |
|-----------|----------------|----------------|
| **BoN**   | ✅ 225/27/23/33/192 | ✅ 198/42/28/50/182 |
| **DVTS**  | ✅ 241/32/21/21/185 | ✅ 219/54/36/29/162 |

*Numbers represent problem counts in difficulty Levels 1/2/3/4/5*

### Source Data

All analyses are based on four pre-computed difficulty analyses:

1. **BoN-ref0.1:** `exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty/`
2. **BoN-ref0.8:** `exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty-ref0.8/`
3. **DVTS-ref0.1:** `exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty/`
4. **DVTS-ref0.8:** `exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty-ref0.8/`

## Main Findings

### Finding 1: BoN is Robust, DVTS is Sensitive

**BoN Temperature Strategy (Consistent):**
- Level 1 (easy): T=0.1
- Level 2: T=0.2
- Level 3: T=0.1-0.2 (both work)
- Level 4 (hard): T=0.4
- Level 5 (hardest): T=0.2

**DVTS Temperature Strategy (Context-Dependent):**
- Level 1 (easy): T=0.1 (consistent)
- Level 2: T=0.8 (consistent, surprisingly high!)
- Levels 3-5: Varies significantly by baseline

**Interpretation:** BoN has intrinsic temperature preferences; DVTS preferences depend on how you measure difficulty.

### Finding 2: Massive Problem Redistribution

Changing baseline from T=0.1 to T=0.8:
- Level 1: -12% to -9% (fewer easy problems)
- Level 2: +56% to +69% (more medium problems)
- Level 3: +22% to +71% (more medium-hard problems)
- Level 4: +38% to +52% (more hard problems)
- Level 5: -5% to -12% (fewer hardest problems)

**Why:** T=0.8 is more variable → lower baseline accuracy → problems appear harder.

### Finding 3: DVTS Outperforms on Base Capability

- T=0.1: +4.5% better than BoN
- T=0.2: +6.2% better (DVTS's best temperature)
- T=0.4: +6.4% better
- T=0.8: +10.1% better

Tree search produces higher-quality completions than simple sampling.

### Finding 4: Weighted Aggregation Paradox

On **easy problems:** Weighted > Naive
On **hard problems:** Naive > Weighted

Recommendation: Use adaptive aggregation based on problem difficulty.

## Practical Recommendations

### For BoN Users
✅ Robust temperature strategy:
- Easy (>80% acc): T=0.1
- Medium (40-80% acc): T=0.2
- Hard (20-40% acc): T=0.4
- Hardest (<20% acc): T=0.2

✅ Use T=0.1 baseline for difficulty classification

### For DVTS Users
⚠️ Context-dependent strategy:
- Easy: T=0.1
- Medium: T=0.8 (high diversity helps tree search!)
- Hard: Test both T=0.4 and T=0.8 (depends on problem mix)

⚠️ Reference baseline matters—test sensitivity

### For Researchers
📌 Always report reference baseline
📌 Test robustness to baseline choice
📌 BoN is better for reproducible comparisons
📌 DVTS requires more careful tuning

## How to Reproduce

```bash
# Run the comprehensive analysis
uv run python exp/scripts/run_comprehensive_temperature_analysis.py

# With custom directories
uv run python exp/scripts/run_comprehensive_temperature_analysis.py \
    --bon-ref01-dir exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty \
    --bon-ref08-dir exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty-ref0.8 \
    --dvts-ref01-dir exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty \
    --dvts-ref08-dir exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty-ref0.8 \
    --output-dir exp/comprehensive_temperature_analysis-MATH500-Qwen2.5-3B
```

## Future Work

### Immediate Next Steps
1. **Problem-level tracking:** Which specific problems migrate between levels?
2. **Conditional analysis:** For problems in both baselines' Level X, what's optimal?
3. **Overlap quantification:** Venn diagrams of problem sets per level

### Methodological Improvements
1. **Continuous difficulty:** Regress optimal temp on actual accuracy (not binned levels)
2. **Problem characteristics:** Correlate with length, domain, complexity
3. **Temperature scheduling:** Test adaptive strategies

### Algorithmic Questions
1. Why does DVTS prefer T=0.8 for Level 2?
2. Can we predict optimal temperature from problem features?
3. Can we improve aggregation with adaptive strategies?

## Citation

If you use this analysis, please cite:

```
Comprehensive Temperature-Difficulty Analysis for Test-Time Compute
Qwen2.5-3B on MATH-500
Analysis Date: 2026-02-03
```

## Contact

For questions or collaboration, see the main repository README.

---

**Last Updated:** 2026-02-03
**Analysis Version:** 1.0
**Status:** ✅ Complete
