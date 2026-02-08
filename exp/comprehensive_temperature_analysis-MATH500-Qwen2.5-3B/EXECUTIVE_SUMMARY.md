# Executive Summary: Temperature-Difficulty Analysis for Qwen2.5-3B

**Dataset:** MATH-500
**Model:** Qwen2.5-3B-Instruct
**Algorithms:** Best-of-N (BoN), Diverse Verifier Tree Search (DVTS)
**Experimental Design:** 2×2 (2 algorithms × 2 reference temperatures)

---

## Research Questions

### Goal 1: Temperature별로 문제를 다르게 풀게 되는가?
**Question:** Do different temperatures solve problems differently? Which problem types perform best at which temperatures?

### Goal 2: Reference Temperature 선택이 분석 결과에 미치는 영향
**Question:** How does the choice of reference temperature (T=0.1 vs T=0.8) affect difficulty stratification and optimal temperature findings?

---

## Critical Insight

**All analyses use the same 500 MATH problems.** Reference temperature is a **stratification method**, not a dataset characteristic. The problems don't change—only how we classify them by difficulty.

---

## Key Findings

### Finding 1: Reference Temperature Significantly Affects Difficulty Stratification

Using T=0.8 baseline instead of T=0.1 redistributes problems across difficulty levels:

**BoN Distribution Changes (ref0.1 → ref0.8):**
- Level 1 (Easy): 225 → 198 problems (-12%)
- Level 2: 27 → 42 problems (+56%)
- Level 3: 23 → 28 problems (+22%)
- Level 4: 33 → 50 problems (+52%)
- Level 5 (Hard): 192 → 182 problems (-5%)

**DVTS Distribution Changes (ref0.1 → ref0.8):**
- Level 1 (Easy): 241 → 219 problems (-9%)
- Level 2: 32 → 54 problems (+69%)
- Level 3: 21 → 36 problems (+71%)
- Level 4: 21 → 29 problems (+38%)
- Level 5 (Hard): 185 → 162 problems (-12%)

**Interpretation:** T=0.8 baseline classifies fewer problems as "easy" (Level 1) and more as medium-difficulty (Levels 2-4). This makes sense because T=0.8 is inherently more variable, making problems appear harder at the reference temperature.

---

### Finding 2: BoN is Robust, DVTS is Sensitive to Reference Temperature

**BoN Optimal Temperature Consistency: 4/5 levels (80%)**

| Level | ref0.1 | ref0.8 | Consistent? |
|-------|--------|--------|-------------|
| 1 | T0.1 | T0.1 | ✓ |
| 2 | T0.2 | T0.2 | ✓ |
| 3 | T0.1 | T0.2 | ✗ (marginal) |
| 4 | T0.4 | T0.4 | ✓ |
| 5 | T0.2 | T0.2 | ✓ |

**DVTS Optimal Temperature Consistency: 2/5 levels (40%)**

| Level | ref0.1 | ref0.8 | Consistent? |
|-------|--------|--------|-------------|
| 1 | T0.1 | T0.1 | ✓ |
| 2 | T0.8 | T0.8 | ✓ |
| 3 | T0.1 | T0.2 | ✗ |
| 4 | T0.4 | **T0.8** | ✗ (major) |
| 5 | **T0.8** | T0.2 | ✗ (major) |

**Critical Observation:** DVTS shows **strong algorithm-baseline interaction**. The optimal temperature for DVTS changes dramatically based on how problems are stratified, particularly for hard problems (Levels 4-5).

---

### Finding 3: Base Capability is Independent of Reference Temperature ✓

As expected, model performance is identical across reference baselines:

**BoN Base Capability:**
- T0.1: 0.534 (both baselines)
- T0.2: 0.533 (both baselines)
- T0.4: 0.531 (both baselines)
- T0.8: 0.504 (both baselines)

**DVTS Base Capability:**
- T0.1: 0.558 (both baselines)
- T0.2: 0.566 (both baselines) ⭐ Best
- T0.4: 0.565 (both baselines)
- T0.8: 0.555 (both baselines)

**DVTS Advantage:** +3.3% to +10.1% higher base accuracy than BoN across all temperatures.

---

### Finding 4: Algorithm Differences in Problem Classification

**At T=0.1 baseline:**
- DVTS classifies 16 more problems as "easy" (Level 1) than BoN
- DVTS has 7 fewer "hardest" problems (Level 5) than BoN

**At T=0.8 baseline:**
- DVTS classifies 21 more problems as "easy" (Level 1) than BoN
- DVTS has 20 fewer "hardest" problems (Level 5) than BoN

**Interpretation:** DVTS is inherently more capable on easier problems, leading to different difficulty stratifications even with the same reference temperature.

---

## Answers to Research Questions

### Goal 1: Do Different Temperatures Solve Problems Differently?

**YES.** Temperature has a significant impact on problem-solving performance, but the effect is **algorithm-dependent**:

**BoN Temperature Strategy (Robust):**
- **Level 1 (Easy):** T0.1 consistently best (0.997-0.998 accuracy)
- **Level 2:** T0.2 consistently best (0.926-0.937 accuracy)
- **Level 3:** T0.1 or T0.2 work well (~0.87-0.89 accuracy)
- **Level 4 (Hard):** T0.4 consistently best (0.758-0.800 accuracy)
- **Level 5 (Hardest):** T0.2 consistently best (0.527-0.536 accuracy)

**Pattern:** BoN prefers low-medium temperatures (T0.1-T0.4) across all difficulty levels. The strategy is **stable regardless of how you measure difficulty**.

**DVTS Temperature Strategy (Context-Dependent):**
- **Level 1 (Easy):** T0.1 consistently best (0.993-0.995 accuracy)
- **Level 2:** **T0.8** surprisingly best (0.917-0.944 accuracy)
- **Level 3-5:** Optimal temperature varies wildly between baselines

**Pattern:** DVTS shows **non-monotonic temperature preferences**. It prefers **high temperatures (T0.8) for medium-difficulty problems** but the optimal choice for hard problems depends on the stratification method.

**Key Insight:** The question "which temperature is best for hard problems?" has **no universal answer for DVTS**—it depends on how you define "hard."

---

### Goal 2: Does Reference Temperature Choice Affect Analysis Conclusions?

**For BoN: NO (mostly).** Only 1/5 levels changes optimal temperature (Level 3: T0.1 → T0.2), and this is a marginal difference. BoN's temperature strategy is **robust to stratification method**.

**For DVTS: YES (significantly).** 3/5 levels change optimal temperature, with major shifts in Levels 4 and 5. DVTS's optimal temperature **depends critically on the stratification method**.

**Why the Difference?**

1. **BoN sampling is simple:** Sample N completions independently, pick the best. Temperature affects diversity, but the best-of-N selection is straightforward.

2. **DVTS is complex:** Tree search with PRM guidance. The interaction between:
   - Temperature-driven exploration
   - PRM-based pruning
   - Diversity requirements

   creates **context-dependent optima**. How you classify problem difficulty affects which temperature setting best balances exploration vs exploitation in the tree search.

---

## Practical Recommendations

### For Practitioners Using BoN

✅ **Use T=0.1 baseline for difficulty stratification** (more stable, classifies more problems as easy)

**Temperature Selection Guide:**
- Easy problems (>80% baseline accuracy): **T=0.1**
- Medium problems (40-80% baseline accuracy): **T=0.2**
- Hard problems (20-40% baseline accuracy): **T=0.4**
- Hardest problems (<20% baseline accuracy): **T=0.2**

**Why T=0.2 for hardest?** Higher temperatures add noise without improving solution quality when problems are fundamentally too difficult.

### For Practitioners Using DVTS

⚠️ **Reference baseline matters!** Choose based on your problem distribution:

**If most problems are easy (>50% at T=0.1):**
- Use T=0.1 baseline
- Easy problems: T=0.1
- Medium problems: T=0.8 (high diversity helps tree search)
- Hard problems: T=0.4-0.8

**If problems are broadly distributed:**
- Use T=0.8 baseline
- Easy problems: T=0.1
- Medium problems: T=0.8
- Hard problems: Test both T0.2 and T0.8

**General DVTS Insight:** DVTS benefits from **high temperature (T=0.8) for medium-difficulty problems** where exploration diversity helps, but the tree search mechanics make optimal temperature **context-sensitive** for very hard problems.

### For Researchers

1. **Always report reference baseline** when presenting difficulty-stratified analyses
2. **Test sensitivity** to baseline choice, especially for complex algorithms like DVTS
3. **BoN is a better baseline** for robust analysis due to its insensitivity to stratification
4. **Consider multiple baselines** when evaluating tree-search algorithms
5. **Problem-level analysis** (tracking individual problems across stratifications) would reveal deeper insights

---

## Open Questions for Future Work

1. **Problem Migration Analysis:** Which specific problems move between difficulty levels when changing baselines? Are there "borderline" problems that are classification-unstable?

2. **Conditional Optimal Temperature:** For problems classified as "Level 3" by BOTH baselines, what is the truly optimal temperature? This would disentangle stratification effects from genuine difficulty.

3. **DVTS Tree Search Mechanics:** Why does DVTS prefer T=0.8 for Level 2 problems? Does high temperature improve PRM-guided exploration in the tree?

4. **Majority Vote Underperformance:** Why does majority voting consistently underperform weighted aggregation? Can we improve voting strategies?

5. **Temperature Scheduling:** Would **adaptive temperature** (starting high, annealing low) outperform fixed-temperature strategies?

6. **Generalization:** Do these findings hold for:
   - Larger models (7B, 32B)?
   - Different datasets (AIME, AMC)?
   - Different PRMs?

---

## Files Generated

### Visualizations
- `difficulty_distributions_2x2_comparison.png` - Problem redistribution across baselines
- `optimal_temperature_heatmap_comparison.png` - Temperature preferences by algorithm and baseline
- `base_capability_comparison_2x2.png` - Model capability verification

### Detailed Reports
- `stratification_comparison.md` - Problem count migrations
- `optimal_temperature_comparison.md` - Temperature consistency analysis
- `base_capability_verification.md` - Verification that base capability is baseline-independent

### Data Tables
- `stratification_comparison.csv` - Numerical data for problem distributions

---

## Conclusion

This analysis reveals a **fundamental algorithmic difference**:

- **BoN** has temperature preferences that are **intrinsic to problem difficulty** (independent of how you measure it)
- **DVTS** has temperature preferences that are **context-dependent** (sensitive to difficulty measurement method)

This has important implications for:
1. **Experimental design:** Always use consistent baselines when comparing algorithms
2. **Algorithm development:** DVTS-like tree search algorithms may need adaptive temperature strategies
3. **Benchmarking:** Report reference baselines and test sensitivity
4. **Practical deployment:** BoN is more robust; DVTS requires careful temperature tuning

**The answer to "what temperature is best?" depends not just on the problem, but on the algorithm and how you measure difficulty.**
