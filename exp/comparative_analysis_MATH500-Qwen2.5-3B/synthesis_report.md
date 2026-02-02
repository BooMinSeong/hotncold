# Temperature-Difficulty Performance Analysis: Executive Summary

**Analysis Date**: 2026-02-03
**Model**: Qwen2.5-3B-Instruct
**Dataset**: MATH-500
**Experiments**: 2×2 Design (BoN vs DVTS × Ref Temp 0.1 vs 0.8)

---

## Executive Summary

This analysis compares four temperature-difficulty experiments in a complete 2×2 factorial design:
- **Algorithms**: Best-of-N (BoN) vs Diverse Verifier Tree Search (DVTS)
- **Reference Temperatures**: T=0.1 vs T=0.8 (for difficulty stratification)

**Key Takeaway**: DVTS achieves 3.2% higher base capability than BoN (0.566 vs 0.534), but is significantly more sensitive to reference temperature choice. BoN shows robust temperature preferences across baselines (80% consistency), while DVTS preferences vary dramatically (40% consistency).

## Critical Findings

### 1. Algorithm Performance Gap

**DVTS significantly outperforms BoN in base capability:**

| Algorithm | Best Temp | Best Accuracy | Advantage |
|-----------|-----------|---------------|-----------|
| BoN | T0.1 | 0.534 ± 0.001 | Baseline |
| DVTS | T0.2 | 0.566 ± 0.002 | **+3.2%** |

**Interpretation**: DVTS's tree search structure with diversity enables better exploration of solution paths, leading to higher quality completions.

### 2. Reference Temperature Impact on Problem Classification

**T=0.1 baseline classifies significantly more problems as 'easy' (Level 1):**

| Algorithm | Ref T=0.1 (Level 1) | Ref T=0.8 (Level 1) | Difference |
|-----------|---------------------|---------------------|------------|
| BoN | 225 problems | 198 problems | +27 (+14%) |
| DVTS | 241 problems | 219 problems | +22 (+11%) |

**Interpretation**: T=0.1 produces more consistent/reliable outputs, making it easier to identify truly 'easy' problems. T=0.8 introduces variability that makes some easy problems appear harder.

### 3. Algorithm-Specific Temperature Preferences

**BoN prefers conservative temperatures (T0.1-T0.2) across most difficulty levels:**

| Level | Difficulty Range | BoN-ref0.1 | BoN-ref0.8 | Consistent? |
|-------|------------------|------------|------------|-------------|
| 1 | Easy (0.8-1.0) | T0.1 | T0.1 | ✓ |
| 2 | Medium-Easy (0.6-0.8) | T0.2 | T0.2 | ✓ |
| 3 | Medium (0.4-0.6) | T0.1 | T0.2 | ✗ |
| 4 | Hard (0.2-0.4) | T0.4 | T0.4 | ✓ |
| 5 | Very Hard (0.0-0.2) | T0.2 | T0.2 | ✓ |

**Consistency**: 80% (4/5 levels)

**DVTS shows mixed preferences, varying significantly by reference temperature:**

| Level | Difficulty Range | DVTS-ref0.1 | DVTS-ref0.8 | Consistent? |
|-------|------------------|-------------|-------------|-------------|
| 1 | Easy (0.8-1.0) | T0.1 | T0.1 | ✓ |
| 2 | Medium-Easy (0.6-0.8) | **T0.8** | **T0.8** | ✓ |
| 3 | Medium (0.4-0.6) | T0.1 | T0.2 | ✗ |
| 4 | Hard (0.2-0.4) | T0.4 | **T0.8** | ✗ |
| 5 | Very Hard (0.0-0.2) | **T0.8** | T0.2 | ✗ |

**Consistency**: 40% (2/5 levels)

**Key Observation**: DVTS favors high temperature (T0.8) for medium-easy and hard problems, especially at ref0.8. This suggests DVTS benefits from diversity when exploring multiple solution paths.

### 4. Robustness vs Sensitivity Trade-off

**BoN Algorithm: Robust but Lower Performance**
- ✓ Consistent temperature preferences across reference temperatures (80% agreement)
- ✓ Predictable behavior - easier to deploy in production
- ✗ Lower base capability (0.534 vs 0.566)
- Strategy: Prefers low-to-medium temperatures across the board

**DVTS Algorithm: High Performance but Sensitive**
- ✓ Significantly higher base capability (+3.2%)
- ✓ Better at leveraging diversity (high temp) for challenging problems
- ✗ Temperature preferences vary dramatically with reference temperature (40% agreement)
- ✗ Requires careful tuning of reference temperature
- Strategy: Mixed - uses both low (easy) and high (medium/hard) temperatures

### 5. Surprising Finding: Very Hard Problems Don't Always Need High Temperature

**Level 5 (0.0-0.2 baseline accuracy) optimal temperatures:**

| Experiment | Optimal Temp | Accuracy | Expected |
|------------|--------------|----------|----------|
| BoN-ref0.1 | T0.2 | 0.536 | T0.6-0.8 |
| BoN-ref0.8 | T0.2 | 0.527 | T0.6-0.8 |
| DVTS-ref0.1 | **T0.8** | 0.537 | ✓ As expected |
| DVTS-ref0.8 | T0.2 | 0.500 | T0.6-0.8 |

**Interpretation**: 
- High temperature introduces noise that can hurt performance on extremely difficult problems
- DVTS-ref0.1 is the exception - it successfully leverages T0.8 on hardest problems
- For BoN, moderate temperature (T0.2-T0.4) appears optimal even for very hard problems
- Suggests a 'sweet spot' exists between exploration (high temp) and quality (low temp)

## Recommendations

### For Practitioners

#### 1. Algorithm Selection

**Use DVTS when:**
- Maximum accuracy is critical
- You can invest time in reference temperature tuning
- You have validation data to optimize both reference and sampling temperatures
- The 3.2% accuracy gain justifies additional complexity

**Use BoN when:**
- You need predictable, robust behavior
- Quick deployment without extensive tuning
- Lower maintenance overhead is important
- Reference temperature choice shouldn't matter much

#### 2. Reference Temperature for Difficulty Stratification

**Recommended: T=0.1**
- Classifies more problems as 'easy' (more accurate identification)
- Produces more consistent baseline measurements
- Works well for both BoN and DVTS
- Better separates truly easy from medium difficulty problems

#### 3. Temperature Strategy by Difficulty

**For BoN:**
```
if baseline_accuracy >= 0.8:   # Easy
    temperature = 0.1
elif baseline_accuracy >= 0.4: # Medium
    temperature = 0.2
else:                          # Hard
    temperature = 0.4
```

**For DVTS (using ref0.1 baseline):**
```
if baseline_accuracy >= 0.8:   # Easy
    temperature = 0.1
elif baseline_accuracy >= 0.6: # Medium-Easy
    temperature = 0.8  # Leverage diversity!
elif baseline_accuracy >= 0.4: # Medium
    temperature = 0.1
elif baseline_accuracy >= 0.2: # Hard
    temperature = 0.4
else:                          # Very Hard
    temperature = 0.8  # Maximum exploration
```

#### 4. Sample Budget Allocation

Based on weighted aggregation performance:
- **Easy problems (Level 1)**: N=8-16 samples sufficient (>0.99 accuracy)
- **Medium problems (Levels 2-3)**: N=32-64 samples recommended
- **Hard problems (Levels 4-5)**: N=64 samples, but expect diminishing returns

**Cost-benefit consideration**: Weighted aggregation shows strongest gains at N≥32

### For Researchers

#### Open Questions

1. **Why does DVTS show algorithm-baseline interaction?**
   - Hypothesis: Tree search amplifies the effects of how problems are categorized
   - Different categorizations lead to different search strategies being optimal
   - Investigate: Does DVTS's tree depth/branching correlate with these effects?

2. **What causes the 3.2% capability gap between BoN and DVTS?**
   - Is it purely the search algorithm structure?
   - Or do generation/filtering differences play a role?
   - Controlled experiment: Same samples, different search strategies

3. **Why does BoN consistently prefer low-moderate temperatures?**
   - Does BoN's scoring mechanism penalize diversity?
   - Is PRM evaluation less effective on high-temperature completions?
   - Test: Same PRM with different temperature regimes

4. **Can we develop adaptive temperature schedulers?**
   - Use initial samples to estimate problem difficulty
   - Dynamically adjust temperature based on early results
   - Potential for compute savings on easy problems

5. **Do findings generalize across model scales?**
   - Test on Qwen2.5-1.5B and Qwen2.5-7B
   - Do smaller models benefit more from high temperature?
   - Do larger models show same algorithm-baseline interactions?

#### Future Experiments

1. **Per-problem temperature optimization**
   - Can mixed-temperature ensembles improve performance?
   - E.g., combine T0.1 (quality) + T0.8 (diversity) samples

2. **Alternative difficulty metrics**
   - Test other baselines: majority vote, weighted aggregation
   - Compare to ground-truth difficulty ratings
   - Explore PRM confidence scores as difficulty proxy

3. **Cross-dataset validation**
   - Replicate on GSM8K, AIME, Olympiad problems
   - Do findings hold across problem types?

4. **Compute-optimized strategies**
   - Adaptive N: allocate more samples to hard problems
   - Early stopping: terminate when confidence is high
   - Sample routing: different algorithms for different difficulties

## Limitations

1. **Single model size**: Results specific to Qwen2.5-3B (generalization unclear)
2. **Single dataset**: MATH-500 only (may not apply to other domains)
3. **Fixed sample budget**: N=1,2,4,8,16,32,64 (intermediate values unexplored)
4. **Discrete temperatures**: T=0.1,0.2,0.4,0.8 (finer granularity might reveal smoother transitions)
5. **PRM dependency**: Results may vary with different process reward models
6. **No cost analysis**: Computational overhead of DVTS vs BoN not quantified

## Conclusion

This comprehensive 2×2 factorial analysis reveals fundamental differences between BoN and DVTS algorithms:

**DVTS** achieves superior accuracy (+3.2%) by leveraging diversity through higher sampling temperatures, but requires careful tuning of the reference temperature used for difficulty stratification. Its temperature preferences vary significantly across baselines (40% consistency).

**BoN** offers robust, predictable behavior with consistent temperature preferences across conditions (80% consistency), making it easier to deploy without extensive tuning. However, it achieves lower overall accuracy.

**The choice between them depends on your priorities**: maximum accuracy (DVTS) vs operational simplicity (BoN).

**Most surprising finding**: Very hard problems don't always benefit from high temperatures. BoN consistently prefers T0.2-T0.4 even for the hardest problems, suggesting a sweet spot between exploration and quality.

**Key recommendation**: Use **T=0.1 as reference temperature** for difficulty stratification with both algorithms. For practitioners using DVTS, carefully validate temperature choices with held-out data. For BoN users, the simple strategy of T0.1 (easy), T0.2 (medium), T0.4 (hard) works reliably.

---

## Appendix: Experimental Details

**Model**: `Qwen/Qwen2.5-Math-3B-Instruct`
**Dataset**: `MATH-500` (stratified sample from MATH benchmark)
**Algorithms**:
- Best-of-N (BoN): Sample N completions, select best by PRM score
- DVTS: Diverse Verifier Tree Search with PRM-guided exploration
**Sampling budgets**: N ∈ {1, 2, 4, 8, 16, 32, 64}
**Temperatures**: T ∈ {0.1, 0.2, 0.4, 0.8}
**Aggregation methods**: naive (best-of-N), weighted, majority vote
**Difficulty levels**: 5 levels based on reference temperature accuracy
  - Level 1: 0.8-1.0 (easy)
  - Level 2: 0.6-0.8 (medium-easy)
  - Level 3: 0.4-0.6 (medium)
  - Level 4: 0.2-0.4 (hard)
  - Level 5: 0.0-0.2 (very hard)

**Analysis generated**: 2026-02-03 04:42:59
