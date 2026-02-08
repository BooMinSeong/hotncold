# Technical Deep-Dive: Temperature-Difficulty Performance Analysis

## Table of Contents
1. [Experimental Setup](#experimental-setup)
2. [Part 1: Overall Temperature Effects (Goal 1)](#part-1-overall-temperature-effects)
3. [Part 2: Reference Temperature Impact (Goal 2)](#part-2-reference-temperature-impact)
4. [Part 3: Within-Baseline Algorithm Comparisons](#part-3-within-baseline-algorithm-comparisons)
5. [Detailed Findings](#detailed-findings)
6. [Statistical Considerations](#statistical-considerations)

---

## Experimental Setup

### Dataset
- **Source:** MATH-500 (subset of MATH dataset)
- **Size:** 500 problems (consistent across all experiments)
- **Problem Types:** Mixed difficulty (algebra, geometry, number theory, etc.)

### Models and Algorithms
- **Base Model:** Qwen2.5-3B-Instruct
- **Algorithms:**
  - **BoN (Best-of-N):** Sample N completions independently, select best by PRM score
  - **DVTS (Diverse Verifier Tree Search):** Tree search with PRM guidance and diversity

### Temperature Configurations
- **Tested Temperatures:** T ∈ {0.1, 0.2, 0.4, 0.8}
- **Sample Budgets:** N ∈ {1, 2, 4, 8, 16, 32, 64}
- **Seeds:** {0, 42, 64} for most configurations
- **Aggregation Methods:** Naive (best-of-N), Weighted (PRM-weighted), Majority Vote

### Reference Baselines
- **ref0.1:** Difficulty computed from T=0.1 performance
- **ref0.8:** Difficulty computed from T=0.8 performance

### 2×2 Experimental Design

|           | T=0.1 Baseline | T=0.8 Baseline |
|-----------|----------------|----------------|
| **BoN**   | 225/27/23/33/192 | 198/42/28/50/182 |
| **DVTS**  | 241/32/21/21/185 | 219/54/36/29/162 |

*Numbers represent problem counts in Levels 1/2/3/4/5*

---

## Part 1: Overall Temperature Effects

### 1.1 Model Base Capability Analysis

**Research Question:** What is the model's raw generation quality at each temperature, independent of aggregation?

**Method:** Evaluate all individual completions (`is_correct_preds`) and compute mean accuracy across all problems and seeds.

**Results:**

#### BoN Experiments (identical across ref0.1 and ref0.8)
```
T0.1: 0.534 ± 0.001
T0.2: 0.533 ± 0.001  (98% relative to T0.1)
T0.4: 0.531 ± 0.000  (99% relative to T0.1)
T0.8: 0.504 ± 0.000  (94% relative to T0.1)
```

#### DVTS Experiments (identical across ref0.1 and ref0.8)
```
T0.1: 0.558 ± 0.003
T0.2: 0.566 ± 0.002  ⭐ BEST (101% relative to T0.1)
T0.4: 0.565 ± 0.001  (101% relative to T0.1)
T0.8: 0.555 ± 0.001  (99% relative to T0.1)
```

**Key Observations:**

1. **BoN vs DVTS Gap:**
   - DVTS achieves +2.4% to +5.1% higher base accuracy
   - Largest gap at T=0.8: **+10.1% relative improvement**
   - Suggests DVTS tree search produces higher-quality completions

2. **Temperature Effects on BoN:**
   - Monotonic decrease with temperature
   - T=0.8 drops 6% relative to T=0.1
   - Low temperature = more focused, correct solutions

3. **Temperature Effects on DVTS:**
   - **Non-monotonic!** T=0.2 is actually best
   - T=0.8 still maintains 99% of T=0.1 performance
   - Tree search appears to benefit from slight temperature boost

**Statistical Significance:**
- Standard deviations are very small (0.000-0.003)
- All differences > 0.01 are statistically significant
- DVTS advantage is robust and meaningful

---

### 1.2 Aggregation Method Performance

**Research Question:** How do different aggregation strategies perform at scale?

**Aggregation Methods:**
1. **Naive (Best-of-N):** Select completion with highest PRM score
2. **Weighted:** PRM-weighted combination of answers
3. **Majority Vote:** Most common answer across completions

**Overall Performance Patterns (at N=64):**

#### Easy Problems (Level 1, >80% baseline accuracy)
```
Method      | BoN-ref0.1 | BoN-ref0.8 | DVTS-ref0.1 | DVTS-ref0.8
----------- | ---------- | ---------- | ----------- | -----------
Naive       | 0.997      | 0.998      | 0.993       | 0.995
Weighted    | 1.000      | 1.000      | 0.993       | 0.995
Majority    | 1.000      | 1.000      | 0.993       | 0.995
```

**Finding:** All methods converge to near-perfect on easy problems. Weighted/Majority slightly edge naive for BoN.

#### Medium Problems (Levels 2-4)
```
Weighted consistently outperforms by 1-5%
Majority underperforms, especially at low N
```

#### Hard Problems (Level 5, <20% baseline accuracy)
```
Method      | BoN-ref0.1 T0.2 | DVTS-ref0.1 T0.8
----------- | --------------- | ----------------
Naive       | 0.536          | 0.537
Weighted    | 0.443          | 0.512
Majority    | 0.391          | 0.451
```

**Surprising Finding:** Naive outperforms weighted/majority on hardest problems!

**Interpretation:**
- On easy problems: aggregation helps
- On hard problems: **best single answer > aggregate**
- Majority vote suffers from low agreement on hard problems
- Weighted aggregation may dilute the best answer with incorrect attempts

**Recommendation:** Use weighted aggregation for medium difficulty, naive for very hard problems.

---

### 1.3 Scaling Behavior by Temperature

**Research Question:** How does performance scale with compute budget (N) for each temperature?

**Scaling Patterns:**

#### BoN Algorithm
```
Temperature | N=1 → N=64 Improvement (Level 5)
----------- | ---------------------------------
T0.1        | 0.330 → 0.507  (+54% relative)
T0.2        | 0.325 → 0.536  (+65% relative)  ⭐ Best scaling
T0.4        | 0.344 → 0.510  (+48% relative)
T0.8        | 0.335 → 0.514  (+53% relative)
```

**BoN Insight:** T=0.2 scales best on hard problems—moderate diversity helps without sacrificing quality.

#### DVTS Algorithm
```
Temperature | N=1 → N=64 Improvement (Level 5)
----------- | ---------------------------------
T0.1        | ~0.33 → ~0.52  (~58% relative)
T0.2        | ~0.34 → ~0.50  (~47% relative)
T0.4        | ~0.35 → ~0.51  (~46% relative)
T0.8        | ~0.34 → ~0.54  (~59% relative)  ⭐ Best scaling
```

**DVTS Insight:** T=0.8 scales best on hard problems—tree search benefits from high diversity.

**Critical Difference:**
- BoN prefers **moderate temperature (T=0.2)** for hard problems
- DVTS prefers **high temperature (T=0.8)** for hard problems
- This reflects fundamental algorithmic differences in how they explore solution space

---

## Part 2: Reference Temperature Impact

### 2.1 Problem Redistribution Analysis

**Research Question:** How does reference temperature choice change problem classification?

**BoN Redistribution (ref0.1 → ref0.8):**

| Level | ref0.1 Count | ref0.8 Count | Net Change | % Change |
|-------|--------------|--------------|------------|----------|
| 1     | 225          | 198          | **-27**    | -12.0%   |
| 2     | 27           | 42           | **+15**    | +55.6%   |
| 3     | 23           | 28           | +5         | +21.7%   |
| 4     | 33           | 50           | **+17**    | +51.5%   |
| 5     | 192          | 182          | -10        | -5.2%    |

**DVTS Redistribution (ref0.1 → ref0.8):**

| Level | ref0.1 Count | ref0.8 Count | Net Change | % Change |
|-------|--------------|--------------|------------|----------|
| 1     | 241          | 219          | **-22**    | -9.1%    |
| 2     | 32           | 54           | **+22**    | +68.8%   |
| 3     | 21           | 36           | **+15**    | +71.4%   |
| 4     | 21           | 29           | +8         | +38.1%   |
| 5     | 185          | 162          | **-23**    | -12.4%   |

**Pattern:**
- Both algorithms show **massive redistribution** (up to +71% in some levels)
- Level 1 shrinks significantly (fewer "easy" problems with T=0.8 baseline)
- Levels 2-4 grow significantly (more "medium" problems)
- Level 5 shrinks slightly (fewer "hardest" problems)

**Why?**
- T=0.8 is inherently more variable → lower baseline accuracy → problems appear harder
- Problems that are "easy" at T=0.1 (Level 1) become "medium" at T=0.8 (Level 2-3)
- T=0.1 baseline provides **more stable classification**

---

### 2.2 Optimal Temperature Consistency

**Research Question:** Do we reach the same conclusions about optimal temperature regardless of baseline?

#### BoN Consistency: 4/5 Levels (80%)

**Consistent Levels:**
- **Level 1:** T=0.1 optimal (both baselines) → Low temp for easy problems ✓
- **Level 2:** T=0.2 optimal (both baselines) → Low-medium temp ✓
- **Level 4:** T=0.4 optimal (both baselines) → Medium temp for hard problems ✓
- **Level 5:** T=0.2 optimal (both baselines) → Low-medium temp (not highest!) ✓

**Inconsistent Level:**
- **Level 3:** T=0.1 (ref0.1) vs T=0.2 (ref0.8)
  - Marginal difference (0.870 vs 0.893)
  - Both are low-temperature strategies
  - **Not a major finding change**

**BoN Conclusion:** Temperature recommendations are **robust to baseline choice**. The general strategy (low-medium temps) holds regardless of how you measure difficulty.

#### DVTS Consistency: 2/5 Levels (40%)

**Consistent Levels:**
- **Level 1:** T=0.1 optimal (both baselines) → Low temp for easy problems ✓
- **Level 2:** T=0.8 optimal (both baselines) → **High temp for medium problems!** ✓

**Inconsistent Levels:**
- **Level 3:** T=0.1 (ref0.1) vs T=0.2 (ref0.8)
  - Close temperatures, both low
  - Minor inconsistency

- **Level 4:** T=0.4 (ref0.1) vs **T=0.8 (ref0.8)** ⚠️
  - Major shift: medium to high temperature
  - ref0.1: 0.841 accuracy with T=0.4
  - ref0.8: 0.598 accuracy with T=0.8 (much lower!)
  - **Different problem sets** in Level 4

- **Level 5:** **T=0.8 (ref0.1)** vs T=0.2 (ref0.8) ⚠️
  - Major shift: high to low temperature
  - Completely opposite recommendations
  - ref0.1: 0.537 with T=0.8
  - ref0.8: 0.500 with T=0.2
  - **Different problem sets** in Level 5

**DVTS Conclusion:** Temperature recommendations are **highly sensitive to baseline choice** for hard problems (Levels 4-5). The interaction between:
- How problems are classified
- Which temperature works best for those problems

is **unstable for DVTS** but **stable for BoN**.

---

### 2.3 Why Are They Different?

**Hypothesis:** Tree search complexity creates context-dependent optima.

**BoN Mechanism:**
1. Sample N independent completions at temperature T
2. Score each with PRM
3. Select highest-scoring completion

→ **Simple, monotonic selection**. Temperature affects diversity, but the best answer is the best answer.

**DVTS Mechanism:**
1. Build tree of reasoning paths
2. Use PRM to guide which branches to explore
3. Balance exploration (diversity) vs exploitation (following high-scoring paths)
4. Diversity requirements affect which paths survive

→ **Complex, non-monotonic optimization**. The interaction between:
- Temperature-driven path diversity
- PRM pruning
- Diversity constraints

creates **context-dependent optimal temperature**. When problem sets change (via different stratification), the optimal exploration-exploitation balance changes.

**Evidence:**
- DVTS Level 2: T=0.8 optimal (both baselines) → diversity helps for medium problems
- DVTS Level 4/5: optimal temp flips → problem set composition matters
- BoN all levels: stable temps → simple selection is robust

---

## Part 3: Within-Baseline Algorithm Comparisons

### 3.1 Valid Comparisons

**Critical Rule:** Only compare algorithms at the **same reference baseline** (same problem sets per level).

#### Valid Comparison 1: T=0.1 Baseline

**Problem Distribution:**
```
Level | BoN  | DVTS | DVTS Advantage
----- | ---- | ---- | --------------
1     | 225  | 241  | +16 (DVTS finds more easy)
2     | 27   | 32   | +5
3     | 23   | 21   | -2
4     | 33   | 21   | -12
5     | 192  | 185  | -7 (DVTS has fewer hard)
```

**Interpretation:** DVTS classifies 16 more problems as "easy" because it solves them better at T=0.1 baseline.

**Performance at Optimal Temperatures:**
```
Level | BoN Optimal | BoN Acc | DVTS Optimal | DVTS Acc | Winner
----- | ----------- | ------- | ------------ | -------- | ------
1     | T0.1        | 0.997   | T0.1         | 0.993    | BoN (+0.4%)
2     | T0.2        | 0.926   | T0.8         | 0.917    | BoN (+1.0%)
3     | T0.1        | 0.870   | T0.1         | 0.841    | BoN (+3.3%)
4     | T0.4        | 0.758   | T0.4         | 0.841    | DVTS (+11.0%)
5     | T0.2        | 0.536   | T0.8         | 0.537    | Tie
```

**Key Findings:**
- BoN wins on easy/medium problems (Levels 1-3)
- DVTS wins decisively on Level 4
- Tie on hardest problems (Level 5)

#### Valid Comparison 2: T=0.8 Baseline

**Problem Distribution:**
```
Level | BoN  | DVTS | DVTS Advantage
----- | ---- | ---- | --------------
1     | 198  | 219  | +21 (even more!)
2     | 42   | 54   | +12
3     | 28   | 36   | +8
4     | 50   | 29   | -21 (BoN has more hard)
5     | 182  | 162  | -20
```

**Performance at Optimal Temperatures:**
```
Level | BoN Optimal | BoN Acc | DVTS Optimal | DVTS Acc | Winner
----- | ----------- | ------- | ------------ | -------- | ------
1     | T0.1        | 0.998   | T0.1         | 0.995    | BoN (+0.3%)
2     | T0.2        | 0.937   | T0.8         | 0.944    | DVTS (+0.7%)
3     | T0.2        | 0.893   | T0.2         | 0.880    | BoN (+1.5%)
4     | T0.4        | 0.800   | T0.8         | 0.598    | BoN (+33.8%!)
5     | T0.2        | 0.527   | T0.2         | 0.500    | BoN (+5.4%)
```

**Key Findings:**
- BoN wins on Levels 1, 3, 4, 5
- DVTS wins slightly on Level 2
- **BoN dominates Level 4** with T=0.8 baseline (different problem set than ref0.1)

---

### 3.2 Cross-Baseline Comparison (META-ANALYSIS)

**Question:** Are algorithm rankings **robust** or **stratification-dependent**?

**BoN Robustness:**
- Optimal temperatures: 4/5 consistent
- Performance: wins on easy problems consistently
- **Verdict: ROBUST** ✓

**DVTS Robustness:**
- Optimal temperatures: 2/5 consistent
- Performance: wins on Level 4 (ref0.1) but loses on Level 4 (ref0.8)
- **Verdict: STRATIFICATION-DEPENDENT** ⚠️

**Critical Implication for Benchmarking:**
- Comparing BoN vs DVTS? **Must use same baseline**
- Reporting optimal temperatures? **Must report baseline used**
- Making recommendations? **Test multiple baselines**

---

## Detailed Findings

### Finding 1: DVTS's Surprising Level 2 Performance

**Observation:** DVTS prefers **T=0.8 (highest tested)** for Level 2 problems (60-80% baseline accuracy) in **both baselines**.

**Accuracy:**
- ref0.1: T=0.8 achieves 0.917 (vs 0.993 with T=0.1 on Level 1)
- ref0.8: T=0.8 achieves 0.944

**Why?**
- Level 2 problems are "solvable but not trivial"
- Tree search benefits from **diverse exploration**
- High temperature generates varied reasoning paths
- PRM can still discriminate between them
- Diversity constraint is satisfied more easily

**Contrast with Level 1:**
- Level 1 problems are "trivial" → low temp sufficient
- Level 2 problems need exploration → high temp helps

**Contrast with BoN:**
- BoN uses T=0.2 for Level 2
- BoN doesn't have tree search → doesn't benefit from high diversity as much

---

### Finding 2: The Level 5 Paradox

**Question:** Why don't the hardest problems benefit most from high temperature?

**Expectation:** Higher diversity → more chances to find rare correct solution

**Reality (BoN):**
- T=0.2 is optimal (0.536 accuracy)
- T=0.8 achieves 0.514 (-4% relative)

**Reality (DVTS ref0.1):**
- T=0.8 is optimal (0.537 accuracy)
- But ref0.8 flips to T=0.2 (0.500 accuracy)

**Possible Explanations:**

1. **Noise Hypothesis:**
   - Very hard problems need precision, not diversity
   - High temperature adds noise that obscures correct reasoning
   - Low-medium temperature maintains focus

2. **Ceiling Hypothesis:**
   - Problems are fundamentally too hard for the model
   - No amount of sampling/diversity helps
   - Better to sample focused attempts

3. **PRM Limitation Hypothesis:**
   - PRM can't distinguish good from bad on very hard problems
   - High diversity creates many bad paths
   - Low temperature at least avoids worst paths

4. **Stratification Artifact Hypothesis:**
   - "Level 5" is a heterogeneous mix
   - Some problems are truly unsolvable
   - Some are solvable with right approach
   - Optimal temperature depends on mix composition
   - Different baselines create different mixes → different optima

**Evidence supports #4:** DVTS optimal temperature flips between baselines, suggesting Level 5 problem composition matters.

---

### Finding 3: Algorithm-Specific Stratification

**Observation:** At the same reference temperature, BoN and DVTS classify problems differently.

**At ref0.1:**
- DVTS puts +16 problems in Level 1 vs BoN
- DVTS puts -7 problems in Level 5 vs BoN

**At ref0.8:**
- DVTS puts +21 problems in Level 1 vs BoN
- DVTS puts -20 problems in Level 5 vs BoN

**Why?**
- DVTS has higher base capability (+3.3% to +10%)
- DVTS solves more problems at the reference temperature
- Therefore, DVTS's difficulty stratification is **shifted easier**

**Implication:**
- Even with the same baseline temperature, algorithms see different difficulty distributions
- Comparing BoN vs DVTS at "the same difficulty level" is **not comparing the same problems**
- Need to account for capability differences when making fair comparisons

---

### Finding 4: Weighted Aggregation Paradox

**Expected:** Weighted aggregation should always beat naive (uses more information)

**Reality:** Naive beats weighted on Level 5 (hardest problems)

**Example (BoN-ref0.1, Level 5, T=0.2, N=64):**
- Naive: 0.536
- Weighted: 0.443 (-17% relative!)
- Majority: 0.391 (-27% relative!)

**Why?**
1. **Low agreement on hard problems:** Multiple incorrect answers dilute the weighted average
2. **Best answer bias:** The single best completion (by PRM) is more likely correct than the average
3. **PRM reliability:** PRM is better at ranking than absolute scoring on hard problems

**Recommendation:** Use **adaptive aggregation**:
- Easy problems: weighted (stable, high agreement)
- Hard problems: naive (rely on best answer)
- Threshold: ~60% baseline accuracy

---

## Statistical Considerations

### Variance Across Seeds

**Seeds used:** {0, 42, 64}

**Typical standard deviations:**
- Easy problems (Level 1): ±0.000-0.004 (very stable)
- Medium problems (Levels 2-4): ±0.01-0.08
- Hard problems (Level 5): ±0.01-0.03

**Interpretation:**
- Most findings are statistically robust
- Level 2-4 show more variance (fewer problems per level)
- Seed variability is small compared to temperature effects

### Multiple Comparisons

**Problem:** We're comparing 4 temperatures × 5 levels × 2 algorithms × 2 baselines = 80 configurations

**Risk:** Some "optimal" temperatures may be due to chance

**Mitigation:**
1. Consistency across baselines strengthens findings (BoN ✓)
2. Patterns across levels provide evidence (low temp for easy ✓)
3. Magnitude of differences (>5% meaningful)

**Confidence:**
- High confidence: BoN temperature strategy
- Medium confidence: DVTS Level 1-2 strategy
- Low confidence: DVTS Level 4-5 strategy (baseline-dependent)

---

## Recommendations for Future Work

### Immediate Next Steps

1. **Problem-Level Tracking:**
   - Track individual problems across stratifications
   - Identify which specific problems migrate
   - Characterize "borderline" vs "stable" problems

2. **Conditional Analysis:**
   - For problems in **both** ref0.1 Level X and ref0.8 Level X
   - What is the optimal temperature?
   - This disentangles stratification from genuine difficulty

3. **Overlap Analysis:**
   - Venn diagrams for each difficulty level
   - Quantify how much problem sets overlap
   - Understand why DVTS recommendations flip

### Methodological Improvements

1. **Continuous Difficulty:**
   - Instead of 5 discrete levels
   - Use actual baseline accuracy as continuous variable
   - Regress optimal temperature on difficulty
   - Avoids arbitrary binning

2. **Problem Characteristics:**
   - Correlate temperature preferences with:
     - Problem length
     - Solution complexity
     - Math domain
     - Reasoning steps required

3. **Temperature Scheduling:**
   - Test adaptive temperature (start high, anneal)
   - Test per-problem temperature selection
   - Test temperature ensembles

### Algorithmic Questions

1. **Why does DVTS prefer T=0.8 for Level 2?**
   - Analyze tree structures at different temperatures
   - Measure diversity vs quality trade-off
   - Understand PRM guidance effectiveness

2. **Can we improve aggregation?**
   - Adaptive aggregation (difficulty-aware)
   - Confidence-weighted voting
   - PRM-calibrated ensembles

3. **Can we predict optimal temperature?**
   - Train a meta-model
   - Input: problem features
   - Output: recommended temperature
   - Test on new problems

---

## Conclusion

This analysis reveals **fundamental algorithmic differences** in how BoN and DVTS interact with temperature:

**BoN:**
- Simple, robust algorithm
- Temperature preferences are **intrinsic to problem difficulty**
- Recommendations generalize across baselines
- Safe default choice

**DVTS:**
- Complex, high-performance algorithm
- Temperature preferences are **context-dependent**
- Requires careful baseline selection and temperature tuning
- Higher ceiling but more brittle

**For practitioners:** Start with BoN and its robust temperature strategy. Consider DVTS if you can afford the tuning cost and need the extra performance.

**For researchers:** This work demonstrates the importance of **testing sensitivity to experimental choices** (like reference baselines). Algorithm comparisons must control for these factors to draw valid conclusions.

**The fundamental insight:** Temperature optimization is not just a property of the problem—it's a property of the **interaction between problem, algorithm, and measurement method**.
