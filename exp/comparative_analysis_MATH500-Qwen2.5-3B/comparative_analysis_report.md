# Temperature-Difficulty Performance: Comparative Analysis

## Executive Summary: 2×2 Experimental Design

| Algorithm | Ref Temp=0.1 | Ref Temp=0.8 |
|-----------|--------------|--------------|
| **BoN** | BoN-ref0.1<br/>(225/27/23/33/192) | BoN-ref0.8<br/>(198/42/28/50/182) |
| **DVTS** | DVTS-ref0.1<br/>(241/32/21/21/185) | DVTS-ref0.8<br/>(219/54/36/29/162) |

Distribution format: Level1/Level2/Level3/Level4/Level5 problem counts

## Phase 1: Difficulty Distribution Analysis

### Problem Classification by Experiment

| Experiment | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 | Total |
|------------|---------|---------|---------|---------|---------|-------|
| BoN-ref0.1 | 225 | 27 | 23 | 33 | 192 | 500 |
| BoN-ref0.8 | 198 | 42 | 28 | 50 | 182 | 500 |
| DVTS-ref0.1 | 241 | 32 | 21 | 21 | 185 | 500 |
| DVTS-ref0.8 | 219 | 54 | 36 | 29 | 162 | 500 |

### Key Observations

- **BoN Algorithm**: T=0.1 baseline classifies 27 more problems as 'easy' (Level 1) than T=0.8 baseline
  - BoN-ref0.1: 225 Level 1 problems
  - BoN-ref0.8: 198 Level 1 problems
- **DVTS Algorithm**: T=0.1 baseline classifies 22 more problems as 'easy' (Level 1) than T=0.8 baseline
  - DVTS-ref0.1: 241 Level 1 problems
  - DVTS-ref0.8: 219 Level 1 problems

- **At T=0.8 baseline**: DVTS classifies 21 more problems as 'easy' than BoN
- **At T=0.1 baseline**: DVTS classifies 16 more problems as 'easy' than BoN

## Phase 2: Optimal Temperature Analysis

### BoN-ref0.1

| Level | Optimal Temp | Best Accuracy |
|-------|--------------|---------------|
| Level 1 | T0.1 | 0.997 |
| Level 2 | T0.2 | 0.926 |
| Level 3 | T0.1 | 0.870 |
| Level 4 | T0.4 | 0.758 |
| Level 5 | T0.2 | 0.536 |

### BoN-ref0.8

| Level | Optimal Temp | Best Accuracy |
|-------|--------------|---------------|
| Level 1 | T0.1 | 0.998 |
| Level 2 | T0.2 | 0.937 |
| Level 3 | T0.2 | 0.893 |
| Level 4 | T0.4 | 0.800 |
| Level 5 | T0.2 | 0.527 |

### DVTS-ref0.1

| Level | Optimal Temp | Best Accuracy |
|-------|--------------|---------------|
| Level 1 | T0.1 | 0.993 |
| Level 2 | T0.8 | 0.917 |
| Level 3 | T0.1 | 0.841 |
| Level 4 | T0.4 | 0.841 |
| Level 5 | T0.8 | 0.537 |

### DVTS-ref0.8

| Level | Optimal Temp | Best Accuracy |
|-------|--------------|---------------|
| Level 1 | T0.1 | 0.995 |
| Level 2 | T0.8 | 0.944 |
| Level 3 | T0.2 | 0.880 |
| Level 4 | T0.8 | 0.598 |
| Level 5 | T0.2 | 0.500 |

### Temperature Preference Patterns

**BoN-ref0.1**:
- T0.1: Levels 1, 3
- T0.2: Levels 2, 5
- T0.4: Levels 4

**BoN-ref0.8**:
- T0.1: Levels 1
- T0.2: Levels 2, 3, 5
- T0.4: Levels 4

**DVTS-ref0.1**:
- T0.1: Levels 1, 3
- T0.4: Levels 4
- T0.8: Levels 2, 5

**DVTS-ref0.8**:
- T0.1: Levels 1
- T0.2: Levels 3, 5
- T0.8: Levels 2, 4

## Phase 3: Base Model Capability

### BoN Algorithm

| Temperature | Accuracy (mean ± std) |
|-------------|----------------------|
| T0.1 | 0.534 ± 0.001 |
| T0.2 | 0.533 ± 0.001 |
| T0.4 | 0.531 ± 0.000 |
| T0.8 | 0.504 ± 0.000 |

### DVTS Algorithm

| Temperature | Accuracy (mean ± std) |
|-------------|----------------------|
| T0.1 | 0.558 ± 0.003 |
| T0.2 | 0.566 ± 0.002 |
| T0.4 | 0.565 ± 0.001 |
| T0.8 | 0.555 ± 0.001 |

**Key Finding**: DVTS achieves 3.2% higher base capability than BoN
- BoN best: 0.534
- DVTS best: 0.566


## Critical Finding: Algorithm-Baseline Interaction Effects

### How Reference Temperature Affects Optimal Temperature Selection

#### BoN: Robust to Reference Temperature

| Level | Optimal @ ref0.1 | Optimal @ ref0.8 | Agreement |
|-------|------------------|------------------|-----------|
| Level 1 | T0.1 (0.997) | T0.1 (0.998) | ✓ |
| Level 2 | T0.2 (0.926) | T0.2 (0.937) | ✓ |
| Level 3 | T0.1 (0.870) | T0.2 (0.893) | ✗ |
| Level 4 | T0.4 (0.758) | T0.4 (0.800) | ✓ |
| Level 5 | T0.2 (0.536) | T0.2 (0.527) | ✓ |

**BoN Consistency**: 4/5 levels (80%) have same optimal temperature regardless of reference

#### DVTS: Sensitive to Reference Temperature

| Level | Optimal @ ref0.1 | Optimal @ ref0.8 | Agreement |
|-------|------------------|------------------|-----------|
| Level 1 | T0.1 (0.993) | T0.1 (0.995) | ✓ |
| Level 2 | T0.8 (0.917) | T0.8 (0.944) | ✓ |
| Level 3 | T0.1 (0.841) | T0.2 (0.880) | ✗ |
| Level 4 | T0.4 (0.841) | T0.8 (0.598) | ✗ |
| Level 5 | T0.8 (0.537) | T0.2 (0.500) | ✗ |

**DVTS Consistency**: 2/5 levels (40%) have same optimal temperature regardless of reference

### Interpretation

- **BoN is robust**: Optimal temperature preferences are stable across different reference temperatures
- **DVTS is sensitive**: Reference temperature choice significantly impacts which temperatures work best
- **Implication**: When using DVTS, the choice of reference temperature for difficulty stratification is critical
- **Recommendation**: For BoN, any reasonable reference works; for DVTS, carefully validate reference temperature choice
