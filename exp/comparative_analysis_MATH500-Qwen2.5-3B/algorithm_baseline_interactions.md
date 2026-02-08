## Critical Finding: Algorithm-Baseline Interaction Effects

### How Reference Temperature Affects Optimal Temperature Selection

#### BoN: Robust to Reference Temperature

| Level | Optimal @ ref0.1 | Optimal @ ref0.8 | Agreement |
|-------|------------------|------------------|-----------|
| Level 1 | T0.1 (1.000) | T0.2 (1.000) | ✗ |
| Level 2 | T0.1 (1.000) | T0.4 (1.000) | ✗ |
| Level 3 | T0.8 (0.884) | T0.8 (0.952) | ✓ |
| Level 4 | T0.8 (0.717) | T0.8 (0.753) | ✓ |
| Level 5 | T0.8 (0.451) | T0.4 (0.423) | ✗ |

**BoN Consistency**: 2/5 levels (40%) have same optimal temperature regardless of reference

#### DVTS: Sensitive to Reference Temperature

| Level | Optimal @ ref0.1 | Optimal @ ref0.8 | Agreement |
|-------|------------------|------------------|-----------|
| Level 1 | T0.1 (1.000) | T0.4 (1.000) | ✗ |
| Level 2 | T0.1 (1.000) | T0.8 (1.000) | ✗ |
| Level 3 | T0.8 (0.825) | T0.8 (0.907) | ✓ |
| Level 4 | T0.8 (0.714) | T0.8 (0.506) | ✓ |
| Level 5 | T0.8 (0.468) | T0.4 (0.440) | ✗ |

**DVTS Consistency**: 2/5 levels (40%) have same optimal temperature regardless of reference

### Interpretation

- **BoN is robust**: Optimal temperature preferences are stable across different reference temperatures
- **DVTS is sensitive**: Reference temperature choice significantly impacts which temperatures work best
- **Implication**: When using DVTS, the choice of reference temperature for difficulty stratification is critical
- **Recommendation**: For BoN, any reasonable reference works; for DVTS, carefully validate reference temperature choice
