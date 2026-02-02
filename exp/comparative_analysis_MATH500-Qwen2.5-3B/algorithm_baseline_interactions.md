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
