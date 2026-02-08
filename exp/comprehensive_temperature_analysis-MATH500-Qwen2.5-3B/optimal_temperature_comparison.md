# Optimal Temperature Comparison

## Research Question: Does Reference Temperature Affect Optimal Temperature Selection?

### BoN Algorithm

| Level | Accuracy Range | ref0.1 Optimal T | ref0.1 Acc | ref0.8 Optimal T | ref0.8 Acc | Consistent? |
|-------|----------------|------------------|------------|------------------|------------|-------------|
| 1 | 0.80-1.00 | T0.1 | 0.997 | T0.1 | 0.998 | ✓ |
| 2 | 0.60-0.80 | T0.2 | 0.926 | T0.2 | 0.937 | ✓ |
| 3 | 0.40-0.60 | T0.1 | 0.870 | T0.2 | 0.893 | ✗ |
| 4 | 0.20-0.40 | T0.4 | 0.758 | T0.4 | 0.800 | ✓ |
| 5 | 0.00-0.20 | T0.2 | 0.536 | T0.2 | 0.527 | ✓ |

### DVTS Algorithm

| Level | Accuracy Range | ref0.1 Optimal T | ref0.1 Acc | ref0.8 Optimal T | ref0.8 Acc | Consistent? |
|-------|----------------|------------------|------------|------------------|------------|-------------|
| 1 | 0.80-1.00 | T0.1 | 0.993 | T0.1 | 0.995 | ✓ |
| 2 | 0.60-0.80 | T0.8 | 0.917 | T0.8 | 0.944 | ✓ |
| 3 | 0.40-0.60 | T0.1 | 0.841 | T0.2 | 0.880 | ✗ |
| 4 | 0.20-0.40 | T0.4 | 0.841 | T0.8 | 0.598 | ✗ |
| 5 | 0.00-0.20 | T0.8 | 0.537 | T0.2 | 0.500 | ✗ |

## Key Findings

- **BoN Consistency**: 4/5 levels have same optimal temperature across baselines
- **DVTS Consistency**: 2/5 levels have same optimal temperature across baselines

- BoN shows **high robustness** to reference temperature choice
- DVTS shows **sensitivity** to reference temperature choice
