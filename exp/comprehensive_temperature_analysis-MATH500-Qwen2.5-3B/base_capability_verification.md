# Base Capability Verification

## Hypothesis: Base capability should be independent of reference temperature

Reference temperature only affects difficulty stratification, not the underlying model quality.

### BoN: ref0.1 vs ref0.8

| Temperature | ref0.1 Mean ± Std | ref0.8 Mean ± Std | Match? |
|-------------|-------------------|-------------------|--------|
| T0.1 | 0.534 ± 0.001 | 0.534 ± 0.001 | ✓ |
| T0.2 | 0.533 ± 0.001 | 0.533 ± 0.001 | ✓ |
| T0.4 | 0.531 ± 0.000 | 0.531 ± 0.000 | ✓ |
| T0.8 | 0.504 ± 0.000 | 0.504 ± 0.000 | ✓ |

### DVTS: ref0.1 vs ref0.8

| Temperature | ref0.1 Mean ± Std | ref0.8 Mean ± Std | Match? |
|-------------|-------------------|-------------------|--------|
| T0.1 | 0.558 ± 0.003 | 0.558 ± 0.003 | ✓ |
| T0.2 | 0.566 ± 0.002 | 0.566 ± 0.002 | ✓ |
| T0.4 | 0.565 ± 0.001 | 0.565 ± 0.001 | ✓ |
| T0.8 | 0.555 ± 0.001 | 0.555 ± 0.001 | ✓ |

### Algorithm Comparison

| Temperature | BoN Mean | DVTS Mean | Delta | DVTS Advantage |
|-------------|----------|-----------|-------|----------------|
| T0.1 | 0.534 | 0.558 | +0.024 | +4.5% |
| T0.2 | 0.533 | 0.566 | +0.033 | +6.2% |
| T0.4 | 0.531 | 0.565 | +0.034 | +6.4% |
| T0.8 | 0.504 | 0.555 | +0.051 | +10.1% |

## Verification Result

✓ **Base capabilities are identical within each algorithm across reference temperatures**

This confirms that reference temperature choice only affects difficulty stratification, not the underlying model performance.
