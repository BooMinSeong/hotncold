# Difficulty Stratification Comparison

## Problem Distribution Across Reference Temperatures

This table shows how the same 500 MATH problems are classified into difficulty levels based on different reference temperature baselines.

| Level   | Accuracy Range   |   BoN-ref0.1 |   BoN-ref0.8 |   DVTS-ref0.1 |   DVTS-ref0.8 |
|:--------|:-----------------|-------------:|-------------:|--------------:|--------------:|
| 1       | 0.80-1.00        |          225 |          198 |           241 |           219 |
| 2       | 0.60-0.80        |           27 |           42 |            32 |            54 |
| 3       | 0.40-0.60        |           23 |           28 |            21 |            36 |
| 4       | 0.20-0.40        |           33 |           50 |            21 |            29 |
| 5       | 0.00-0.20        |          192 |          182 |           185 |           162 |
| Total   | 0.0-1.0          |          500 |          500 |           500 |           500 |

## Reference Temperature Impact Analysis

### BoN: ref0.1 vs ref0.8

| Level | ref0.1 | ref0.8 | Delta | Change |
|-------|--------|--------|-------|--------|
| 1 | 225 | 198 | -27 | -12.0% |
| 2 | 27 | 42 | +15 | +55.6% |
| 3 | 23 | 28 | +5 | +21.7% |
| 4 | 33 | 50 | +17 | +51.5% |
| 5 | 192 | 182 | -10 | -5.2% |

### DVTS: ref0.1 vs ref0.8

| Level | ref0.1 | ref0.8 | Delta | Change |
|-------|--------|--------|-------|--------|
| 1 | 241 | 219 | -22 | -9.1% |
| 2 | 32 | 54 | +22 | +68.8% |
| 3 | 21 | 36 | +15 | +71.4% |
| 4 | 21 | 29 | +8 | +38.1% |
| 5 | 185 | 162 | -23 | -12.4% |

### Algorithm Comparison at Same Baseline

#### T=0.1 Baseline: BoN vs DVTS

| Level | BoN | DVTS | Delta | Notes |
|-------|-----|------|-------|-------|
| 1 | 225 | 241 | +16 | DVTS classifies more as easy |
| 2 | 27 | 32 | +5 | |
| 3 | 23 | 21 | -2 | |
| 4 | 33 | 21 | -12 | |
| 5 | 192 | 185 | -7 | DVTS has fewer hardest problems |

#### T=0.8 Baseline: BoN vs DVTS

| Level | BoN | DVTS | Delta | Notes |
|-------|-----|------|-------|-------|
| 1 | 198 | 219 | +21 | DVTS classifies more as easy |
| 2 | 42 | 54 | +12 | |
| 3 | 28 | 36 | +8 | |
| 4 | 50 | 29 | -21 | |
| 5 | 182 | 162 | -20 | DVTS has fewer hardest problems |
