# Per-Problem Temperature Sensitivity Analysis

## Metadata

- **Analysis Date**: 2026-01-08 21:40:31
- **Baseline**: Default datasets (seeds [0, 42, 64], single temp=0.8)
- **HNC Analysis**: Multi-temp datasets (seeds [128, 192, 256], temps [0.4, 0.8, 1.2, 1.6])
- **Approaches**: bon, dvts, beam_search
- **Total Problems**: 500

## Summary Statistics

### BEAM_SEARCH

**Category Distribution:**

| Category | Count | Percentage |
|----------|-------|------------|
| unsolvable | 94 | 18.8% |
| single-temp | 8 | 1.6% |
| multi-temp | 398 | 79.6% |

**Temperature Preferences (Multi-Temp Problems):**

| Best Temperature | Count |
|------------------|-------|
| 0.4 | 48 |
| 0.8 | 125 |
| 1.2 | 92 |
| 1.6 | 133 |

**Temperature Coverage:**

| Temperature | Total Solves | Unique Solves |
|-------------|--------------|---------------|
| 0.4 | 381 | 5 |
| 0.8 | 389 | 0 |
| 1.2 | 389 | 0 |
| 1.6 | 388 | 3 |

**Difficulty-Temperature Correlation:** -0.0957

### BON

**Category Distribution:**

| Category | Count | Percentage |
|----------|-------|------------|
| unsolvable | 73 | 14.6% |
| single-temp | 35 | 7.0% |
| multi-temp | 392 | 78.4% |

**Temperature Preferences (Multi-Temp Problems):**

| Best Temperature | Count |
|------------------|-------|
| 0.4 | 289 |
| 0.8 | 99 |
| 1.2 | 4 |
| 1.6 | 0 |

**Temperature Coverage:**

| Temperature | Total Solves | Unique Solves |
|-------------|--------------|---------------|
| 0.4 | 392 | 10 |
| 0.8 | 407 | 20 |
| 1.2 | 305 | 4 |
| 1.6 | 45 | 1 |

**Difficulty-Temperature Correlation:** 0.3622

### DVTS

**Category Distribution:**

| Category | Count | Percentage |
|----------|-------|------------|
| unsolvable | 55 | 11.0% |
| single-temp | 17 | 3.4% |
| multi-temp | 428 | 85.6% |

**Temperature Preferences (Multi-Temp Problems):**

| Best Temperature | Count |
|------------------|-------|
| 0.4 | 289 |
| 0.8 | 108 |
| 1.2 | 31 |
| 1.6 | 0 |

**Temperature Coverage:**

| Temperature | Total Solves | Unique Solves |
|-------------|--------------|---------------|
| 0.4 | 431 | 5 |
| 0.8 | 423 | 5 |
| 1.2 | 427 | 6 |
| 1.6 | 392 | 1 |

**Difficulty-Temperature Correlation:** 0.2022

## Cross-Approach Comparison

- **Consistent Problems**: 64 problems have same best temperature across approaches
- **Inconsistent Problems**: 436 problems have different best temperatures

**Approach-Specific Advantages:**

- **BEAM_SEARCH**: 6 problems solved uniquely by this approach
- **BON**: 4 problems solved uniquely by this approach
- **DVTS**: 8 problems solved uniquely by this approach

## Example Problems

### BEAM_SEARCH

**single-temp:**

- Problem: On a particular map, $3$ inches on the map equates to $10$ miles in real life. If you know that the ...
  - Accuracies: 0.4=0.00, 0.8=0.00, 1.2=0.00, 1.6=0.02
  - Best: 1.6, Category: 1.6-only
- Problem: If $a$ and $b$ are positive integers such that $\gcd(a,b)=210$, $\mathop{\text{lcm}}[a,b]=210^3$, an...
  - Accuracies: 0.4=0.21, 0.8=0.00, 1.2=0.00, 1.6=0.00
  - Best: 0.4, Category: 0.4-only
- Problem: Let $\mathbf{a}$ and $\mathbf{b}$ be vectors such that the angle between $\mathbf{a}$ and $\mathbf{b...
  - Accuracies: 0.4=0.08, 0.8=0.00, 1.2=0.00, 1.6=0.00
  - Best: 0.4, Category: 0.4-only

**multi-temp-robust:**

- Problem: Let $N$ be the units digit of the number $21420N$. Which nonzero value of $N$ makes this number divi...
  - Accuracies: 0.4=0.85, 0.8=0.92, 1.2=0.85, 1.6=0.92
  - Best: 0.8, Category: 0.8-best
- Problem: Six witches and ten sorcerers are at an arcane mixer. The witches have decided to shake hands with e...
  - Accuracies: 0.4=0.75, 0.8=0.77, 1.2=0.88, 1.6=0.94
  - Best: 1.6, Category: 1.6-best
- Problem: Simplify $\frac{(10r^3)(4r^6)}{8r^4}$.
  - Accuracies: 0.4=1.00, 0.8=0.96, 1.2=1.00, 1.6=0.98
  - Best: 0.4, Category: 0.4-best

**unsolvable:**

- Problem: In the circle with center $Q$, radii $AQ$ and $BQ$ form a right angle. The two smaller regions are t...
  - Accuracies: 0.4=0.00, 0.8=0.00, 1.2=0.00, 1.6=0.00
  - Best: N/A, Category: N/A
- Problem: Let $z$ be a complex number such that $|z| = 1.$  Find the maximum value of
\[|1 + z| + |1 - z + z^2...
  - Accuracies: 0.4=0.00, 0.8=0.00, 1.2=0.00, 1.6=0.00
  - Best: N/A, Category: N/A

### BON

**single-temp:**

- Problem: Let $a,$ $b,$ $c,$ and $d$ be positive real numbers such that $a + b + c + d = 10.$  Find the maximu...
  - Accuracies: 0.4=0.00, 0.8=0.02, 1.2=0.00, 1.6=0.00
  - Best: 0.8, Category: 0.8-only
- Problem: When rolling a certain unfair six-sided die with faces numbered 1, 2, 3, 4, 5, and 6, the probabilit...
  - Accuracies: 0.4=0.02, 0.8=0.00, 1.2=0.00, 1.6=0.00
  - Best: 0.4, Category: 0.4-only
- Problem: Tom got a Mr. Potato Head for his birthday. It came with 3 hairstyles, 2 sets of eyebrows, 1 pair of...
  - Accuracies: 0.4=0.00, 0.8=0.02, 1.2=0.00, 1.6=0.00
  - Best: 0.8, Category: 0.8-only

**multi-temp-robust:**

- Problem: On a particular map, $3$ inches on the map equates to $10$ miles in real life. If you know that the ...
  - Accuracies: 0.4=0.04, 0.8=0.04, 1.2=0.02, 1.6=0.00
  - Best: 0.4, Category: 0.4-best
- Problem: In how many ways can  5 students be selected from a group of 6 students?
  - Accuracies: 0.4=0.42, 0.8=0.38, 1.2=0.35, 1.6=0.00
  - Best: 0.4, Category: 0.4-best
- Problem: Below is a magic square, meaning that the sum of the numbers in each row, in each column, and in eac...
  - Accuracies: 0.4=0.06, 0.8=0.10, 1.2=0.04, 1.6=0.00
  - Best: 0.8, Category: 0.8-best

**unsolvable:**

- Problem: A matrix $\mathbf{M}$ takes $\begin{pmatrix} 2 \\ -1 \end{pmatrix}$ to $\begin{pmatrix} 9 \\ 3 \end{...
  - Accuracies: 0.4=0.00, 0.8=0.00, 1.2=0.00, 1.6=0.00
  - Best: N/A, Category: N/A
- Problem: Fake gold bricks are made by covering concrete cubes with gold paint, so the cost of the paint is pr...
  - Accuracies: 0.4=0.00, 0.8=0.00, 1.2=0.00, 1.6=0.00
  - Best: N/A, Category: N/A

### DVTS

**single-temp:**

- Problem: Let $f(x)$ be an odd function, and let $g(x)$ be an even function.  Is $f(f(g(f(g(f(x))))))$ even, o...
  - Accuracies: 0.4=0.00, 0.8=0.00, 1.2=0.04, 1.6=0.00
  - Best: 1.2, Category: 1.2-only
- Problem: Point $A$ lies somewhere within or on the square which has opposite corners at $(0,0)$ and $(2,2)$. ...
  - Accuracies: 0.4=0.00, 0.8=0.00, 1.2=0.04, 1.6=0.00
  - Best: 1.2, Category: 1.2-only
- Problem: A right cylindrical tank with circular bases is being filled with water at a rate of $20\pi$ cubic m...
  - Accuracies: 0.4=0.00, 0.8=0.02, 1.2=0.00, 1.6=0.00
  - Best: 0.8, Category: 0.8-only

**multi-temp-robust:**

- Problem: For the eight counties listed below, what was the median number of students in $2005?$

\begin{tabul...
  - Accuracies: 0.4=0.06, 0.8=0.06, 1.2=0.02, 1.6=0.00
  - Best: 0.4, Category: 0.4-best
- Problem: At what value of $y$ is there a horizontal asymptote for the graph of the equation $y=\frac{4x^3+2x-...
  - Accuracies: 0.4=0.06, 0.8=0.10, 1.2=0.08, 1.6=0.04
  - Best: 0.8, Category: 0.8-best
- Problem: A strictly increasing sequence of positive integers $a_1$, $a_2$, $a_3$, $\dots$ has the property th...
  - Accuracies: 0.4=0.02, 0.8=0.02, 1.2=0.02, 1.6=0.00
  - Best: 0.4, Category: 0.4-best

**unsolvable:**

- Problem: The equation
\[x^{10}+(13x-1)^{10}=0\,\]has 10 complex roots $r_1,$ $\overline{r}_1,$ $r_2,$ $\overl...
  - Accuracies: 0.4=0.00, 0.8=0.00, 1.2=0.00, 1.6=0.00
  - Best: N/A, Category: N/A
- Problem: Let $\lambda$ be a constant, $0 \le \lambda \le 4,$ and let $f : [0,1] \to [0,1]$ be defined by
\[f(...
  - Accuracies: 0.4=0.00, 0.8=0.00, 1.2=0.00, 1.6=0.00
  - Best: N/A, Category: N/A

## Key Findings

### Temperature Allocation Efficiency

**BEAM_SEARCH:**
- Temperature 0.4: 5 unique solves / 381 total solves (1.3% unique)
- Temperature 0.8: 0 unique solves / 389 total solves (0.0% unique)
- Temperature 1.2: 0 unique solves / 389 total solves (0.0% unique)
- Temperature 1.6: 3 unique solves / 388 total solves (0.8% unique)

**BON:**
- Temperature 0.4: 10 unique solves / 392 total solves (2.6% unique)
- Temperature 0.8: 20 unique solves / 407 total solves (4.9% unique)
- Temperature 1.2: 4 unique solves / 305 total solves (1.3% unique)
- Temperature 1.6: 1 unique solves / 45 total solves (2.2% unique)

**DVTS:**
- Temperature 0.4: 5 unique solves / 431 total solves (1.2% unique)
- Temperature 0.8: 5 unique solves / 423 total solves (1.2% unique)
- Temperature 1.2: 6 unique solves / 427 total solves (1.4% unique)
- Temperature 1.6: 1 unique solves / 392 total solves (0.3% unique)

## Recommendations

Based on the analysis:

1. **Temperature Diversity**: Low approach-specific advantage suggests approaches may be redundant.
2. **Temperature Range**: BON shows significant extreme temperature value (11 unique problems). DVTS shows significant extreme temperature value (6 unique problems). BEAM_SEARCH shows significant extreme temperature value (8 unique problems). 
3. **Difficulty-Aware Selection**: Positive correlation (0.156) suggests harder problems benefit from higher temperatures.

