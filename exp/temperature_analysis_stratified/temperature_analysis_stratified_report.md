# Temperature Analysis with Difficulty Stratification

## Metadata

- **Analysis Date**: 2026-01-08 16:18:54
- **Baseline**: Default datasets (seeds [0, 42, 64], single temp=0.8)
- **HNC Analysis**: Multi-temp datasets (seeds [128, 192, 256], temps [0.4, 0.8, 1.2, 1.6])
- **Approaches**: bon, beam_search, dvts

## Baseline Establishment

- **Total Problems**: 500
- **Mean Accuracy**: 0.3984 ± 0.3009
- **Min Accuracy**: 0.0000
- **Max Accuracy**: 0.9635

## Difficulty Stratification

| Level | Description | Accuracy Range | Problem Count |
|-------|-------------|----------------|---------------|
| 5 | Hardest | [0.000, 0.047] | 98 |
| 4 | Hard | [0.047, 0.285] | 101 |
| 3 | Medium | [0.285, 0.508] | 101 |
| 2 | Easy | [0.508, 0.726] | 100 |
| 1 | Easiest | [0.726, 0.964] | 100 |

## BEAM_SEARCH - Temperature Analysis

### Overall Performance by Temperature

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.4017 ± 0.0116 |
| 0.8 | 0.4740 ± 0.0051 |
| 1.2 | 0.4999 ± 0.0037 |
| 1.6 | 0.5102 ± 0.0085 |

### Performance by Difficulty Level

#### Level 5 (Hardest)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.0121 ± 0.0060 |
| 0.8 | 0.0202 ± 0.0083 |
| 1.2 | 0.0151 ± 0.0071 |
| 1.6 | 0.0123 ± 0.0053 |

#### Level 4 (Hard)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.1258 ± 0.0110 |
| 0.8 | 0.1654 ± 0.0213 |
| 1.2 | 0.1881 ± 0.0089 |
| 1.6 | 0.2024 ± 0.0096 |

#### Level 3 (Medium)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.4425 ± 0.0226 |
| 0.8 | 0.5512 ± 0.0158 |
| 1.2 | 0.5895 ± 0.0139 |
| 1.6 | 0.6196 ± 0.0128 |

#### Level 2 (Easy)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.6402 ± 0.0180 |
| 0.8 | 0.7625 ± 0.0177 |
| 1.2 | 0.8181 ± 0.0232 |
| 1.6 | 0.8277 ± 0.0064 |

#### Level 1 (Easiest)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.7827 ± 0.0123 |
| 0.8 | 0.8640 ± 0.0066 |
| 1.2 | 0.8810 ± 0.0063 |
| 1.6 | 0.8808 ± 0.0207 |

## BON - Temperature Analysis

### Overall Performance by Temperature

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.3568 ± 0.0028 |
| 0.8 | 0.2954 ± 0.0018 |
| 1.2 | 0.0924 ± 0.0039 |
| 1.6 | 0.0041 ± 0.0005 |

### Performance by Difficulty Level

#### Level 5 (Hardest)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.0085 ± 0.0006 |
| 0.8 | 0.0072 ± 0.0013 |
| 1.2 | 0.0034 ± 0.0008 |
| 1.6 | 0.0002 ± 0.0003 |

#### Level 4 (Hard)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.0738 ± 0.0072 |
| 0.8 | 0.0732 ± 0.0032 |
| 1.2 | 0.0198 ± 0.0005 |
| 1.6 | 0.0006 ± 0.0000 |

#### Level 3 (Medium)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.2898 ± 0.0144 |
| 0.8 | 0.2523 ± 0.0029 |
| 1.2 | 0.0703 ± 0.0037 |
| 1.6 | 0.0029 ± 0.0015 |

#### Level 2 (Easy)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.5648 ± 0.0062 |
| 0.8 | 0.4550 ± 0.0027 |
| 1.2 | 0.1321 ± 0.0085 |
| 1.6 | 0.0073 ± 0.0039 |

#### Level 1 (Easiest)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.8438 ± 0.0087 |
| 0.8 | 0.6862 ± 0.0031 |
| 1.2 | 0.2356 ± 0.0124 |
| 1.6 | 0.0094 ± 0.0005 |

## DVTS - Temperature Analysis

### Overall Performance by Temperature

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.3793 ± 0.0070 |
| 0.8 | 0.3669 ± 0.0035 |
| 1.2 | 0.3249 ± 0.0052 |
| 1.6 | 0.1600 ± 0.0008 |

### Performance by Difficulty Level

#### Level 5 (Hardest)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.0157 ± 0.0006 |
| 0.8 | 0.0121 ± 0.0016 |
| 1.2 | 0.0130 ± 0.0003 |
| 1.6 | 0.0055 ± 0.0003 |

#### Level 4 (Hard)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.1101 ± 0.0108 |
| 0.8 | 0.1085 ± 0.0098 |
| 1.2 | 0.0963 ± 0.0120 |
| 1.6 | 0.0456 ± 0.0070 |

#### Level 3 (Medium)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.3370 ± 0.0056 |
| 0.8 | 0.3302 ± 0.0037 |
| 1.2 | 0.2853 ± 0.0079 |
| 1.6 | 0.1322 ± 0.0054 |

#### Level 2 (Easy)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.6210 ± 0.0145 |
| 0.8 | 0.5913 ± 0.0132 |
| 1.2 | 0.5171 ± 0.0046 |
| 1.6 | 0.2321 ± 0.0074 |

#### Level 1 (Easiest)

| Temperature | Accuracy (mean ± std) |
|-------------|-----------------------|
| 0.4 | 0.8081 ± 0.0117 |
| 0.8 | 0.7883 ± 0.0069 |
| 1.2 | 0.7092 ± 0.0074 |
| 1.6 | 0.3827 ± 0.0016 |

