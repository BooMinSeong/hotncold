# Changes: Majority Method Analysis

## 수정 사항

분석 방법을 **Majority Vote를 메인**으로 변경했습니다.

### 이전 문제점
- Naive, Weighted, Majority 방법을 섞어서 비교
- 한 실험에서는 naive의 최적 온도, 다른 실험에서는 weighted의 최적 온도 사용
- Apple-to-orange 비교로 결과가 부정확

### 현재 접근
- **Majority Vote (N=64) 기준**으로 모든 분석 통일
- 동일한 aggregation method 내에서 일관된 비교
- Naive/Weighted는 부록으로 처리

---

## 주요 발견사항 변화

### 1. 온도 선호도 패턴 (극적인 변화!)

#### 이전 (Mixed Methods):
```
BoN-ref0.1:  T0.1(L1,L3), T0.2(L2,L5), T0.4(L4)
DVTS-ref0.1: T0.1(L1,L3), T0.4(L4), T0.8(L2,L5)
```

#### 현재 (Majority Method):
```
BoN-ref0.1:  T0.1(L1,L2), T0.8(L3,L4,L5)  ← 어려운 문제 = 높은 온도!
DVTS-ref0.1: T0.1(L1,L2), T0.8(L3,L4,L5) ← 동일한 패턴!
```

**새로운 인사이트**: Majority vote에서는 **어려운 문제(Level 3-5)가 일관되게 T0.8을 선호**합니다!

### 2. 알고리즘 일관성

#### 이전 (Mixed Methods):
- **BoN**: 80% 일관성 (4/5 levels) - 매우 견고
- **DVTS**: 40% 일관성 (2/5 levels) - 민감

#### 현재 (Majority Method):
- **BoN**: 40% 일관성 (2/5 levels) ← 감소!
- **DVTS**: 40% 일관성 (2/5 levels) - 동일

**새로운 인사이트**: **둘 다 동일하게 민감**합니다! 이전의 "BoN이 견고하다"는 주장은 method를 섞은 결과였습니다.

### 3. "Surprising Finding" 소멸

#### 이전:
> "Very hard problems don't always prefer high temperature!"
> BoN-ref0.1 Level 5: T0.2 optimal (not T0.8!)

#### 현재:
> "Hard problems consistently prefer high temperature"
> BoN-ref0.1 Level 5: **T0.8** → 0.451
> DVTS-ref0.1 Level 5: **T0.8** → 0.468

**새로운 인사이트**: Majority vote에서는 **더 직관적인 패턴** - 어려울수록 높은 온도 필요!

---

## 온도 전략 업데이트

### 이전 권장사항 (Mixed, 틀림!):
```python
# BoN
if baseline_acc >= 0.8: temp = 0.1  # Easy
elif baseline_acc >= 0.4: temp = 0.2  # Medium
else: temp = 0.4  # Hard
```

### 현재 권장사항 (Majority, 올바름):
```python
# BoN (ref0.1 기준)
if baseline_acc >= 0.6: temp = 0.1  # Easy (L1-2)
else: temp = 0.8  # Medium-Hard (L3-5)

# DVTS (ref0.1 기준)
if baseline_acc >= 0.6: temp = 0.1  # Easy (L1-2)
else: temp = 0.8  # Medium-Hard (L3-5)
```

**핵심**: 어려운 문제는 **높은 온도(T0.8)가 필요**!

---

## Level별 최적 온도 (Majority @ N=64)

### BoN-ref0.1
| Level | Range | Optimal | Accuracy |
|-------|-------|---------|----------|
| 1 | 0.8-1.0 | **T0.1** | 1.000 |
| 2 | 0.6-0.8 | **T0.1** | 1.000 |
| 3 | 0.4-0.6 | **T0.8** | 0.884 |
| 4 | 0.2-0.4 | **T0.8** | 0.717 |
| 5 | 0.0-0.2 | **T0.8** | 0.451 |

### DVTS-ref0.1
| Level | Range | Optimal | Accuracy |
|-------|-------|---------|----------|
| 1 | 0.8-1.0 | **T0.1** | 1.000 |
| 2 | 0.6-0.8 | **T0.1** | 1.000 |
| 3 | 0.4-0.6 | **T0.8** | 0.825 |
| 4 | 0.2-0.4 | **T0.8** | 0.714 |
| 5 | 0.0-0.2 | **T0.8** | 0.468 |

**패턴**: 쉬운 문제 = 낮은 온도, 어려운 문제 = 높은 온도 (직관적!)

---

## 핵심 교훈

### 1. Aggregation Method가 결론에 큰 영향
- **Naive/Weighted**: 낮은-중간 온도 선호
- **Majority**: 어려운 문제에서 높은 온도 선호
- 방법을 섞으면 잘못된 결론!

### 2. 일관성 평가도 방법에 의존
- Mixed methods: BoN이 견고해 보임 (80%)
- Majority only: 둘 다 동일하게 민감 (40%)

### 3. Majority Vote가 더 직관적
- 어려운 문제 → 높은 온도 (탐색 필요)
- 쉬운 문제 → 낮은 온도 (정확성 중요)
- 이전의 "surprising finding"은 method mixing 아티팩트

---

## 기술적 변경사항

### 코드 수정
1. `comparative_analysis.py`:
   - `parse_report()`: Majority method의 N=64 데이터 파싱
   - `optimal_temps`: Majority 최적 온도 추출

2. `scaling_analysis.py`:
   - `BEST_TEMPS_MAJORITY`: Majority 기반 최적 온도 딕셔너리
   - 모든 plot 함수: 'majority' method 사용
   - 라벨/타이틀: "Majority Vote" 명시

3. 새 스크립트:
   - `exp/scripts/extract_majority_optimal_temps.py`: 자동 추출

### 재생성된 파일
- `comparative_analysis_report.md`
- `algorithm_baseline_interactions.md`
- `compute_efficiency_analysis.md`
- `scaling_curves_by_level.png`
- `algorithm_scaling_comparison.png`
- `optimal_temperatures_comparison.png`

---

**날짜**: 2026-02-03
**방법론**: Majority Vote (N=64) 기준, 일관된 비교
