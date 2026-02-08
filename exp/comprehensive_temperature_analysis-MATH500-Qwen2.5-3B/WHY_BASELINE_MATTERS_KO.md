# 왜 기준선마다 분석 결과가 달라지는가?

## 핵심 개념: 기준선은 "자"를 바꾸는 것입니다

동일한 500개 문제를 **다른 자(ruler)로 측정**하면 **다른 분류**를 얻게 됩니다.

### 비유: 신장 측정

상상해보세요:
- **자 A (T=0.1 기준선)**: 정확하고 일관된 자 → 대부분의 사람을 정확하게 측정
- **자 B (T=0.8 기준선)**: 약간 불안정한 자 → 같은 사람이 더 작게 측정될 수 있음

"키 180cm 이상" 그룹을 정의하면:
- 자 A로 측정: 225명
- 자 B로 측정: 198명 (-12%)

**같은 사람들인데, 측정 도구가 달라서 분류가 바뀝니다!**

## 구체적 예시: Level 3 문제들

### Scenario 1: T=0.1 기준선 사용

1. **난이도 측정**: T=0.1에서 각 문제의 정확도 계산
2. **분류**: 40-60% 정확도 → Level 3
3. **결과**: 23개 문제가 Level 3에 속함

이제 이 23개 문제에 대해 다양한 온도를 테스트:
```
T=0.1: 0.870 accuracy  ⭐ 최적
T=0.2: 0.870 accuracy  ⭐ 최적 (동점)
T=0.4: 0.870 accuracy  ⭐ 최적 (동점)
T=0.8: 0.812 accuracy
```

**결론 (ref0.1)**: Level 3에서는 T=0.1이 최적

### Scenario 2: T=0.8 기준선 사용

1. **난이도 측정**: T=0.8에서 각 문제의 정확도 계산
2. **분류**: 40-60% 정확도 → Level 3
3. **결과**: **28개 문제**가 Level 3에 속함 (+5개!)

이제 이 28개 문제에 대해 다양한 온도를 테스트:
```
T=0.1: 0.841 accuracy
T=0.2: 0.893 accuracy  ⭐ 최적
T=0.4: 0.870 accuracy
T=0.8: 0.880 accuracy
```

**결론 (ref0.8)**: Level 3에서는 T=0.2가 최적

### 왜 결론이 다른가?

**핵심: 23개 ≠ 28개. 다른 문제 세트입니다!**

#### 문제 재분배

T=0.1 기준선:
```
문제 A: T=0.1에서 55% 정확도 → Level 3 (40-60% 범위)
문제 B: T=0.1에서 85% 정확도 → Level 1 (80-100% 범위)
```

T=0.8 기준선:
```
문제 A: T=0.8에서 45% 정확도 → Level 3 (40-60% 범위) ✓ 여전히 Level 3
문제 B: T=0.8에서 65% 정확도 → Level 2 (60-80% 범위) ✗ Level 1에서 이동!
```

**문제 B가 Level 1에서 Level 2로 이동했습니다!**

그리고 새로운 문제들이 Level 3로 들어옵니다:
```
문제 C: T=0.1에서 35% 정확도 → Level 4 (20-40%)
문제 C: T=0.8에서 50% 정확도 → Level 3 (40-60%) ✓ Level 4에서 이동!
```

#### 결과적으로

- **23개 문제 세트 (ref0.1 Level 3)**: 특정 특성을 가진 문제들
- **28개 문제 세트 (ref0.8 Level 3)**: 다른 특성을 가진 문제들 (5개 새 문제, 일부 기존 문제 제외)

**다른 문제들이므로, 다른 최적 온도를 가지는 것이 당연합니다!**

## BoN vs DVTS: 왜 차이가 나는가?

### BoN: 견고함 (4/5 수준 일관성)

**BoN의 메커니즘:**
```python
# 의사코드
completions = sample_n_times(problem, temperature=T)
scores = [prm_score(c) for c in completions]
answer = completions[argmax(scores)]  # 가장 높은 점수 선택
```

**특징:**
- 단순한 선택 과정
- 각 완성은 독립적
- 최고 답변은 문제의 고유한 특성에 의존

**결과:**
- 문제가 재분류되어도, **유사한 난이도의 문제는 유사한 온도를 선호**
- "쉬운" 문제 (어떻게 측정하든): T=0.1 선호
- "중간" 문제 (어떻게 측정하든): T=0.2 선호
- "어려운" 문제 (어떻게 측정하든): T=0.4 선호

→ **온도 선호도가 문제의 본질적 난이도에 의존**

### DVTS: 민감함 (2/5 수준 일관성)

**DVTS의 메커니즘:**
```python
# 의사코드
tree = initialize_search_tree()
for step in range(max_steps):
    # 온도가 경로 탐색에 영향
    candidates = expand_nodes(tree, temperature=T)
    # PRM이 가지치기 안내
    prune_low_scoring_paths(candidates, prm)
    # 다양성 제약 적용
    maintain_diversity(tree)
answer = best_leaf_node(tree)
```

**특징:**
- 복잡한 트리 탐색
- 온도가 **탐색 전략**에 영향:
  - 낮은 T: 집중된 탐색 (활용)
  - 높은 T: 다양한 탐색 (탐험)
- PRM 가지치기와 다양성 제약 사이의 상호작용

**결과:**
- 최적 온도는 **문제 세트 구성**에 의존:
  - "모두 유사한 해법이 있는" 문제들 → 낮은 T (집중)
  - "다양한 접근이 필요한" 문제들 → 높은 T (탐험)
  - "혼합된" 문제들 → 중간 T

→ **온도 선호도가 문제 세트의 구성에 의존**

### 구체적 예시: Level 5 (최난이도)

#### ref0.1 기준선: 185개 문제 (DVTS)
```
문제 구성: T=0.1에서 0-20% 정확도를 가진 185개 문제
특성: T=0.1에서 매우 어려움
최적 온도: T=0.8 (0.537 정확도)
해석: 다양한 탐색이 도움됨
```

#### ref0.8 기준선: 162개 문제 (DVTS)
```
문제 구성: T=0.8에서 0-20% 정확도를 가진 162개 문제
특성: T=0.8에서 매우 어려움 (T=0.1에서는 더 쉬울 수 있음!)
최적 온도: T=0.2 (0.500 정확도)
해석: 집중된 접근이 더 나음
```

**왜 반대 결론인가?**

- **ref0.1 Level 5**: "T=0.1에서 실패하지만 다른 전략으로 풀 수 있는" 문제들
  - 높은 T로 다양한 경로 탐색 → 성공 가능성 증가

- **ref0.8 Level 5**: "T=0.8에서 실패하는" 문제들
  - 이미 매우 어려움
  - 더 높은 T는 노이즈만 추가
  - 낮은 T로 집중된 시도가 더 나음

**다른 문제 세트 → 다른 전략 → 다른 최적 온도**

## 이것이 문제인가? 아니면 통찰인가?

### 문제처럼 보이는 이유
- "최적 온도"가 기준선 선택에 의존
- 재현 불가능한 것처럼 보임
- "올바른" 답이 없는 것처럼 보임

### 실제로는 깊은 통찰
- **BoN**: 온도 선호도가 문제 본질에 의존 → 견고함
- **DVTS**: 온도 선호도가 문제 구성에 의존 → 맥락 의존성

이는 **알고리즘의 근본적 차이**를 드러냅니다:

```
BoN:  단순하지만 견고한 전략
      → 문제가 "본질적으로" 어렵다면, 특정 온도가 작동

DVTS: 복잡하지만 적응적인 전략
      → 문제 세트에 따라 탐험-활용 균형 조정 필요
```

## 실무적 시사점

### 1. 연구자의 경우

**항상 기준선을 보고하세요:**
```markdown
❌ 나쁜 예: "Level 3 문제에는 T=0.2가 최적"
✓ 좋은 예: "T=0.1 기준선으로 정의된 Level 3 문제에는 T=0.2가 최적"
```

**민감도 테스트:**
```python
# 여러 기준선으로 테스트
for baseline_temp in [0.1, 0.2, 0.8]:
    results = analyze_with_baseline(baseline_temp)
    if algorithm == "BoN":
        assert results_are_consistent()  # BoN은 일관되어야 함
    elif algorithm == "DVTS":
        # DVTS는 다를 수 있음 - 이것을 문서화하세요
        document_sensitivity(results)
```

### 2. 실무자의 경우

**BoN 사용 시:**
- ✓ 어떤 기준선이든 일관된 전략 사용 가능
- ✓ 단순하고 예측 가능
- ✓ 권장: T=0.1 기준선 (더 안정적)

**DVTS 사용 시:**
- ⚠️ 기준선 선택이 중요
- ⚠️ 데이터셋의 문제 분포에 맞게 조정
- ⚠️ 여러 온도 테스트하여 검증

### 3. 올바른 기준선 선택

**T=0.1 기준선을 권장하는 이유:**

1. **안정성**: 낮은 분산 → 일관된 분류
2. **직관성**: "쉬운" = T=0.1에서 높은 정확도
3. **재현성**: 더 많은 문제를 Level 1로 분류 (안정적)

**언제 T=0.8 기준선을 고려할까?**
- 매우 어려운 데이터셋 (대부분 문제가 T=0.1에서 실패)
- 높은 다양성이 중요한 경우
- 극한 조건 테스트

## 수학적 관점

### 문제 난이도의 정의

**절대 난이도 (존재하지 않음):**
```
difficulty(problem) = ??? (모델, 온도, 방법에 독립적인 값은 없음)
```

**조건부 난이도 (실제로 측정하는 것):**
```
difficulty(problem | model, temp, method) = 1 - accuracy(problem, model, temp, method)
```

**핵심: 난이도는 문제의 속성이 아니라 관계입니다!**

### 분류 함수

기준선이 분류 함수를 정의합니다:

```python
def classify_ref01(problem):
    acc = accuracy(problem, temp=0.1)
    if acc > 0.8: return "Level 1"
    elif acc > 0.6: return "Level 2"
    # ...

def classify_ref08(problem):
    acc = accuracy(problem, temp=0.8)  # 다른 측정값!
    if acc > 0.8: return "Level 1"
    elif acc > 0.6: return "Level 2"
    # ...
```

**같은 문제, 다른 정확도 값 → 다른 레벨**

### 최적 온도 함수

```python
optimal_temp(level, baseline, algorithm) =
    argmax_{T} accuracy(problems_in_level, temp=T, alg=algorithm)
```

**level의 정의가 baseline에 의존하므로, optimal_temp도 baseline에 의존합니다!**

## 핵심 교훈

### 1. "난이도"는 측정 방법에 의존합니다
- 절대적 난이도는 없음
- 모든 난이도 측정은 특정 컨텍스트에 상대적

### 2. BoN은 본질적 특성을 측정합니다
- 문제의 고유한 특성에 의존
- 측정 방법에 견고함
- 더 나은 벤치마킹 기준선

### 3. DVTS는 상호작용을 측정합니다
- 문제-알고리즘-온도 상호작용
- 맥락에 민감함
- 실무 배포 시 주의 깊은 조정 필요

### 4. 이것은 버그가 아니라 특성입니다
- 기준선 민감도 = 알고리즘 복잡도의 증거
- 단순한 알고리즘 = 견고한 권장사항
- 복잡한 알고리즘 = 맥락별 최적화

## 다음 단계: 더 나은 이해를 위해

### 문제 수준 추적
```python
# 각 문제가 어떻게 이동하는지 추적
for problem_id in all_problems:
    level_ref01 = classify_ref01(problem_id)
    level_ref08 = classify_ref08(problem_id)

    if level_ref01 != level_ref08:
        print(f"문제 {problem_id}: Level {level_ref01} → {level_ref08}")
        analyze_why_migrated(problem_id)
```

### 조건부 분석
```python
# "진정한" 최적 온도 찾기
level3_both_baselines = (
    set(problems_in_level3_ref01) &
    set(problems_in_level3_ref08)
)

# 이 안정적인 문제들에 대해서만 최적 온도 찾기
optimal_temp = find_best_temp(level3_both_baselines)
```

### 연속 난이도
```python
# 이산 레벨 대신 연속 난이도 사용
for problem in all_problems:
    continuous_difficulty = 1 - accuracy_at_baseline(problem)
    optimal_temp = predict_optimal_temp(continuous_difficulty)
```

---

## 결론

**기준선마다 분석 결과가 다른 이유:**

1. **기준선이 문제 분류 방법을 변경** (다른 측정 자)
2. **다른 문제 세트 → 다른 특성**
3. **BoN은 견고함** (본질적 난이도 측정)
4. **DVTS는 민감함** (맥락 의존적 최적화)

이것은 **버그가 아니라, 알고리즘 차이에 대한 깊은 통찰**입니다.

**실무 조언:**
- 간단함과 견고함을 원한다면: **BoN 사용**
- 최고 성능이 필요하고 조정 비용을 감당할 수 있다면: **DVTS 사용**
- 항상 **기준선을 명시적으로 보고**
- 여러 기준선에서 **민감도 테스트**
