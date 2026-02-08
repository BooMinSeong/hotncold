# 포괄적 온도-난이도 분석

## 개요

이 디렉토리는 Qwen2.5-3B 모델이 MATH-500 문제에서 온도와 난이도가 어떻게 상호작용하는지에 대한 포괄적 분석을 포함합니다. Best-of-N (BoN) 및 Diverse Verifier Tree Search (DVTS) 알고리즘을 모두 사용합니다.

## 연구 질문

1. **목표 1 (온도 효과):** 다양한 온도에서 문제를 다르게 푸는가? 어떤 온도가 어떤 문제 유형에 가장 잘 작동하는가?

2. **목표 2 (기준선 영향):** 기준 온도(T=0.1 vs T=0.8) 선택이 난이도 분층화 및 최적 온도 발견에 어떻게 영향을 미치는가?

## 주요 파일

### 요약
📄 **`EXECUTIVE_SUMMARY.md`** - 여기서 시작하세요!
- 두 연구 질문 모두에 답변
- 실무자를 위한 실질적 권장사항
- 주요 발견 및 알고리즘 통찰
- ~10분 읽기

### 기술 심화 분석
📄 **`TECHNICAL_ANALYSIS.md`**
- 상세한 실험 분석
- 통계학적 고려사항
- 방법론적 논의
- 향후 연구 방향
- ~30분 읽기

### 비교 데이터
📊 **`stratification_comparison.md`**
- 기준선 간 문제 재분배
- 동일 기준선에서 BoN vs DVTS
- 델타 분석

📊 **`optimal_temperature_comparison.md`**
- 기준선 간 일관성 분석
- BoN: 80% 일관성 (4/5 수준)
- DVTS: 40% 일관성 (2/5 수준)

📊 **`base_capability_verification.md`**
- 기본 능력이 기준선과 무관함을 확인
- DVTS가 BoN보다 3-10% 우위를 보여줌

### 시각화
🖼️ **`difficulty_distributions_2x2_comparison.png`**
- 500개 문제가 어떻게 다르게 분층화되는지 보여줌
- 2×2 격자: BoN vs DVTS × ref0.1 vs ref0.8

🖼️ **`optimal_temperature_heatmap_comparison.png`**
- 알고리즘, 기준선, 난이도 수준별 최적 온도
- 견고성 차이를 시각화

🖼️ **`base_capability_comparison_2x2.png`**
- 각 온도에서의 모델 성능
- 기준선 간 일관성을 확인

### 데이터 테이블
📁 **`stratification_comparison.csv`**
- 기계 판독 가능한 문제 분포 데이터

## 실험 설계

### 2×2 완전 인수분해 설계

|           | T=0.1 기준선 | T=0.8 기준선 |
|-----------|-------------|-------------|
| **BoN**   | ✅ 225/27/23/33/192 | ✅ 198/42/28/50/182 |
| **DVTS**  | ✅ 241/32/21/21/185 | ✅ 219/54/36/29/162 |

*숫자는 난이도 Level 1/2/3/4/5의 문제 수를 나타냅니다*

### 원본 데이터

모든 분석은 4개의 사전 계산된 난이도 분석을 기반으로 합니다:

1. **BoN-ref0.1:** `exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty/`
2. **BoN-ref0.8:** `exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty-ref0.8/`
3. **DVTS-ref0.1:** `exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty/`
4. **DVTS-ref0.8:** `exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty-ref0.8/`

## 주요 발견

### 발견 1: BoN은 견고함, DVTS는 민감함

**BoN 온도 전략 (일관성):**
- Level 1 (쉬움): T=0.1
- Level 2: T=0.2
- Level 3: T=0.1-0.2 (둘 다 작동)
- Level 4 (어려움): T=0.4
- Level 5 (가장 어려움): T=0.2

**DVTS 온도 전략 (문맥 의존적):**
- Level 1 (쉬움): T=0.1 (일관성)
- Level 2: T=0.8 (일관성, 놀랍게도 높음!)
- Level 3-5: 기준선에 따라 크게 변함

**해석:** BoN은 내재적 온도 선호도를 가짐; DVTS 선호도는 난이도를 측정하는 방식에 따라 달라집니다.

### 발견 2: 대규모 문제 재분배

기준선을 T=0.1에서 T=0.8로 변경:
- Level 1: -12%~-9% (쉬운 문제 감소)
- Level 2: +56%~+69% (중간 문제 증가)
- Level 3: +22%~+71% (중간-어려운 문제 증가)
- Level 4: +38%~+52% (어려운 문제 증가)
- Level 5: -5%~-12% (가장 어려운 문제 감소)

**이유:** T=0.8이 더 가변적 → 낮은 기준선 정확도 → 문제가 더 어려워 보임.

### 발견 3: DVTS가 기본 능력에서 우수

- T=0.1: BoN보다 +4.5% 우수
- T=0.2: +6.2% 우수 (DVTS의 최고 온도)
- T=0.4: +6.4% 우수
- T=0.8: +10.1% 우수

트리 탐색이 단순 샘플링보다 더 높은 품질의 완성을 생성합니다.

### 발견 4: 가중치 결합 역설

**쉬운 문제에서:** 가중치 > 나이브
**어려운 문제에서:** 나이브 > 가중치

권장사항: 문제 난이도에 따라 적응형 결합을 사용하세요.

## 실무적 권장사항

### BoN 사용자를 위해
✅ 견고한 온도 전략:
- 쉬움 (>80% 정확도): T=0.1
- 중간 (40-80% 정확도): T=0.2
- 어려움 (20-40% 정확도): T=0.4
- 가장 어려움 (<20% 정확도): T=0.2

✅ 난이도 분류에 T=0.1 기준선 사용

### DVTS 사용자를 위해
⚠️ 문맥 의존적 전략:
- 쉬움: T=0.1
- 중간: T=0.8 (높은 다양성이 트리 탐색을 도움!)
- 어려움: T=0.4 및 T=0.8 모두 테스트 (문제 조합에 따라 다름)

⚠️ 기준선이 중요합니다—민감도 테스트

### 연구자를 위해
📌 항상 기준선을 보고하세요
📌 기준선 선택에 대한 견고성 테스트
📌 재현 가능한 비교를 위해 BoN이 더 좋음
📌 DVTS는 더 신중한 조정이 필요함

## 재현 방법

```bash
# 포괄적 분석 실행
uv run python exp/scripts/run_comprehensive_temperature_analysis.py

# 커스텀 디렉토리 사용
uv run python exp/scripts/run_comprehensive_temperature_analysis.py \
    --bon-ref01-dir exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty \
    --bon-ref08-dir exp/analysis_output-MATH500-Qwen2.5-3B-bon-difficulty-ref0.8 \
    --dvts-ref01-dir exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty \
    --dvts-ref08-dir exp/analysis_output-MATH500-Qwen2.5-3B-dvts-difficulty-ref0.8 \
    --output-dir exp/comprehensive_temperature_analysis-MATH500-Qwen2.5-3B
```

## 향후 작업

### 즉시 다음 단계
1. **문제 수준 추적:** 어떤 특정 문제가 수준 간 마이그레이션하는가?
2. **조건부 분석:** 두 기준선의 Level X에 모두 분류되는 문제의 경우 최적은?
3. **겹침 정량화:** 수준별 문제 집합의 벤 다이어그램

### 방법론적 개선
1. **연속 난이도:** 실제 정확도에 최적 온도를 회귀 (분류된 수준 아님)
2. **문제 특성:** 길이, 영역, 복잡성과 상관관계
3. **온도 스케줄링:** 적응형 전략 테스트

### 알고리즘 질문
1. DVTS가 Level 2에서 T=0.8을 선호하는 이유?
2. 문제 특성에서 최적 온도를 예측할 수 있는가?
3. 적응형 전략으로 결합을 개선할 수 있는가?

## 인용

이 분석을 사용하면, 다음과 같이 인용하세요:

```
테스트 시간 계산 스케일링을 위한 포괄적 온도-난이도 분석
MATH-500의 Qwen2.5-3B
분석 날짜: 2026-02-03
```

## 연락처

질문이나 협력은 주 저장소 README를 참조하세요.

---

**마지막 업데이트:** 2026-02-03
**분석 버전:** 1.0
**상태:** ✅ 완료
