# Experiment Analysis Module

실험 결과 분석을 위한 모듈화된 코드 구조입니다.

## 디렉토리 구조

```
exp/
├── configs/           # 설정 관리
│   ├── registry.yaml  # 실험 결과 레지스트리
│   └── schemas.py     # 설정 데이터클래스
├── analysis/          # 분석 모듈
│   ├── core.py        # 평가 함수 (evaluate_answer, evaluate_result)
│   ├── datasets.py    # 데이터셋 로딩
│   ├── metrics.py     # 메트릭 계산 (accuracy, pass@k)
│   ├── difficulty.py  # 난이도 계층화
│   └── visualization.py
├── scripts/           # 실행 스크립트
│   └── analyze_results.py
└── legacy/            # 기존 스크립트 (참조용)
```

## 빠른 시작

### 레지스트리 로드 및 필터링

```python
from exp.configs import load_registry

reg = load_registry('exp/configs/registry.yaml')

# 필터링
results = reg.filter(
    model='Qwen2.5-1.5B-Instruct',
    dataset='MATH-500',
    strategy='hnc'
)

for r in results:
    print(f"{r.name}: {r.hub_path}")
```

### 데이터셋 로드

```python
from exp.analysis import load_from_registry
from exp.configs import load_registry

reg = load_registry('exp/configs/registry.yaml')
result = reg.get_by_name('hnc-math500-1.5B-bon')

# 시드별 데이터셋 로드
datasets = load_from_registry(result)  # {seed: Dataset}
```

### 분석 실행

```python
from exp.analysis import analyze_single_dataset, evaluate_answer

# 단일 데이터셋 분석
metrics = analyze_single_dataset(dataset, "hnc-bon", seed=128)
# {'naive': {1: 0.45, 2: 0.52, ...}, 'weighted': {...}, 'maj': {...}}

# 답변 평가
is_correct = evaluate_answer(completion, gold_answer)
```

## CLI 사용법

### 분석 타입

| 타입 | 설명 | 대상 데이터셋 |
|------|------|--------------|
| `hnc_comparison` | HNC vs Default (T=0.8, T=0.4) 비교 | MATH-500 |
| `temperature_comparison` | T=0.4 vs T=0.8 비교 | AIME25 |
| `model_comparison` | 모델 크기 비교 (1.5B vs 3B) | AIME25 |
| `scaling` | 스케일링 곡선 생성 | 모두 |
| `default` | 기본 정확도 분석 | 모두 |

### MATH-500 HNC vs Default 비교

```bash
python exp/scripts/analyze_results.py \
    --filter-dataset="MATH-500" \
    --analysis-type="hnc_comparison" \
    --output-dir="exp/math500_analysis"
```

출력 파일:
- `{approach}-scaling_curves.png` - 전략별 스케일링 곡선
- `{approach}-{method}-hnc_vs_default.png` - HNC vs Default 비교
- `{approach}-pass_at_k_curves.png` - pass@k 곡선
- `{approach}-pass_at_k_comparison.png` - pass@k 바 차트
- `analysis_report.md` - 마크다운 리포트

### AIME25 온도 비교 (T=0.4 vs T=0.8)

```bash
python exp/scripts/analyze_results.py \
    --filter-dataset="aime25" \
    --filter-model="Qwen2.5-1.5B-Instruct" \
    --analysis-type="temperature_comparison" \
    --output-dir="exp/aime25_temp_analysis"
```

출력 파일:
- `aime25-{model}-{approach}-scaling_curves.png` - 온도별 스케일링 곡선
- `aime25-{model}-{approach}-{method}-temp_comparison.png` - 온도 비교
- `aime25-{model}-{approach}-pass_at_k_curves.png` - pass@k 곡선
- `analysis_report.md` - 마크다운 리포트

### AIME25 모델 크기 비교 (1.5B vs 3B)

```bash
python exp/scripts/analyze_results.py \
    --filter-dataset="aime25" \
    --analysis-type="model_comparison" \
    --output-dir="exp/aime25_model_analysis"
```

출력 파일:
- `aime25-{approach}-T{temp}-model_scaling.png` - 모델별 스케일링 곡선
- `aime25-{approach}-{method}-T{temp}-model_comparison.png` - 모델 비교
- `aime25-{approach}-T{temp}-model_pass_at_k.png` - 모델별 pass@k
- `analysis_report.md` - 마크다운 리포트

### 기본 옵션

```bash
# 플롯 없이 리포트만 생성
python exp/scripts/analyze_results.py \
    --filter-dataset="MATH-500" \
    --analysis-type="hnc_comparison" \
    --no-plots

# 상세 출력
python exp/scripts/analyze_results.py \
    --filter-dataset="aime25" \
    --analysis-type="model_comparison" \
    --verbose

# 도움말
python exp/scripts/analyze_results.py --help
```

## 레지스트리에 새 실험 추가

`exp/configs/registry.yaml`에 항목 추가:

```yaml
results:
  - name: "my-new-experiment"
    hub_path: "USER/dataset-name"
    model: "Qwen2.5-1.5B-Instruct"
    dataset: "MATH-500"
    approach: "bon"
    strategy: "hnc"
    seeds: [0, 42, 64]
    temperatures: [0.4, 0.8]
```

## 모듈 API

| 모듈 | 주요 함수 |
|------|-----------|
| `analysis.core` | `evaluate_answer()`, `evaluate_result()` |
| `analysis.datasets` | `load_datasets_by_seed()`, `load_from_registry()` |
| `analysis.metrics` | `analyze_single_dataset()`, `analyze_pass_at_k()` |
| `analysis.difficulty` | `compute_problem_baselines()`, `stratify_by_difficulty()` |
| `analysis.visualization` | `plot_comparison()`, `plot_bar_comparison()` |

## 그래프 타입

### 스케일링 곡선 (Scaling Curves)
- X축: Number of Samples (n), 로그 스케일
- Y축: Accuracy
- 선: 전략/온도/모델별로 구분
- 오차대역: ± 1 std (seeds 평균)

### HNC/온도/모델 비교
- X축: Number of Samples (n)
- Y축: Accuracy
- 실선+마커: 주요 비교 대상 (HNC, T=0.4, 더 큰 모델)
- 점선: 베이스라인 (Default, T=0.8, 더 작은 모델)

### Pass@k 곡선
- X축: k (number of samples), 로그 스케일
- Y축: Pass@k probability
- 상한선을 나타내는 이론적 성능 지표

### Pass@k 바 차트
- X축: 선택된 k 값 [1, 8, 32, 64]
- Y축: Pass@k probability
- 그룹: 전략/온도/모델별로 비교
