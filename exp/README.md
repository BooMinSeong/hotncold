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

```bash
# 기본 분석
python exp/scripts/analyze_results.py \
    --filter-model="Qwen2.5-1.5B-Instruct" \
    --filter-strategy="hnc"

# 모델 비교 분석
python exp/scripts/analyze_results.py \
    --filter-dataset="aime25" \
    --analysis-type="model_comparison"

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
