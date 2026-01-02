# Hot and Cold Temperature Testing Guide

이 가이드는 hot and cold temperature 기능을 테스트하는 방법을 설명합니다.

## 빠른 시작

### 기본 테스트 실행

```bash
./test_hot_cold.sh
```

이 명령은 다음을 테스트합니다:
1. ✅ best_of_n (multi-temperature)
2. ✅ best_of_n (equal distribution)
3. ✅ beam_search (multi-temperature)
4. ✅ beam_search (baseline - single temperature)
5. ✅ DVTS (multi-temperature)
6. ✅ DVTS (baseline - single temperature)

### 커스터마이징

환경 변수를 통해 설정을 변경할 수 있습니다:

```bash
# GPU ID 변경
GPU_ID=0 ./test_hot_cold.sh

# 더 많은 샘플로 테스트
NUM_SAMPLES=10 ./test_hot_cold.sh

# 다른 모델 사용
MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct" ./test_hot_cold.sh

# 모든 설정 조합
GPU_ID=0 NUM_SAMPLES=5 MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct" ./test_hot_cold.sh
```

## 상세 설정

### 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GPU_ID` | 1 | 사용할 GPU ID |
| `MODEL_PATH` | meta-llama/Llama-3.2-1B-Instruct | 모델 경로 |
| `DATASET_NAME` | HuggingFaceH4/MATH-500 | 데이터셋 이름 |
| `NUM_SAMPLES` | 2 | 테스트할 샘플 개수 |

### Temperature 설정

스크립트 내부에서 다음 설정을 사용합니다:
- **Temperatures**: `0.6, 0.8, 1.0`
- **Temperature Ratios**: `0.33, 0.34, 0.33` (약 1:1:1 비율)

## 결과 확인

테스트가 완료되면 `./test_outputs_hot_cold/` 디렉토리에 결과가 저장됩니다:

```
test_outputs_hot_cold/
├── best_of_n/
│   └── results.json
├── best_of_n_equal/
│   └── results.json
├── beam_search/
│   └── results.json
├── beam_search_baseline/
│   └── results.json
├── dvts/
│   └── results.json
└── dvts_baseline/
    └── results.json
```

### 결과 분석

각 `results.json` 파일의 주요 필드:
- `completions`: 생성된 모든 완성본
- `pred`: 선택된 최종 답변
- `scores`: PRM 점수
- `completion_tokens`: 사용된 토큰 수

### 예제: 결과 비교

```bash
# Multi-temperature vs Baseline 비교
echo "=== best_of_n multi-temp ==="
jq '.pred[0]' test_outputs_hot_cold/best_of_n/results.json

echo "=== beam_search multi-temp ==="
jq '.pred[0]' test_outputs_hot_cold/beam_search/results.json

echo "=== beam_search baseline ==="
jq '.pred[0]' test_outputs_hot_cold/beam_search_baseline/results.json
```

## 개별 테스트 실행

특정 approach만 테스트하고 싶다면:

### best_of_n만 테스트

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/test_time_compute.py \
    --approach best_of_n \
    --model_path "meta-llama/Llama-3.2-1B-Instruct" \
    --num_samples 2 \
    --n 12 \
    --temperatures "0.6,0.8,1.0" \
    --temperature_ratios "0.33,0.34,0.33" \
    --output_dir "./test_outputs/best_of_n"
```

### beam_search만 테스트

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/test_time_compute.py \
    --approach beam_search \
    --model_path "meta-llama/Llama-3.2-1B-Instruct" \
    --num_samples 2 \
    --n 12 \
    --beam_width 3 \
    --num_iterations 3 \
    --temperatures "0.6,0.8,1.0" \
    --output_dir "./test_outputs/beam_search"
```

### DVTS만 테스트

```bash
CUDA_VISIBLE_DEVICES=1 uv run python scripts/test_time_compute.py \
    --approach dvts \
    --model_path "meta-llama/Llama-3.2-1B-Instruct" \
    --num_samples 2 \
    --n 12 \
    --beam_width 4 \
    --num_iterations 3 \
    --temperatures "0.6,0.8,1.0" \
    --output_dir "./test_outputs/dvts"
```

## Temperature 설정 이해하기

### best_of_n
- `n=12`, `temperatures=[0.6, 0.8, 1.0]`, `ratios=[0.33, 0.34, 0.33]`
- 생성: 4개(T=0.6) + 4개(T=0.8) + 4개(T=1.0) = 12개

### beam_search
- `n=12`, `beam_width=3`
- search_per_beam = 12 // 3 = 4
- 12개 beam을 4개씩 그룹핑 → 각 그룹이 다른 temperature 사용
- 예: Beam 0-3 사용 T=[0.6, 0.8, 1.0, 0.6] (cycling)

### DVTS
- `n=12`, `beam_width=4`
- n_beams = 12 // 4 = 3
- 각 beam이 4개 diverse path 생성
- 각 path가 다른 temperature 사용
- 예: 각 beam의 4개 path가 T=[0.6, 0.8, 1.0, 0.6] 사용

## 문제 해결

### GPU 메모리 부족
```bash
# 더 작은 모델 사용
MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct" ./test_hot_cold.sh

# 샘플 수 줄이기
NUM_SAMPLES=1 ./test_hot_cold.sh
```

### Config 검증 오류
- `n`이 `temperatures` 개수로 나누어떨어지는지 확인
- beam_search: `n`이 `beam_width`로 나누어떨어지는지 확인
- DVTS: `n`이 `beam_width`로 나누어떨어지는지 확인

## 다음 단계

테스트가 성공하면:
1. 더 큰 데이터셋으로 실험 (`NUM_SAMPLES` 증가)
2. 다양한 temperature 조합 시도
3. 결과 분석 및 성능 비교
