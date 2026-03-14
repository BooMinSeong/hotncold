<p align="center">
  <img style="width:200px" src="https://raw.githubusercontent.com/huggingface/search-and-learn/main/assets/logo.png">
</p>

<p align="center">
      🤗 <a href="https://huggingface.co/collections/HuggingFaceH4/scaling-test-time-compute-with-open-models-675c3b475a0d6eb4528fec23" target="_blank">Models & Datasets</a> |
      📃 <a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute" target="_blank">Blog Post</a>
</p>

# hotncold: Temperature-Aware Test-Time Scaling

Test-time compute scaling이 결국 **탐색(search)** 의 반복이라는 관점에서, LLM 디코딩의 핵심 변수인 **샘플링 온도**가 탐색 효율에 미치는 영향을 분석하는 연구 프레임워크입니다.

> *"The two methods that seem to scale arbitrarily are **search** and **learning**."*
> — Rich Sutton, [The Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf)

---

## 연구 배경 및 가설

Test-time compute scaling(TTS)은 모델이 하나의 문제에 대해 N번의 추론을 수행함으로써, 추가 학습 없이 성능을 향상시키는 기법입니다. 이 과정은 본질적으로 **해 공간(solution space)을 반복적으로 탐색**하는 것과 동일합니다.

이때 샘플링 온도 $T$는 탐색 범위를 직접적으로 결정합니다:

| 온도 | 탐색 특성 | 예상 강점 |
| :--- | :--- | :--- |
| **저온 (Cold, $T \to 0$)** | 좁고 집중된 탐색 — 고확률 경로에 수렴 | 단일 정답이 명확한 문제, 논리가 연쇄적인 문제 |
| **고온 (Hot, $T \to 1+$)** | 넓고 다양한 탐색 — 저확률 경로까지 탐색 | 여러 접근법이 가능한 문제, 창의적 추론이 필요한 문제 |

**핵심 가설**: 동일한 계산 예산(N) 하에서, 문제의 특성에 따라 최적 샘플링 온도가 다르며, 저온과 고온의 탐색 범위 차이가 문제별 TTS 효율 차이를 설명한다.

---

## 접근 방법

세 가지 탐색 알고리즘 모두 PRM(Process Reward Model) 기반으로 온도별 탐색 효율을 측정합니다:

| 알고리즘 | 핵심 아이디어 |
| :--- | :--- |
| **Best-of-N** | N개 완성본 샘플링 후 PRM 최고 점수 선택 |
| **Beam Search** | 각 추론 단계에서 PRM 점수로 빔 확장 |
| **DVTS** | 다양성과 검증을 균형있게 결합한 트리 탐색 |

온도가 각 알고리즘의 탐색 다양성과 수렴 속도에 미치는 영향을 비교함으로써, **"언제 hot이 cold보다 유리한가"** 를 규명합니다.

---

## 설치

```shell
conda create -n sal python=3.11 && conda activate sal
pip install -e '.[dev]'
huggingface-cli login
```

---

## 빠른 시작

```shell
export CONFIG=recipes/Qwen2.5-1.5B-Instruct/best_of_n.yaml
uv run python scripts/test_time_compute.py $CONFIG
```

기본 설정으로 MATH-500의 첫 10문제에 Best-of-N(`n=4`)을 실행하고 결과를 `data/`에 저장합니다.

커맨드라인으로 설정을 오버라이드:

```shell
uv run python scripts/test_time_compute.py $CONFIG \
    --model_path=meta-llama/Llama-3.2-8B-Instruct \
    --prm_path=Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B \
    --dataset_name=AI-MO/aimo-validation-aime \
    --dataset_split=train \
    --n=64 \
    --seed=42
```

> **참고:** 기본 config는 Llama 3 수학 추론 최적화 채팅 템플릿을 사용합니다. 다른 모델 계열은 `--custom_chat_template=none` 설정이 필요합니다.

---

## 온도 실험

### 단일 온도 실험

```shell
# 저온 (집중 탐색)
uv run python scripts/test_time_compute.py $CONFIG --temperature=0.4

# 고온 (다양성 탐색)
uv run python scripts/test_time_compute.py $CONFIG --temperature=1.2
```

### 멀티 온도 샘플링

탐색 예산 N을 여러 온도에 분배하여 탐색 다양성을 극대화합니다:

```shell
# Best-of-N: 온도별 비율로 n 분배
uv run python scripts/test_time_compute.py $CONFIG \
    --temperatures "0.4,0.8,1.2" \
    --temperature_ratios "0.33,0.34,0.33" \
    --n 12

# Beam Search / DVTS: 각 빔이 온도 목록을 순환
uv run python scripts/test_time_compute.py $CONFIG \
    --approach beam_search \
    --temperatures "0.4,0.8,1.2" \
    --beam_width 3 --n 12
```

설정 제약: `n`은 온도 수로 나누어 떨어져야 하며, beam_search/dvts의 경우 `beam_width`로도 나누어 떨어져야 합니다. 전체 가이드는 [`TEST_GUIDE.md`](TEST_GUIDE.md)를 참고하세요.

### Hot/Cold 비교 실험 (핵심 실험)

```shell
# 전체 온도 스펙트럼에 대한 대규모 실험 제출
./run_hnc.sh
```

`run_hnc.sh`는 저온(cold)과 고온(hot) 조건을 포함한 실험 배치를 Slurm 어레이 잡으로 제출합니다.

---

## 지원 모델 및 PRM

**생성 모델:**

| 모델 | 제공 레시피 |
| :--- | :--- |
| `Qwen/Qwen2.5-3B-Instruct` | best_of_n, beam_search, dvts |
| `Qwen/Qwen2.5-1.5B-Instruct` | best_of_n, beam_search, dvts |
| `meta-llama/Llama-3.2-3B-Instruct` | best_of_n, beam_search, dvts |
| `meta-llama/Llama-3.2-1B-Instruct` | best_of_n, beam_search, dvts |
| `nvidia/AceMath-7B-Instruct` | best_of_n, beam_search, dvts |

호환 채팅 템플릿을 가진 모든 모델은 `--model_path`로 사용 가능합니다.

**Process Reward Models:**

- `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` (기본값)
- `peiyi9979/math-shepherd-mistral-7b-prm`
- `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B`
- `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B`
- TRL로 직접 훈련한 PRM (see [`recipes/training/`](recipes/training/))

---

## 대규모 실험

500문제, `n=256`, 다중 시드/온도 조건의 전체 실험은 병렬화가 필요합니다.

### 병렬 생성

```shell
# 데이터셋을 분할하여 어레이 잡 제출
sbatch recipes/launch_array_default.slurm recipes/Qwen2.5-3B-Instruct/best_of_n.yaml \
    --n=256 --seed=0 \
    --hub_dataset_id=<YOUR_ORG>/Qwen2.5-3B-best_of_n-completions

# 전체 완료 후 결과 병합
python scripts/merge_chunks.py \
    --dataset_name=<YOUR_ORG>/Qwen2.5-3B-best_of_n-completions \
    --filter_strings seed-0
```

### 자동화 스크립트

```shell
./run_default.sh                               # 기본 실험 잡 전체 제출
./run_hnc.sh                                   # Hot/Cold 온도 실험 제출
./merge_default.sh                             # 완료된 병렬 결과 병합
python scripts/run_missing_auto.py --dry-run   # 누락 범위 탐지 및 제출
```

---

## PRM 훈련

TRL로 직접 PRM을 파인튜닝:

```shell
pip install -e '.[trl]'
# recipes/training/ 의 모델별 훈련 스크립트 참고
```

---

## 프로젝트 구조

```
├── src/sal/               # 핵심 라이브러리
│   ├── config.py          # 중앙 Config 데이터클래스
│   ├── models/            # PRM 로딩 및 추론
│   ├── search/            # best_of_n, beam_search, dvts 알고리즘
│   └── utils/             # 데이터 로딩, 스코어링, 온도 스케줄링
├── scripts/               # 실험 진입점 및 자동화
│   ├── test_time_compute.py   # 메인 실험 러너
│   ├── merge_chunks.py        # 병렬 잡 결과 병합
│   └── run_missing_auto.py    # 누락 잡 자동 탐지/제출
├── recipes/               # 모델/알고리즘별 YAML 설정 + Slurm 런처
├── prm-toolkit/           # PRM 서버 인프라 (git 서브모듈)
└── TEST_GUIDE.md          # 멀티 온도 테스트 가이드
```

---

## Citation

```bibtex
@misc{beeching2024scalingtesttimecompute,
      title={Scaling test-time compute with open models},
      author={Edward Beeching and Lewis Tunstall and Sasha Rush},
      url={https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute},
}
```

```bibtex
@misc{snell2024scalingllmtesttimecompute,
      title={Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters},
      author={Charlie Snell and Jaehoon Lee and Kelvin Xu and Aviral Kumar},
      year={2024},
      eprint={2408.03314},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.03314},
}
```
