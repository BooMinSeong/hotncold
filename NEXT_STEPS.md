# Hot and Cold Temperature - 다음 확장 계획

## 현재 구현 상태 (Phase 1 - 완료)

✅ **best_of_n 알고리즘 Multi-Temperature 지원**
- Helper 함수: `src/sal/utils/temperature.py`
- Config 확장: `temperatures`, `temperature_ratios` 필드
- SamplingParams 리스트를 사용한 구현
- 균등 분배 및 사용자 지정 비율 지원

## Phase 2: beam_search 확장

### 핵심 설계 원칙

**기존 방식의 문제점**:
- 각 beam에 하나의 temperature를 할당하면, beam이 pruning될 때 해당 temperature가 사라져 search space가 제한됨

**새로운 접근**:
- 각 beam이 매 iteration마다 temperature_list 비율대로 여러 next_text를 생성
- **고정값 사용**: `continuations_per_beam = config.n // config.beam_width`
- 예: n=16, beam_width=4 → 각 beam당 항상 4개의 next_text
  - temperature_list=[0.6, 0.8, 1.0], temperature_ratios=[1,1,1]
  - 각 beam의 next_texts: 1개(T=0.6) + 2개(T=0.8) + 1개(T=1.0) = 4개
- 각 beam의 next_texts 중 PRM 점수가 가장 높은 것을 선택

**장점**:
- Beam이 pruning되어도 살아있는 beam들이 모든 temperature 유지
- active_beams가 줄면 총 computation도 자연스럽게 감소
- best_of_n과 동일한 temperature 분배 메커니즘 재사용

### 필요한 변경사항

1. **generate_k_steps 제거 및 직접 generation** (`beam_search.py`):

   **이유**:
   - lookahead 불필요 (temperature 다양성으로 대체)
   - 각 beam마다 다른 temperature 리스트 적용하기 위함
   - 코드 단순화

   ```python
   # 각 beam당 next_text 개수 (고정값)
   continuations_per_beam = config.n // config.beam_width

   # temperature_list 비율대로 분배
   from sal.utils.temperature import get_temperature_assignment
   temp_config = copy.copy(config)
   temp_config.n = continuations_per_beam
   temps = get_temperature_assignment(temp_config)

   # 각 beam마다 temperature별로 여러 프롬프트 준비
   prompts = []
   sampling_params_list = []

   for beam in active_beams:
       conv = build_conv(beam.prompt, beam.current_text, config.system_prompt)
       templated_conv = tokenizer.apply_chat_template(
           conv,
           add_generation_prompt=(i == 0),
           continue_final_message=(i > 0),
           tokenize=False
       )

       for t in temps:
           prompts.append(templated_conv)
           is_last_iteration = (i == config.num_iterations - 1)
           sampling_params_list.append(
               SamplingParams(
                   temperature=t,
                   max_tokens=config.max_tokens,
                   top_p=config.top_p,
                   stop=["\n\n"] if not is_last_iteration else None,
                   include_stop_str_in_output=True,
                   n=1
               )
           )

   # vLLM 직접 호출
   outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)
   ```

2. **Beam별 next_texts 할당**:

   ```python
   # outputs를 beam별로 그룹핑하여 next_texts에 할당
   # 총 len(active_beams) * continuations_per_beam 개의 output

   for beam_idx, beam in enumerate(active_beams):
       start_idx = beam_idx * continuations_per_beam
       end_idx = start_idx + continuations_per_beam
       beam_outputs = outputs[start_idx:end_idx]

       # Beam 클래스의 기존 필드에 저장
       beam.next_texts = [out.outputs[0].text for out in beam_outputs]
       beam.stop_reasons = [out.outputs[0].finish_reason for out in beam_outputs]
       beam.completion_tokens += sum(len(out.outputs[0].token_ids) for out in beam_outputs)
   ```

3. **PRM Scoring** (기존 방식 유지):

   ```python
   # 기존과 동일하게 PRM scoring
   prompts, completions = [], []

   for beam in active_beams:
       for next_text in beam.next_texts:
           prompts.append(beam.prompt)
           completions.append([beam.current_text + next_text])

   scores = prm.score(prompts, completions)
   ```

4. **각 Beam의 최고점 next_text 선택 및 적용**:

   ```python
   # 각 beam마다 최고점만 선택하여 current_text에 반영
   score_idx = 0
   for beam in active_beams:
       beam_scores = []
       for _ in beam.next_texts:
           agg_score = aggregate_scores(scores[score_idx][0], config.agg_strategy)
           beam_scores.append(agg_score)
           score_idx += 1

       # 최고점 선택
       best_idx = np.argmax(beam_scores)

       # Beam에 최고점 적용
       beam.current_text += beam.next_texts[best_idx]
       beam.history.append(beam.next_texts[best_idx])
       beam.all_scores = scores[beam_idx * continuations_per_beam + best_idx][0]

       # 완료 체크
       if beam.stop_reasons[best_idx] in ["stop", "length"] or beam.next_texts[best_idx] == "":
           beam.completed = True
           completed_beams.append(beam)
   ```

5. **Beam Pruning** (기존 방식 유지):

   ```python
   # 완료된 beam 제거
   active_beams = [b for b in active_beams if not b.completed]

   # 중복 제거 (config.filter_duplicates)
   # ...기존 로직 유지...

   # 각 beam의 최종 스코어로 top k 선택
   agg_scores = [aggregate_scores(b.all_scores, config.agg_strategy) for b in active_beams]
   top_indices = np.argsort(agg_scores)[-(config.n // config.beam_width):]

   for idx, beam in enumerate(active_beams):
       if idx not in top_indices:
           beam.pruned = True
   ```

### 변경되지 않는 부분

- **Beam 클래스**: 기존 구조 그대로 사용 (next_texts는 이미 list[str])
- **PRM scoring 알고리즘**: 동일
- **Pruning 로직**: 동일
- **완료 처리**: 동일

### 핵심 개념

- **Beam은 temperature를 소유하지 않음**
- 매 iteration마다 모든 beam이 동일한 temperature 분포 사용
- 각 beam은 여러 next_texts 중 PRM 최고점만 선택
- **고정된 continuations_per_beam으로 computation budget 제어**
- **lookahead 제거** (temperature 다양성으로 대체)
- **generate_k_steps 제거** (직접 llm.generate 사용)
- best_of_n의 `get_temperature_assignment()` 로직 재사용

### Computation Budget 예시

- n=16, beam_width=4 → search = n / beam_width = 4 (각 beam당 continuation 개수)
- 매 iteration마다 active_beams를 n=16개로 유지 (duplication, 기존과 동일)
- 각 beam당 search=4개 continuation 생성
- 총 16×4=64개 generation per iteration
- PRM으로 스코어링 후 각 beam의 최고점 선택
- Top n//beam_width=4개 beam 선택 (pruning)

**핵심**:
- 기존 코드의 duplication/pruning 로직 유지
- 변경: 각 beam당 1개 → search개 continuation 생성

### 예상 구현 난이도

**중간**: generate_k_steps 제거, 직접 generation, beam별 그룹핑 및 최고점 선택 추가

## Phase 3: dvts 확장

### 핵심 설계 원칙

**DVTS 구조**:
- `n_beams`: 메인 beam 개수
- `beam_width`: 각 beam의 diverse path 개수
- 각 메인 beam이 beam_width개의 다양한 경로를 유지

**Temperature 적용 방식 (beam_search와 유사)**:
- 각 diverse path가 temperature_list 비율대로 여러 continuation 생성
- DVTS의 경우 beam_width가 다른 의미이므로 search 계산 방식 확인 필요
- Diverse path가 사라져도 남은 path들이 모든 temperature 유지

### 필요한 변경사항

1. **Search 개수 계산 및 generate_k_steps 제거**:
   ```python
   # DVTS의 경우 config.n과 beam_width 관계 확인 필요
   # beam_search: search = config.n // config.beam_width
   # DVTS: search = ? (코드 분석 후 결정)

   continuations_per_path = # DVTS에 맞는 계산식

   from sal.utils.temperature import get_temperature_assignment
   temp_config = copy.copy(config)
   temp_config.n = continuations_per_path
   temps = get_temperature_assignment(temp_config)

   # 각 diverse path마다 temperature별로 여러 continuation 생성
   prompts = []
   sampling_params_list = []

   for beam in active_beams:
       for diverse_path_idx in range(beam_width):
           conv = build_conv(...)
           templated_conv = tokenizer.apply_chat_template(...)

           for t in temps:
               prompts.append(templated_conv)
               sampling_params_list.append(SamplingParams(temperature=t, ...))

   outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)
   ```

2. **PRM Scoring 및 최고점 선택** (beam_search와 동일 패턴):
   ```python
   # 각 diverse path의 여러 continuation 중 PRM 최고점 선택
   # beam_search의 로직을 diverse path에 적용
   ```

3. **DVTS 고유의 Diversity/Verification 로직 유지**:
   - Diversity-based selection
   - Verification-based pruning
   - 기존 DVTS 알고리즘은 유지, temperature 부분만 확장

### 핵심 개념

- **Beam/Path는 temperature를 소유하지 않음** (beam_search와 동일)
- 각 diverse path가 동일한 temperature 분포 사용
- **lookahead 제거, generate_k_steps 제거** (beam_search와 동일)
- best_of_n의 `get_temperature_assignment()` 로직 재사용

### 예상 구현 난이도

**낮음**: beam_search 패턴 거의 그대로 재사용 가능 (DVTS 특유의 diverse path 처리만 추가)

## 구현 순서

1. ✅ Phase 1: best_of_n (완료)
2. 🔜 Phase 2: beam_search
   - generate_k_steps 제거
   - 직접 llm.generate() 호출로 변경
   - 각 beam당 search개 continuation 생성 (temperature별)
   - PRM 스코어링 및 최고점 선택
   - 기존 duplication/pruning 로직 유지
   - 테스트 및 검증
3. 🔜 Phase 3: dvts
   - beam_search 패턴 적용
   - DVTS 구조에 맞게 search 계산
   - generate_k_steps 제거
   - 각 diverse path당 continuation 생성 (temperature별)
   - 테스트 및 검증

## 참고 자료

- Plan 파일: `/home/b.ms/.claude/plans/eager-stirring-cocoa.md`
- vLLM 문서: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
- 현재 구현 커밋: [이 파일과 함께 커밋됨]
