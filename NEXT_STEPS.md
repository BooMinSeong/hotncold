# Hot and Cold Temperature - 다음 확장 계획

## 현재 구현 상태 (Phase 1 - 완료)

✅ **best_of_n 알고리즘 Multi-Temperature 지원**
- Helper 함수: `src/sal/utils/temperature.py`
- Config 확장: `temperatures`, `temperature_ratios` 필드
- SamplingParams 리스트를 사용한 구현
- 균등 분배 및 사용자 지정 비율 지원

## Phase 2: beam_search 확장

### 필요한 변경사항

1. **Beam 클래스 확장** (`src/sal/search/utils.py`):
   ```python
   @dataclass
   class Beam:
       # ... 기존 필드들 ...
       temperature: float = 0.8  # NEW
   ```

2. **Beam 초기화 시 Temperature 할당** (`src/sal/search/beam_search.py`):
   - `get_temperature_assignment(config)` 호출
   - 각 beam에 고유 temperature 할당
   - copy.deepcopy 시 temperature 자동 보존

3. **Iteration별 SamplingParams 리스트 생성**:
   - 각 active beam의 temperature로 개별 SamplingParams 생성
   - Last iteration 여부에 따라 stop token 조정

4. **generate_k_steps 수정**:
   - `sampling_params` 인자가 리스트도 받을 수 있도록 확장
   - beam_width만큼 SamplingParams 확장
   - Lookahead는 greedy (T=0.0) 유지

### 핵심 개념

- 각 beam은 초기화 시 특정 temperature 소유
- 모든 iteration에서 동일한 temperature 유지 (Hot/Cold 일관성)
- Pruning/duplication 시에도 temperature 보존
- beam_width=1이므로 구현이 비교적 단순

### 예상 구현 난이도

**중간**: generate_k_steps에서 리스트 처리 로직 추가 필요

## Phase 3: dvts 확장

### 필요한 변경사항

1. **n_beams에 대한 Temperature 할당**:
   ```python
   temp_config = copy.copy(config)
   temp_config.n = config.n_beams
   temp_assignment = get_temperature_assignment(temp_config)
   ```

2. **Beam 초기화**:
   - n_beams (not n)에 대해 temperature 할당
   - beam_search와 동일한 패턴

3. **Iteration 로직**:
   - beam_search와 거의 동일
   - beam_width diverse continuations는 각 beam의 temperature 사용

### 핵심 개념

- Temperature는 main beams (n_beams)에 할당
- beam_width 개의 diverse continuation은 해당 beam의 temperature 공유
- beam_search 구현 완료 후 상대적으로 쉽게 확장 가능

### 예상 구현 난이도

**낮음**: beam_search 패턴 재사용 가능

## 구현 순서

1. ✅ Phase 1: best_of_n (완료)
2. 🔜 Phase 2: beam_search
   - Beam 클래스 확장
   - beam_search.py 수정
   - generate_k_steps 수정
   - 테스트 및 검증
3. 🔜 Phase 3: dvts
   - n_beams temperature 할당
   - dvts.py 수정
   - 테스트 및 검증

## 참고 자료

- Plan 파일: `/home/b.ms/.claude/plans/eager-stirring-cocoa.md`
- vLLM 문서: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
- 현재 구현 커밋: [이 파일과 함께 커밋됨]
