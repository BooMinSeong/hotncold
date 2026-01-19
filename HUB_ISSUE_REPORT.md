# Hub Dataset Consistency Issue Report

## Summary

6개의 job이 잘못된 hub_dataset_id로 실행되어 데이터가 잘못된 데이터셋에 업로드되었습니다.

### Problem Overview

- **Total problematic jobs**: 6
- **Affected datasets**: 2
- **Root cause**: `SEARCH_METHODS` 배열과 `METHOD_NAMES` 배열의 불일치

## Problematic Datasets

### 1. ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-bon

**Issue**: Beam search 결과가 best-of-n 데이터셋에 업로드됨

**Problematic Revisions** (3개):
```
HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-0--agg_strategy--last--chunk-0_50
HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-42--agg_strategy--last--chunk-0_50
HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-64--agg_strategy--last--chunk-0_50
```

**Source Jobs**: 99460, 99461, 99462
- Actual approach: `beam_search`
- Wrong hub_dataset_id: ends with `-bon`

### 2. ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-beam_search

**Issue**: DVTS 결과가 beam search 데이터셋에 업로드됨

**Problematic Revisions** (3개):
```
HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-0--agg_strategy--last--chunk-0_50
HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-42--agg_strategy--last--chunk-0_50
HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-64--agg_strategy--last--chunk-0_50
```

**Source Jobs**: 99463, 99464, 99465
- Actual approach: `dvts`
- Wrong hub_dataset_id: ends with `-beam_search`

## Root Cause Analysis

실행 당시 `run_default.sh`가 다음과 같이 수정되었을 가능성:

```bash
# ❌ 잘못된 수정 (추정)
SEARCH_METHODS=("beam_search.yaml" "dvts.yaml")  # best_of_n 제거
METHOD_NAMES=("bon" "beam_search" "dvts")         # 그대로 유지 (실수!)
```

이로 인해:
- `SEARCH_METHODS[0]` = beam_search.yaml → `METHOD_NAMES[0]` = bon (❌)
- `SEARCH_METHODS[1]` = dvts.yaml → `METHOD_NAMES[1]` = beam_search (❌)

## Recommended Actions

### 1. Delete Problematic Revisions

**Option A: Using huggingface-cli**
```bash
# Delete from bon dataset (beam_search data)
huggingface-cli repo delete ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-bon --repo-type dataset --revision "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-0--agg_strategy--last--chunk-0_50"
huggingface-cli repo delete ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-bon --repo-type dataset --revision "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-42--agg_strategy--last--chunk-0_50"
huggingface-cli repo delete ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-bon --repo-type dataset --revision "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-64--agg_strategy--last--chunk-0_50"

# Delete from beam_search dataset (dvts data)
huggingface-cli repo delete ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-beam_search --repo-type dataset --revision "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-0--agg_strategy--last--chunk-0_50"
huggingface-cli repo delete ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-beam_search --repo-type dataset --revision "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-42--agg_strategy--last--chunk-0_50"
huggingface-cli repo delete ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-beam_search --repo-type dataset --revision "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-64--agg_strategy--last--chunk-0_50"
```

**Option B: Using Python**
```python
from huggingface_hub import HfApi

api = HfApi()

# Delete from bon dataset
bon_revisions = [
    "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-0--agg_strategy--last--chunk-0_50",
    "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-42--agg_strategy--last--chunk-0_50",
    "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-64--agg_strategy--last--chunk-0_50",
]

for rev in bon_revisions:
    api.delete_branch(
        repo_id="ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-bon",
        repo_type="dataset",
        branch=rev
    )

# Delete from beam_search dataset
beam_revisions = [
    "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-0--agg_strategy--last--chunk-0_50",
    "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-42--agg_strategy--last--chunk-0_50",
    "HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-64--agg_strategy--last--chunk-0_50",
]

for rev in beam_revisions:
    api.delete_branch(
        repo_id="ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-beam_search",
        repo_type="dataset",
        branch=rev
    )
```

### 2. Re-run Missing Experiments

올바른 approach와 hub_dataset_id로 다음 실험을 재실행해야 합니다:

**Beam Search** (T=0.4, seeds 0, 42, 64):
```bash
sbatch recipes/launch_array_default.slurm \
    recipes/Llama-3.2-1B-Instruct/beam_search.yaml \
    --seed=0 --temperature=0.4 --dataset_name=HuggingFaceH4/MATH-500

sbatch recipes/launch_array_default.slurm \
    recipes/Llama-3.2-1B-Instruct/beam_search.yaml \
    --seed=42 --temperature=0.4 --dataset_name=HuggingFaceH4/MATH-500

sbatch recipes/launch_array_default.slurm \
    recipes/Llama-3.2-1B-Instruct/beam_search.yaml \
    --seed=64 --temperature=0.4 --dataset_name=HuggingFaceH4/MATH-500
```

**DVTS** (T=0.4, seeds 0, 42, 64):
```bash
sbatch recipes/launch_array_default.slurm \
    recipes/Llama-3.2-1B-Instruct/dvts.yaml \
    --seed=0 --temperature=0.4 --dataset_name=HuggingFaceH4/MATH-500

sbatch recipes/launch_array_default.slurm \
    recipes/Llama-3.2-1B-Instruct/dvts.yaml \
    --seed=42 --temperature=0.4 --dataset_name=HuggingFaceH4/MATH-500

sbatch recipes/launch_array_default.slurm \
    recipes/Llama-3.2-1B-Instruct/dvts.yaml \
    --seed=64 --temperature=0.4 --dataset_name=HuggingFaceH4/MATH-500
```

### 3. Prevent Future Issues

**Option 1**: `run_default.sh`를 수정할 때 METHOD_NAMES도 함께 수정
```bash
# ✅ 올바른 수정
SEARCH_METHODS=("beam_search.yaml" "dvts.yaml")
METHOD_NAMES=("beam_search" "dvts")  # bon 제거!
```

**Option 2**: `--hub_dataset_id`를 제거하고 config.py가 자동 생성하도록 함
- config.py:108-110에서 approach 기반으로 자동 생성
- 이렇게 하면 불일치 문제가 원천적으로 방지됨

## Verification Scripts

이 보고서는 다음 스크립트들로 생성되었습니다:

1. `check_hub_consistency.py` - 로컬 로그에서 불일치 발견
2. `check_hub_revisions.py` - Hub에서 실제 revision 확인
3. `identify_problematic_revisions.py` - 삭제할 정확한 revision 식별

---

**Report generated**: 2026-01-19
**Analysis based on**: logs/default_run/ directory
