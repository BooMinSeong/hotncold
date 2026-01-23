# Dataset Preprocessing Commands

This document contains all commands for preprocessing experiment datasets.

## Quick Start: Preprocess Everything

```bash
# Preprocess all experiments in registry (2-4 hours)
bash preprocess_all.sh
```

## Category-by-Category Preprocessing

### 1. MATH-500 Qwen2.5-1.5B (HNC Strategy)

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-1.5B_hnc \
    --push-to-hub \
    --validate
```

**Experiments:**
- `ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon`
- `ENSEONG/hnc-Qwen2.5-1.5B-Instruct-beam_search`
- `ENSEONG/hnc-Qwen2.5-1.5B-Instruct-dvts`

**Output:**
- `ENSEONG/preprocessed-hnc-Qwen2.5-1.5B-Instruct-bon`
- `ENSEONG/preprocessed-hnc-Qwen2.5-1.5B-Instruct-beam_search`
- `ENSEONG/preprocessed-hnc-Qwen2.5-1.5B-Instruct-dvts`

---

### 2. MATH-500 Qwen2.5-1.5B (Default Strategy)

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-1.5B \
    --push-to-hub \
    --validate
```

**Experiments:**
- `ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon`
- `ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-beam_search`
- `ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-dvts`

**Output:**
- `ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-bon`
- `ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-beam_search`
- `ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-dvts`

---

### 3. MATH-500 Qwen2.5-3B (Default Strategy)

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-3B \
    --push-to-hub \
    --validate
```

**Experiments:**
- `ENSEONG/default-MATH-500-Qwen2.5-3B-Instruct-bon`
- `ENSEONG/default-MATH-500-Qwen2.5-3B-Instruct-beam_search`
- `ENSEONG/default-MATH-500-Qwen2.5-3B-Instruct-dvts`

**Output:**
- `ENSEONG/preprocessed-default-MATH-500-Qwen2.5-3B-Instruct-bon`
- `ENSEONG/preprocessed-default-MATH-500-Qwen2.5-3B-Instruct-beam_search`
- `ENSEONG/preprocessed-default-MATH-500-Qwen2.5-3B-Instruct-dvts`

---

### 4. AIME25 Qwen2.5-1.5B

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --category aime25_Qwen2.5-1.5B \
    --push-to-hub \
    --validate
```

**Experiments:**
- `ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-bon`
- `ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-beam_search`
- `ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-dvts`

**Output:**
- `ENSEONG/preprocessed-default-aime25-Qwen2.5-1.5B-Instruct-bon`
- `ENSEONG/preprocessed-default-aime25-Qwen2.5-1.5B-Instruct-beam_search`
- `ENSEONG/preprocessed-default-aime25-Qwen2.5-1.5B-Instruct-dvts`

---

### 5. AIME25 Qwen2.5-3B

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --category aime25_Qwen2.5-3B \
    --push-to-hub \
    --validate
```

**Experiments:**
- `ENSEONG/default-aime25-Qwen2.5-3B-Instruct-bon`
- `ENSEONG/default-aime25-Qwen2.5-3B-Instruct-beam_search`
- `ENSEONG/default-aime25-Qwen2.5-3B-Instruct-dvts`

**Output:**
- `ENSEONG/preprocessed-default-aime25-Qwen2.5-3B-Instruct-bon`
- `ENSEONG/preprocessed-default-aime25-Qwen2.5-3B-Instruct-beam_search`
- `ENSEONG/preprocessed-default-aime25-Qwen2.5-3B-Instruct-dvts`

---

## Individual Experiment Preprocessing

### Preprocess a single experiment

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --push-to-hub \
    --validate
```

### Preprocess specific subsets only

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --subsets "HuggingFaceH4_MATH-500--T-0.4--top_p-1.0--n-64--seed-42--agg_strategy-last" \
    --push-to-hub \
    --validate
```

### Save locally (for testing)

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --output-dir /tmp/preprocessed-test
```

### Skip validation (faster)

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-1.5B \
    --push-to-hub
```

### Force reprocessing

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --push-to-hub \
    --force
```

---

## After Preprocessing: Update Registry

After preprocessing, update `exp/configs/registry.yaml` to use preprocessed paths:

```yaml
# Before
math500_Qwen2.5-1.5B:
  - ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon
  - ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-beam_search
  - ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-dvts

# After
math500_Qwen2.5-1.5B:
  - ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-bon
  - ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-beam_search
  - ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-dvts
```

Or create a helper script:

```bash
# Update all paths in registry to use preprocessed versions
sed -i 's|ENSEONG/|ENSEONG/preprocessed-|g' exp/configs/registry.yaml
```

---

## Verification Commands

### Check preprocessing status

```bash
uv run python -c "
from exp.analysis.preprocessing import get_preprocessing_stats
from datasets import load_dataset

dataset = load_dataset('ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-bon',
                       'HuggingFaceH4_MATH-500--T-0.4--top_p-1.0--n-64--seed-42--agg_strategy-last')
stats = get_preprocessing_stats(dataset['train'])
print(stats)
"
```

### List all preprocessed datasets

```bash
# List all datasets with 'preprocessed-' prefix in your org
huggingface-cli repo list --organization ENSEONG | grep preprocessed
```

---

## Estimated Processing Time

| Category | Experiments | Subsets | Est. Time |
|----------|-------------|---------|-----------|
| math500_Qwen2.5-1.5B_hnc | 3 | ~36 | 30-60 min |
| math500_Qwen2.5-1.5B | 3 | ~36 | 30-60 min |
| math500_Qwen2.5-3B | 3 | ~36 | 30-60 min |
| aime25_Qwen2.5-1.5B | 3 | ~12 | 10-20 min |
| aime25_Qwen2.5-3B | 3 | ~12 | 10-20 min |
| **Total** | **15** | **~132** | **2-4 hours** |

*Times are estimates and depend on network speed and compute resources.*

---

## Troubleshooting

### Error: "Dataset is not preprocessed"

You're trying to analyze a raw dataset. Run preprocessing first:

```bash
uv run python exp/scripts/preprocess_dataset.py \
    --hub-path <YOUR_DATASET> \
    --push-to-hub
```

### Error: "No module named 'datasets'"

Make sure dependencies are installed:

```bash
uv sync
```

### Hub authentication required

```bash
huggingface-cli login
```

### Out of memory during preprocessing

Process smaller batches or single experiments at a time instead of categories.
