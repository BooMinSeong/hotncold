from datasets import load_dataset

# 1. GSM8K test split 로드
ds = load_dataset("openai/gsm8k", "main", split="test")

# 2. 컬럼 변환
def transform(example):
    parts = example["answer"].split("#### ")
    solution = parts[0].strip()
    answer = parts[1].strip() if len(parts) > 1 else ""
    return {
        "problem": example["question"],
        "solution": solution,
        "answer": answer,
    }

ds = ds.map(transform, remove_columns=["question"])

# 3. 검증: 샘플 출력
print(f"Columns: {ds.column_names}")
print(f"Num examples: {len(ds)}")
for i in range(3):
    print(f"\n--- Example {i} ---")
    print(f"problem: {ds[i]['problem'][:100]}...")
    print(f"solution: {ds[i]['solution'][:100]}...")
    print(f"answer: {ds[i]['answer']}")

# 4. Push to Hub
ds.push_to_hub("ENSEONG/gsm8k-private", split="test")
print("\nPushed to ENSEONG/gsm8k-private")
