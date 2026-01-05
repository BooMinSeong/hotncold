from huggingface_hub import list_repo_refs

# 데이터셋 이름 지정
dataset_name = "ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon"

# 브랜치와 태그 정보 가져오기
refs = list_repo_refs(dataset_name, repo_type="dataset")

# 브랜치 이름만 출력
branch_names = [branch.name for branch in refs.branches]

for name in branch_names:
    if 'seed-42' in name:
        print(name)
