#!/usr/bin/env python3
"""Check what revisions exist on Hub for problematic datasets."""

from huggingface_hub import HfApi
import re

api = HfApi()

# Problematic datasets
datasets = [
    "ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-bon",
    "ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-beam_search",
]

print("Hub Revision Analysis")
print("=" * 120)

for dataset_id in datasets:
    try:
        refs = api.list_repo_refs(dataset_id, repo_type="dataset")

        print(f"\n📦 Dataset: {dataset_id}")
        print("-" * 120)

        # Get all branches
        branches = [b.name for b in refs.branches if b.name != "main"]

        if not branches:
            print("  No branches found (only main)")
            continue

        # Group by method name pattern
        bon_revisions = []
        beam_revisions = []
        dvts_revisions = []

        for branch in sorted(branches):
            if "--seed-0" in branch or "--seed-42" in branch or "--seed-64" in branch:
                # Extract seed
                seed_match = re.search(r'--seed-(\d+)', branch)
                seed = seed_match.group(1) if seed_match else "?"

                # Check what method it should be based on the dataset name
                if "bon" in dataset_id:
                    bon_revisions.append((branch, seed))
                elif "beam_search" in dataset_id:
                    beam_revisions.append((branch, seed))
                elif "dvts" in dataset_id:
                    dvts_revisions.append((branch, seed))

        if bon_revisions:
            print(f"\n  Best-of-N revisions (should be in bon dataset):")
            for rev, seed in bon_revisions:
                print(f"    ✓ {rev}")

        if beam_revisions:
            print(f"\n  Beam Search revisions (should be in beam_search dataset):")
            for rev, seed in beam_revisions:
                print(f"    ✓ {rev}")

        if dvts_revisions:
            print(f"\n  DVTS revisions (should be in dvts dataset):")
            for rev, seed in dvts_revisions:
                print(f"    ✓ {rev}")

        # Now check for problematic ones based on our log analysis
        print(f"\n  🔍 Cross-checking with job logs:")

        if "bon" in dataset_id:
            # This dataset should only have best_of_n results
            # But jobs 99460-99462 pushed beam_search here
            problematic = [b for b in branches if any(s in b for s in ["--seed-0", "--seed-42", "--seed-64"])]
            # We need to check the actual content or use our log analysis
            print(f"    ⚠️  Jobs 99460-99462 (beam_search) incorrectly pushed to this dataset")
            print(f"    ⚠️  These revisions may contain beam_search data instead of best_of_n")

        elif "beam_search" in dataset_id:
            # This dataset should only have beam_search results
            # But jobs 99463-99465 pushed dvts here
            print(f"    ⚠️  Jobs 99463-99465 (dvts) incorrectly pushed to this dataset")
            print(f"    ⚠️  These revisions may contain dvts data instead of beam_search")

    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 120)
print("\n🔴 RECOMMENDED ACTIONS:")
print("=" * 120)
print("\n1. Delete problematic revisions from ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-bon:")
print("   - Revisions with --seed-0, --seed-42, --seed-64 that were pushed by jobs 99460-99462")
print("   - These contain beam_search data, not best_of_n data")
print("\n2. Delete problematic revisions from ENSEONG/default-MATH-500-Llama-3.2-1B-Instruct-beam_search:")
print("   - Revisions with --seed-0, --seed-42, --seed-64 that were pushed by jobs 99463-99465")
print("   - These contain dvts data, not beam_search data")
print("\n3. Re-run the correct jobs:")
print("   - beam_search with seeds 0, 42, 64")
print("   - dvts with seeds 0, 42, 64")
