#!/usr/bin/env python3
"""Identify exact problematic revisions by matching job logs with Hub revisions."""

import re
from pathlib import Path
from huggingface_hub import HfApi

api = HfApi()
log_dir = Path("logs/default_run")

# First, collect all job information
jobs = {}
for job_dir in sorted(log_dir.iterdir()):
    if not job_dir.is_dir():
        continue

    err_file = job_dir / "task_1.err"
    if not err_file.exists():
        continue

    try:
        content = err_file.read_text()

        # Find the Config line
        config_match = re.search(
            r"Config\(approach='([^']*)'.*hub_dataset_id='([^']*)'.*"
            r"dataset_start=(\d+).*dataset_end=(\d+).*seed=(\d+)",
            content
        )
        if not config_match:
            continue

        approach = config_match.group(1)
        hub_dataset_id = config_match.group(2)
        dataset_start = config_match.group(3)
        dataset_end = config_match.group(4)
        seed = config_match.group(5)

        jobs[job_dir.name] = {
            "approach": approach,
            "hub_dataset_id": hub_dataset_id,
            "dataset_start": dataset_start,
            "dataset_end": dataset_end,
            "seed": seed,
        }

    except Exception as e:
        continue

# Filter for problematic jobs
problematic_jobs = {
    job_id: info
    for job_id, info in jobs.items()
    if (
        (info["approach"] == "beam_search" and "-bon" in info["hub_dataset_id"])
        or (info["approach"] == "dvts" and "-beam_search" in info["hub_dataset_id"])
    )
}

print("Problematic Jobs and Their Expected Revisions")
print("=" * 120)

revision_to_delete = {}

for job_id, info in sorted(problematic_jobs.items()):
    print(f"\n🔴 Job {job_id}:")
    print(f"   Approach: {info['approach']}")
    print(f"   Hub Dataset ID: {info['hub_dataset_id']}")
    print(f"   Chunk: {info['dataset_start']}-{info['dataset_end']}")
    print(f"   Seed: {info['seed']}")

    # Construct the expected revision name
    # Based on config.py logic for beam_search/dvts
    if info["approach"] in ["beam_search", "dvts"]:
        # From config.py line 124
        revision_pattern = (
            f"HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--"
            f"look-0--seed-{info['seed']}--agg_strategy--last--"
            f"chunk-{info['dataset_start']}_{info['dataset_end']}"
        )
    else:
        # From config.py line 126
        revision_pattern = (
            f"HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--"
            f"seed-{info['seed']}--agg_strategy-last--"
            f"chunk-{info['dataset_start']}_{info['dataset_end']}"
        )

    print(f"   Expected revision: {revision_pattern}")

    # Store for deletion
    dataset_id = info["hub_dataset_id"]
    if dataset_id not in revision_to_delete:
        revision_to_delete[dataset_id] = []
    revision_to_delete[dataset_id].append(revision_pattern)

# Verify these revisions exist on Hub
print("\n" + "=" * 120)
print("\n✅ VERIFIED REVISIONS TO DELETE:")
print("=" * 120)

for dataset_id, revisions in revision_to_delete.items():
    print(f"\n📦 Dataset: {dataset_id}")

    try:
        refs = api.list_repo_refs(dataset_id, repo_type="dataset")
        existing_branches = {b.name for b in refs.branches}

        for rev in sorted(set(revisions)):
            if rev in existing_branches:
                print(f"   ✓ {rev} (EXISTS - should be deleted)")
            else:
                print(f"   ✗ {rev} (NOT FOUND - may have different name)")

    except Exception as e:
        print(f"   ❌ Error checking dataset: {e}")

# Generate deletion commands
print("\n" + "=" * 120)
print("\n🔧 DELETION COMMANDS:")
print("=" * 120)
print("\n# Use huggingface-cli or HfApi to delete these revisions:\n")

for dataset_id, revisions in revision_to_delete.items():
    print(f"\n# Dataset: {dataset_id}")
    for rev in sorted(set(revisions)):
        print(f'huggingface-cli repo delete {dataset_id} --repo-type dataset --revision "{rev}"')

print("\n# Or use Python:")
print("from huggingface_hub import HfApi")
print("api = HfApi()")
for dataset_id, revisions in revision_to_delete.items():
    for rev in sorted(set(revisions)):
        print(f'api.delete_branch(repo_id="{dataset_id}", repo_type="dataset", branch="{rev}")')
