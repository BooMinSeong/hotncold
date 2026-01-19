#!/usr/bin/env python3
"""Check consistency between approach and hub_dataset_id in submitted jobs."""

import re
from pathlib import Path

log_dir = Path("logs/default_run")

results = []

for job_dir in sorted(log_dir.iterdir()):
    if not job_dir.is_dir():
        continue

    err_file = job_dir / "task_1.err"
    if not err_file.exists():
        continue

    try:
        content = err_file.read_text()

        # Find the Config line
        config_match = re.search(r"Config\(approach='([^']*)'.*hub_dataset_id='([^']*)'.*seed=(\d+)", content)
        if not config_match:
            continue

        approach = config_match.group(1)
        hub_dataset_id = config_match.group(2)
        seed = config_match.group(3)

        # Extract method name from hub_dataset_id
        hub_method = hub_dataset_id.split("-")[-1]

        # Check consistency
        expected_method = {
            "best_of_n": "bon",
            "beam_search": "beam_search",
            "dvts": "dvts"
        }.get(approach, "unknown")

        consistent = (hub_method == expected_method)

        results.append({
            "job_id": job_dir.name,
            "approach": approach,
            "hub_method": hub_method,
            "expected": expected_method,
            "consistent": consistent,
            "seed": seed,
            "hub_dataset_id": hub_dataset_id
        })

    except Exception as e:
        continue

# Print results
print("Job Analysis Report")
print("=" * 120)
print(f"{'Job ID':<10} {'Approach':<15} {'Hub Method':<15} {'Expected':<15} {'Seed':<6} {'Status':<10} {'Hub Dataset ID'}")
print("=" * 120)

for r in results:
    status = "✅ OK" if r["consistent"] else "❌ MISMATCH"
    print(f"{r['job_id']:<10} {r['approach']:<15} {r['hub_method']:<15} {r['expected']:<15} {r['seed']:<6} {status:<10} {r['hub_dataset_id']}")

# Summary
print("\n" + "=" * 120)
mismatches = [r for r in results if not r["consistent"]]
print(f"Total jobs: {len(results)}")
print(f"Consistent: {len(results) - len(mismatches)}")
print(f"Mismatched: {len(mismatches)}")

if mismatches:
    print("\n🔴 PROBLEMATIC DATASETS ON HUB:")
    print("=" * 120)
    for r in mismatches:
        print(f"  - {r['hub_dataset_id']}")
        print(f"    Job {r['job_id']}: approach='{r['approach']}' but pushed as '{r['hub_method']}'")
