#!/usr/bin/env python
"""
Automated missing dataset detection and job submission.

Usage:
    # Dry-run: show what would be submitted
    python scripts/run_missing_auto.py

    # Submit all missing ranges
    python scripts/run_missing_auto.py --submit

    # Filter by specific method/seed/temperature
    python scripts/run_missing_auto.py --submit --method bon --seed 42

    # Interactive mode (prompt before each submission)
    python scripts/run_missing_auto.py --submit --interactive
"""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from exp.ds_chehck import check_dataset


@dataclass
class JobConfig:
    """Configuration for a single missing range job."""
    method: str          # "bon", "beam_search", "dvts"
    method_yaml: str     # "best_of_n.yaml", etc.
    seed: int
    temperature: float
    dataset_start: int
    dataset_end: int
    hub_dataset_id: str
    model_name: str


@dataclass
class AutoRunConfig:
    """Overall automation configuration."""
    model_name: str = "Qwen2.5-3B-Instruct"
    dataset_name: str = "MATH-500"
    hub_org: str = "ENSEONG"
    total_samples: int = 500
    chunk_size: int = 50  # Split missing ranges into chunks of this size

    seeds: List[int] = field(default_factory=lambda: [0, 42, 64])
    temperatures: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.4, 0.8])
    methods: Dict[str, str] = field(default_factory=lambda: {
        "bon": "best_of_n.yaml",
        "beam_search": "beam_search.yaml",
        "dvts": "dvts.yaml"
    })


class MissingJobFinder:
    """Finds all missing dataset ranges across methods/seeds/temperatures."""

    def __init__(self, config: AutoRunConfig):
        self.config = config

    def build_dataset_name(self, method: str) -> str:
        """Construct HuggingFace dataset repo name."""
        return f"{self.config.hub_org}/default-{self.config.dataset_name}-{self.config.model_name}-{method}"

    def split_range(self, start: int, end: int) -> List[tuple]:
        """Split a range into chunks of chunk_size.

        Example: split_range(100, 300) with chunk_size=50 returns:
            [(100, 150), (150, 200), (200, 250), (250, 300)]
        """
        chunks = []
        current = start
        while current < end:
            chunk_end = min(current + self.config.chunk_size, end)
            chunks.append((current, chunk_end))
            current = chunk_end
        return chunks

    def find_all_missing(
        self,
        method_filter: Optional[str] = None,
        seed_filter: Optional[int] = None,
        temp_filter: Optional[float] = None
    ) -> List[JobConfig]:
        """Scan all combinations and return list of jobs for missing ranges."""
        jobs = []

        for method, method_yaml in self.config.methods.items():
            if method_filter and method != method_filter:
                continue

            dataset_repo = self.build_dataset_name(method)

            for temp in self.config.temperatures:
                if temp_filter is not None and temp != temp_filter:
                    continue

                for seed in self.config.seeds:
                    if seed_filter is not None and seed != seed_filter:
                        continue

                    # Check this specific combination
                    filters = [f"seed-{seed}", f"T-{temp}"]
                    result = check_dataset(
                        dataset_repo,
                        filters=filters,
                        total=self.config.total_samples
                    )

                    # Skip if NOT_FOUND (dataset doesn't exist)
                    if result.status == "NOT_FOUND":
                        print(f"  Warning: {dataset_repo} not found, skipping...")
                        continue

                    # Create jobs for each missing range, split into chunks
                    for start, end in result.missing_ranges:
                        for chunk_start, chunk_end in self.split_range(start, end):
                            jobs.append(JobConfig(
                                method=method,
                                method_yaml=method_yaml,
                                seed=seed,
                                temperature=temp,
                                dataset_start=chunk_start,
                                dataset_end=chunk_end,
                                hub_dataset_id=dataset_repo,
                                model_name=self.config.model_name
                            ))

        return jobs


class JobSubmitter:
    """Handles job submission via sbatch."""

    def __init__(self, dry_run: bool = True, interactive: bool = False):
        self.dry_run = dry_run
        self.interactive = interactive

    def submit(self, job: JobConfig) -> bool:
        """Submit a single job. Returns True if submitted."""
        cmd = [
            "sbatch", "recipes/launch_single_default.slurm",
            f"recipes/{job.model_name}/{job.method_yaml}",
            f"--hub_dataset_id={job.hub_dataset_id}",
            f"--dataset_start={job.dataset_start}",
            f"--dataset_end={job.dataset_end}",
            f"--seed={job.seed}",
            f"--temperature={job.temperature}"
        ]

        print(f"  [{job.method}] seed={job.seed} T={job.temperature} "
              f"range=[{job.dataset_start},{job.dataset_end})")

        if self.dry_run:
            print(f"    DRY-RUN: {' '.join(cmd)}")
            return False

        if self.interactive:
            response = input("    Submit? [y/N]: ").strip().lower()
            if response != 'y':
                print("    Skipped.")
                return False

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    Submitted: {result.stdout.strip()}")
            return True
        else:
            print(f"    ERROR: {result.stderr}")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Auto-detect and submit missing dataset jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run (default) - show what would be submitted
  python scripts/run_missing_auto.py

  # Submit all missing ranges
  python scripts/run_missing_auto.py --submit

  # Interactive mode
  python scripts/run_missing_auto.py --submit --interactive

  # Filter by method/seed/temperature
  python scripts/run_missing_auto.py --submit --method bon --seed 42
  python scripts/run_missing_auto.py --submit --temperature 0.8
        """
    )
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit jobs (default: dry-run)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Prompt before each submission")
    parser.add_argument("--method", choices=["bon", "beam_search", "dvts"],
                        help="Filter by method")
    parser.add_argument("--seed", type=int, help="Filter by seed")
    parser.add_argument("--temperature", type=float, help="Filter by temperature")
    parser.add_argument("--model", default="Qwen2.5-3B-Instruct",
                        help="Model name (default: Qwen2.5-3B-Instruct)")

    args = parser.parse_args()

    # Configure
    config = AutoRunConfig(model_name=args.model)
    finder = MissingJobFinder(config)
    submitter = JobSubmitter(
        dry_run=not args.submit,
        interactive=args.interactive
    )

    # Find missing
    print("Scanning for missing dataset ranges...")
    print()
    jobs = finder.find_all_missing(
        method_filter=args.method,
        seed_filter=args.seed,
        temp_filter=args.temperature
    )

    if not jobs:
        print("\nNo missing ranges found. All datasets are complete!")
        return 0

    # Summarize
    print(f"\nFound {len(jobs)} missing ranges:")

    # Group by method for display
    by_method: Dict[str, List[JobConfig]] = {}
    for job in jobs:
        by_method.setdefault(job.method, []).append(job)

    for method, method_jobs in by_method.items():
        print(f"\n{method.upper()}:")
        for job in method_jobs:
            print(f"  seed={job.seed} T={job.temperature}: "
                  f"[{job.dataset_start},{job.dataset_end})")

    # Submit
    if args.submit or args.interactive:
        print(f"\n{'='*60}")
        print("Submitting jobs..." if args.submit else "Interactive submission mode:")
        print(f"{'='*60}")

        submitted = sum(1 for job in jobs if submitter.submit(job))
        print(f"\nSubmitted {submitted}/{len(jobs)} jobs")
    else:
        print(f"\nDry-run mode. Use --submit to actually submit jobs.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
