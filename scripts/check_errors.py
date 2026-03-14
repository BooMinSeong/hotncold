#!/usr/bin/env python
"""Scan default_run logs for errors and map them to temperature/seed combinations."""

import argparse
import re
from collections import defaultdict
from pathlib import Path


def parse_config_from_err(err_path: Path) -> dict | None:
    """Extract Config parameters from the fingerprint WARNING in .err file."""
    text = err_path.read_text(errors="replace")

    # The Config(...) dump appears in the fingerprint WARNING line
    m = re.search(
        r"Config\(approach='(\w+)'.*?"
        r"temperature=([0-9.]+).*?"
        r"seed=(\d+).*?"
        r"dataset_start=(\d+).*?"
        r"dataset_end=(\d+)",
        text,
        re.DOTALL,
    )
    if not m:
        return None

    # Also extract hub_dataset_id if present
    hub_m = re.search(r"hub_dataset_id='([^']*)'", text)

    return {
        "approach": m.group(1),
        "temperature": float(m.group(2)),
        "seed": int(m.group(3)),
        "dataset_start": int(m.group(4)),
        "dataset_end": int(m.group(5)),
        "hub_dataset_id": hub_m.group(1) if hub_m else "",
    }


def infer_config_from_order(
    job_ids: list[str],
    seeds: list[int] = [0, 42, 64, 128, 256, 512],
    temp_start: float = 0.1,
    temp_step: float = 0.1,
    temp_count: int = 12,
    dataset_end: int = 5000,
) -> dict[str, dict]:
    """Infer temperature/seed from submission order for jobs missing Config dump.

    run_default.sh submits in order: for temp in 0.1..1.2, for seed in (0,42,64).
    Jobs are submitted sequentially, so consecutive job IDs form contiguous batches.
    """
    # Detect contiguous batches of job IDs
    ids = sorted(int(j) for j in job_ids)
    batches = []
    batch = [ids[0]]
    for i in range(1, len(ids)):
        if ids[i] - ids[i - 1] <= 1:
            batch.append(ids[i])
        else:
            batches.append(batch)
            batch = [ids[i]]
    batches.append(batch)

    expected_per_batch = len(seeds) * temp_count
    mapping = {}

    for batch in batches:
        if len(batch) == expected_per_batch:
            # Perfect batch: use positional index
            for idx, jid in enumerate(batch):
                temp_idx = idx // len(seeds)
                seed_idx = idx % len(seeds)
                temp = round(temp_start + temp_idx * temp_step, 2)
                mapping[str(jid)] = {
                    "approach": "best_of_n",
                    "temperature": temp,
                    "seed": seeds[seed_idx],
                    "dataset_start": 0,
                    "dataset_end": dataset_end,
                    "hub_dataset_id": "(inferred)",
                }
        elif len(batch) <= expected_per_batch:
            # Incomplete batch (gaps from failed jobs): use offset from first ID
            base_id = batch[0]
            for jid in batch:
                idx = jid - base_id
                if idx >= expected_per_batch:
                    break
                temp_idx = idx // len(seeds)
                seed_idx = idx % len(seeds)
                temp = round(temp_start + temp_idx * temp_step, 2)
                mapping[str(jid)] = {
                    "approach": "best_of_n",
                    "temperature": temp,
                    "seed": seeds[seed_idx],
                    "dataset_start": 0,
                    "dataset_end": dataset_end,
                    "hub_dataset_id": "(inferred)",
                }

    return mapping


def classify_job(err_path: Path) -> tuple[str, str]:
    """Classify job status and extract error message from last lines.

    Returns (status, error_message) where status is 'failed', 'running', or 'success'.
    """
    text = err_path.read_text(errors="replace")
    lines = text.strip().splitlines()

    if not lines:
        return "empty", ""

    last_lines = "\n".join(lines[-10:])

    # Check for fatal errors in the last lines
    if re.search(r"(RuntimeError|Exception|Error):", last_lines):
        # Extract the last error line
        for line in reversed(lines[-10:]):
            line = line.strip()
            if re.search(r"(RuntimeError|Exception|Error):", line):
                # Clean ANSI codes
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line).strip()
                return "failed", clean
        return "failed", "Unknown error"

    # Check if still running (progress bar in last lines)
    if re.search(r"Running search:.*\d+%", last_lines):
        return "running", ""

    # Check for successful completion markers
    if re.search(r"100%\|", last_lines) or re.search(r"Pushing dataset", last_lines):
        return "success", ""

    return "success", ""


def scan_logs(log_dir: Path, seeds: list[int] | None = None, dataset_end: int = 5000) -> list[dict]:
    """Scan all job directories and collect results."""
    results = []
    missing_config_ids = []

    for job_dir in sorted(log_dir.iterdir()):
        if not job_dir.is_dir():
            continue

        job_id = job_dir.name

        # Find all task err files
        err_files = sorted(job_dir.glob("task_*.err"))
        if not err_files:
            continue

        for err_file in err_files:
            task_id = re.search(r"task_(\d+)", err_file.name)
            task_id = task_id.group(1) if task_id else "?"

            config = parse_config_from_err(err_file)
            status, error_msg = classify_job(err_file)

            if config is None:
                missing_config_ids.append(job_id)

            results.append(
                {
                    "job_id": job_id,
                    "task_id": task_id,
                    "status": status,
                    "error_msg": error_msg,
                    **(config or {}),
                }
            )

    # Fill in missing configs using submission order inference
    if missing_config_ids:
        all_ids = [r["job_id"] for r in results]
        inferred = infer_config_from_order(all_ids, seeds=seeds or [0, 42, 64, 128, 256, 512], dataset_end=dataset_end)
        for r in results:
            if "temperature" not in r and r["job_id"] in inferred:
                r.update(inferred[r["job_id"]])

    return results


def print_summary(results: list[dict]):
    """Print overall summary counts."""
    total = len(results)
    failed = sum(1 for r in results if r["status"] == "failed")
    running = sum(1 for r in results if r["status"] == "running")
    success = sum(1 for r in results if r["status"] == "success")
    empty = sum(1 for r in results if r["status"] == "empty")

    print("=" * 60)
    print(f"  Total: {total} jobs | Failed: {failed} | Running: {running} | Success: {success}", end="")
    if empty:
        print(f" | Empty: {empty}", end="")
    print()
    print("=" * 60)


def print_failed(results: list[dict]):
    """Print table of failed jobs."""
    failed = [r for r in results if r["status"] == "failed"]
    if not failed:
        print("\nNo failed jobs found.")
        return

    print(f"\n{'=' * 60}")
    print(f"  Failed Jobs ({len(failed)})")
    print(f"{'=' * 60}")

    header = f"{'Job ID':<10} {'Task':<5} {'Temp':<6} {'Seed':<6} {'Range':<12} {'Error'}"
    print(header)
    print("-" * len(header) + "-" * 40)

    for r in failed:
        temp = f"{r.get('temperature', '?')}" if "temperature" in r else "?"
        seed = f"{r.get('seed', '?')}" if "seed" in r else "?"
        ds_range = f"{r.get('dataset_start', '?')}-{r.get('dataset_end', '?')}" if "dataset_start" in r else "?"
        error = r["error_msg"][:60] if r["error_msg"] else ""
        print(f"{r['job_id']:<10} {r['task_id']:<5} {temp:<6} {seed:<6} {ds_range:<12} {error}")


def print_running(results: list[dict]):
    """Print table of running jobs."""
    running = [r for r in results if r["status"] == "running"]
    if not running:
        return

    print(f"\n{'=' * 60}")
    print(f"  Running Jobs ({len(running)})")
    print(f"{'=' * 60}")

    header = f"{'Job ID':<10} {'Task':<5} {'Temp':<6} {'Seed':<6} {'Range':<12}"
    print(header)
    print("-" * len(header))

    for r in running:
        temp = f"{r.get('temperature', '?')}" if "temperature" in r else "?"
        seed = f"{r.get('seed', '?')}" if "seed" in r else "?"
        ds_range = f"{r.get('dataset_start', '?')}-{r.get('dataset_end', '?')}" if "dataset_start" in r else "?"
        print(f"{r['job_id']:<10} {r['task_id']:<5} {temp:<6} {seed:<6} {ds_range:<12}")


def detect_batches(results: list[dict]) -> list[list[dict]]:
    """Split results into contiguous batches based on job ID gaps."""
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: int(r["job_id"]))
    batches = [[sorted_results[0]]]

    for r in sorted_results[1:]:
        prev_id = int(batches[-1][-1]["job_id"])
        curr_id = int(r["job_id"])
        if curr_id - prev_id > 1:
            batches.append([])
        batches[-1].append(r)

    return batches


def print_matrix(results: list[dict]):
    """Print temperature × seed matrix showing status, one per batch."""
    with_config = [r for r in results if "temperature" in r]
    if not with_config:
        print("\nNo config data found to build matrix.")
        return

    batches = detect_batches(with_config)
    status_char = {"success": "✓", "failed": "✗", "running": "…", "empty": "?"}

    for batch_idx, batch in enumerate(batches):
        job_range = f"{batch[0]['job_id']}-{batch[-1]['job_id']}"

        temps = sorted(set(r["temperature"] for r in batch))
        seeds = sorted(set(r["seed"] for r in batch))

        lookup = defaultdict(list)
        for r in batch:
            lookup[(r["temperature"], r["seed"])].append(r)

        print(f"\n{'=' * 60}")
        print(f"  Batch {batch_idx + 1} — Jobs {job_range} ({len(batch)} jobs)")
        print(f"{'=' * 60}")

        seed_width = 8
        print(f"{'Temp':<8}", end="")
        for s in seeds:
            print(f"{s:<{seed_width}}", end="")
        print()
        print("-" * (8 + seed_width * len(seeds)))

        for t in temps:
            print(f"{t:<8.1f}", end="")
            for s in seeds:
                entries = lookup.get((t, s), [])
                if not entries:
                    cell = "-"
                else:
                    cell = status_char.get(entries[0]["status"], "?")
                print(f"{cell:<{seed_width}}", end="")
            print()

    print()
    print("Legend: ✓=success  ✗=failed  …=running  -=no data")


def print_error_summary(results: list[dict]):
    """Group and count distinct error types."""
    failed = [r for r in results if r["status"] == "failed" and r["error_msg"]]
    if not failed:
        return

    error_counts = defaultdict(int)
    for r in failed:
        # Normalize error message for grouping
        msg = r["error_msg"]
        # Truncate to the error type
        m = re.match(r"(\w+Error:\s*[^.{]+)", msg)
        key = m.group(1).strip() if m else msg[:80]
        error_counts[key] += 1

    print(f"\n{'=' * 60}")
    print("  Error Types")
    print(f"{'=' * 60}")
    for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
        print(f"  [{count:>3}x] {err}")


def main():
    parser = argparse.ArgumentParser(description="Scan default_run logs for errors")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/default_run"),
        help="Log directory to scan (default: logs/default_run)",
    )
    parser.add_argument("--show-all", action="store_true", help="Show successful jobs too")
    parser.add_argument("--matrix", action="store_true", help="Show only the temperature × seed matrix")
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,42,64,128,256,512",
        help="Comma-separated seed list for order inference (default: 0,42,64,128,256,512)",
    )
    parser.add_argument(
        "--dataset-end",
        type=int,
        default=5000,
        help="Dataset end index for order inference (default: 5000)",
    )
    args = parser.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]

    if not args.log_dir.exists():
        print(f"Error: {args.log_dir} does not exist")
        return 1

    results = scan_logs(args.log_dir, seeds=args.seeds, dataset_end=args.dataset_end)

    if not results:
        print("No log files found.")
        return 1

    if args.matrix:
        print_matrix(results)
        return 0

    print_summary(results)
    print_failed(results)
    print_running(results)
    print_error_summary(results)
    print_matrix(results)

    if args.show_all:
        success = [r for r in results if r["status"] == "success"]
        if success:
            print(f"\n{'=' * 60}")
            print(f"  Successful Jobs ({len(success)})")
            print(f"{'=' * 60}")
            for r in success:
                temp = f"{r.get('temperature', '?')}" if "temperature" in r else "?"
                seed = f"{r.get('seed', '?')}" if "seed" in r else "?"
                print(f"  {r['job_id']} task={r['task_id']} temp={temp} seed={seed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
