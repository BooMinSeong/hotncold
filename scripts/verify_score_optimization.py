#!/usr/bin/env python
"""Verify that the optimized score() produces identical results to the original code.

Checks:
  1. 'preds'     — per-completion safe_parse_answer output (completion-level)
  2. 'pred_maj@n' — majority prediction for each subset size (row-level)

Reference dataset: ENSEONG/stratified-solvable-1k-math-private-Qwen2.5-3B-Instruct-bon
"""

import argparse

from datasets import load_dataset

from sal.config import Config
from sal.utils.math import safe_parse_answer
from sal.utils.score import score


def compute_preds(x):
    return {"preds_new": [safe_parse_answer(c) for c in x["completions"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ENSEONG/stratified-solvable-1k-math-private-Qwen2.5-3B-Instruct-bon",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="ENSEONG_math-private--T-0.6--top_p-1.0--n-64--seed-42--agg_strategy-last",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset_name} [{args.config_name}] (split={args.split})")
    ds_ref = load_dataset(args.dataset_name, args.config_name, split=args.split)
    print(f"Dataset loaded: {len(ds_ref)} rows, columns: {ds_ref.column_names}")

    total_rows = len(ds_ref)
    all_match = True

    # ── 1. preds: per-completion safe_parse_answer ──────────────────────────
    print("\n[1] Checking preds (per-completion parse) ...")
    if "preds" not in ds_ref.column_names:
        print("  [SKIP] 'preds' column not found.")
    else:
        ds_parsed = ds_ref.map(compute_preds, num_proc=args.num_proc, desc="Parsing completions")
        total_completions = 0
        mismatch_rows = 0
        mismatch_completions = 0
        for i in range(total_rows):
            ref = ds_ref["preds"][i]
            new = ds_parsed["preds_new"][i]
            row_mismatches = sum(r != n for r, n in zip(ref, new))
            total_completions += len(ref)
            if row_mismatches > 0:
                mismatch_rows += 1
                mismatch_completions += row_mismatches
                if mismatch_rows <= 3:
                    for j, (r, n) in enumerate(zip(ref, new)):
                        if r != n:
                            print(f"  row {i}, completion {j}: ref={repr(r)}  new={repr(n)}")
        status = "OK" if mismatch_rows == 0 else "MISMATCH"
        print(f"  rows: {mismatch_rows}/{total_rows} mismatches, "
              f"completions: {mismatch_completions}/{total_completions}  [{status}]")
        if mismatch_rows > 0:
            all_match = False

    # ── 2. pred_maj@n: majority prediction per subset ───────────────────────
    print(f"\n[2] Checking pred_maj@n (majority prediction, n={args.n}) ...")
    subsets = [2**i for i in range(args.n) if 2**i <= args.n]
    ref_cols = [f"pred_maj@{n}" for n in subsets]
    missing = [c for c in ref_cols if c not in ds_ref.column_names]
    if missing:
        print(f"  [WARN] Missing reference columns: {missing}")
        subsets = [n for n in subsets if f"pred_maj@{n}" in ds_ref.column_names]
        ref_cols = [f"pred_maj@{n}" for n in subsets]

    config = Config(n=args.n, num_proc=args.num_proc)
    ds_scored = score(ds_ref.select_columns(["completions"]), config)

    for col in ref_cols:
        mismatches = sum(ds_ref[col][i] != ds_scored[col][i] for i in range(total_rows))
        status = "OK" if mismatches == 0 else "MISMATCH"
        print(f"  {col}: {mismatches}/{total_rows} mismatches  [{status}]")
        if mismatches > 0:
            all_match = False

    # ── Summary ─────────────────────────────────────────────────────────────
    print()
    if all_match:
        print("All checks passed. Optimization is correct.")
    else:
        print("Some checks failed. See details above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
