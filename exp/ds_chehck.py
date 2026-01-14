import argparse
import re
import sys
from huggingface_hub import list_repo_refs
from huggingface_hub.utils import RepositoryNotFoundError


def parse_chunk_range(branch_name):
    """Extract (start, end) from branch name containing 'chunk-X_Y'"""
    match = re.search(r'chunk-(\d+)_(\d+)', branch_name)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def find_missing_ranges(ranges, total):
    """Given list of (start, end) tuples, find gaps from 0 to total"""
    if not ranges:
        return [(0, total)]

    # Sort ranges by start position
    sorted_ranges = sorted(ranges, key=lambda x: x[0])

    # Merge continuous/overlapping ranges
    merged = [sorted_ranges[0]]
    for current in sorted_ranges[1:]:
        last = merged[-1]
        # If current range overlaps or is continuous with last range, merge them
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    # Find gaps
    missing = []

    # Check if there's a gap at the beginning
    if merged[0][0] > 0:
        missing.append((0, merged[0][0]))

    # Check gaps between ranges
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]
        if gap_start < gap_end:
            missing.append((gap_start, gap_end))

    # Check if there's a gap at the end
    if merged[-1][1] < total:
        missing.append((merged[-1][1], total))

    return missing


def format_missing_ranges(missing):
    """Format missing ranges as compact string: [0,100),[200,250)"""
    if not missing:
        return ""
    return ",".join(f"[{s},{e})" for s, e in missing)


def print_oneline(dataset_name, filters, coverage_pct, missing, status="OK"):
    """Print single line status output"""
    filter_display = ",".join(filters) if filters else "all"

    if status == "NOT_FOUND":
        print(f"{dataset_name} | {filter_display} | - | NOT_FOUND")
    elif status == "NO_BRANCHES":
        print(f"{dataset_name} | {filter_display} | 0.0% | NO_BRANCHES")
    elif missing:
        missing_str = format_missing_ranges(missing)
        print(f"{dataset_name} | {filter_display} | {coverage_pct:.1f}% | MISSING: {missing_str}")
    else:
        print(f"{dataset_name} | {filter_display} | {coverage_pct:.1f}% | OK")


def main():
    # 인자 파서 설정
    parser = argparse.ArgumentParser(
        description='Check HuggingFace dataset branches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List branches with single filter
  python ds_chehck.py ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon --filter seed-42

  # List branches with multiple filters (AND condition)
  python ds_chehck.py ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon --filter seed-42 --filter T-0.8

  # List all branches
  python ds_chehck.py ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon
        """
    )
    parser.add_argument('dataset_name', type=str,
                       help='HuggingFace dataset name (e.g., org/dataset-name)')
    parser.add_argument('--filter', type=str, action='append', default=None,
                       help='Filter string for branch names (can be used multiple times for AND condition)')
    parser.add_argument('--total', type=int, default=500,
                       help='Total range to check for gaps (default: 500)')
    parser.add_argument('--format', type=str, choices=['human', 'oneline'],
                       default='human', help='Output format (default: human)')

    args = parser.parse_args()

    # 브랜치와 태그 정보 가져오기
    try:
        refs = list_repo_refs(args.dataset_name, repo_type="dataset")
    except RepositoryNotFoundError:
        if args.format == 'oneline':
            print_oneline(args.dataset_name, args.filter, 0, [], status="NOT_FOUND")
            sys.exit(1)
        else:
            print(f"Error: Dataset '{args.dataset_name}' not found")
            sys.exit(1)

    # 브랜치 이름만 추출
    branch_names = [branch.name for branch in refs.branches]

    # 필터링 및 출력
    if args.filter:
        # 필터링된 브랜치 찾기 (모든 필터 조건을 AND로 적용)
        filtered_branches = [
            name for name in branch_names
            if all(f in name for f in args.filter)
        ]

        # chunk 범위 파싱
        chunk_ranges = []
        for name in filtered_branches:
            chunk_range = parse_chunk_range(name)
            if chunk_range:
                chunk_ranges.append(chunk_range)

        # 빠진 범위 찾기
        missing = find_missing_ranges(chunk_ranges, args.total) if chunk_ranges else [(0, args.total)]
        total_missing = sum(e - s for s, e in missing)
        covered = args.total - total_missing
        coverage_pct = (covered / args.total) * 100

        if args.format == 'oneline':
            if not filtered_branches:
                print_oneline(args.dataset_name, args.filter, 0, [], status="NO_BRANCHES")
            else:
                print_oneline(args.dataset_name, args.filter, coverage_pct, missing)
        else:
            # Human-readable format
            filter_display = ", ".join(args.filter)
            print("=" * 70)
            print(f"Filtered branches (matching '{filter_display}'):")
            print("=" * 70)
            for name in filtered_branches:
                print(name)
            print(f"\nFound {len(filtered_branches)} branches")

            if chunk_ranges:
                # 범위 정렬
                sorted_ranges = sorted(chunk_ranges, key=lambda x: x[0])

                # 커버된 범위 출력
                print("\n" + "=" * 70)
                print(f"Covered ranges:")
                print("=" * 70)
                for start, end in sorted_ranges:
                    print(f"  [{start:4d}, {end:4d})")

                print("\n" + "=" * 70)
                if missing:
                    print(f"Missing ranges (total: {args.total}):")
                    print("=" * 70)
                    for start, end in missing:
                        size = end - start
                        print(f"  [{start:4d}, {end:4d}) - {size} items")

                    # 통계 출력
                    print("\n" + "=" * 70)
                    print(f"Coverage Statistics:")
                    print("=" * 70)
                    print(f"  Total range:    [0, {args.total})")
                    print(f"  Covered:        {covered} items ({coverage_pct:.1f}%)")
                    print(f"  Missing:        {total_missing} items ({100-coverage_pct:.1f}%)")
                else:
                    print(f"Complete coverage! All ranges from [0, {args.total}) are covered.")
                    print("=" * 70)
            else:
                print("\nNo chunk ranges found in filtered branches.")

    else:
        # 필터 없음 - 모든 브랜치 나열
        for name in branch_names:
            print(name)
        print(f"\nTotal {len(branch_names)} branches")

if __name__ == "__main__":
    main()
