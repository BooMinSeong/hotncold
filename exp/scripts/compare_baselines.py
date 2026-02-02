#!/usr/bin/env python3
"""Compare difficulty baselines between T=0.1 and T=0.8 reference temperatures."""

import json
from pathlib import Path


def get_level(accuracy):
    """Get difficulty level from accuracy."""
    if accuracy >= 0.8:
        return 1
    elif accuracy >= 0.6:
        return 2
    elif accuracy >= 0.4:
        return 3
    elif accuracy >= 0.2:
        return 4
    else:
        return 5


def main():
    # Load both baseline files
    with open('exp/analysis_output-MATH500-Qwen2.5-1.5B-bon-difficulty/difficulty_baselines.json') as f:
        baseline_t01 = json.load(f)

    with open('exp/analysis_output-MATH500-Qwen2.5-1.5B-bon-difficulty-ref0.8/difficulty_baselines.json') as f:
        baseline_t08 = json.load(f)

    print("=" * 70)
    print("Baseline Comparison: T=0.1 vs T=0.8 Reference")
    print("=" * 70)
    print()

    # Compare metadata
    print("Metadata:")
    print(f"  T=0.1 reference temp: {baseline_t01['metadata']['reference_temperature']}")
    print(f"  T=0.8 reference temp: {baseline_t08['metadata']['reference_temperature']}")
    print(f"  Number of problems: {baseline_t01['metadata']['num_problems']}")
    print()

    # Extract baselines
    problems_t01 = baseline_t01['problems']
    problems_t08 = baseline_t08['problems']

    # Compare levels
    shifts = {
        'easier': [],  # Moved to easier level at T=0.8
        'harder': [],  # Moved to harder level at T=0.8
        'same': []     # Same level at both
    }

    for problem_id in problems_t01.keys():
        acc_t01 = problems_t01[problem_id]['mean_accuracy']
        acc_t08 = problems_t08[problem_id]['mean_accuracy']

        level_t01 = get_level(acc_t01)
        level_t08 = get_level(acc_t08)

        if level_t08 < level_t01:  # Lower level number = easier
            shifts['easier'].append({
                'id': problem_id,
                'acc_t01': acc_t01,
                'acc_t08': acc_t08,
                'level_t01': level_t01,
                'level_t08': level_t08,
                'diff': acc_t08 - acc_t01
            })
        elif level_t08 > level_t01:
            shifts['harder'].append({
                'id': problem_id,
                'acc_t01': acc_t01,
                'acc_t08': acc_t08,
                'level_t01': level_t01,
                'level_t08': level_t08,
                'diff': acc_t08 - acc_t01
            })
        else:
            shifts['same'].append({
                'id': problem_id,
                'acc_t01': acc_t01,
                'acc_t08': acc_t08,
                'level': level_t01
            })

    print("Problem Category Shifts:")
    print(f"  Problems that became EASIER at T=0.8: {len(shifts['easier'])}")
    print(f"  Problems that became HARDER at T=0.8: {len(shifts['harder'])}")
    print(f"  Problems that stayed SAME level: {len(shifts['same'])}")
    print()

    # Top problems that benefited from diversity (much easier at T=0.8)
    print("Top 10 Problems That MOST Benefit from Diversity (easier at T=0.8):")
    easier_sorted = sorted(shifts['easier'], key=lambda x: x['diff'], reverse=True)
    for i, p in enumerate(easier_sorted[:10], 1):
        print(f"  {i}. {p['id']}")
        print(f"      Level {p['level_t01']}→{p['level_t08']}, "
              f"Acc: {p['acc_t01']:.3f}→{p['acc_t08']:.3f} (+{p['diff']:.3f})")
    print()

    # Top problems that got harder (worse at T=0.8)
    print("Top 10 Problems That HURT by Diversity (harder at T=0.8):")
    harder_sorted = sorted(shifts['harder'], key=lambda x: x['diff'])
    for i, p in enumerate(harder_sorted[:10], 1):
        print(f"  {i}. {p['id']}")
        print(f"      Level {p['level_t01']}→{p['level_t08']}, "
              f"Acc: {p['acc_t01']:.3f}→{p['acc_t08']:.3f} ({p['diff']:.3f})")
    print()

    # Level transition matrix
    print("Level Transition Matrix (T=0.1 → T=0.8):")
    print("         | L1    L2    L3    L4    L5")
    print("---------+---------------------------")
    transition_matrix = {}
    for source_level in range(1, 6):
        transition_matrix[source_level] = {target: 0 for target in range(1, 6)}

    for p in shifts['easier'] + shifts['harder']:
        transition_matrix[p['level_t01']][p['level_t08']] += 1
    for p in shifts['same']:
        transition_matrix[p['level']][p['level']] += 1

    for source in range(1, 6):
        row = f"Level {source} |"
        for target in range(1, 6):
            count = transition_matrix[source][target]
            row += f" {count:4d}"
        print(row)
    print()

    # Statistics by level
    print("Accuracy Change Statistics by Original Level (T=0.1):")
    for level in range(1, 6):
        problems_at_level = [p for p in problems_t01.keys()
                             if get_level(problems_t01[p]['mean_accuracy']) == level]
        if not problems_at_level:
            continue

        acc_changes = [problems_t08[p]['mean_accuracy'] - problems_t01[p]['mean_accuracy']
                       for p in problems_at_level]
        avg_change = sum(acc_changes) / len(acc_changes)

        print(f"  Level {level}: {len(problems_at_level)} problems, "
              f"Avg accuracy change: {avg_change:+.3f}")
    print()

    # Save detailed comparison
    output = {
        'metadata': {
            'ref_t01_temp': baseline_t01['metadata']['reference_temperature'],
            'ref_t08_temp': baseline_t08['metadata']['reference_temperature'],
            'num_problems': baseline_t01['metadata']['num_problems']
        },
        'summary': {
            'easier_at_t08': len(shifts['easier']),
            'harder_at_t08': len(shifts['harder']),
            'same_level': len(shifts['same'])
        },
        'transition_matrix': transition_matrix,
        'top_diversity_benefiting': easier_sorted[:20],
        'top_diversity_hurting': harder_sorted[:20]
    }

    output_path = 'exp/analysis_output-MATH500-Qwen2.5-1.5B-bon-difficulty/baseline_comparison.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Saved detailed comparison to: {output_path}")


if __name__ == '__main__':
    main()
