"""
Analyze experimental results across multiple datasets and seeds.
Computes metrics for different approaches (bon, beam_search, dvts) with
hnc and default temperature strategies, averaging across seeds 0, 42, 64.
"""

from math_verify import parse, verify
from datasets import load_dataset, Dataset
from sal.utils.math import extract_completion_answers
from sal.utils.score import compute_pass_at_k
import re
from tqdm import tqdm
import numpy as np
from collections import defaultdict


def evaluate_result(data, key="answer"):
    """Evaluate a single prediction against gold answer."""
    gold_answer = data["answer"]
    gold_answer = parse("\\boxed{" + gold_answer + "}")

    prediction = data[key]
    prediction = parse(prediction)

    return verify(gold_answer, prediction)


def score_pass_at_k(dataset: Dataset, n: int = 256) -> Dataset:
    """Compute pass@k metrics for different k values."""
    dataset = dataset.map(
        extract_completion_answers,
        fn_kwargs={"n": None},
        num_proc=1,
        desc=f"Extract answers for Pass@k",
    )

    subsets = [2**i for i in range(n) if 2**i <= n]
    for k in subsets:
        dataset = dataset.map(
            compute_pass_at_k,
            fn_kwargs={"k": k},
            num_proc=1,
            desc=f"Compute Pass@{k}",
        )
    return dataset


def analyze_single_dataset(dataset, dataset_name, seed):
    """Analyze a single dataset and return results."""
    results_by_method = {
        'naive': {},
        'weighted': {},
        'maj': {},
        'pass@k': {}
    }

    if 'train' in dataset:
        dataset = dataset['train']

    # Find all prediction keys
    pred_keys = [key for key in dataset.features.keys() if key.startswith('pred_')]
    print(f"  Found {len(pred_keys)} prediction keys to evaluate")

    # Accumulate results for each key
    results_accumulator = {key: [] for key in pred_keys}

    # Evaluate all predictions
    print(f"  Evaluating predictions...")
    for data in tqdm(dataset, desc=f"  {dataset_name} (seed {seed})", leave=False):
        for key in pred_keys:
            result = evaluate_result(data, key)
            results_accumulator[key].append(result)

    # Calculate accuracy for each key
    for key in pred_keys:
        results = results_accumulator[key]
        accuracy = sum(results) / len(results)

        # Parse key: pred_method@number format
        match = re.match(r'pred_(naive|weighted|maj)@(\d+)', key)
        if match:
            method = match.group(1)
            n_samples = int(match.group(2))
            results_by_method[method][n_samples] = accuracy

    # Process pass@k keys
    pass_keys = [key for key in dataset.features.keys() if key.startswith('pass@')]
    for key in pass_keys:
        k = int(key.split('@')[-1])
        accuracy = sum(dataset[key]) / len(dataset[key])
        results_by_method['pass@k'][k] = accuracy

    return results_by_method


def main():
    # Dataset configurations
    datasets_config = {
        'hnc-bon': {
            'path': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon',
            'subset_template': 'HuggingFaceH4_MATH-500--temps_0.6_0.8_1.0_1.2__r_0.25_0.25_0.25_0.25--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last'
        },
        'hnc-beam_search': {
            'path': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-beam_search',
            'subset_template': 'HuggingFaceH4_MATH-500--temps_0.6_0.8_1.0_1.2__r_0.25_0.25_0.25_0.25--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last'
        },
        'hnc-dvts': {
            'path': 'ENSEONG/hnc-Qwen2.5-1.5B-Instruct-dvts',
            'subset_template': 'HuggingFaceH4_MATH-500--temps_0.6_0.8_1.0_1.2__r_0.25_0.25_0.25_0.25--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last'
        },
        'default-bon': {
            'path': 'ENSEONG/default-Qwen2.5-1.5B-Instruct-bon',
            'subset_template': 'HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last'
        },
        'default-beam_search': {
            'path': 'ENSEONG/default-Qwen2.5-1.5B-Instruct-beam_search',
            'subset_template': 'HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last'
        },
        'default-dvts': {
            'path': 'ENSEONG/default-Qwen2.5-1.5B-Instruct-dvts',
            'subset_template': 'HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--m-4--iters-40--look-0--seed-{seed}--agg_strategy--last'
        }
    }

    seeds = [0, 42, 64]

    # Store all results: dataset_name -> seed -> method -> n_samples -> accuracy
    all_results = defaultdict(lambda: defaultdict(dict))

    # Process each dataset
    for dataset_name, config in datasets_config.items():
        print(f"\n{'='*80}")
        print(f"Processing: {dataset_name}")
        print(f"{'='*80}")

        for seed in seeds:
            subset_name = config['subset_template'].format(seed=seed)
            print(f"\nLoading seed {seed}...")

            try:
                # Load dataset
                dataset = load_dataset(config['path'], subset_name)

                # Compute pass@k if not already computed
                if 'pass@1' not in dataset['train'].features:
                    print(f"  Computing pass@k metrics...")
                    dataset['train'] = score_pass_at_k(dataset['train'], n=64)

                # Analyze
                results = analyze_single_dataset(dataset, dataset_name, seed)
                all_results[dataset_name][seed] = results

            except Exception as e:
                print(f"  Error processing {dataset_name} seed {seed}: {e}")
                continue

    # Calculate statistics and print results
    print("\n\n" + "="*80)
    print("FINAL RESULTS - COMPARISON ACROSS DATASETS")
    print("="*80)

    # Group by approach type
    approaches = ['bon', 'beam_search', 'dvts']
    strategies = ['hnc', 'default']

    for approach in approaches:
        print(f"\n\n{'#'*80}")
        print(f"# APPROACH: {approach.upper()}")
        print(f"{'#'*80}")

        for strategy in strategies:
            dataset_name = f"{strategy}-{approach}"

            if dataset_name not in all_results:
                continue

            print(f"\n{'='*60}")
            print(f"Strategy: {strategy.upper()}")
            print(f"{'='*60}")

            # Aggregate results across seeds
            methods = ['naive', 'weighted', 'maj', 'pass@k']

            for method in methods:
                # Collect all n_samples and k values across seeds
                all_keys = set()
                for seed in seeds:
                    if seed in all_results[dataset_name]:
                        all_keys.update(all_results[dataset_name][seed][method].keys())

                if not all_keys:
                    continue

                print(f"\n{method.upper()} Method:")
                print("-" * 40)

                for key in sorted(all_keys):
                    values = []
                    for seed in seeds:
                        if seed in all_results[dataset_name]:
                            if key in all_results[dataset_name][seed][method]:
                                values.append(all_results[dataset_name][seed][method][key])

                    if values:
                        mean_acc = np.mean(values)
                        std_acc = np.std(values)
                        print(f"  {method}@{key:2d}: {mean_acc:.4f} ± {std_acc:.4f} "
                              f"({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")
                        print(f"           Individual seeds: {[f'{v:.4f}' for v in values]}")

    # Summary comparison table
    print("\n\n" + "="*80)
    print("SUMMARY TABLE - BEST RESULTS FOR EACH CONFIGURATION")
    print("="*80)
    print(f"\n{'Configuration':<30} {'Best Method':<15} {'Mean Accuracy':<20} {'Std Dev'}")
    print("-" * 85)

    for approach in approaches:
        for strategy in strategies:
            dataset_name = f"{strategy}-{approach}"

            if dataset_name not in all_results:
                continue

            # Find best result across all methods
            best_mean = 0
            best_std = 0
            best_method_name = ""

            for method in ['naive', 'weighted', 'maj', 'pass@k']:
                for seed in seeds:
                    if seed not in all_results[dataset_name]:
                        continue
                    for key, acc in all_results[dataset_name][seed][method].items():
                        # Collect values for this method@key
                        values = []
                        for s in seeds:
                            if s in all_results[dataset_name]:
                                if key in all_results[dataset_name][s][method]:
                                    values.append(all_results[dataset_name][s][method][key])

                        if values:
                            mean_acc = np.mean(values)
                            std_acc = np.std(values)

                            if mean_acc > best_mean:
                                best_mean = mean_acc
                                best_std = std_acc
                                best_method_name = f"{method}@{key}"

            if best_method_name:
                print(f"{dataset_name:<30} {best_method_name:<15} "
                      f"{best_mean:.4f} ({best_mean*100:.2f}%){'':<3} ± {best_std:.4f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
