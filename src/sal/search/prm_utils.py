"""Utility functions for prm-toolkit compatibility."""
from typing import List, Tuple


def flatten_completions(
    prompts: List[str],
    completions: List[List[str]]
) -> Tuple[List[str], List[str], List[int]]:
    """
    Flatten nested completions structure for prm-toolkit.

    Args:
        prompts: List of N prompts
        completions: List of N lists, where completions[i] contains
                    candidates for prompts[i]

    Returns:
        flat_prompts: Flattened prompts repeated for each candidate
        flat_responses: Flattened all candidate responses
        structure: Number of candidates per prompt for unflattening

    Example:
        prompts = ["Q1", "Q2"]
        completions = [["A1", "A2"], ["B1"]]

        Returns:
            (["Q1", "Q1", "Q2"], ["A1", "A2", "B1"], [2, 1])
    """
    flat_prompts = []
    flat_responses = []
    structure = []

    for prompt, candidates in zip(prompts, completions):
        structure.append(len(candidates))
        for candidate in candidates:
            flat_prompts.append(prompt)
            flat_responses.append(candidate)

    return flat_prompts, flat_responses, structure


def unflatten_scores(
    flat_scores: List[List[float]],
    structure: List[int]
) -> List[List[List[float]]]:
    """
    Unflatten prm-toolkit scores back to nested structure.

    Args:
        flat_scores: Flattened scores from prm.score_batch()
        structure: Number of candidates per prompt (from flatten_completions)

    Returns:
        Nested scores: List[prompt][candidate][step_scores]

    Example:
        flat_scores = [[0.9], [0.8], [0.95]]
        structure = [2, 1]

        Returns:
            [[[0.9], [0.8]], [[0.95]]]
    """
    nested_scores = []
    idx = 0

    for num_candidates in structure:
        prompt_scores = []
        for _ in range(num_candidates):
            prompt_scores.append(flat_scores[idx])
            idx += 1
        nested_scores.append(prompt_scores)

    return nested_scores
