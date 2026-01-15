"""Core evaluation functions for experiment analysis.

This module contains the fundamental evaluation functions used across all analysis scripts.
Extracted from:
- exp/analyze_all_results.py (evaluate_result)
- exp/analyze_aime25_results.py (evaluate_result)
- exp/temperature_analysis.py (evaluate_answer)
- exp/temperature_analysis_per_problem.py (evaluate_answer)
- exp/temperature_analysis_stratified.py (evaluate_answer)
"""

import re

from math_verify import parse, verify


def evaluate_answer(completion: str, gold_answer: str) -> bool:
    """Evaluate if a completion contains the correct answer.

    This function extracts the answer from \\boxed{...} format in the completion
    and compares it with the gold answer using math_verify.

    Args:
        completion: Generated completion text
        gold_answer: Gold standard answer (without \\boxed{} wrapper)

    Returns:
        True if the extracted answer matches the gold answer, False otherwise

    Examples:
        >>> evaluate_answer("The answer is \\\\boxed{42}", "42")
        True
        >>> evaluate_answer("Therefore, x = \\\\boxed{\\\\frac{1}{2}}", "0.5")
        True
    """
    try:
        # Parse gold answer
        gold = parse("\\boxed{" + gold_answer + "}")

        # Extract content in \\boxed{...}
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        match = re.search(boxed_pattern, completion)
        if match:
            pred_text = match.group(1)
        else:
            # Fallback: try to extract last number-like string
            pred_text = completion.strip().split()[-1] if completion.strip() else ""

        pred = parse("\\boxed{" + pred_text + "}")

        return verify(gold, pred)
    except Exception:
        return False


def evaluate_result(data: dict, key: str = "answer") -> bool:
    """Evaluate a single prediction against gold answer.

    This function evaluates predictions stored in dataset rows against
    the gold answer field.

    Args:
        data: Dictionary containing 'answer' field and prediction field
        key: Key for the prediction in data dict (default: "answer")

    Returns:
        True if prediction matches gold answer, False otherwise

    Examples:
        >>> data = {"answer": "42", "pred_naive@1": "42"}
        >>> evaluate_result(data, "pred_naive@1")
        True
    """
    gold_answer = data["answer"]
    gold_answer = parse("\\boxed{" + gold_answer + "}")

    prediction = data[key]
    prediction = parse(prediction)

    return verify(gold_answer, prediction)


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{...} format.

    Args:
        text: Text containing \\boxed{answer}

    Returns:
        Extracted answer string, or last word if no boxed format found
    """
    boxed_pattern = r"\\boxed\{([^}]+)\}"
    match = re.search(boxed_pattern, text)
    if match:
        return match.group(1)
    return text.strip().split()[-1] if text.strip() else ""
