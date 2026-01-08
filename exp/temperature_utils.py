"""
Temperature inference utilities for analyzing multi-temperature search strategies.

This module provides utilities to infer temperature values from completion positions
in datasets where temperature metadata is not explicitly stored. Different search
approaches (BoN, DVTS) use different temperature assignment strategies.
"""


def infer_temperature_from_position(
    position: int,
    approach: str,
    n: int = 64,
    temperatures: list[float] = None,
    beam_width: int = 16,
) -> float:
    """
    Map completion position to temperature value based on approach.

    Temperature assignment varies by search approach:
    - BoN: Sequential assignment by chunks (position → temps[position // chunk_size])
    - DVTS: Cyclic assignment per beam (position → temps[position % beam_width])
    - Beam Search: Sequential assignment by chunks (same as BoN)

    Args:
        position: Index in completions list (0-based)
        approach: Search approach ('bon', 'dvts', or 'beam_search')
        n: Total number of samples (used for BoN)
        temperatures: List of temperature values. Defaults to [0.4, 0.8, 1.2, 1.6]
        beam_width: Beam width for DVTS (number of diverse paths per beam)

    Returns:
        Temperature value for the given position

    Raises:
        ValueError: If approach is unknown or position is out of range

    Examples:
        >>> # BoN with n=64, temps=[0.4, 0.8, 1.2, 1.6]
        >>> infer_temperature_from_position(0, 'bon', n=64)
        0.4
        >>> infer_temperature_from_position(15, 'bon', n=64)
        0.4
        >>> infer_temperature_from_position(16, 'bon', n=64)
        0.8

        >>> # DVTS with beam_width=16, temps=[0.4, 0.8, 1.2, 1.6]
        >>> infer_temperature_from_position(0, 'dvts', beam_width=16)
        0.4
        >>> infer_temperature_from_position(1, 'dvts', beam_width=16)
        0.8
        >>> infer_temperature_from_position(16, 'dvts', beam_width=16)
        0.4
    """
    if temperatures is None:
        temperatures = [0.4, 0.8, 1.2, 1.6]

    approach = approach.lower()

    if approach == "bon":
        # Sequential assignment: divide n into equal chunks for each temperature
        samples_per_temp = n // len(temperatures)
        if position >= n:
            raise ValueError(f"Position {position} out of range for n={n}")
        temp_idx = position // samples_per_temp
        # Handle edge case where position might exceed due to rounding
        temp_idx = min(temp_idx, len(temperatures) - 1)
        return temperatures[temp_idx]

    elif approach == "dvts":
        # Cyclic assignment: each beam uses same temperature sequence
        # beam_width diverse paths, each with different temperature
        temp_idx = position % beam_width
        if temp_idx >= len(temperatures):
            # Handle case where beam_width > len(temperatures)
            # In this case, temperatures cycle within beam_width
            temp_idx = temp_idx % len(temperatures)
        return temperatures[temp_idx]

    elif approach == "beam_search":
        # Sequential assignment like BoN: divide n into equal chunks
        # This matches the implementation in beam_search.py where
        # temp_group = beam_idx // beams_per_temp (sequential grouping)
        samples_per_temp = n // len(temperatures)
        if position >= n:
            raise ValueError(f"Position {position} out of range for n={n}")
        temp_idx = position // samples_per_temp
        # Handle edge case where position might exceed due to rounding
        temp_idx = min(temp_idx, len(temperatures) - 1)
        return temperatures[temp_idx]

    else:
        raise ValueError(
            f"Unknown approach: {approach}. " f"Supported: 'bon', 'dvts', 'beam_search'"
        )


def assign_temperatures_to_completions(
    completions: list,
    scores: list,
    approach: str,
    n: int = 64,
    temperatures: list[float] = None,
    beam_width: int = 16,
) -> dict:
    """
    Assign temperature metadata to completions and scores based on position.

    Args:
        completions: List of completion texts
        scores: List of PRM scores corresponding to completions
        approach: Search approach ('bon', 'dvts', or 'beam_search')
        n: Total number of samples (used for BoN)
        temperatures: List of temperature values. Defaults to [0.4, 0.8, 1.2, 1.6]
        beam_width: Beam width for DVTS

    Returns:
        Dictionary with temperature assignments:
        {
            'completions': original completions list,
            'scores': original scores list,
            'temperatures': [temp for each completion],
        }

    Raises:
        ValueError: If completions and scores lengths don't match
    """
    if len(completions) != len(scores):
        raise ValueError(
            f"Completions ({len(completions)}) and scores ({len(scores)}) "
            f"must have same length"
        )

    if temperatures is None:
        temperatures = [0.4, 0.8, 1.2, 1.6]

    # Infer temperature for each position
    assigned_temps = [
        infer_temperature_from_position(
            i, approach, n=n, temperatures=temperatures, beam_width=beam_width
        )
        for i in range(len(completions))
    ]

    return {"completions": completions, "scores": scores, "temperatures": assigned_temps}


def group_by_temperature(
    completions: list, scores: list, temperatures: list[float]
) -> dict[float, dict]:
    """
    Group completions and scores by their temperature value.

    Args:
        completions: List of completion texts
        scores: List of PRM scores
        temperatures: List of temperature values (one per completion)

    Returns:
        Dictionary mapping temperature to completions and scores:
        {
            0.4: {'completions': [...], 'scores': [...]},
            0.8: {'completions': [...], 'scores': [...]},
            ...
        }

    Raises:
        ValueError: If list lengths don't match
    """
    if not (len(completions) == len(scores) == len(temperatures)):
        raise ValueError(
            f"All lists must have same length: "
            f"completions={len(completions)}, scores={len(scores)}, "
            f"temperatures={len(temperatures)}"
        )

    grouped = {}
    for completion, score, temp in zip(completions, scores, temperatures):
        if temp not in grouped:
            grouped[temp] = {"completions": [], "scores": []}
        grouped[temp]["completions"].append(completion)
        grouped[temp]["scores"].append(score)

    return grouped


def validate_temperature_config(
    approach: str,
    n: int,
    temperatures: list[float],
    beam_width: int = None,
) -> bool:
    """
    Validate that temperature configuration is compatible with approach parameters.

    Args:
        approach: Search approach ('bon', 'dvts', or 'beam_search')
        n: Total number of samples
        temperatures: List of temperature values
        beam_width: Beam width for DVTS (required for dvts approach)

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid or incompatible
    """
    approach = approach.lower()

    if approach == "bon":
        # For BoN, n must be evenly divisible by number of temperatures
        if n % len(temperatures) != 0:
            raise ValueError(
                f"For BoN approach, n ({n}) must be divisible by "
                f"number of temperatures ({len(temperatures)})"
            )

    elif approach == "dvts":
        if beam_width is None:
            raise ValueError("beam_width is required for DVTS approach")
        # For DVTS, beam_width should accommodate all temperatures
        # Temperature cycles if beam_width > len(temperatures)
        # This is valid, just a warning case
        if beam_width % len(temperatures) != 0:
            # Warning: temperatures will cycle unevenly
            pass

    elif approach == "beam_search":
        # For beam search, same validation as BoN (sequential grouping)
        if n % len(temperatures) != 0:
            raise ValueError(
                f"For beam_search approach, n ({n}) must be divisible by "
                f"number of temperatures ({len(temperatures)})"
            )

    else:
        raise ValueError(f"Unknown approach: {approach}")

    return True


def supports_temperature_analysis(approach: str) -> bool:
    """
    Check if the given approach supports temperature analysis.

    Args:
        approach: Search approach name

    Returns:
        True if approach supports temperature analysis, False otherwise

    Examples:
        >>> supports_temperature_analysis('bon')
        True
        >>> supports_temperature_analysis('dvts')
        True
        >>> supports_temperature_analysis('beam_search')
        True
    """
    return approach.lower() in ["bon", "dvts", "beam_search"]


def get_approach_from_dataset_name(dataset_name: str) -> str:
    """
    Extract approach name from dataset identifier.

    Args:
        dataset_name: Dataset name (e.g., 'hnc-bon', 'default-dvts')

    Returns:
        Approach name ('bon', 'dvts', or 'beam_search')

    Examples:
        >>> get_approach_from_dataset_name('hnc-bon')
        'bon'
        >>> get_approach_from_dataset_name('default-beam_search')
        'beam_search'
    """
    dataset_name = dataset_name.lower()
    if "bon" in dataset_name:
        return "bon"
    elif "beam_search" in dataset_name or "beam-search" in dataset_name:
        return "beam_search"
    elif "dvts" in dataset_name:
        return "dvts"
    else:
        raise ValueError(f"Cannot determine approach from dataset name: {dataset_name}")
