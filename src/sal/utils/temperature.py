#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sal.config import Config


def get_temperature_assignment(config: Config) -> list[float]:
    """
    Assign temperatures to config.beam_width samples based on configuration.

    Returns:
        List of temperatures assigned to each beam (length = config.beam_width)

    Examples:
        >>> config = Config(beam_width=6, temperatures=[0.5, 1.0, 1.5])
        >>> get_temperature_assignment(config)
        [0.5, 1.0, 1.5, 0.5, 1.0, 1.5]

        >>> config = Config(beam_width=8, temperatures=[0.5, 1.0], temperature_ratios=[0.25, 0.75])
        >>> get_temperature_assignment(config)
        [0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

        >>> config = Config(beam_width=4, temperature=0.8)
        >>> get_temperature_assignment(config)
        [0.8, 0.8, 0.8, 0.8]
    """
    # Legacy mode: single temperature
    if config.temperatures is None:
        return [config.temperature] * config.beam_width

    # Equal distribution with cyclic repetition
    if config.temperature_ratios is None:
        return [
            config.temperatures[i % len(config.temperatures)]
            for i in range(config.beam_width)
        ]

    # Custom ratios
    assignment = []
    for temp, ratio in zip(config.temperatures, config.temperature_ratios):
        count = int(config.beam_width * ratio)
        assignment.extend([temp] * count)

    # Handle rounding: add remaining to highest ratio
    while len(assignment) < config.beam_width:
        max_ratio_idx = config.temperature_ratios.index(max(config.temperature_ratios))
        assignment.insert(
            sum(
                int(config.beam_width * r)
                for r in config.temperature_ratios[: max_ratio_idx + 1]
            ),
            config.temperatures[max_ratio_idx],
        )

    return assignment
