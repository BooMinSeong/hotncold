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
import copy
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
from sal.utils.temperature import get_temperature_assignment

from .utils import Beam, build_conv, last

logger = logging.getLogger()


def _beam_search(batch_of_prompts, config: Config, llm: LLM, prm: PRM) -> list[Beam]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,  # New flag to track completion
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                )
            )

    completed_beams: list[Beam] = []

    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        # Calculate number of continuations per beam (search)
        continuations_per_beam = config.n // config.beam_width

        # Get temperature assignment for continuations
        temp_config = copy.copy(config)
        temp_config.n = continuations_per_beam
        temps = get_temperature_assignment(temp_config)

        # Prepare prompts and sampling params for each beam and temperature
        prompts = []
        sampling_params_list = []
        is_last_iteration = i == config.num_iterations - 1

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        for beam in active_beams:
            conv = build_conv(beam.prompt, beam.current_text, config.system_prompt)
            templated_conv = tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=(i == 0),
                continue_final_message=(i > 0),
                tokenize=False,
            )

            for t in temps:
                prompts.append(templated_conv)
                sampling_params_list.append(
                    SamplingParams(
                        temperature=t,
                        max_tokens=config.max_tokens,
                        top_p=config.top_p,
                        stop=["\n\n"] if not is_last_iteration else None,
                        include_stop_str_in_output=True,
                        n=1,
                    )
                )

        # Generate all continuations
        outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)

        # Group outputs by beam and assign to next_texts
        for beam_idx, beam in enumerate(active_beams):
            start_idx = beam_idx * continuations_per_beam
            end_idx = start_idx + continuations_per_beam
            beam_outputs = outputs[start_idx:end_idx]

            # Store all continuations for this beam
            beam.next_texts = [out.outputs[0].text for out in beam_outputs]
            beam.stop_reasons = [out.outputs[0].finish_reason for out in beam_outputs]
            beam.completion_tokens += sum(
                len(out.outputs[0].token_ids) for out in beam_outputs
            )

        # Score all continuations with PRM
        prm_prompts, prm_completions = [], []
        for beam in active_beams:
            for next_text in beam.next_texts:
                prm_prompts.append(beam.prompt)
                prm_completions.append([beam.current_text + next_text])

        scores = prm.score(prm_prompts, prm_completions)

        # Select best continuation for each beam
        score_idx = 0
        for beam in active_beams:
            beam_scores = []
            for _ in beam.next_texts:
                agg_score = aggregate_scores(scores[score_idx][0], config.agg_strategy)
                beam_scores.append(agg_score)
                score_idx += 1

            # Select best continuation
            best_idx = np.argmax(beam_scores)

            # Apply best continuation to beam
            beam.current_text += beam.next_texts[best_idx]
            beam.history.append(beam.next_texts[best_idx])
            beam.all_scores = scores[
                (active_beams.index(beam) * continuations_per_beam) + best_idx
            ][0]

            # Check if completed
            if (
                beam.stop_reasons[best_idx] in ["stop", "length"]
                or beam.next_texts[best_idx] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)

        # Filter out completed beams
        active_beams = [b for b in active_beams if not b.completed]

        # Early stopping if all beams are completed
        if len(active_beams) == 0:
            break

        # Filter duplicate active beams
        if config.filter_duplicates:
            # Create a dictionary to filter duplicates and retain order
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = (
                        i  # Map the unique text to its index
                    )
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]

        # Calculate aggregate scores for each beam based on their all_scores
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in active_beams
        ]

        # Get indices for top (config.n / config.beam_width) completions
        top_indices = np.argsort(agg_scores)[-(config.n // config.beam_width) :]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams


def beam_search(examples, config: Config, llm: LLM, prm: PRM):
    problems = examples["problem"]
    beam_results = _beam_search(problems, config, llm, prm)

    # Group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
