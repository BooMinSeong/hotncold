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

from .utils import Beam, build_conv

logger = logging.getLogger()


def _beam_search(batch_of_prompts, config: Config, llm: LLM, prm: PRM) -> list[Beam]:
    SamplingParams(
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

        # Get temperature assignment - each temperature will be used by multiple beams
        temps = get_temperature_assignment(config)
        # beam_width= 4

        # Calculate how many beams should use each temperature
        beams_per_temp = config.n // config.beam_width

        # Prepare prompts and sampling params for each beam and temperature
        prompts = []
        sampling_params_list = []
        is_last_iteration = i == config.num_iterations - 1

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        for beam_idx, beam in enumerate(active_beams):
            conv = build_conv(beam.prompt, beam.current_text, config.system_prompt)
            templated_conv = tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=(i == 0),
                continue_final_message=(i > 0),
                tokenize=False,
            )

            # Assign temperature based on beam group
            # Each temperature is used by beams_per_temp consecutive beams
            temp_group = beam_idx // beams_per_temp
            t = temps[temp_group]

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

        # Generate one continuation per beam (total = config.n)
        outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)

        # Assign outputs to beams and score with PRM
        prompts, completions = [], []
        for beam, output in zip(active_beams, outputs, strict=True):
            beam.next_texts = [output.outputs[0].text]
            beam.stop_reasons = [output.outputs[0].finish_reason]
            beam.completion_tokens += len(output.outputs[0].token_ids)
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.stop_reasons[0] == "length"
                or beam.next_texts[0] == ""
            ):
                beam.completed = True
                completed_beams.append(beam)
            #             if beam.stop_reasons[0] == "length" or beam.next_texts[0] == "":
            #                 beam.completed = True
            #                 completed_beams.append(beam)
            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        # PRM scoring
        scores = prm.score(prompts, completions)

        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]

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
