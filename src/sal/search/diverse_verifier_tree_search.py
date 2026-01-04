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


def _dvts(batch_of_prompts: list[str], config: Config, llm: LLM, prm: PRM):
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=2048,
        top_p=config.top_p,
        stop=[
            "\n\n"
        ],  # we consider that a step in the problem is indicated by a double newline
        include_stop_str_in_output=True,
        n=1,
    )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n_beams):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    best_scores=[0.0],
                    all_scores=[],
                    previous_text=None,
                    pruned=False,
                    stop_reasons=None,
                    history=[],
                )
            )

    for i in tqdm(range(config.num_iterations), desc="Beam search iterations"):
        # generation
        gen_beams = [b for b in beams if not b.pruned]
        if len(gen_beams) == 0:
            break

        # Get temperature assignment for diverse paths
        # Each beam generates beam_width diverse paths
        temp_config = copy.copy(config)
        temp_config.n = config.beam_width
        temps = get_temperature_assignment(temp_config)

        # Prepare prompts and sampling params for each beam's diverse paths
        prompts = []
        sampling_params_list = []
        is_last_iteration = i == config.num_iterations - 1

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        # For each beam, generate beam_width diverse paths
        # Each diverse path gets ONE temperature based on its path index
        # Total: len(gen_beams) * beam_width = config.n generations

        for beam in gen_beams:
            conv = build_conv(beam.prompt, beam.current_text, config.system_prompt)
            templated_conv = tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=(i == 0),
                continue_final_message=(i > 0),
                tokenize=False,
            )

            # Generate beam_width diverse paths with different temperatures
            for path_idx in range(config.beam_width):
                # Assign temperature based on path index
                # Each path gets a different temperature
                t = temps[path_idx]

                prompts.append(templated_conv)
                sampling_params_list.append(
                    SamplingParams(
                        temperature=t,
                        max_tokens=2048,
                        top_p=config.top_p,
                        stop=["\n\n"] if not is_last_iteration else None,
                        include_stop_str_in_output=True,
                        n=1,
                    )
                )

        # Generate all diverse paths (total = config.n)
        outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)

        # Process outputs for each beam
        prompts_prm = []
        completions_prm = []
        output_idx = 0

        for beam in gen_beams:
            # Extract beam_width diverse paths for this beam
            beam.next_texts = []
            beam.stop_reasons = []

            for path_idx in range(config.beam_width):
                output = outputs[output_idx]
                beam.next_texts.append(output.outputs[0].text)
                beam.stop_reasons.append(output.outputs[0].finish_reason)
                output_idx += 1

                # Prepare for PRM scoring
                prompts_prm.append(beam.prompt)
                completions_prm.append([beam.current_text + beam.next_texts[path_idx]])

            if len(beam.next_texts) != config.beam_width:
                beam.pruned = True
                logger.warning(
                    f"beam {beam.index} has {len(beam.next_texts)} completions"
                )

        # Score all diverse paths with PRM
        all_scores = prm.score(prompts_prm, completions_prm)

        # Assign scores and select best path for each beam
        score_idx = 0
        for beam in gen_beams:
            beam_scores = []
            for path_idx in range(config.beam_width):
                beam_scores.append(all_scores[score_idx][0])
                score_idx += 1

            # Select best path based on aggregate score
            agg_scores = [aggregate_scores(s, config.agg_strategy) for s in beam_scores]
            best_score_ind = np.argmax(agg_scores)

            beam.all_scores = beam_scores
            beam.previous_text = beam.current_text
            beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
            beam.history.append(beam.next_texts[best_score_ind])
            beam.best_scores = beam_scores[best_score_ind]

            if beam.next_texts[best_score_ind] == "":
                # stopped on EOS, prune
                beam.pruned = True

        # filter / prune
        for beam in gen_beams:
            if "boxed{" in beam.current_text:
                beam.pruned = True

    # we need to copy the results from the last iteration in to beam_width beams as otherwise we would only have n/m results
    output: list[Beam] = []
    for beam in beams:
        for i in range(config.beam_width):
            output.append(
                Beam(
                    prompt=beam.prompt,
                    index=beam.index,
                    current_text=beam.previous_text + beam.next_texts[i],
                    next_texts=None,
                    lookahead_texts=None,
                    stop_reasons=None,
                    best_scores=beam.all_scores[i],
                    all_scores=beam.all_scores,
                    previous_text=beam.current_text,
                    pruned=beam.pruned,
                    history=beam.history,
                )
            )

    return output


def dvts(examples, config: Config, llm: LLM, prm: PRM):
    problems = examples["problem"]
    beam_results = _dvts(problems, config, llm, prm)

    # group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        beams = grouped_results[p]
        results["completions"].append([b.current_text for b in beams])
        results["pred"].append(
            beams[
                np.argmax(
                    [
                        aggregate_scores(b.best_scores, config.agg_strategy)
                        for b in beams
                    ]
                )
            ].current_text
        )
        results["scores"].append([b.best_scores for b in beams])
        results["completion_tokens"].append(-1)

    # TODO: construct and store the tree

    return results
