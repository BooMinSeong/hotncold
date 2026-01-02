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

        # Calculate number of continuations per diverse path
        continuations_per_path = config.n // config.beam_width

        # Get temperature assignment for continuations
        temp_config = copy.copy(config)
        temp_config.n = continuations_per_path
        temps = get_temperature_assignment(temp_config)

        # Prepare prompts and sampling params for each beam's diverse paths
        prompts = []
        sampling_params_list = []
        is_last_iteration = i == config.num_iterations - 1

        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        # For each beam, we'll generate beam_width diverse paths
        # Each diverse path will have continuations_per_path temperature variants
        # Total: len(gen_beams) * beam_width * continuations_per_path generations

        # First, generate initial diverse paths (using original method with beam_width)
        # We need to maintain diversity, so we'll use n=beam_width for the first step
        for beam in gen_beams:
            conv = build_conv(beam.prompt, beam.current_text, config.system_prompt)
            templated_conv = tokenizer.apply_chat_template(
                conv,
                add_generation_prompt=(i == 0),
                continue_final_message=(i > 0),
                tokenize=False,
            )

            # For each diverse path slot, generate continuations_per_path variants with different temps
            for _ in range(config.beam_width):
                for t in temps:
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

        # Generate all continuations
        outputs = llm.generate(prompts, sampling_params_list, use_tqdm=False)

        # Process outputs for each beam
        output_idx = 0
        for beam in gen_beams:
            # For this beam, we have beam_width diverse paths
            # Each path has continuations_per_path temperature variants
            diverse_paths_texts = []
            diverse_paths_stops = []

            for path_idx in range(config.beam_width):
                path_outputs = outputs[output_idx : output_idx + continuations_per_path]
                path_texts = [out.outputs[0].text for out in path_outputs]
                path_stops = [out.outputs[0].finish_reason for out in path_outputs]

                diverse_paths_texts.append(path_texts)
                diverse_paths_stops.append(path_stops)
                output_idx += continuations_per_path

            # Score all variants of all diverse paths
            prm_prompts = []
            prm_completions = []
            for path_texts in diverse_paths_texts:
                for text in path_texts:
                    prm_prompts.append(beam.prompt)
                    prm_completions.append([beam.current_text + text])

            path_scores = prm.score(prm_prompts, prm_completions)

            # Select best variant for each diverse path
            best_texts = []
            best_stops = []
            all_path_scores = []

            score_idx = 0
            for path_idx in range(config.beam_width):
                path_variant_scores = []
                for _ in range(continuations_per_path):
                    agg_score = aggregate_scores(
                        path_scores[score_idx][0], config.agg_strategy
                    )
                    path_variant_scores.append(agg_score)
                    score_idx += 1

                best_variant_idx = np.argmax(path_variant_scores)
                best_texts.append(diverse_paths_texts[path_idx][best_variant_idx])
                best_stops.append(diverse_paths_stops[path_idx][best_variant_idx])
                all_path_scores.append(
                    path_scores[path_idx * continuations_per_path + best_variant_idx][0]
                )

            # Store the best variants for each diverse path
            beam.next_texts = best_texts
            beam.stop_reasons = best_stops

            if len(beam.next_texts) != config.beam_width:
                beam.pruned = True
                logger.warning(
                    f"beam {beam.index} has {len(beam.next_texts)} completions"
                )

            # Select the overall best path among the diverse paths
            agg_scores = [
                aggregate_scores(s, config.agg_strategy) for s in all_path_scores
            ]
            best_score_ind = np.argmax(agg_scores)

            beam.all_scores = all_path_scores
            beam.previous_text = beam.current_text
            beam.current_text = beam.current_text + beam.next_texts[best_score_ind]
            beam.history.append(beam.next_texts[best_score_ind])
            beam.best_scores = all_path_scores[best_score_ind]

            if (
                beam.next_texts[best_score_ind] == ""
                or beam.stop_reasons[best_score_ind] == "stop"
            ):
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
