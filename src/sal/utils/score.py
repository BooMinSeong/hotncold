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


import math
from typing import Literal

from datasets import Dataset

from sal.config import Config
from sal.utils.math import (
    compute_maj_pred,
    # compute_naive_pred,
    compute_pass_at_k,
    # compute_weighted_pred,
    extract_completion_answers,
    score_all_subsets,
    subsample_completions,
)


def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


def score(dataset: Dataset, config: Config) -> Dataset:
    subsets = [2**i for i in range(config.n) if 2**i <= config.n]
    dataset = dataset.map(
        score_all_subsets,
        fn_kwargs={"subsets": subsets},
        num_proc=config.num_proc,
        desc="Computing majority predictions",
    )
    return dataset


def score_pass_at_k(dataset: Dataset, config: Config) -> Dataset:
    dataset = dataset.map(
        extract_completion_answers,
        fn_kwargs={"n": None},
        num_proc=config.num_proc,
        desc=f"Extract answers for Pass@k",
    )

    subsets = [2**i for i in range(config.n) if 2**i <= config.n]
    for k in subsets:
        dataset = dataset.map(
            compute_pass_at_k,
            fn_kwargs={"k": k},
            num_proc=config.num_proc,
            desc=f"Compute Pass@{k}",
        )
    return dataset
