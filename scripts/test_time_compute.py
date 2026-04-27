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

import logging

import datasets
import torch
from datasets import load_dataset
from vllm import LLM

from sal.config import Config
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.hub import revision_exists
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score, score_pass_at_k

# from prm_toolkit import PrmConfig, PrmServer, load_prm_server

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


def main():
    datasets.disable_caching()

    parser = H4ArgumentParser(Config)
    config = parser.parse()

    logger.info("Config: %s", config)

    approach_fn = APPROACHES[config.approach]
    gen_revision = f"{config.revision}-gen"

    if config.push_to_hub and revision_exists(config.hub_dataset_id, gen_revision):
        logger.info(
            f"Found existing {gen_revision}; loading and skipping generation"
        )
        dataset = load_dataset(
            config.hub_dataset_id, revision=gen_revision, split="train"
        )
    else:
        dataset = get_dataset(config)
        num_gpus = torch.cuda.device_count()
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            max_model_len=config.max_model_len,
            seed=config.seed,
            tensor_parallel_size=num_gpus,
        )
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": llm},
            desc="Running search",
            load_from_cache_file=False,
        )
        save_dataset(dataset, config, suffix="-gen")

    dataset = score(dataset, config)
    dataset = score_pass_at_k(dataset, config)

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
