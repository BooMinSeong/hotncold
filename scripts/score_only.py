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

"""Recovery script for stranded `-gen` checkpoints.

Loads a `<revision>-gen` branch pushed by `scripts/test_time_compute.py`,
runs scoring (`score()` + `score_pass_at_k()`), and pushes the final
`<revision>` branch. No GPU is used.

Reuses identical YAML/CLI as `scripts/test_time_compute.py` so that
`config.revision` is byte-for-byte identical (must pass the same
`--dataset_start`/`--dataset_end` as the original generation invocation).

If the final revision already exists on the Hub, `Config.__post_init__`
exits silently (idempotent).
"""

import logging

import datasets
from datasets import load_dataset

from sal.config import Config
from sal.utils.data import save_dataset
from sal.utils.hub import revision_exists
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score, score_pass_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def main():
    datasets.disable_caching()
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    assert config.push_to_hub, "score_only.py requires --push_to_hub=true"

    gen_revision = f"{config.revision}-gen"
    if not revision_exists(config.hub_dataset_id, gen_revision):
        raise RuntimeError(
            f"No -gen branch found at {config.hub_dataset_id}@{gen_revision}"
        )

    dataset = load_dataset(
        config.hub_dataset_id, revision=gen_revision, split="train"
    )
    dataset = score(dataset, config)
    dataset = score_pass_at_k(dataset, config)
    save_dataset(dataset, config)
    logger.info("Score-only recovery done 🔥!")


if __name__ == "__main__":
    main()
