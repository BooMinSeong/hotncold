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
import random
import time
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import (
    create_branch,
    list_repo_commits,
    repo_exists,
)

from sal.config import Config

logger = logging.getLogger()


def get_dataset(config: Config) -> Dataset:
    dataset = load_dataset(config.dataset_name, split=config.dataset_split)

    if config.dataset_start is not None and config.dataset_end is not None:
        dataset_end = min(config.dataset_end, len(dataset))
        dataset = dataset.select(range(config.dataset_start, dataset_end))
    if config.num_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.num_samples)))

    return dataset


def save_dataset(dataset, config, suffix: str = ""):
    revision = f"{config.revision}{suffix}"
    if config.push_to_hub:
        # Concurrent pushes get rejected by HF free-tier rate limits.
        # Use exponential backoff with jitter; fall back to local jsonl if all retries fail.
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(config.hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        config.hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=config.hub_dataset_id,
                        branch=revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    config.hub_dataset_id,
                    revision=revision,
                    split="train",
                    private=config.hub_dataset_private,
                    commit_message=f"Add {revision}",
                )
                logger.info(f"Pushed dataset to {url}")
                return
            except Exception as e:
                # cap at 120s; jitter to spread out concurrent retries
                base = min(120, 5 * (2 ** attempt))
                sleep_s = base * random.uniform(0.7, 1.3)
                logger.error(
                    f"Error pushing dataset (attempt {attempt+1}/{max_attempts}): {e}; sleeping {sleep_s:.1f}s"
                )
                time.sleep(sleep_s)
        # Fallback: persist generation/score results locally so they aren't lost.
        fallback_dir = Path("data/failed-push") / config.hub_dataset_id.replace("/", "_")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / f"{revision}.jsonl"
        dataset.to_json(str(fallback_path), lines=True)
        logger.warning(
            f"All {max_attempts} push attempts failed for {revision}; saved locally to {fallback_path}"
        )
    else:
        if config.output_dir is None:
            config.output_dir = f"data/{config.model_path}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        filename = f"{config.approach}_completions{suffix}.jsonl"
        dataset.to_json(f"{config.output_dir}/{filename}", lines=True)
        logger.info(f"Saved completions to {config.output_dir}/{filename}")
