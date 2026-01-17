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

import json
import logging
import time
from itertools import accumulate
from pathlib import Path

import httpx
import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sal.config import Config

logger = logging.getLogger(__name__)
from sal.models.skywork_o1_prm.io_utils import (
    derive_step_rewards,
    prepare_batch_input_for_model,
    prepare_input,
)
from sal.models.skywork_o1_prm.prm_model import SkyworkPRMModel

CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


def batched_math_shepherd_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: list[str],
    batch_size: int,
) -> list[list[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, CANDIDATE_TOKENS]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores_flat = scores[inputs_batch.input_ids == STEP_TAG_ID].tolist()
            # Split scores into sublist based on number of \n in the input
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(STEP_TAG_ID)
                step_scores.append(step_scores_flat[counter : counter + count])
                counter += count

        # Store the step scores for this batch
        output_scores.extend(step_scores)

    return output_scores


class PRM:
    def __init__(self, search_config: Config, **model_kwargs):
        self.search_config = search_config
        self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError

    def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass


class MathShepherd(PRM):
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # For batched inference
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.search_config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_math_shepherd_inference(
            self.model,
            self.tokenizer,
            inputs_for_prm,
            self.search_config.prm_batch_size,
        )
        cumulative_lengths = list(accumulate(lengths))
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(output), (
                f"{len(output_score)} != {len(output)}"
            )

        return output_scores


class RLHFFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = True,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(
                questions, outputs, batch_size=self.search_config.prm_batch_size
            )
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to(self.model.device)
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        step_scores = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        # print(scores)
                        single_step_score.append(
                            step_scores[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        inputs2_batch[i, 1:] == special_tok_id
                    ].tolist()
                    output_scores.append(step_scores_flat)

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores


class SkyworkO1(PRM):
    @classmethod
    def _load_model_and_tokenizer(
        cls, prm_model_path, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            prm_model_path, trust_remote_code=True
        )
        model = SkyworkPRMModel.from_pretrained(
            prm_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()

        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        # reference code: https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B#huggingface-inference
        # Prepare all inputs
        all_processed_data = []
        lengths = []
        for question, answers in zip(questions, outputs):
            for answer in answers:
                processed = prepare_input(
                    question, answer, tokenizer=self.tokenizer, step_token="\n"
                )
                all_processed_data.append(processed)
            lengths.append(len(answers))

        # Mini-batch processing
        all_step_scores = []
        batch_size = self.search_config.prm_batch_size
        device = self.model.pretrained_model.device
        for i in range(0, len(all_processed_data), batch_size):
            batch_data = all_processed_data[i : i + batch_size]
            input_ids, steps, reward_flags = zip(*batch_data)
            input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(
                input_ids, reward_flags, self.tokenizer.pad_token_id
            )
            with torch.no_grad():
                _, _, rewards = self.model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    return_probs=True,
                )
                batch_scores = derive_step_rewards(
                    rewards.detach().to("cpu", dtype=torch.float32), reward_flags
                )
                all_step_scores.extend(batch_scores)

        # Reshape to match input structure
        all_scores = []
        cumulative_lengths = list(accumulate(lengths))
        for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths):
            all_scores.append(all_step_scores[i:j])

        return all_scores


class Qwen_2_5_Math(PRM):
    @classmethod
    def _load_model_and_tokenizer(
        cls, prm_model_path, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            prm_model_path, trust_remote_code=True
        )
        model = AutoModel.from_pretrained(
            prm_model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **model_kwargs,
        ).eval()

        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        # Prepare all inputs
        all_processed_responses = []
        lengths = []
        for question, answers in zip(questions, outputs):
            for answer in answers:
                messages = [
                    {
                        "role": "system",
                        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    },
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": answer.replace("\n\n", "<extra_0>") + "<extra_0>",
                    },
                ]
                conversation_str = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                all_processed_responses.append(conversation_str)
            lengths.append(len(answers))

        # Mini-batch processing
        all_step_rewards = []
        batch_size = self.search_config.prm_batch_size
        step_sep_id = self.tokenizer.encode("<extra_0>")[0]

        for i in range(0, len(all_processed_responses), batch_size):
            batch_responses = all_processed_responses[i : i + batch_size]
            input_ids = self.tokenizer(
                batch_responses, return_tensors="pt", padding=True, truncation=True
            )["input_ids"].to(self.model.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)

            token_masks = input_ids == step_sep_id
            batch_rewards = self.make_step_rewards(outputs[0], token_masks)
            all_step_rewards.extend(batch_rewards)

        # Reshape to match input structure
        all_scores = []
        cumulative_lengths = list(accumulate(lengths))
        for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths):
            all_scores.append(all_step_rewards[i:j])

        return all_scores

    @staticmethod
    def make_step_rewards(logits, token_masks):
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(
            -1
        )  # bs, seq_len, num_labels

        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]  # seq_len, num_labels
            positive_probs = sample[sample != 0].view(-1, 2)[
                :, 1
            ]  # valid_tokens, num_labels
            all_scores_res.append(positive_probs.cpu().tolist())

        return all_scores_res


class SkyworkO1_1_5B(SkyworkO1):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        prm_model_path = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
        return SkyworkO1._load_model_and_tokenizer(prm_model_path, **model_kwargs)


class SkyworkO1_7B(SkyworkO1):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        prm_model_path = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
        return SkyworkO1._load_model_and_tokenizer(prm_model_path, **model_kwargs)


class Qwen_2_5_Math_7B(Qwen_2_5_Math):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        prm_model_path = "Qwen/Qwen2.5-Math-PRM-7B"
        return Qwen_2_5_Math._load_model_and_tokenizer(prm_model_path, **model_kwargs)


class VLLMRewardBase:
    """Base class for vLLM-based reward models."""

    def __init__(self, search_config: Config):
        self.search_config = search_config
        self._tokenizer = None

    def _prepare_inputs_qwen(
        self, questions: list[str], outputs: list[list[str]]
    ) -> tuple[list[str], list[int]]:
        """
        Prepare formatted inputs for Qwen PRM model.
        Returns (formatted_texts, lengths) where lengths tracks
        how many completions per question for reshaping outputs.
        """
        formatted_texts = []
        lengths = []

        for question, answers in zip(questions, outputs):
            for answer in answers:
                messages = [
                    {
                        "role": "system",
                        "content": "Please reason step by step, and put your final answer within \\boxed{}.",
                    },
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": answer.replace("\n\n", "<extra_0>") + "<extra_0>",
                    },
                ]
                formatted_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                formatted_texts.append(formatted_text)
            lengths.append(len(answers))

        return formatted_texts, lengths

    def _reshape_scores(
        self, flat_scores: list[list[float]], lengths: list[int]
    ) -> list[list[list[float]]]:
        """Reshape flat scores back to [questions][completions][steps] structure."""
        result = []
        idx = 0
        for length in lengths:
            result.append(flat_scores[idx : idx + length])
            idx += length
        return result

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[list[float]]]:
        raise NotImplementedError

    def close(self) -> None:
        """Clean up resources. Override in subclasses if needed."""
        pass


class VLLMRewardOffline(VLLMRewardBase):
    """
    vLLM-based PRM using local offline inference.
    Designed for dual-GPU setups where LLM and PRM run on separate GPUs.
    """

    def __init__(self, search_config: Config, **model_kwargs):
        super().__init__(search_config)
        self._load_model(**model_kwargs)

    def _load_model(self, **model_kwargs):
        from vllm import LLM

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.search_config.prm_path, trust_remote_code=True
        )

        # Configure vLLM for reward model
        self.llm = LLM(
            model=self.search_config.prm_path,
            task="reward",
            gpu_memory_utilization=self.search_config.prm_gpu_memory_utilization,
            tensor_parallel_size=self.search_config.prm_tensor_parallel_size,
            trust_remote_code=True,
            dtype="bfloat16",
            **model_kwargs,
        )

        self.step_sep_id = self._tokenizer.encode("<extra_0>")[0]

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[list[float]]]:
        """Score using vLLM offline inference."""
        formatted_texts, lengths = self._prepare_inputs_qwen(questions, outputs)

        # Use vLLM's reward scoring
        results = self.llm.encode(formatted_texts)

        # Extract step-level scores
        flat_scores = self._extract_step_scores(results, formatted_texts)

        return self._reshape_scores(flat_scores, lengths)

    def _extract_step_scores(
        self, results, formatted_texts: list[str]
    ) -> list[list[float]]:
        """
        Extract step-level scores from vLLM encode results.
        Model-specific logic for extracting scores at step separator positions.
        """
        step_scores = []

        for result, text in zip(results, formatted_texts):
            input_ids = self._tokenizer.encode(text)
            step_positions = [
                i for i, tid in enumerate(input_ids) if tid == self.step_sep_id
            ]

            if hasattr(result.outputs, "data"):
                scores_tensor = result.outputs.data
                if scores_tensor.dim() == 2:
                    # Token classification output: [num_tokens, num_classes]
                    # Use softmax and take positive class probability
                    probs = F.softmax(scores_tensor, dim=-1)
                    scores = [
                        float(probs[pos, 1]) if pos < len(probs) else 0.0
                        for pos in step_positions
                    ]
                else:
                    # Single score per position
                    scores = [
                        float(scores_tensor[pos]) if pos < len(scores_tensor) else 0.0
                        for pos in step_positions
                    ]
                step_scores.append(scores)
            else:
                # Fallback: use uniform scores
                step_scores.append([1.0] * len(step_positions))

        return step_scores

    def close(self):
        """Clean up vLLM resources."""
        if hasattr(self, "llm"):
            del self.llm
            torch.cuda.empty_cache()


class VLLMRewardAPI(VLLMRewardBase):
    """
    vLLM-based PRM using remote API server.
    Designed for cluster setups where PRM runs as a separate job.
    """

    def __init__(self, search_config: Config, **model_kwargs):
        super().__init__(search_config)
        self._setup_client()
        self._load_tokenizer()

    def _setup_client(self):
        """Initialize HTTP client and resolve service endpoint."""
        self.base_urls = self._resolve_service_urls()
        self.clients = [
            httpx.Client(
                base_url=url,
                timeout=self.search_config.prm_api_timeout,
            )
            for url in self.base_urls
        ]
        self._wait_for_servers()

    def _resolve_service_urls(self) -> list[str]:
        """Resolve API URLs from config or service discovery file."""
        if self.search_config.prm_api_base_urls:
            return self.search_config.prm_api_base_urls

        if self.search_config.prm_api_base_url:
            return [self.search_config.prm_api_base_url]

        # Service discovery from file
        service_file = Path(self.search_config.prm_service_file)
        max_wait = 300  # 5 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            if service_file.exists():
                with open(service_file) as f:
                    service_info = json.load(f)
                return [f"http://{service_info['host']}:{service_info['port']}"]
            logger.info(f"Waiting for service file: {service_file}")
            time.sleep(5)

        raise TimeoutError(
            f"Service discovery file {service_file} not found after {max_wait}s"
        )

    def _wait_for_servers(self, timeout: float = 300.0):
        """Wait for vLLM servers to be ready."""
        start_time = time.time()

        for i, (client, url) in enumerate(zip(self.clients, self.base_urls)):
            while time.time() - start_time < timeout:
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        logger.info(f"vLLM server {i} at {url} is ready")
                        break
                except httpx.RequestError:
                    pass
                time.sleep(5)
            else:
                raise TimeoutError(f"vLLM server at {url} not ready after {timeout}s")

    def _load_tokenizer(self):
        """Load tokenizer for input preparation."""
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.search_config.prm_path, trust_remote_code=True
        )
        self.step_sep_id = self._tokenizer.encode("<extra_0>")[0]

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[list[float]]]:
        """Score using vLLM API server."""
        formatted_texts, lengths = self._prepare_inputs_qwen(questions, outputs)

        # Batch requests to API
        flat_scores = []
        batch_size = self.search_config.prm_batch_size

        for i in range(0, len(formatted_texts), batch_size):
            batch = formatted_texts[i : i + batch_size]
            batch_scores = self._score_batch(batch)
            flat_scores.extend(batch_scores)

        return self._reshape_scores(flat_scores, lengths)

    def _score_batch(self, texts: list[str]) -> list[list[float]]:
        """Send batch to API and extract step scores."""
        # Use round-robin client selection for load balancing
        client = self.clients[hash(texts[0]) % len(self.clients)]

        request_data = {
            "model": self.search_config.prm_path,
            "input": texts,
        }

        for attempt in range(self.search_config.prm_api_max_retries):
            try:
                response = client.post(
                    "/pooling",
                    json=request_data,
                )
                response.raise_for_status()
                result = response.json()

                return self._parse_api_response(result, texts)

            except httpx.RequestError as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt == self.search_config.prm_api_max_retries - 1:
                    raise
                time.sleep(2**attempt)  # Exponential backoff

        return [[0.0] for _ in texts]

    def _parse_api_response(
        self, result: dict, texts: list[str]
    ) -> list[list[float]]:
        """Parse API response to extract step-level scores."""
        step_scores = []

        for i, text in enumerate(texts):
            input_ids = self._tokenizer.encode(text)
            step_positions = [
                j for j, tid in enumerate(input_ids) if tid == self.step_sep_id
            ]

            # Extract scores from API response
            if "data" in result and i < len(result["data"]):
                scores_data = result["data"][i]
                if isinstance(scores_data, dict):
                    if "scores" in scores_data:
                        # Token-level scores
                        all_scores = scores_data["scores"]
                        step_scores.append(
                            [
                                all_scores[pos] if pos < len(all_scores) else 0.0
                                for pos in step_positions
                            ]
                        )
                    elif "score" in scores_data:
                        # Sequence-level score - replicate for all steps
                        seq_score = scores_data["score"]
                        step_scores.append([seq_score] * len(step_positions))
                    else:
                        step_scores.append([0.0] * len(step_positions))
                elif isinstance(scores_data, (list, tuple)):
                    # Direct token scores
                    step_scores.append(
                        [
                            scores_data[pos] if pos < len(scores_data) else 0.0
                            for pos in step_positions
                        ]
                    )
                else:
                    step_scores.append([float(scores_data)] * len(step_positions))
            else:
                step_scores.append([0.0] * len(step_positions))

        return step_scores

    def close(self):
        """Close HTTP clients."""
        for client in self.clients:
            client.close()


def load_prm(config: Config) -> PRM | VLLMRewardBase:
    """
    Load PRM based on configuration.
    Supports three backends: transformers, vllm_offline, vllm_api
    """
    # Route based on backend selection
    if config.prm_backend == "vllm_offline":
        logger.info(f"Loading PRM with vLLM offline backend: {config.prm_path}")
        return VLLMRewardOffline(config)

    if config.prm_backend == "vllm_api":
        logger.info(f"Loading PRM with vLLM API backend: {config.prm_path}")
        return VLLMRewardAPI(config)

    # Default: Transformers-based implementation (backward compatible)
    logger.info(f"Loading PRM with transformers backend: {config.prm_path}")

    if config.prm_path == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if config.prm_path == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        return RLHFFlow(config)

    if config.prm_path == "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B":
        return SkyworkO1_1_5B(config)

    if config.prm_path == "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B":
        return SkyworkO1_7B(config)

    if config.prm_path == "Qwen/Qwen2.5-Math-PRM-7B":
        return Qwen_2_5_Math_7B(config)

    raise NotImplementedError(f"PRM {config.prm_path} not implemented")
