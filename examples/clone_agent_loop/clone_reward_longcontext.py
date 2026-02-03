# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""
Drop-in reward function for ToolAgentLoop clone rollouts.

The agent loop now propagates dataset metadata (data_source, reward_model)
into `AgentLoopOutput.extra_fields`, so this function can fetch the ground-truth
answer that was stored in the dataset row.
"""

from __future__ import annotations

from typing import Any

import atexit
import json
import math
import os
import re
import signal
import threading
import uuid


def _coerce_ground_truth(ground_truth: Any) -> Any:
    if isinstance(ground_truth, list) and len(ground_truth) == 1:
        return ground_truth[0]
    return ground_truth


def _normalize_answer_text(value: Any) -> str:
    return str(value).strip().lower()


ROOT_TOKEN_PENALTY_THRESHOLD = 512
ROOT_TOKEN_PENALTY_RAMP = 256
ROOT_TOKEN_PENALTY_MAX = 0.3

CLONE_TOKEN_PENALTY_THRESHOLD = 512
CLONE_TOKEN_PENALTY_RAMP = 512
CLONE_TOKEN_PENALTY_MAX = 0.2

CLONE_ANSWER_FORMAT_MAX_TOKENS = 192
CLONE_ANSWER_FORMAT_PENALTY = 0.05

_INT_PATTERN = re.compile(r"-?\d[\d,]*")
_CLONE_TURN_PATTERN = re.compile(r"turn(\d+)")

_REWARD_STATS: list[dict[str, Any]] = []
_REWARD_STATS_DUMP_PATH: str | None = None
_DUMP_HANDLERS_REGISTERED = False
_FLUSH_EVERY = int(os.environ.get("VERL_CLONE_REWARD_STATS_FLUSH_EVERY", "50"))
_PREV_SIGNAL_HANDLERS: dict[int, Any] = {}


def _sum_mask_tokens(mask: Any) -> int:
    if mask is None:
        return 0
    if hasattr(mask, "sum"):
        try:
            mask_sum = mask.sum()
            return int(mask_sum.item()) if hasattr(mask_sum, "item") else int(mask_sum)
        except TypeError:
            pass
    return int(sum(int(value) for value in mask))


def _count_prompt_tokens(output: Any) -> int:
    prompt_ids = getattr(output, "prompt_ids", None)
    return len(prompt_ids) if prompt_ids is not None else 0


def _count_completion_tokens(output: Any) -> int:
    response_mask = getattr(output, "response_mask", None)
    if response_mask:
        return _sum_mask_tokens(response_mask)
    response_ids = getattr(output, "response_ids", None)
    return len(response_ids) if response_ids is not None else 0


def _root_token_penalty(root_generated_tokens: int, threshold: int, ramp: int, max_penalty: float) -> float:
    if root_generated_tokens <= threshold:
        return 0.0
    if ramp <= 0:
        return float(max_penalty)
    excess = float(root_generated_tokens - threshold)
    return float(max_penalty) * (1.0 - math.exp(-excess / float(ramp)))


def _extract_int(text: str) -> int | None:
    if not text:
        return None
    matches = _INT_PATTERN.findall(text)
    if not matches:
        return None
    candidate = matches[-1].replace(",", "")
    try:
        return int(candidate)
    except (TypeError, ValueError):
        return None


def _clone_turn_index(label: str | None) -> int | None:
    if not label:
        return None
    match = _CLONE_TURN_PATTERN.search(label)
    if match is None:
        return None
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return None


def clone_accuracy_reward(
    root_answer: str,
    root_output: Any,
    clone_rollouts: list[Any],
    root_text: str | None = None,
    clone_texts: dict[str, str] | None = None,
) -> dict[str, Any] | float:
    """
    Compute a single scalar reward for the root + all clones.

    Args:
        root_answer: decoded root response text with chain-of-thought stripped.
        clone_rollouts: list of AgentLoopOutput objects returned by spawn_clone.
        metadata: dict passed by ToolAgentLoop._assign_rewards. Contains:
            - root_extra: extra_fields from the root rollout (now includes
              reward_model/data_source/extra_info from the dataset row).
            - num_clones: number of clone rollouts.

    Returns:
        Either a float or a dict with a `reward` key; any extra keys will be
        recorded in reward_extra_info.
    """

    root_extra = root_output.extra_fields
    reward_model = root_extra.get("reward_model") or {}
    data_source = root_extra.get("data_source") or reward_model.get("data_source")
    ground_truth = _coerce_ground_truth(reward_model.get("ground_truth"))

    if ground_truth is None:
        return {"reward": 0.0, "clone_rewards": [], "reason": "missing_ground_truth"}

    root_answer_norm = _normalize_answer_text(root_answer)
    ground_truth_norm = _normalize_answer_text(ground_truth)

    if data_source == "musique":
        extra_info = root_extra.get("extra_info") or {}
        answer_aliases = extra_info["answer_aliases"]
        if not isinstance(answer_aliases, list):
            raise TypeError(f"answer_aliases must be a list, got {type(answer_aliases)}")
        alias_norms = [_normalize_answer_text(alias) for alias in answer_aliases]
        valid_answers = {ground_truth_norm, *alias_norms}
        score = 1.0 if root_answer_norm in valid_answers else 0.0
    else:
        score = 1.0 if root_answer_norm == ground_truth_norm else 0.0

    penalty_cfg = root_extra.get("root_token_penalty") or reward_model.get("root_token_penalty") or {
        "threshold": 512, "ramp": 256, "max_penalty": 0.3
    }
    threshold = int(penalty_cfg.get("threshold", ROOT_TOKEN_PENALTY_THRESHOLD))
    ramp = int(penalty_cfg.get("ramp", ROOT_TOKEN_PENALTY_RAMP))
    max_penalty = float(penalty_cfg.get("max_penalty", ROOT_TOKEN_PENALTY_MAX))
    root_generated_tokens = _sum_mask_tokens(getattr(root_output, "response_mask", None))
    token_penalty = _root_token_penalty(root_generated_tokens, threshold, ramp, max_penalty)

    # NEW: Evaluate each clone
    clone_scores = {}
    turn_penalty_max = 0.0
    if clone_rollouts:
        clone_texts = clone_texts or {}

        format_cfg = root_extra.get("clone_answer_format_penalty") or reward_model.get("clone_answer_format_penalty") or {
            "max_tokens": CLONE_ANSWER_FORMAT_MAX_TOKENS,
            "penalty": CLONE_ANSWER_FORMAT_PENALTY,
            "answer_prefix": "Answer:",
        }
        format_max_tokens = int(format_cfg.get("max_tokens", CLONE_ANSWER_FORMAT_MAX_TOKENS))
        format_penalty = float(format_cfg.get("penalty", CLONE_ANSWER_FORMAT_PENALTY))
        answer_prefix = str(format_cfg.get("answer_prefix", "Answer:"))

        clone_penalty_cfg = root_extra.get("clone_token_penalty") or reward_model.get("clone_token_penalty") or {
            "threshold": CLONE_TOKEN_PENALTY_THRESHOLD,
            "ramp": CLONE_TOKEN_PENALTY_RAMP,
            "max_penalty": CLONE_TOKEN_PENALTY_MAX,
        }
        clone_threshold = int(clone_penalty_cfg.get("threshold", CLONE_TOKEN_PENALTY_THRESHOLD))
        clone_ramp = int(clone_penalty_cfg.get("ramp", CLONE_TOKEN_PENALTY_RAMP))
        clone_max_penalty = float(clone_penalty_cfg.get("max_penalty", CLONE_TOKEN_PENALTY_MAX))

        turn_max_tokens: dict[Any, int] = {}
        clone_info = []
        for clone in clone_rollouts:
            extra = getattr(clone, "extra_fields", {}) or {}
            request_id = extra.get("request_id")
            label = extra.get("clone_label") or extra.get("clone_id") or ""
            turn_index = _clone_turn_index(label)
            turn_key = turn_index if turn_index is not None else label or request_id or id(clone)
            tokens = _sum_mask_tokens(getattr(clone, "response_mask", None))
            prev = turn_max_tokens.get(turn_key, 0)
            if tokens > prev:
                turn_max_tokens[turn_key] = tokens
            clone_info.append((clone, request_id, turn_key, tokens))

        turn_penalties: dict[Any, float] = {}
        if clone_max_penalty > 0:
            for turn_key, max_tokens in turn_max_tokens.items():
                turn_penalties[turn_key] = _root_token_penalty(
                    int(max_tokens), clone_threshold, clone_ramp, clone_max_penalty
                )
        if turn_penalties:
            turn_penalty_max = max(turn_penalties.values())

    score -= token_penalty
    score -= turn_penalty_max
    if root_text:
        score -= max(0.0, root_text.count("<tool_call>") * 0.1 - 0.2)

    if clone_rollouts:
        for clone, request_id, turn_key, tokens in clone_info:
            clone_reward = score
            if format_penalty > 0 and request_id in clone_texts:
                clone_text = clone_texts[request_id] or ""
                has_prefix = answer_prefix.lower() in clone_text.lower()
                if not has_prefix and tokens > format_max_tokens:
                    # Malformed clone: do not grant positive reward.
                    # clone_reward = min(0.0, score)
                    clone_reward = 0.0 # N.B. either appears okay
            if request_id is not None:
                clone_scores[request_id] = clone_reward

    return {
        "reward": score,
        "clone_rewards": clone_scores,
        "score": score,
        "num_clones": len(clone_rollouts),
        # "root_generated_tokens": root_generated_tokens,
        # "root_token_penalty": token_penalty,
        # "root_token_penalty_threshold": threshold,
        # "root_token_penalty_ramp": ramp,
        "ground_truth": ground_truth,
    }


__all__ = ["clone_accuracy_reward"]
