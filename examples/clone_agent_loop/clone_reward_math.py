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

import math
import re

from verl.utils.reward_score.math_reward import is_equiv


def _coerce_ground_truth(ground_truth: Any) -> Any:
    if isinstance(ground_truth, list) and len(ground_truth) == 1:
        return ground_truth[0]
    return ground_truth


ROOT_TOKEN_PENALTY_THRESHOLD = 512
ROOT_TOKEN_PENALTY_RAMP = 256
ROOT_TOKEN_PENALTY_MAX = 0.3

CLONE_TOKEN_PENALTY_THRESHOLD = 512
CLONE_TOKEN_PENALTY_RAMP = 512
CLONE_TOKEN_PENALTY_MAX = 0.2

CLONE_ANSWER_FORMAT_MAX_TOKENS = 192
CLONE_ANSWER_FORMAT_PENALTY = 0.05

_INT_PATTERN = re.compile(r"-?\d[\d,]*")
_SIMPLE_INT_PATTERN = re.compile(r"^-?\d[\d,]*$")
_CLONE_TURN_PATTERN = re.compile(r"turn(\d+)")
_ANSWER_PREFIXES = ("answer:", "final answer:", "result:", "output:", "#### ")
_BOX_TOKENS = ("\\boxed", "\\fbox")
_LATEX_NOISE = ("\\left", "\\right", "\\,", "\\!", "\\;", "\\:", "\\quad", "\\qquad")


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


def _strip_answer_prefix(text: str) -> str:
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in _ANSWER_PREFIXES:
        if lowered.startswith(prefix):
            return stripped[len(prefix) :].strip()
    return stripped


def _extract_boxed(text: str) -> str | None:
    if not text:
        return None
    start = 0
    last_answer = None
    while start < len(text):
        next_pos = None
        next_token = None
        for token in _BOX_TOKENS:
            pos = text.find(token, start)
            if pos != -1 and (next_pos is None or pos < next_pos):
                next_pos = pos
                next_token = token
        if next_pos is None or next_token is None:
            break
        cursor = next_pos + len(next_token)
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            start = next_pos + len(next_token)
            continue
        depth = 0
        for i in range(cursor, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_answer = text[cursor + 1 : i]
                    start = i + 1
                    break
        else:
            break
    return last_answer


def _strip_answer_wrappers(text: str) -> str:
    if not text:
        return ""
    stripped = _strip_answer_prefix(text)
    boxed = _extract_boxed(stripped)
    if boxed is not None:
        stripped = boxed.strip()
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) > 1:
        stripped = stripped[1:-1].strip()
    return stripped


def _parse_simple_int(text: str) -> int | None:
    cleaned = _strip_answer_wrappers(text).replace(" ", "")
    if not cleaned or not _SIMPLE_INT_PATTERN.match(cleaned):
        return None
    try:
        return int(cleaned.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _normalize_answer_text(text: str) -> str:
    cleaned = _strip_answer_wrappers(text)
    for token in _LATEX_NOISE:
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace("\\cdot", "*").replace("\\times", "*")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace(" ", "")
    return cleaned.lower()


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

    gt_int = _parse_simple_int(str(ground_truth))
    root_int = _extract_int(str(root_answer)) if gt_int is not None else None
    if root_int is not None and gt_int is not None:
        A, B = root_int, gt_int
        # score = 1.0 if A == B else 0.2 / math.log2(2 + abs(A - B) / abs(2 + max(A, B)) ** 0.5)
        score = 1.0 if A == B else 0.0
    else:
        # Fallback: exact/substring match (comma-insensitive)
        # root_norm = _normalize_answer_text(str(root_answer))
        # gt_norm = _normalize_answer_text(str(ground_truth))
        # score = 0.1 if gt_norm and (gt_norm == root_norm or gt_norm in root_norm) else 0.0
        score = 1.0 if is_equiv(str(root_answer), str(ground_truth)) else 0.0

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

    total_penalty = token_penalty + turn_penalty_max
    # if root_text:
    #     total_penalty += max(0.0, root_text.count("<tool_call>") * 0.05 - 0.15)
    score = max(0, score - min(total_penalty, 0.5))

    if clone_rollouts:
        for clone, request_id, turn_key, tokens in clone_info:
            clone_reward = score / 2.5
            if format_penalty > 0 and request_id in clone_texts:
                clone_text = clone_texts[request_id] or ""
                has_prefix = answer_prefix.lower() in clone_text.lower()
                if not has_prefix and tokens > format_max_tokens:
                    # Malformed clone: do not grant positive reward.
                    clone_reward = min(0.0, score)
                    # clone_reward = 0.0 # N.B. either appears okay
            if request_id is not None:
                clone_scores[request_id] = clone_reward

    # clone_penalty_cfg = root_extra.get("clone_token_penalty") or reward_model.get("clone_token_penalty") or {
    #     "threshold": CLONE_TOKEN_PENALTY_THRESHOLD,
    #     "ramp": CLONE_TOKEN_PENALTY_RAMP,
    #     "max_penalty": CLONE_TOKEN_PENALTY_MAX,
    # }

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
