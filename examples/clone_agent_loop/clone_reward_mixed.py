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
Reward router for ToolAgentLoop clone rollouts across mixed datasets.
"""

from __future__ import annotations

import os
from typing import Any

from verl.utils.import_utils import load_extern_object

_THIS_DIR = os.path.dirname(__file__)
_ARITHMETIC_PATH = os.path.join(_THIS_DIR, "clone_reward_2.py")
_MATH_PATH = os.path.join(_THIS_DIR, "clone_reward_math.py")

arithmetic_clone_reward = load_extern_object(_ARITHMETIC_PATH, "clone_accuracy_reward")
math_clone_reward = load_extern_object(_MATH_PATH, "clone_accuracy_reward")


_MATH_DATA_SOURCE_HINTS = (
    "aime",
    "lighteval",
    "math",
)


def _resolve_data_source(root_output: Any) -> str:
    root_extra = getattr(root_output, "extra_fields", {}) or {}
    reward_model = root_extra.get("reward_model") or {}
    data_source = root_extra.get("data_source") or reward_model.get("data_source")
    if data_source is None:
        return ""
    return str(data_source)


def _is_math_source(data_source: str) -> bool:
    if not data_source:
        return False
    lowered = data_source.lower()
    return any(hint in lowered for hint in _MATH_DATA_SOURCE_HINTS) and "arithmetic" not in lowered


def clone_accuracy_reward(
    root_answer: str,
    root_output: Any,
    clone_rollouts: list[Any],
    root_text: str | None = None,
    clone_texts: dict[str, str] | None = None,
) -> dict[str, Any] | float:
    data_source = _resolve_data_source(root_output)
    reward_fn = math_clone_reward if _is_math_source(data_source) else arithmetic_clone_reward
    return reward_fn(
        root_answer,
        root_output,
        clone_rollouts,
        root_text=root_text,
        clone_texts=clone_texts,
    )


__all__ = ["clone_accuracy_reward"]
