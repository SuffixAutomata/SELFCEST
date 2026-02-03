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
Preprocess the 2WikiMultihopQA dataset into parquet format for ToolAgentLoop.

Input format (as provided in this repo):
  - train.parquet, dev.parquet, test.parquet
  - context/supporting_facts/evidences are JSON strings
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import datasets

from verl.utils.hdfs_io import copy, makedirs


DEFAULT_SYSTEM_PROMPT = (
    "You are the root agent. Use the tool `spawn_clone` to delegate multi-hop reasoning. "
    "When sharing context with clones, pass only the context key(s) from the list; "
    "the agent loop expands keys into full paragraphs. "
    "Provide the final answer in the format `Answer: <text>`."
)


def _parse_json_field(value: Any, field_name: str) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse {field_name}: {exc}") from exc
    return value


def _normalize_supporting_facts(items: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        title = str(item[0]).strip()
        if not title:
            continue
        try:
            sentence_idx = int(item[1])
        except (TypeError, ValueError):
            sentence_idx = -1
        normalized.append({"title": title, "sentence_idx": sentence_idx})
    return normalized


def _normalize_evidences(items: Any) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in items or []:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        subj = str(item[0]).strip()
        rel = str(item[1]).strip()
        obj = str(item[2]).strip()
        if not subj and not rel and not obj:
            continue
        normalized.append({"subject": subj, "relation": rel, "object": obj})
    return normalized


def _build_context_items(context_items: Any) -> tuple[list[dict[str, str]], list[str]]:
    context_map: dict[str, str] = {}
    context_keys: list[str] = []
    for item in context_items or []:
        if not isinstance(item, list) or len(item) < 2:
            continue
        title = str(item[0]).strip()
        if not title:
            continue
        sentences = item[1] or []
        if isinstance(sentences, str):
            sentences = [sentences]
        if not isinstance(sentences, list):
            continue
        paragraph = " ".join(sentence.strip() for sentence in sentences if isinstance(sentence, str) and sentence.strip())
        if not paragraph:
            continue
        if title in context_map:
            context_map[title] = f"{context_map[title]}\n{paragraph}"
        else:
            context_map[title] = paragraph
            context_keys.append(title)
    context_entries = [{"key": key, "text": context_map[key]} for key in context_keys]
    return context_entries, context_keys


def _build_user_prompt(question: str, context_keys: list[str]) -> str:
    sections = ["Question:", question.strip()]
    if context_keys:
        sections.append("Context keys:")
        sections.extend(f"- {key}" for key in context_keys)
    return "\n".join(sections)


def make_map_fn(split: str, data_source: str, system_prompt: str):
    def process_fn(example, idx: int):
        question = (example.pop("question") or "").strip()
        answer = (example.pop("answer") or "").strip()
        if not question:
            raise ValueError("Missing question text")
        missing_answer = not answer
        if missing_answer and split != "test":
            raise ValueError("Missing answer text")
        if missing_answer:
            answer = ""
        context_raw = example.pop("context", None)
        supporting_raw = example.pop("supporting_facts", None)
        evidences_raw = example.pop("evidences", None)

        context_items = _parse_json_field(context_raw, "context")
        context_entries, context_keys = _build_context_items(context_items)
        supporting_facts = _parse_json_field(supporting_raw, "supporting_facts")
        evidences = _parse_json_field(evidences_raw, "evidences")
        supporting_facts = _normalize_supporting_facts(supporting_facts)
        evidences = _normalize_evidences(evidences)

        user_prompt = _build_user_prompt(question, context_keys)
        return {
            "data_source": data_source,
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "ability": "multihop_qa",
            "reward_model": {"style": "rule", "ground_truth": answer, "data_source": data_source},
            "extra_info": {
                "split": split,
                "index": idx,
                "source_id": example.pop("_id", None),
                "question": question,
                "answer": answer,
                "has_answer": not missing_answer,
                "context_keys": context_keys,
                "context_items": context_entries,
                "context_source": data_source,
                "question_type": example.pop("type", None),
                "supporting_facts": supporting_facts,
                "evidences": evidences,
            },
        }

    return process_fn


def _resolve_path(dataset_root: str, override: str | None, filename: str) -> str:
    if override:
        return override
    return os.path.join(dataset_root, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="<HOME_DIR>/xanhho-2WikiMultihopQA",
        help="Path to the 2WikiMultihopQA dataset root.",
    )
    parser.add_argument("--train_file", default=None, help="Override the train parquet file path.")
    parser.add_argument("--dev_file", default=None, help="Override the dev parquet file path.")
    parser.add_argument("--test_file", default=None, help="Override the test parquet file path.")
    parser.add_argument("--data_source", default="2WikiMultihopQA", help="Dataset identifier for reward scoring.")
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for the root agent.",
    )
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="<HOME_DIR>/xanhho-2WikiMultihopQA/preprocessed",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    splits = {
        "train": _resolve_path(args.dataset_root, args.train_file, "train.parquet"),
        "dev": _resolve_path(args.dataset_root, args.dev_file, "dev.parquet"),
        "test": _resolve_path(args.dataset_root, args.test_file, "test.parquet"),
    }

    datasets_out: dict[str, datasets.Dataset] = {}
    for split, path in splits.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected parquet file for {split}: {path}")
        dataset = datasets.load_dataset("parquet", data_files=path)["train"]
        datasets_out[split] = dataset.map(
            function=make_map_fn(split, args.data_source, args.system_prompt),
            with_indices=True,
        )

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    os.makedirs(local_save_dir, exist_ok=True)
    for split, dataset in datasets_out.items():
        dataset.to_parquet(os.path.join(local_save_dir, f"{split}.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
