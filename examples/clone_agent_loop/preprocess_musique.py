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
Preprocess the Musique dataset into parquet format for ToolAgentLoop.

Input format:
  - musique_ans_v1.0_train.jsonl
  - musique_ans_v1.0_dev.jsonl
"""

from __future__ import annotations

import argparse
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


def _build_context_items(paragraphs: Any) -> tuple[list[dict[str, str]], list[str], list[str]]:
    context_map: dict[str, str] = {}
    context_keys: list[str] = []
    supporting_titles: list[str] = []
    for para in paragraphs or []:
        if not isinstance(para, dict):
            continue
        title = str(para.get("title") or "").strip()
        text = str(para.get("paragraph_text") or "").strip()
        if not title or not text:
            continue
        if title in context_map:
            context_map[title] = f"{context_map[title]}\n{text}"
        else:
            context_map[title] = text
            context_keys.append(title)
        if para.get("is_supporting"):
            supporting_titles.append(title)
    context_entries = [{"key": key, "text": context_map[key]} for key in context_keys]
    return context_entries, context_keys, supporting_titles


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
        if not answer:
            raise ValueError("Missing answer text")
        paragraphs = example.pop("paragraphs", [])
        context_entries, context_keys, supporting_titles = _build_context_items(paragraphs)

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
                "source_id": example.pop("id", None),
                "question": question,
                "answer": answer,
                "context_keys": context_keys,
                "context_items": context_entries,
                "context_source": data_source,
                "answer_aliases": example.pop("answer_aliases", None),
                "answerable": example.pop("answerable", None),
                "question_decomposition": example.pop("question_decomposition", None),
                "supporting_titles": supporting_titles,
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
        default="<HOME_DIR>/dgslibisey-musique",
        help="Path to the Musique dataset root.",
    )
    parser.add_argument("--train_file", default=None, help="Override the train jsonl file path.")
    parser.add_argument("--dev_file", default=None, help="Override the dev jsonl file path.")
    parser.add_argument("--data_source", default="musique", help="Dataset identifier for reward scoring.")
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for the root agent.",
    )
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="<HOME_DIR>/dgslibisey-musique/preprocessed",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    splits = {
        "train": _resolve_path(args.dataset_root, args.train_file, "musique_ans_v1.0_train.jsonl"),
        "dev": _resolve_path(args.dataset_root, args.dev_file, "musique_ans_v1.0_dev.jsonl"),
    }

    datasets_out: dict[str, datasets.Dataset] = {}
    for split, path in splits.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected jsonl file for {split}: {path}")
        dataset = datasets.load_dataset("json", data_files=path)["train"]
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
