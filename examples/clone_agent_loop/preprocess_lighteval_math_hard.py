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
Preprocess the lighteval MATH-Hard dataset into parquet format.

Input format:
    ../lighteval-MATH-Hard/{train,test}/*.jsonl (JSON arrays)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


DEFAULT_SYSTEM_PROMPT = (
    "You are the root agent. Delegate the math by calling the tool `spawn_clone` as many times as needed. "
    "Coordinate the steps until the final answer is computed. "
    "Put the final answer in the format `Answer: <value>`."
)

_BOX_TOKENS = ("\\boxed", "\\fbox")


def _load_json_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read().strip()
    if not raw:
        return []
    if raw.lstrip().startswith("["):
        return json.loads(raw)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def _extract_boxed_answers(text: str) -> list[str]:
    answers: list[str] = []
    start = 0
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
                    answers.append(text[cursor + 1 : i])
                    start = i + 1
                    break
        else:
            break
    return answers


def _extract_final_answer(solution: str, source: str) -> str:
    answers = _extract_boxed_answers(solution)
    if not answers:
        raise ValueError(f"No boxed answer found in {source}")
    return answers[-1].strip()


def _load_split_records(split_dir: str) -> list[dict[str, str]]:
    if not os.path.isdir(split_dir):
        raise ValueError(f"Split directory not found: {split_dir}")
    files = sorted(glob.glob(os.path.join(split_dir, "*.jsonl")))
    if not files:
        raise ValueError(f"No jsonl files found in {split_dir}")

    records: list[dict[str, str]] = []
    for path in files:
        category = os.path.splitext(os.path.basename(path))[0]
        items = _load_json_records(path)
        for item in items:
            problem = (item.get("problem") or "").strip()
            solution = (item.get("solution") or "").strip()
            if not problem or not solution:
                raise ValueError(f"Missing problem/solution in {path}")
            answer = _extract_final_answer(solution, path)
            records.append(
                {
                    "problem": problem,
                    "solution": solution,
                    "answer": answer,
                    "level": (item.get("level") or "").strip(),
                    "type": (item.get("type") or "").strip(),
                    "category": category,
                    "source_file": os.path.basename(path),
                }
            )

    if not records:
        raise ValueError(f"No records were loaded from {split_dir}")
    return records


def make_map_fn(split: str, data_source: str, system_prompt: str):
    def process_fn(example, idx: int):
        problem = example.pop("problem")
        solution = example.pop("solution")
        answer = example.pop("answer")
        level = example.pop("level")
        problem_type = example.pop("type")
        category = example.pop("category")
        source_file = example.pop("source_file")
        return {
            "data_source": data_source,
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer, "data_source": data_source},
            "extra_info": {
                "split": split,
                "index": idx,
                "category": category,
                "level": level,
                "type": problem_type,
                "source_file": source_file,
                "problem": problem,
                "solution": solution,
                "answer": answer,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="../lighteval-MATH-Hard",
        help="Path to the lighteval MATH-Hard dataset root.",
    )
    parser.add_argument("--train_dir", default=None, help="Override the train split directory.")
    parser.add_argument("--test_dir", default=None, help="Override the test split directory.")
    parser.add_argument("--data_source", default="lighteval_math_hard", help="Dataset identifier for reward scoring.")
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for the root agent.",
    )
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="~/data/lighteval_math_hard",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    train_dir = args.train_dir or os.path.join(args.dataset_root, "train")
    test_dir = args.test_dir or os.path.join(args.dataset_root, "test")

    train_records = _load_split_records(train_dir)
    test_records = _load_split_records(test_dir)

    train_dataset = datasets.Dataset.from_list(train_records)
    test_dataset = datasets.Dataset.from_list(test_records)

    train_dataset = train_dataset.map(
        function=make_map_fn("train", args.data_source, args.system_prompt),
        with_indices=True,
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test", args.data_source, args.system_prompt),
        with_indices=True,
    )

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
