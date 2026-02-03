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
Preprocess the AIME CSV dataset into parquet format.

Input format:
    ../gneubig-aime/AIME_Dataset_1983_2024.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


DEFAULT_SYSTEM_PROMPT = (
    "You are the root agent. Delegate the math by calling the tool `spawn_clone` as many times as needed. "
    "Coordinate the steps until the final answer is computed. "
    "Put the final answer in the format `Answer: <value>`."
)


def parse_test_size(raw: str) -> float | int:
    if raw.isdigit():
        return int(raw)
    return float(raw)


def load_samples(input_path: str) -> list[dict[str, str]]:
    with open(input_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"ID", "Year", "Problem Number", "Question", "Answer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

        samples: list[dict[str, str]] = []
        for row in reader:
            question = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()
            if not question or not answer:
                raise ValueError("Encountered an empty question/answer row in the AIME CSV.")
            samples.append(
                {
                    "id": (row.get("ID") or "").strip(),
                    "year": (row.get("Year") or "").strip(),
                    "problem_number": (row.get("Problem Number") or "").strip(),
                    "question": question,
                    "answer": answer,
                    "part": (row.get("Part") or "").strip(),
                }
            )

    if not samples:
        raise ValueError("No samples were loaded from the AIME CSV.")
    return samples


def make_map_fn(split: str, data_source: str, system_prompt: str):
    def process_fn(example, idx: int):
        question = example.pop("question")
        answer = example.pop("answer")
        problem_id = example.pop("id")
        year = example.pop("year")
        problem_number = example.pop("problem_number")
        part = example.pop("part")
        return {
            "data_source": data_source,
            "agent_name": "tool_agent",
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": answer, "data_source": data_source},
            "extra_info": {
                "split": split,
                "index": idx,
                "id": problem_id,
                "year": year,
                "problem_number": problem_number,
                "part": part,
                "question": question,
                "answer": answer,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="../gneubig-aime/AIME_Dataset_1983_2024.csv",
        help="Path to the AIME CSV file.",
    )
    parser.add_argument("--data_source", default="aime_1983_2024", help="Dataset identifier for reward scoring.")
    parser.add_argument(
        "--system_prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used for the root agent.",
    )
    parser.add_argument("--test_size", type=parse_test_size, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir",
        default="~/data/aime_1983_2024",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()

    samples = load_samples(args.input_file)
    dataset = datasets.Dataset.from_list(samples)

    if len(dataset) < 2 or args.test_size == 0:
        train_dataset = dataset
        test_dataset = dataset.select([])
    else:
        split = dataset.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
        train_dataset = split["train"]
        test_dataset = split["test"]

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
