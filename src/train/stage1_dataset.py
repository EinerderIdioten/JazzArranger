"""Dataset and collator for stage-one melody-to-harmony SFT."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset


def split_files(data_dir: Path, split: str) -> list[Path]:
    return sorted(data_dir.glob(f"*_{split}.jsonl"))


def build_prompt(row: dict) -> str:
    return f"<MELODY>\n{row['abc_melody'].strip()}\n</MELODY>\n\n"


def build_target(row: dict) -> str:
    return row["harmony_tokens"].strip()


@dataclass
class Stage1Example:
    input_ids: list[int]
    labels: list[int]
    weight: float
    row_id: str
    dataset: str


class Stage1HarmonyDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir: Path,
        split: str,
        tokenizer,
        max_length: int,
        include_zero_weight: bool = False,
        dataset_names: set[str] | None = None,
    ) -> None:
        self.examples: list[Stage1Example] = []
        self.skipped: Counter = Counter()
        self.source_files = split_files(data_dir, split)
        eos_id = tokenizer.eos_token_id

        for path in self.source_files:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    dataset = str(row.get("dataset") or "")
                    if dataset_names and dataset not in dataset_names:
                        self.skipped["dataset_filter"] += 1
                        continue
                    weight = float(row.get("dataset_weight", 1.0))
                    if weight <= 0.0 and not include_zero_weight:
                        self.skipped["zero_weight"] += 1
                        continue
                    if "harmony_tokens" not in row:
                        self.skipped["missing_harmony_tokens"] += 1
                        continue

                    prompt_ids = tokenizer.encode(build_prompt(row), add_special_tokens=False)
                    target_ids = tokenizer.encode(build_target(row), add_special_tokens=False)
                    if eos_id is not None:
                        target_ids.append(eos_id)
                    input_ids = prompt_ids + target_ids
                    labels = [-100] * len(prompt_ids) + target_ids
                    if len(input_ids) > max_length:
                        self.skipped["over_max_length"] += 1
                        continue
                    self.examples.append(
                        Stage1Example(
                            input_ids=input_ids,
                            labels=labels,
                            weight=weight,
                            row_id=str(row.get("id") or ""),
                            dataset=dataset,
                        )
                    )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "labels": example.labels,
            "example_weights": example.weight,
        }


class Stage1DataCollator:
    def __init__(self, tokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id is required")

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        weights = []

        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
            weights.append(float(feature["example_weights"]))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "example_weights": torch.tensor(weights, dtype=torch.float),
        }


def dataset_names_from_csv(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def describe_dataset(dataset: Stage1HarmonyDataset) -> dict:
    counter = Counter(example.dataset for example in dataset.examples)
    return {
        "examples": len(dataset),
        "source_files": [str(path) for path in dataset.source_files],
        "datasets": dict(sorted(counter.items())),
        "skipped": dict(sorted(dataset.skipped.items())),
    }
