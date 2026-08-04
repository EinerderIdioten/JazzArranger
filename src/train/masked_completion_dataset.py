"""Masked harmony completion datasets for stage-one curriculum training."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.harmony_tokens import quality_to_token, root_to_token
from src.train.stage1_dataset import dataset_names_from_csv, split_files


MASK_RATIOS = {
    "root": [
        ("single_both", 0.40),
        ("single_one_side", 0.20),
        ("adjacent_two", 0.20),
        ("phrase_partial", 0.10),
        ("progression_251", 0.10),
    ],
    "quality": [
        ("single", 0.60),
        ("adjacent_two", 0.20),
        ("same_root_contrast", 0.15),
        ("tail_quality", 0.05),
    ],
    "chord": [
        ("single", 0.50),
        ("adjacent_two", 0.25),
        ("phrase_partial", 0.15),
        ("all", 0.10),
    ],
}

TAIL_QUALITIES = {"dim", "hdim7", "aug", "sus2", "sus4"}


@dataclass
class MaskedCompletionExample:
    input_ids: list[int]
    labels: list[int]
    weight: float
    row_id: str
    dataset: str
    mask_pattern: str
    stage: str
    answer_text: str


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _weighted_choice(rng: random.Random, weighted_items: list[tuple[str, float]]) -> str:
    total = sum(weight for _, weight in weighted_items)
    point = rng.random() * total
    cursor = 0.0
    for item, weight in weighted_items:
        cursor += weight
        if point <= cursor:
            return item
    return weighted_items[-1][0]


def _root_pc(root: str) -> int:
    roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return roots.index(root)


def _progression_251_indices(chords: list[dict]) -> list[int]:
    candidates: list[int] = []
    for idx in range(len(chords) - 2):
        a = _root_pc(chords[idx]["root"])
        b = _root_pc(chords[idx + 1]["root"])
        c = _root_pc(chords[idx + 2]["root"])
        if (b - a) % 12 == 5 and (c - b) % 12 == 5:
            candidates.extend([idx, idx + 1, idx + 2])
    return sorted(set(candidates))


def _same_root_quality_indices(chords: list[dict]) -> list[int]:
    by_root: dict[str, set[str]] = {}
    for chord in chords:
        by_root.setdefault(chord["root"], set()).add(chord["quality"])
    roots_with_contrast = {root for root, qualities in by_root.items() if len(qualities) > 1}
    return [idx for idx, chord in enumerate(chords) if chord["root"] in roots_with_contrast]


def _tail_quality_indices(chords: list[dict]) -> list[int]:
    return [idx for idx, chord in enumerate(chords) if chord["quality"] in TAIL_QUALITIES]


def _sample_indices(
    *,
    chords: list[dict],
    stage: str,
    rng: random.Random,
) -> tuple[list[int], str, str]:
    n = len(chords)
    if n <= 0:
        return [], "empty", "full"

    pattern = _weighted_choice(rng, MASK_RATIOS[stage])
    context_mode = "full"

    if stage == "root":
        if pattern == "progression_251":
            candidates = _progression_251_indices(chords)
            if candidates:
                return [rng.choice(candidates)], pattern, context_mode
            pattern = "single_both"
        if pattern == "adjacent_two" and n >= 2:
            start = rng.randrange(n - 1)
            return [start, start + 1], pattern, context_mode
        if pattern == "phrase_partial" and n >= 2:
            count = max(1, min(n, round(n * rng.uniform(0.4, 0.6))))
            return sorted(rng.sample(range(n), count)), pattern, context_mode
        if pattern == "single_one_side":
            context_mode = rng.choice(["left", "right"])
            return [rng.randrange(n)], pattern, context_mode
        return [rng.randrange(n)], pattern, context_mode

    if stage == "quality":
        if pattern == "same_root_contrast":
            candidates = _same_root_quality_indices(chords)
            if candidates:
                return [rng.choice(candidates)], pattern, context_mode
            pattern = "single"
        if pattern == "tail_quality":
            candidates = _tail_quality_indices(chords)
            if candidates:
                return [rng.choice(candidates)], pattern, context_mode
            pattern = "single"
        if pattern == "adjacent_two" and n >= 2:
            start = rng.randrange(n - 1)
            return [start, start + 1], pattern, context_mode
        return [rng.randrange(n)], pattern, context_mode

    if pattern == "all":
        return list(range(n)), pattern, context_mode
    if pattern == "adjacent_two" and n >= 2:
        start = rng.randrange(n - 1)
        return [start, start + 1], pattern, context_mode
    if pattern == "phrase_partial" and n >= 2:
        count = max(1, min(n, round(n * rng.uniform(0.4, 0.6))))
        return sorted(rng.sample(range(n), count)), pattern, context_mode
    return [rng.randrange(n)], pattern, context_mode


def _visible_indices(n: int, masked: set[int], context_mode: str) -> list[int]:
    if context_mode == "full" or not masked:
        return list(range(n))
    pivot = min(masked)
    if context_mode == "left":
        return list(range(0, pivot + 1))
    if context_mode == "right":
        return list(range(pivot, n))
    return list(range(n))


def _context_line(chord: dict, *, index: int, stage: str, masked: set[int]) -> str:
    start = int(chord["start"])
    end = int(chord["end"])
    if index not in masked:
        return f"<SPAN> @{start}-{end} {root_to_token(chord['root'])} {quality_to_token(chord['quality'])} </SPAN>"
    if stage == "root":
        return f"<SPAN> @{start}-{end} <MASK_ROOT> {quality_to_token(chord['quality'])} </SPAN>"
    if stage == "quality":
        return f"<SPAN> @{start}-{end} {root_to_token(chord['root'])} <MASK_QUALITY> </SPAN>"
    return f"<SPAN> @{start}-{end} <MASK_CHORD> </SPAN>"


def build_masked_completion_texts(
    row: dict,
    *,
    stage: str,
    seed: int,
    variant: int,
) -> tuple[str, str, str] | None:
    chords = list(row.get("chords") or [])
    if not chords:
        return None
    rng = random.Random(_stable_seed(f"{row.get('id', '')}:{stage}:{variant}", seed))
    mask_indices, pattern, context_mode = _sample_indices(chords=chords, stage=stage, rng=rng)
    if not mask_indices:
        return None
    masked = set(mask_indices)
    visible = _visible_indices(len(chords), masked, context_mode)
    context_lines = [_context_line(chords[idx], index=idx, stage=stage, masked=masked) for idx in visible]

    if stage == "root":
        prediction_open = "<ROOT_PREDICTION>"
        prediction_close = "</ROOT_PREDICTION>"
        answers = [root_to_token(chords[idx]["root"]) for idx in mask_indices]
    elif stage == "quality":
        prediction_open = "<QUALITY_PREDICTION>"
        prediction_close = "</QUALITY_PREDICTION>"
        answers = [quality_to_token(chords[idx]["quality"]) for idx in mask_indices]
    else:
        prediction_open = "<CHORD_PREDICTION>"
        prediction_close = "</CHORD_PREDICTION>"
        answers = [
            f"{root_to_token(chords[idx]['root'])} {quality_to_token(chords[idx]['quality'])}"
            for idx in mask_indices
        ]

    prompt = "\n".join(
        [
            "<MELODY>",
            str(row["abc_melody"]).strip(),
            "</MELODY>",
            "",
            "<HARMONY_CONTEXT>",
            *context_lines,
            "</HARMONY_CONTEXT>",
            "",
            prediction_open,
        ]
    )
    answer = "\n".join(answers)
    suffix = f"\n{prediction_close}"
    return prompt + "\n", answer, suffix


class Stage1MaskedCompletionDataset(Dataset):
    def __init__(
        self,
        *,
        data_dir: Path,
        split: str,
        tokenizer,
        max_length: int,
        stage: str,
        seed: int = 13,
        examples_per_row: int = 1,
        include_zero_weight: bool = False,
        dataset_names: set[str] | None = None,
    ) -> None:
        if stage not in {"root", "quality", "chord"}:
            raise ValueError(f"unsupported masked-completion stage: {stage}")
        self.examples: list[MaskedCompletionExample] = []
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
                    if not row.get("chords"):
                        self.skipped["missing_chords"] += 1
                        continue
                    for variant in range(examples_per_row):
                        built = build_masked_completion_texts(row, stage=stage, seed=seed, variant=variant)
                        if built is None:
                            self.skipped["build_failed"] += 1
                            continue
                        prompt, answer, suffix = built
                        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
                        input_ids = prompt_ids + answer_ids + suffix_ids
                        labels = [-100] * len(prompt_ids) + answer_ids + [-100] * len(suffix_ids)
                        if eos_id is not None:
                            input_ids.append(eos_id)
                            labels.append(-100)
                        if len(input_ids) > max_length:
                            self.skipped["over_max_length"] += 1
                            continue
                        self.examples.append(
                            MaskedCompletionExample(
                                input_ids=input_ids,
                                labels=labels,
                                weight=weight,
                                row_id=str(row.get("id") or ""),
                                dataset=dataset,
                                mask_pattern="mixed",
                                stage=stage,
                                answer_text=answer,
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


class MaskedCompletionCollator:
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


def describe_masked_dataset(dataset: Stage1MaskedCompletionDataset) -> dict:
    counter = Counter(example.dataset for example in dataset.examples)
    return {
        "examples": len(dataset),
        "source_files": [str(path) for path in dataset.source_files],
        "datasets": dict(sorted(counter.items())),
        "skipped": dict(sorted(dataset.skipped.items())),
    }


__all__ = [
    "MaskedCompletionCollator",
    "Stage1MaskedCompletionDataset",
    "build_masked_completion_texts",
    "dataset_names_from_csv",
    "describe_masked_dataset",
]
