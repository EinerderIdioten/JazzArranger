#!/usr/bin/env python3
"""Train Qwen2.5 scope-1 harmony masks: change timing, root, and quality."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from realbook_ingestion.scripts.prepare_qwen_chord_tokenizer import (  # noqa: E402
    QUALITIES,
    ROOTS,
    parse_chord_symbol,
)
from realbook_ingestion.scripts.run_qwen3_harmony_rhythm_baseline import (  # noqa: E402
    build_candidate_grid,
    max_bar,
    normalize_beat,
    time_key,
)


CHANGE_TOKENS = ("<CHANGE:0>", "<CHANGE:1>")
ROOT_TOKENS = tuple(f"<ROOT:{root}>" for root in ROOTS)
QUALITY_TOKENS = tuple(f"<QUAL:{quality}>" for quality in QUALITIES)
CONTROL_TOKENS = ("<HOLD>", "<NEW>", "<CHORD>", "</CHORD>")
IGNORE_INDEX = -100


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    row.setdefault("logged_at", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def single_token_id(tokenizer: Any, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{token!r} must encode to exactly one token, got {token_ids}")
    return int(token_ids[0])


def source_group_for(path: Path, context: dict[str, Any]) -> str:
    transposition = context.get("transposition") or {}
    source_file = transposition.get("source_file")
    if source_file:
        return Path(str(source_file)).name.removesuffix(".canonical.json")
    return path.name.removesuffix(".canonical.json")


def candidate_rows(candidates: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            candidate["bar"],
            candidate["beat"],
            candidate["type"],
            candidate["pitch"],
            candidate["note_duration"],
        ]
        for candidate in candidates
    ]


@dataclass(frozen=True)
class Scope1Case:
    case_id: str
    source_file: str
    source_group: str
    title: str
    context: dict[str, Any]
    bar_start: int
    bar_end: int
    candidates: list[dict[str, Any]]
    change_labels: list[int]
    root_labels: list[int]
    quality_labels: list[int]
    gold_symbols: list[str | None]
    total_harmony_changes: int
    covered_harmony_changes: int
    uncovered_harmony_events: list[dict[str, Any]]


@dataclass(frozen=True)
class Scope1Example:
    case: Scope1Case
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    mask_positions: list[list[int]]


def build_scope1_case(
    path: Path,
    data: dict[str, Any],
    bar_start: int,
    window_bars: int,
    root_to_index: dict[str, int],
    quality_to_index: dict[str, int],
) -> Scope1Case | None:
    context = data["context"]
    candidates, change_labels, total_changes, covered_changes, uncovered_events = build_candidate_grid(
        context,
        data["melody_stream"],
        data["harmony_stream"],
        bar_start=bar_start,
        window_bars=window_bars,
    )
    if not candidates:
        return None

    harmony_by_key: dict[tuple[int, float], str] = {}
    bar_end = bar_start + window_bars - 1
    for event in data.get("harmony_stream", []):
        if len(event) < 4:
            continue
        bar = int(event[0])
        beat = float(event[1])
        if bar_start <= bar <= bar_end:
            harmony_by_key[time_key(bar, beat)] = str(event[2])

    root_labels = [IGNORE_INDEX for _ in candidates]
    quality_labels = [IGNORE_INDEX for _ in candidates]
    gold_symbols: list[str | None] = [None for _ in candidates]
    for index, candidate in enumerate(candidates):
        symbol = harmony_by_key.get(time_key(candidate["bar"], float(candidate["beat"])))
        if symbol is None:
            continue
        parsed = parse_chord_symbol(symbol)
        gold_symbols[index] = symbol
        if parsed.root is not None:
            root_token = f"<ROOT:{parsed.root['spelling']}>"
            root_labels[index] = root_to_index.get(root_token, IGNORE_INDEX)
        quality_token = f"<QUAL:{parsed.quality}>"
        quality_labels[index] = quality_to_index.get(quality_token, quality_to_index["<QUAL:unknown>"])

    case_id = f"{path.stem}:bars_{bar_start}_{bar_end}"
    return Scope1Case(
        case_id=case_id,
        source_file=path.as_posix(),
        source_group=source_group_for(path, context),
        title=str(context.get("title", path.stem)),
        context=context,
        bar_start=bar_start,
        bar_end=bar_end,
        candidates=candidates,
        change_labels=change_labels,
        root_labels=root_labels,
        quality_labels=quality_labels,
        gold_symbols=gold_symbols,
        total_harmony_changes=total_changes,
        covered_harmony_changes=covered_changes,
        uncovered_harmony_events=uncovered_events,
    )


def load_cases(
    canonical_dir: Path,
    window_bars: int,
    stride_bars: int,
    root_to_index: dict[str, int],
    quality_to_index: dict[str, int],
) -> list[Scope1Case]:
    cases: list[Scope1Case] = []
    for path in sorted(canonical_dir.glob("*.canonical.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        tune_max_bar = max_bar(data)
        if tune_max_bar < window_bars:
            continue
        for bar_start in range(1, tune_max_bar - window_bars + 2, stride_bars):
            case = build_scope1_case(
                path,
                data,
                bar_start=bar_start,
                window_bars=window_bars,
                root_to_index=root_to_index,
                quality_to_index=quality_to_index,
            )
            if case is not None and any(candidate["type"] in {"N", "R"} for candidate in case.candidates):
                cases.append(case)
    return cases


def split_cases_by_group(
    cases: list[Scope1Case],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[Scope1Case]]:
    by_group: dict[str, list[Scope1Case]] = {}
    for case in cases:
        by_group.setdefault(case.source_group, []).append(case)
    groups = sorted(by_group)
    rng = random.Random(seed)
    rng.shuffle(groups)
    group_count = len(groups)
    test_count = max(1, round(group_count * test_fraction)) if test_fraction > 0 and groups else 0
    val_count = max(1, round(group_count * val_fraction)) if val_fraction > 0 and groups else 0
    if group_count > 1 and val_count + test_count >= group_count:
        overflow = val_count + test_count - (group_count - 1)
        reduce_test = min(test_count, overflow)
        test_count -= reduce_test
        overflow -= reduce_test
        val_count = max(0, val_count - overflow)
    test_groups = set(groups[:test_count])
    val_groups = set(groups[test_count : test_count + val_count])
    splits = {"train": [], "val": [], "test": []}
    for group in groups:
        if group in test_groups:
            split = "test"
        elif group in val_groups:
            split = "val"
        else:
            split = "train"
        splits[split].extend(by_group[group])
    for split_cases in splits.values():
        rng.shuffle(split_cases)
    return splits


def split_manifest(splits: dict[str, list[Scope1Case]], args: argparse.Namespace) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "unit": "source_group_original_tune",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "splits": {},
    }
    for split_name, split_cases in splits.items():
        groups = sorted({case.source_group for case in split_cases})
        manifest["splits"][split_name] = {
            "num_groups": len(groups),
            "num_windows": len(split_cases),
            "groups": groups,
        }
    return manifest


def maybe_limit_cases(cases: list[Scope1Case], limit: int | None) -> list[Scope1Case]:
    if limit is None:
        return cases
    return cases[:limit]


def render_scope1_text(case: Scope1Case, mask_token: str) -> str:
    label_rows = [
        [candidate["bar"], candidate["beat"], mask_token, mask_token, mask_token]
        for candidate in case.candidates
    ]
    return "\n".join(
        [
            "task:harmony_scope1_change_root_quality",
            f"key:{case.context.get('key')}|meter:{case.context.get('meter')}",
            "type:N=note_onset,R=rest_onset,C=continuation_checkpoint",
            "candidate_cols:[bar,beat,type,pitch,duration]",
            "label_cols:[bar,beat,change,root,quality]",
            "change_vocab:" + ",".join(CHANGE_TOKENS),
            "root_vocab:" + ",".join(ROOT_TOKENS),
            "quality_vocab:" + ",".join(QUALITY_TOKENS),
            f"input_length:{len(case.candidates)}",
            "candidates:" + compact_json(candidate_rows(case.candidates)),
            "label_slots:" + compact_json(label_rows),
        ]
    )


class Scope1Dataset(Dataset[Scope1Example]):
    def __init__(
        self,
        cases: list[Scope1Case],
        tokenizer: Any,
        mask_token: str,
        max_seq_length: int | None,
    ):
        self.examples: list[Scope1Example] = []
        self.skipped_too_long = 0
        mask_token_id = single_token_id(tokenizer, mask_token)
        for case in cases:
            text = render_scope1_text(case, mask_token=mask_token)
            encoded = tokenizer(text, add_special_tokens=True, truncation=False)
            input_ids = list(encoded["input_ids"])
            if max_seq_length is not None and len(input_ids) > max_seq_length:
                self.skipped_too_long += 1
                continue
            flat_mask_positions = [
                index for index, token_id in enumerate(input_ids) if token_id == mask_token_id
            ]
            expected = len(case.candidates) * 3
            if len(flat_mask_positions) != expected:
                raise ValueError(
                    f"{case.case_id}: expected {expected} mask positions, "
                    f"found {len(flat_mask_positions)}"
                )
            mask_positions = [
                flat_mask_positions[index : index + 3]
                for index in range(0, len(flat_mask_positions), 3)
            ]
            self.examples.append(
                Scope1Example(
                    case=case,
                    text=text,
                    input_ids=input_ids,
                    attention_mask=list(encoded["attention_mask"]),
                    mask_positions=mask_positions,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Scope1Example:
        return self.examples[index]


def collate_examples(examples: list[Scope1Example], pad_token_id: int) -> dict[str, Any]:
    batch_size = len(examples)
    max_input_len = max(len(example.input_ids) for example in examples)
    max_slots = max(len(example.case.change_labels) for example in examples)

    input_ids = torch.full((batch_size, max_input_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_input_len), dtype=torch.long)
    mask_positions = torch.zeros((batch_size, max_slots, 3), dtype=torch.long)
    change_labels = torch.zeros((batch_size, max_slots), dtype=torch.float)
    root_labels = torch.full((batch_size, max_slots), IGNORE_INDEX, dtype=torch.long)
    quality_labels = torch.full((batch_size, max_slots), IGNORE_INDEX, dtype=torch.long)
    label_mask = torch.zeros((batch_size, max_slots), dtype=torch.bool)

    for row_index, example in enumerate(examples):
        input_len = len(example.input_ids)
        slot_count = len(example.case.change_labels)
        input_ids[row_index, :input_len] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[row_index, :input_len] = torch.tensor(
            example.attention_mask,
            dtype=torch.long,
        )
        mask_positions[row_index, :slot_count] = torch.tensor(
            example.mask_positions,
            dtype=torch.long,
        )
        change_labels[row_index, :slot_count] = torch.tensor(
            example.case.change_labels,
            dtype=torch.float,
        )
        root_labels[row_index, :slot_count] = torch.tensor(
            example.case.root_labels,
            dtype=torch.long,
        )
        quality_labels[row_index, :slot_count] = torch.tensor(
            example.case.quality_labels,
            dtype=torch.long,
        )
        label_mask[row_index, :slot_count] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask_positions": mask_positions,
        "change_labels": change_labels,
        "root_labels": root_labels,
        "quality_labels": quality_labels,
        "label_mask": label_mask,
        "examples": examples,
    }


def make_loader(
    dataset: Scope1Dataset,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda examples: collate_examples(examples, tokenizer.pad_token_id),
    )


class QwenScope1MaskModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        change_token_ids: list[int],
        root_token_ids: list[int],
        quality_token_ids: list[int],
    ):
        super().__init__()
        self.model = model
        self.register_buffer("change_token_ids", torch.tensor(change_token_ids, dtype=torch.long))
        self.register_buffer("root_token_ids", torch.tensor(root_token_ids, dtype=torch.long))
        self.register_buffer("quality_token_ids", torch.tensor(quality_token_ids, dtype=torch.long))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        lm_logits = outputs.logits
        positions = mask_positions.to(device=lm_logits.device).clamp_min(0)
        batch_index = torch.arange(lm_logits.size(0), device=lm_logits.device).view(-1, 1, 1)
        selected = lm_logits[batch_index, positions]
        return {
            "change_logits": selected[:, :, 0, :].index_select(
                -1,
                self.change_token_ids.to(lm_logits.device),
            ),
            "root_logits": selected[:, :, 1, :].index_select(
                -1,
                self.root_token_ids.to(lm_logits.device),
            ),
            "quality_logits": selected[:, :, 2, :].index_select(
                -1,
                self.quality_token_ids.to(lm_logits.device),
            ),
        }


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class Scope1Loss(nn.Module):
    def __init__(
        self,
        pos_weight: float,
        root_weight: float,
        quality_weight: float,
        alpha: float,
        beta: float,
    ):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.root_weight = root_weight
        self.quality_weight = quality_weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        label_mask = batch["label_mask"].to(device=outputs["change_logits"].device)
        change_labels = batch["change_labels"].to(device=outputs["change_logits"].device)
        change_binary_logits = outputs["change_logits"][..., 1] - outputs["change_logits"][..., 0]
        change_loss_values = F.binary_cross_entropy_with_logits(
            change_binary_logits.float(),
            change_labels.float(),
            pos_weight=self.pos_weight.to(device=change_binary_logits.device),
            reduction="none",
        )
        change_loss = masked_mean(change_loss_values, label_mask)

        root_labels = batch["root_labels"].to(device=outputs["root_logits"].device)
        quality_labels = batch["quality_labels"].to(device=outputs["quality_logits"].device)
        chord_mask = label_mask & (change_labels > 0.5)
        root_mask = chord_mask & (root_labels != IGNORE_INDEX)
        quality_mask = chord_mask & (quality_labels != IGNORE_INDEX)
        root_loss_values = F.cross_entropy(
            outputs["root_logits"].float().reshape(-1, outputs["root_logits"].size(-1)),
            root_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape_as(root_labels)
        quality_loss_values = F.cross_entropy(
            outputs["quality_logits"].float().reshape(-1, outputs["quality_logits"].size(-1)),
            quality_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape_as(quality_labels)
        root_loss = masked_mean(root_loss_values, root_mask)
        quality_loss = masked_mean(quality_loss_values, quality_mask)

        change_probs = torch.sigmoid(change_binary_logits.float())
        sparsity_loss = masked_mean(change_probs, label_mask)
        if change_probs.size(1) > 1:
            transition_mask = label_mask[:, 1:] & label_mask[:, :-1]
            smoothness_loss = masked_mean(
                change_probs[:, 1:] * change_probs[:, :-1],
                transition_mask,
            )
        else:
            smoothness_loss = change_probs.new_tensor(0.0)

        total = (
            change_loss
            + self.root_weight * root_loss
            + self.quality_weight * quality_loss
            + self.alpha * sparsity_loss
            + self.beta * smoothness_loss
        )
        parts = {
            "change_loss": float(change_loss.detach().cpu()),
            "root_loss": float(root_loss.detach().cpu()),
            "quality_loss": float(quality_loss.detach().cpu()),
            "sparsity_loss": float(sparsity_loss.detach().cpu()),
            "smoothness_loss": float(smoothness_loss.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }
        return total, parts


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def merge_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in counts:
        for key, value in item.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def rates(counts: dict[str, int]) -> dict[str, float]:
    tp = counts.get("tp", 0)
    fp = counts.get("fp", 0)
    fn = counts.get("fn", 0)
    tn = counts.get("tn", 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def score_change(gold: list[int], pred: list[int]) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for gold_value, pred_value in zip(gold, pred, strict=True):
        if gold_value == 1 and pred_value == 1:
            tp += 1
        elif gold_value == 0 and pred_value == 1:
            fp += 1
        elif gold_value == 1 and pred_value == 0:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def prediction_rows_from_batch(
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    threshold: float,
    root_tokens: tuple[str, ...],
    quality_tokens: tuple[str, ...],
) -> list[dict[str, Any]]:
    change_logits = outputs["change_logits"].detach().cpu()
    root_logits = outputs["root_logits"].detach().cpu()
    quality_logits = outputs["quality_logits"].detach().cpu()
    change_probs = torch.sigmoid(change_logits[..., 1] - change_logits[..., 0])
    rows: list[dict[str, Any]] = []
    for row_index, example in enumerate(batch["examples"]):
        slot_count = len(example.case.change_labels)
        probabilities = [float(value) for value in change_probs[row_index, :slot_count].tolist()]
        pred_change = [1 if value >= threshold else 0 for value in probabilities]
        pred_root_indices = root_logits[row_index, :slot_count].argmax(dim=-1).tolist()
        pred_quality_indices = quality_logits[row_index, :slot_count].argmax(dim=-1).tolist()
        pred_roots_all = [root_tokens[index] for index in pred_root_indices]
        pred_qualities_all = [quality_tokens[index] for index in pred_quality_indices]
        pred_roots = [pred_roots_all[pos] if pred_change[pos] else None for pos in range(slot_count)]
        pred_qualities = [
            quality_tokens[index] if pred_change[pos] else None
            for pos, index in enumerate(pred_quality_indices)
        ]
        gold_roots = [
            root_tokens[index] if index != IGNORE_INDEX else None
            for index in example.case.root_labels
        ]
        gold_qualities = [
            quality_tokens[index] if index != IGNORE_INDEX else None
            for index in example.case.quality_labels
        ]
        rows.append(
            {
                "case_id": example.case.case_id,
                "source_file": example.case.source_file,
                "source_group": example.case.source_group,
                "title": example.case.title,
                "key": example.case.context.get("key"),
                "meter": example.case.context.get("meter"),
                "bar_start": example.case.bar_start,
                "bar_end": example.case.bar_end,
                "num_candidates": slot_count,
                "input_rows": candidate_rows(example.case.candidates),
                "gold_change": example.case.change_labels,
                "pred_change": pred_change,
                "change_probabilities": probabilities,
                "gold_symbols": example.case.gold_symbols,
                "gold_roots": gold_roots,
                "pred_roots_all": pred_roots_all,
                "pred_roots": pred_roots,
                "gold_qualities": gold_qualities,
                "pred_qualities_all": pred_qualities_all,
                "pred_qualities": pred_qualities,
                "change_counts": score_change(example.case.change_labels, pred_change),
                "total_harmony_changes": example.case.total_harmony_changes,
                "covered_harmony_changes": example.case.covered_harmony_changes,
                "uncovered_harmony_events": example.case.uncovered_harmony_events,
            }
        )
    return rows


def score_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    change_counts = merge_counts([row["change_counts"] for row in rows])
    total_candidates = sum(row["num_candidates"] for row in rows)
    gold_positive = sum(sum(row["gold_change"]) for row in rows)
    pred_positive = sum(sum(row["pred_change"]) for row in rows)
    gold_chord_positions = 0
    root_correct_on_gold = 0
    quality_correct_on_gold = 0
    root_quality_correct_on_gold = 0
    joint_correct_on_gold = 0
    for row in rows:
        for index, gold_change in enumerate(row["gold_change"]):
            if not gold_change:
                continue
            gold_chord_positions += 1
            root_correct = row["gold_roots"][index] == row["pred_roots_all"][index]
            quality_correct = row["gold_qualities"][index] == row["pred_qualities_all"][index]
            if root_correct:
                root_correct_on_gold += 1
            if quality_correct:
                quality_correct_on_gold += 1
            if root_correct and quality_correct:
                root_quality_correct_on_gold += 1
            if row["pred_change"][index] and root_correct and quality_correct:
                joint_correct_on_gold += 1
    total_harmony_changes = sum(row["total_harmony_changes"] for row in rows)
    covered_harmony_changes = sum(row["covered_harmony_changes"] for row in rows)
    return {
        "num_examples": len(rows),
        "change_counts": change_counts,
        "change": rates(change_counts),
        "label_distribution": {
            "gold_positive": gold_positive,
            "pred_positive": pred_positive,
            "total_candidates": total_candidates,
            "gold_positive_rate": gold_positive / total_candidates if total_candidates else 0.0,
            "pred_positive_rate": pred_positive / total_candidates if total_candidates else 0.0,
        },
        "root_quality": {
            "gold_chord_positions": gold_chord_positions,
            "root_accuracy_on_gold": root_correct_on_gold / gold_chord_positions
            if gold_chord_positions
            else 0.0,
            "quality_accuracy_on_gold": quality_correct_on_gold / gold_chord_positions
            if gold_chord_positions
            else 0.0,
            "root_quality_accuracy_on_gold": root_quality_correct_on_gold / gold_chord_positions
            if gold_chord_positions
            else 0.0,
            "joint_change_root_quality_recall_on_gold": joint_correct_on_gold / gold_chord_positions
            if gold_chord_positions
            else 0.0,
        },
        "candidate_coverage": {
            "total_harmony_changes": total_harmony_changes,
            "covered_harmony_changes": covered_harmony_changes,
            "uncovered_harmony_changes": total_harmony_changes - covered_harmony_changes,
            "coverage_rate": covered_harmony_changes / total_harmony_changes
            if total_harmony_changes
            else 0.0,
        },
    }


def evaluate(
    scope_model: QwenScope1MaskModel,
    loader: DataLoader[dict[str, Any]],
    loss_fn: Scope1Loss,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scope_model.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_tensor_batch(batch, device)
            outputs = scope_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                mask_positions=batch["mask_positions"],
            )
            loss, _parts = loss_fn(outputs, batch)
            losses.append(float(loss.detach().cpu()))
            rows.extend(
                prediction_rows_from_batch(
                    batch,
                    outputs,
                    threshold=threshold,
                    root_tokens=ROOT_TOKENS,
                    quality_tokens=QUALITY_TOKENS,
                )
            )
    scores = score_prediction_rows(rows)
    scores["loss"] = sum(losses) / len(losses) if losses else 0.0
    return scores, rows


def token_anchor_texts(token: str) -> list[str]:
    if token.startswith("<ROOT:") and token.endswith(">"):
        value = token[len("<ROOT:") : -1]
        return [value, " " + value]
    if token.startswith("<BASS:") and token.endswith(">"):
        value = token[len("<BASS:") : -1]
        return [value, " " + value, "bass"]
    if token.startswith("<QUAL:") and token.endswith(">"):
        value = token[len("<QUAL:") : -1]
        quality_anchors = {
            "maj": ["major", " maj"],
            "min": ["minor", " min", "m"],
            "maj7": ["maj7", "major 7", "major seventh"],
            "m7": ["m7", "minor 7", "minor seventh"],
            "7": ["7", "dominant 7", "seventh"],
            "6": ["6", "sixth"],
            "m6": ["m6", "minor 6"],
            "dim": ["dim", "diminished"],
            "dim7": ["dim7", "diminished 7"],
            "m7b5": ["m7b5", "half diminished"],
            "aug": ["aug", "augmented"],
            "sus": ["sus", "suspended"],
            "sus7": ["sus7", "suspended 7"],
            "unknown": ["unknown"],
        }
        return quality_anchors.get(value, [value])
    if token.startswith("<TENS:") and token.endswith(">"):
        value = token[len("<TENS:") : -1]
        return [value, value.replace("#", "sharp ").replace("b", "flat "), "tension"]
    anchors = {
        "<HOLD>": ["hold", "0", "same"],
        "<NEW>": ["new", "1", "change"],
        "<CHORD>": ["chord"],
        "</CHORD>": ["chord"],
        "<CHANGE:0>": ["0", "hold", "no"],
        "<CHANGE:1>": ["1", "new", "change"],
        "<ROOT:none>": ["none", "root"],
        "<QUAL:none>": ["none", "quality"],
        "<TENS:none>": ["none", "tension"],
        "<BASS:none>": ["none", "bass"],
    }
    return anchors.get(token, [token.strip("<>")])


def mean_embedding_for_texts(
    tokenizer: Any,
    weight: torch.Tensor,
    texts: list[str],
    old_vocab_size: int,
) -> torch.Tensor | None:
    vectors: list[torch.Tensor] = []
    for text in texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids = [token_id for token_id in token_ids if token_id < old_vocab_size]
        if token_ids:
            vectors.append(weight[token_ids].detach().mean(dim=0))
    if not vectors:
        return None
    return torch.stack(vectors, dim=0).mean(dim=0)


def initialize_new_token_embeddings(
    model: nn.Module,
    tokenizer: Any,
    old_vocab_size: int,
    tokens: list[str],
) -> dict[str, Any]:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    initialized: list[str] = []
    skipped: list[str] = []
    with torch.no_grad():
        for token in tokens:
            token_id = single_token_id(tokenizer, token)
            if token_id < old_vocab_size:
                continue
            input_vector = mean_embedding_for_texts(
                tokenizer,
                input_embeddings.weight,
                token_anchor_texts(token),
                old_vocab_size=old_vocab_size,
            )
            if input_vector is None:
                skipped.append(token)
                continue
            input_embeddings.weight[token_id].copy_(input_vector)
            if output_embeddings is not None and output_embeddings.weight.size(0) > token_id:
                output_vector = mean_embedding_for_texts(
                    tokenizer,
                    output_embeddings.weight,
                    token_anchor_texts(token),
                    old_vocab_size=old_vocab_size,
                )
                if output_vector is not None:
                    output_embeddings.weight[token_id].copy_(output_vector)
            initialized.append(token)
    return {
        "old_vocab_size": old_vocab_size,
        "initialized_count": len(initialized),
        "initialized_tokens": initialized,
        "skipped_tokens": skipped,
    }


def mask_token_row_grads(
    parameter: torch.nn.Parameter,
    trainable_token_ids: list[int],
) -> Any:
    allowed_rows = sorted(
        token_id
        for token_id in set(trainable_token_ids)
        if 0 <= token_id < parameter.shape[0]
    )

    def hook(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
        grad = grad.clone()
        row_mask = torch.zeros(grad.shape[0], dtype=torch.bool, device=grad.device)
        row_mask[allowed_rows] = True
        grad[~row_mask].zero_()
        return grad

    return parameter.register_hook(hook)


def set_trainable_phase(
    model: nn.Module,
    phase: str,
    trainable_token_ids: list[int],
    new_token_rows_only: bool,
) -> list[Any]:
    hooks: list[Any] = []
    for parameter in model.parameters():
        parameter.requires_grad = phase == "full"
    if phase == "embedding":
        input_embeddings = model.get_input_embeddings()
        input_embeddings.weight.requires_grad = True
        if new_token_rows_only:
            hooks.append(mask_token_row_grads(input_embeddings.weight, trainable_token_ids))
        output_embeddings = model.get_output_embeddings()
        if output_embeddings is not None and output_embeddings.weight is not input_embeddings.weight:
            output_embeddings.weight.requires_grad = True
            if new_token_rows_only:
                hooks.append(mask_token_row_grads(output_embeddings.weight, trainable_token_ids))
    return hooks


def make_optimizer(
    parameters: list[nn.Parameter],
    args: argparse.Namespace,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if args.optimizer == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(
            parameters,
            lr=learning_rate,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=args.weight_decay)


def make_tensorboard_writer(args: argparse.Namespace) -> Any | None:
    if not args.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard is not installed; continuing without TensorBoard logs.", file=sys.stderr)
        return None
    log_dir = args.tensorboard_log_dir or args.output_dir / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def log_train_step_to_tensorboard(writer: Any | None, row: dict[str, Any], learning_rate: float) -> None:
    if writer is None:
        return
    phase = row["phase"]
    step = int(row["global_step"])
    writer.add_scalar(f"{phase}/learning_rate", learning_rate, step)
    for key, value in row.items():
        if key.startswith("avg_") and isinstance(value, int | float):
            writer.add_scalar(f"{phase}/train/{key.removeprefix('avg_')}", value, step)
    writer.flush()


def log_val_epoch_to_tensorboard(writer: Any | None, row: dict[str, Any]) -> None:
    if writer is None:
        return
    phase = row["phase"]
    step = int(row["global_step"])
    for key, value in row.items():
        if key in {"phase", "epoch", "global_step", "is_phase_best"}:
            continue
        if isinstance(value, int | float):
            writer.add_scalar(f"{phase}/val/{key}", value, step)
    writer.flush()


def log_eval_scores_to_tensorboard(writer: Any | None, scores: dict[str, Any], step: int = 0) -> None:
    if writer is None:
        return
    writer.add_scalar("eval/loss", scores["loss"], step)
    for key, value in scores["change"].items():
        writer.add_scalar(f"eval/change_{key}", value, step)
    for key, value in scores["root_quality"].items():
        writer.add_scalar(f"eval/root_quality_{key}", value, step)
    for key, value in scores["label_distribution"].items():
        writer.add_scalar(f"eval/label_distribution_{key}", value, step)
    for key, value in scores["candidate_coverage"].items():
        writer.add_scalar(f"eval/candidate_coverage_{key}", value, step)
    writer.flush()


def train_one_phase(
    phase: str,
    scope_model: QwenScope1MaskModel,
    tokenizer: Any,
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]],
    loss_fn: Scope1Loss,
    args: argparse.Namespace,
    device: torch.device,
    start_epoch: int,
    num_epochs: int,
    progress_path: Path,
    trainable_token_ids: list[int],
    writer: Any | None,
) -> list[dict[str, Any]]:
    hooks = set_trainable_phase(
        scope_model.model,
        phase=phase,
        trainable_token_ids=trainable_token_ids,
        new_token_rows_only=args.warmup_new_token_rows_only,
    )
    trainable_parameters = [parameter for parameter in scope_model.parameters() if parameter.requires_grad]
    learning_rate = args.warmup_learning_rate if phase == "embedding" else args.learning_rate
    optimizer = make_optimizer(trainable_parameters, args, learning_rate=learning_rate)
    metrics: list[dict[str, Any]] = []
    global_step = 0
    best_metric = -1.0
    for local_epoch in range(1, num_epochs + 1):
        epoch = start_epoch + local_epoch - 1
        scope_model.train()
        running: Counter[str] = Counter()
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_tensor_batch(batch, device)
            outputs = scope_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                mask_positions=batch["mask_positions"],
            )
            loss, parts = loss_fn(outputs, batch)
            (loss / args.grad_accum_steps).backward()
            for key, value in parts.items():
                running[key] += value
            steps += 1
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if args.log_every_steps > 0 and (
                step == 1 or step % args.log_every_steps == 0 or step == len(train_loader)
            ):
                row = {
                    "event": "train_step",
                    "phase": phase,
                    "epoch": epoch,
                    "step": step,
                    "steps_per_epoch": len(train_loader),
                    "global_step": global_step,
                    **{f"avg_{key}": running[key] / steps for key in running},
                }
                append_jsonl(progress_path, row)
                log_train_step_to_tensorboard(writer, row, learning_rate=learning_rate)
                print(
                    f"phase={phase} epoch={epoch} step={step}/{len(train_loader)} "
                    f"avg_loss={row.get('avg_total_loss', 0.0):.4f}",
                    flush=True,
                )
        val_scores, _rows = evaluate(
            scope_model,
            val_loader,
            loss_fn=loss_fn,
            threshold=args.threshold,
            device=device,
        )
        metric_row = {
            "phase": phase,
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running["total_loss"] / steps if steps else 0.0,
            "val_loss": val_scores["loss"],
            "val_change_f1": val_scores["change"]["f1"],
            "val_change_precision": val_scores["change"]["precision"],
            "val_change_recall": val_scores["change"]["recall"],
            "val_root_acc_on_gold": val_scores["root_quality"]["root_accuracy_on_gold"],
            "val_quality_acc_on_gold": val_scores["root_quality"]["quality_accuracy_on_gold"],
            "val_root_quality_acc_on_gold": val_scores["root_quality"]["root_quality_accuracy_on_gold"],
            "val_joint_recall_on_gold": val_scores["root_quality"][
                "joint_change_root_quality_recall_on_gold"
            ],
        }
        if metric_row["val_joint_recall_on_gold"] > best_metric:
            best_metric = metric_row["val_joint_recall_on_gold"]
            metric_row["is_phase_best"] = True
            if args.save_best_model and phase == "full":
                save_model(scope_model.model, tokenizer, args.output_dir / "best_model")
        else:
            metric_row["is_phase_best"] = False
        metrics.append(metric_row)
        append_jsonl(progress_path, {"event": "val_epoch", **metric_row})
        log_val_epoch_to_tensorboard(writer, metric_row)
        print(
            f"phase={phase} epoch={epoch} train_loss={metric_row['train_loss']:.4f} "
            f"val_change_f1={metric_row['val_change_f1']:.3f} "
            f"val_root_acc={metric_row['val_root_acc_on_gold']:.3f} "
            f"val_quality_acc={metric_row['val_quality_acc_on_gold']:.3f} "
            f"val_joint_recall={metric_row['val_joint_recall_on_gold']:.3f}",
            flush=True,
        )
    for hook in hooks:
        hook.remove()
    return metrics


def save_model(model: nn.Module, tokenizer: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def write_report(path: Path, metadata: dict[str, Any], scores: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    worst = sorted(
        rows,
        key=lambda row: row["change_counts"]["fn"] + row["change_counts"]["fp"],
        reverse=True,
    )[:12]
    lines = [
        "# Qwen2.5 Scope-1 Chord Mask Training",
        "",
        "## Summary",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Model path | `{metadata['model_path']}` |",
        f"| Tokenizer path | `{metadata['tokenizer_path']}` |",
        f"| Canonical source | `{metadata['canonical_dir']}` |",
        f"| Eval split | `{metadata['eval_split']}` |",
        f"| Examples | `{scores['num_examples']}` |",
        f"| Window size | `{metadata['window_bars']} bars` |",
        f"| Threshold | `{metadata['threshold']}` |",
        f"| Candidate coverage | `{scores['candidate_coverage']['coverage_rate']:.2%}` |",
        f"| Loss | `{scores['loss']:.4f}` |",
        "",
        "## Metrics",
        "",
        "| Task | Precision | Recall | F1 | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Change timing | {scores['change']['precision']:.3f} | {scores['change']['recall']:.3f} | {scores['change']['f1']:.3f} | {scores['change']['accuracy']:.3f} |",
        "",
        "## Root / Quality",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Gold chord positions | {scores['root_quality']['gold_chord_positions']} |",
        f"| Root accuracy on gold changes | {scores['root_quality']['root_accuracy_on_gold']:.3f} |",
        f"| Quality accuracy on gold changes | {scores['root_quality']['quality_accuracy_on_gold']:.3f} |",
        f"| Root+quality accuracy on gold changes | {scores['root_quality']['root_quality_accuracy_on_gold']:.3f} |",
        f"| Joint change+root+quality recall on gold | {scores['root_quality']['joint_change_root_quality_recall_on_gold']:.3f} |",
        "",
        "## Highest Change-Error Samples",
        "",
        "| Case | Gold symbols | Change probabilities preview |",
        "| --- | --- | --- |",
    ]
    for row in worst:
        symbols = [
            [candidate[0], candidate[1], symbol]
            for candidate, symbol in zip(row["input_rows"], row["gold_symbols"], strict=True)
            if symbol is not None
        ][:12]
        preview = [round(value, 3) for value in row["change_probabilities"][:24]]
        lines.append(f"| `{row['case_id']}` | `{symbols}` | `{preview}` |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metadata.json`: run configuration",
            "- `split_manifest.json`: group-level train/validation/test split",
            "- `train_metrics.jsonl`: warmup/full epoch metrics",
            "- `train_progress.jsonl`: step-level progress",
            "- `scores.json`: aggregate eval metrics",
            "- `predictions.jsonl`: per-window predictions",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_tokenizer_and_model(args: argparse.Namespace) -> tuple[Any, nn.Module, dict[str, Any]]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    base_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    base_tokenizer_vocab_size = len(base_tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_config_vocab_size = int(getattr(config, "vocab_size"))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    embedding_rows_before_resize = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    embedding_rows_after_resize = model.get_input_embeddings().weight.shape[0]
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    trainable_new_token_ids = [
        single_token_id(tokenizer, token)
        for token in CONTROL_TOKENS + CHANGE_TOKENS + ROOT_TOKENS + QUALITY_TOKENS
    ]
    old_vocab_size = base_tokenizer_vocab_size
    if args.initialize_new_token_embeddings:
        init_summary = initialize_new_token_embeddings(
            model,
            tokenizer,
            old_vocab_size=old_vocab_size,
            tokens=list(CONTROL_TOKENS + CHANGE_TOKENS + ROOT_TOKENS + QUALITY_TOKENS),
        )
    else:
        init_summary = {
            "old_vocab_size": old_vocab_size,
            "initialized_count": 0,
            "initialized_tokens": [],
            "skipped_tokens": [],
        }
    metadata = {
        "old_vocab_size": old_vocab_size,
        "base_tokenizer_vocab_size": base_tokenizer_vocab_size,
        "model_config_vocab_size": model_config_vocab_size,
        "embedding_rows_before_resize": embedding_rows_before_resize,
        "embedding_rows_after_resize": embedding_rows_after_resize,
        "new_vocab_size": len(tokenizer),
        "trainable_new_token_ids": trainable_new_token_ids,
        "embedding_initialization": init_summary,
        "change_token_ids": [single_token_id(tokenizer, token) for token in CHANGE_TOKENS],
        "root_token_ids": [single_token_id(tokenizer, token) for token in ROOT_TOKENS],
        "quality_token_ids": [single_token_id(tokenizer, token) for token in QUALITY_TOKENS],
        "mask_token": args.mask_token,
        "mask_token_id": single_token_id(tokenizer, args.mask_token),
    }
    return tokenizer, model, metadata


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--window-bars", type=int, default=4)
    parser.add_argument("--stride-bars", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--mask-token", default="M")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--warmup-learning-rate", type=float, default=5e-4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=1.5)
    parser.add_argument("--root-weight", type=float, default=1.0)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw8bit")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--initialize-new-token-embeddings", action="store_true", default=True)
    parser.add_argument("--no-initialize-new-token-embeddings", dest="initialize_new_token_embeddings", action="store_false")
    parser.add_argument("--warmup-new-token-rows-only", action="store_true", default=True)
    parser.add_argument("--tensorboard", action="store_true", default=True)
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false")
    parser.add_argument("--tensorboard-log-dir", type=Path)
    parser.add_argument("--save-warmup-model", action="store_true")
    parser.add_argument("--save-best-model", action="store_true")
    parser.add_argument("--save-final-model", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "train_progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer, model, tokenizer_metadata = load_tokenizer_and_model(args)
    root_to_index = {token: index for index, token in enumerate(ROOT_TOKENS)}
    quality_to_index = {token: index for index, token in enumerate(QUALITY_TOKENS)}
    cases = load_cases(
        args.canonical_dir,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        root_to_index=root_to_index,
        quality_to_index=quality_to_index,
    )
    if not cases:
        print("No training cases generated.", file=sys.stderr)
        return 2
    splits = split_cases_by_group(
        cases,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )
    train_cases = maybe_limit_cases(splits["train"], args.max_train_examples)
    val_cases = maybe_limit_cases(splits["val"], args.max_eval_examples)
    eval_cases = maybe_limit_cases(splits[args.eval_split], args.max_eval_examples)

    train_dataset = Scope1Dataset(train_cases, tokenizer, args.mask_token, args.max_seq_length)
    val_dataset = Scope1Dataset(val_cases, tokenizer, args.mask_token, args.max_seq_length)
    eval_dataset = Scope1Dataset(eval_cases, tokenizer, args.mask_token, args.max_seq_length)
    if len(val_dataset) == 0 or len(eval_dataset) == 0:
        print("Validation/evaluation dataset is empty after filtering.", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scope_model = QwenScope1MaskModel(
        model,
        change_token_ids=tokenizer_metadata["change_token_ids"],
        root_token_ids=tokenizer_metadata["root_token_ids"],
        quality_token_ids=tokenizer_metadata["quality_token_ids"],
    )
    loss_fn = Scope1Loss(
        pos_weight=args.pos_weight,
        root_weight=args.root_weight,
        quality_weight=args.quality_weight,
        alpha=args.alpha,
        beta=args.beta,
    )
    train_loader = make_loader(train_dataset, tokenizer, args.batch_size, shuffle=True)
    val_loader = make_loader(val_dataset, tokenizer, args.eval_batch_size, shuffle=False)
    eval_loader = make_loader(eval_dataset, tokenizer, args.eval_batch_size, shuffle=False)
    writer = make_tensorboard_writer(args)
    tensorboard_log_dir = (
        str(args.tensorboard_log_dir or args.output_dir / "tb")
        if writer is not None
        else None
    )

    manifest = split_manifest(splits, args)
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model_path": str(args.model_path),
        "tokenizer_path": str(args.tokenizer_path),
        "canonical_dir": str(args.canonical_dir),
        "window_bars": args.window_bars,
        "stride_bars": args.stride_bars,
        "seed": args.seed,
        "eval_split": args.eval_split,
        "threshold": args.threshold,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "eval_examples": len(eval_dataset),
        "skipped_too_long": {
            "train": train_dataset.skipped_too_long,
            "val": val_dataset.skipped_too_long,
            "eval": eval_dataset.skipped_too_long,
        },
        "tokenizer": tokenizer_metadata,
        "root_tokens": ROOT_TOKENS,
        "quality_tokens": QUALITY_TOKENS,
        "tensorboard": {
            "enabled": writer is not None,
            "log_dir": tensorboard_log_dir,
        },
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "split_manifest.json", manifest)

    train_metrics: list[dict[str, Any]] = []
    if args.mode == "train":
        if args.warmup_epochs > 0:
            train_metrics.extend(
                train_one_phase(
                    "embedding",
                    scope_model,
                    tokenizer,
                    train_loader,
                    val_loader,
                    loss_fn,
                    args,
                    device,
                    start_epoch=1,
                    num_epochs=args.warmup_epochs,
                    progress_path=progress_path,
                    trainable_token_ids=tokenizer_metadata["trainable_new_token_ids"],
                    writer=writer,
                )
            )
            if args.save_warmup_model:
                save_model(model, tokenizer, args.output_dir / "warmup_model")
        if args.epochs > 0:
            train_metrics.extend(
                train_one_phase(
                    "full",
                    scope_model,
                    tokenizer,
                    train_loader,
                    val_loader,
                    loss_fn,
                    args,
                    device,
                    start_epoch=args.warmup_epochs + 1,
                    num_epochs=args.epochs,
                    progress_path=progress_path,
                    trainable_token_ids=tokenizer_metadata["trainable_new_token_ids"],
                    writer=writer,
                )
            )
        write_jsonl(args.output_dir / "train_metrics.jsonl", train_metrics)
        if args.save_final_model:
            save_model(model, tokenizer, args.output_dir / "final_model")

    scores, rows = evaluate(scope_model, eval_loader, loss_fn, args.threshold, device)
    log_eval_scores_to_tensorboard(writer, scores)
    write_json(args.output_dir / "scores.json", scores)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    write_report(args.output_dir / "REPORT.md", metadata, scores, rows)
    if writer is not None:
        writer.close()
    print(
        f"done mode={args.mode} eval_split={args.eval_split} "
        f"change_f1={scores['change']['f1']:.3f} "
        f"root_acc={scores['root_quality']['root_accuracy_on_gold']:.3f} "
        f"quality_acc={scores['root_quality']['quality_accuracy_on_gold']:.3f} "
        f"joint_recall={scores['root_quality']['joint_change_root_quality_recall_on_gold']:.3f} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
