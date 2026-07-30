#!/usr/bin/env python3
"""Train or evaluate Qwen as a token-level harmony-rhythm classifier."""

from __future__ import annotations

import argparse
import json
import random
import sys
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

from realbook_ingestion.scripts.run_qwen3_harmony_rhythm_baseline import (  # noqa: E402
    WindowCase,
    label_positions,
    load_cases,
    merge_counts,
    rates,
    strict_counts,
    tolerant_counts,
    write_json,
    write_jsonl,
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def candidate_feature_rows(case: WindowCase) -> list[list[Any]]:
    return [
        [
            candidate["bar"],
            candidate["beat"],
            candidate["type"],
            candidate["pitch"],
            candidate["note_duration"],
        ]
        for candidate in case.candidates
    ]


def render_classifier_text(
    case: WindowCase,
    mask_token: str,
    slot_layout: str,
) -> str:
    rows = candidate_feature_rows(case)
    if slot_layout == "target_tail":
        label_rows = [
            [candidate["bar"], candidate["beat"], mask_token]
            for candidate in case.candidates
        ]
        return "\n".join(
            [
                "任务:判断爵士lead sheet每个候选点是否换新和弦。",
                "类型:N=新音符开始,R=休止开始,C=骨架延续检查点。",
                "候选行格式:[小节,拍数,类型,音高,时值]",
                f"输入长度:{len(rows)}",
                f"候选:{compact_json(rows)}",
                "标签位格式:[小节,拍数,占位符]",
                f"标签位:{compact_json(label_rows)}",
            ]
        )
    if slot_layout == "inline":
        masked_rows = [row + [mask_token] for row in rows]
        return "\n".join(
            [
                "任务:判断爵士lead sheet每个候选点是否换新和弦。",
                "类型:N=新音符开始,R=休止开始,C=骨架延续检查点。",
                "候选行格式:[小节,拍数,类型,音高,时值,占位符]",
                f"输入长度:{len(rows)}",
                f"候选:{compact_json(masked_rows)}",
            ]
        )
    raise ValueError(f"Unsupported slot layout: {slot_layout}")


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=values.dtype, device=values.device)
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


class HarmonicRhythmLoss(nn.Module):
    def __init__(self, pos_weight: float = 1.5, alpha: float = 0.1, beta: float = 0.3):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        logits: [batch_size, seq_len], binary change logits at mask-token slots.
        labels: [batch_size, seq_len], gold 0/1 labels.
        label_mask: [batch_size, seq_len], valid slots; padding is ignored.
        """
        logits = logits.float()
        labels = labels.float().to(device=logits.device)
        if label_mask is None:
            label_mask = torch.ones_like(labels, dtype=torch.bool, device=logits.device)
        else:
            label_mask = label_mask.to(device=logits.device, dtype=torch.bool)

        base_loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=self.pos_weight.to(device=logits.device, dtype=logits.dtype),
            reduction="none",
        )
        base_loss = masked_mean(base_loss, label_mask)

        probs = torch.sigmoid(logits)
        sparsity_loss = masked_mean(probs, label_mask)

        if probs.size(1) > 1:
            transition_mask = label_mask[:, 1:] & label_mask[:, :-1]
            smoothness_loss = masked_mean(probs[:, 1:] * probs[:, :-1], transition_mask)
        else:
            smoothness_loss = probs.new_tensor(0.0)

        return base_loss + self.alpha * sparsity_loss + self.beta * smoothness_loss


@dataclass(frozen=True)
class TokenClassifierExample:
    case: WindowCase
    text: str
    input_ids: list[int]
    attention_mask: list[int]
    mask_positions: list[int]
    labels: list[int]


class HarmonyRhythmTokenDataset(Dataset[TokenClassifierExample]):
    def __init__(
        self,
        cases: list[WindowCase],
        tokenizer: Any,
        mask_token_id: int,
        mask_token: str,
        slot_layout: str,
        max_seq_length: int | None,
    ):
        self.examples: list[TokenClassifierExample] = []
        self.skipped_too_long = 0
        for case in cases:
            text = render_classifier_text(case, mask_token=mask_token, slot_layout=slot_layout)
            encoded = tokenizer(text, add_special_tokens=True, truncation=False)
            input_ids = list(encoded["input_ids"])
            if max_seq_length is not None and len(input_ids) > max_seq_length:
                self.skipped_too_long += 1
                continue
            mask_positions = [
                index for index, token_id in enumerate(input_ids) if token_id == mask_token_id
            ]
            if len(mask_positions) != len(case.labels):
                raise ValueError(
                    f"{case.case_id}: expected {len(case.labels)} mask positions, "
                    f"found {len(mask_positions)}"
                )
            self.examples.append(
                TokenClassifierExample(
                    case=case,
                    text=text,
                    input_ids=input_ids,
                    attention_mask=list(encoded["attention_mask"]),
                    mask_positions=mask_positions,
                    labels=case.labels,
                )
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenClassifierExample:
        return self.examples[index]


def collate_examples(
    examples: list[TokenClassifierExample],
    pad_token_id: int,
) -> dict[str, Any]:
    batch_size = len(examples)
    max_input_len = max(len(example.input_ids) for example in examples)
    max_labels = max(len(example.labels) for example in examples)

    input_ids = torch.full((batch_size, max_input_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_input_len), dtype=torch.long)
    mask_positions = torch.zeros((batch_size, max_labels), dtype=torch.long)
    labels = torch.zeros((batch_size, max_labels), dtype=torch.float)
    label_mask = torch.zeros((batch_size, max_labels), dtype=torch.bool)

    for row_index, example in enumerate(examples):
        input_len = len(example.input_ids)
        label_len = len(example.labels)
        input_ids[row_index, :input_len] = torch.tensor(example.input_ids, dtype=torch.long)
        attention_mask[row_index, :input_len] = torch.tensor(
            example.attention_mask,
            dtype=torch.long,
        )
        mask_positions[row_index, :label_len] = torch.tensor(
            example.mask_positions,
            dtype=torch.long,
        )
        labels[row_index, :label_len] = torch.tensor(example.labels, dtype=torch.float)
        label_mask[row_index, :label_len] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mask_positions": mask_positions,
        "labels": labels,
        "label_mask": label_mask,
        "examples": examples,
    }


class QwenHarmonyRhythmClassifier(nn.Module):
    def __init__(self, model: nn.Module, zero_token_id: int, one_token_id: int):
        super().__init__()
        self.model = model
        self.zero_token_id = int(zero_token_id)
        self.one_token_id = int(one_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        lm_logits = outputs.logits
        positions = mask_positions.to(device=lm_logits.device).clamp_min(0)
        batch_index = torch.arange(lm_logits.size(0), device=lm_logits.device).unsqueeze(1)
        selected = lm_logits[batch_index, positions]
        return selected[..., self.one_token_id] - selected[..., self.zero_token_id]


def single_token_id(tokenizer: Any, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Token {token!r} must encode to one token, got {token_ids}")
    return int(token_ids[0])


def ensure_mask_token(tokenizer: Any, mask_token: str) -> tuple[int, bool]:
    token_ids = tokenizer.encode(mask_token, add_special_tokens=False)
    if len(token_ids) == 1:
        return int(token_ids[0]), False

    additional = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    if mask_token not in additional:
        tokenizer.add_special_tokens({"additional_special_tokens": additional + [mask_token]})
    token_ids = tokenizer.encode(mask_token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Failed to register {mask_token!r} as a single token")
    return int(token_ids[0]), True


def initialize_mask_embedding(
    model: nn.Module,
    tokenizer: Any,
    mask_token_id: int,
    init_token: str,
) -> bool:
    init_ids = tokenizer.encode(init_token, add_special_tokens=False)
    if len(init_ids) != 1:
        return False
    init_token_id = int(init_ids[0])
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    with torch.no_grad():
        input_embeddings.weight[mask_token_id].copy_(input_embeddings.weight[init_token_id])
        if output_embeddings is not None and output_embeddings.weight.size(0) > mask_token_id:
            output_embeddings.weight[mask_token_id].copy_(output_embeddings.weight[init_token_id])
    return True


def load_tokenizer_and_model(args: argparse.Namespace) -> tuple[Any, nn.Module, dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    mask_token_id, mask_token_added = ensure_mask_token(tokenizer, args.mask_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    if mask_token_added:
        model.resize_token_embeddings(len(tokenizer))
    mask_embedding_initialized = initialize_mask_embedding(
        model,
        tokenizer,
        mask_token_id=mask_token_id,
        init_token=args.mask_init_token,
    )
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.adapter_path is not None:
        from peft import PeftModel, prepare_model_for_kbit_training

        if args.load_in_4bit and args.mode == "train":
            model = prepare_model_for_kbit_training(model)
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=args.mode == "train")
    elif args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[item.strip() for item in args.lora_target_modules.split(",")],
        )
        model = get_peft_model(model, lora_config)

    zero_token_id = single_token_id(tokenizer, "0")
    one_token_id = single_token_id(tokenizer, "1")
    metadata = {
        "mask_token": args.mask_token,
        "mask_token_id": mask_token_id,
        "mask_token_added": mask_token_added,
        "mask_embedding_initialized_from": (
            args.mask_init_token if mask_embedding_initialized else None
        ),
        "zero_token_id": zero_token_id,
        "one_token_id": one_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    return tokenizer, model, metadata


def primary_device(model: nn.Module) -> torch.device:
    try:
        return torch.device(model.device)
    except AttributeError:
        return next(model.parameters()).device


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def split_cases_by_tune(
    cases: list[WindowCase],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[WindowCase]]:
    by_tune: dict[str, list[WindowCase]] = {}
    for case in cases:
        by_tune.setdefault(case.source_file, []).append(case)
    tune_ids = sorted(by_tune)
    rng = random.Random(seed)
    rng.shuffle(tune_ids)
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("--val-fraction and --test-fraction must be non-negative and sum to < 1")

    tune_count = len(tune_ids)
    test_count = max(1, round(tune_count * test_fraction)) if test_fraction > 0 and tune_ids else 0
    val_count = max(1, round(tune_count * val_fraction)) if val_fraction > 0 and tune_ids else 0
    if tune_count > 1 and val_count + test_count >= tune_count:
        overflow = val_count + test_count - (tune_count - 1)
        reduce_test = min(test_count, overflow)
        test_count -= reduce_test
        overflow -= reduce_test
        val_count = max(0, val_count - overflow)

    test_tunes = set(tune_ids[:test_count])
    val_tunes = set(tune_ids[test_count : test_count + val_count])
    train_tunes = set(tune_ids[test_count + val_count :])
    splits: dict[str, list[WindowCase]] = {"train": [], "val": [], "test": []}
    for tune_id in tune_ids:
        if tune_id in test_tunes:
            split_name = "test"
        elif tune_id in val_tunes:
            split_name = "val"
        elif tune_id in train_tunes:
            split_name = "train"
        else:
            raise AssertionError(f"Unassigned tune: {tune_id}")
        splits[split_name].extend(by_tune[tune_id])
    for split_cases in splits.values():
        rng.shuffle(split_cases)
    return splits


def split_manifest_from_cases(splits: dict[str, list[WindowCase]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "unit": "source_file_tune",
        "splits": {},
    }
    for split_name, split_cases in splits.items():
        tune_ids = sorted({case.source_file for case in split_cases})
        manifest["splits"][split_name] = {
            "num_tunes": len(tune_ids),
            "num_windows": len(split_cases),
            "tunes": tune_ids,
        }
    return manifest


def build_dataset(
    cases: list[WindowCase],
    tokenizer: Any,
    mask_token_id: int,
    args: argparse.Namespace,
) -> HarmonyRhythmTokenDataset:
    return HarmonyRhythmTokenDataset(
        cases=cases,
        tokenizer=tokenizer,
        mask_token_id=mask_token_id,
        mask_token=args.mask_token,
        slot_layout=args.slot_layout,
        max_seq_length=args.max_seq_length,
    )


def make_loader(
    dataset: HarmonyRhythmTokenDataset,
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


def prediction_rows_from_batch(
    batch: dict[str, Any],
    probs: torch.Tensor,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probs = probs.detach().cpu()
    for row_index, example in enumerate(batch["examples"]):
        label_len = len(example.labels)
        probabilities = [float(value) for value in probs[row_index, :label_len].tolist()]
        predictions = [1 if value >= threshold else 0 for value in probabilities]
        positions = [candidate["absolute_position"] for candidate in example.case.candidates]
        strict = strict_counts(example.labels, predictions)
        tolerant = tolerant_counts(example.labels, predictions, positions, tolerance_beats=0.5)
        rows.append(
            {
                "case_id": example.case.case_id,
                "source_file": example.case.source_file,
                "title": example.case.title,
                "meter": example.case.context.get("meter"),
                "key": example.case.context.get("key"),
                "bar_start": example.case.bar_start,
                "bar_end": example.case.bar_end,
                "num_candidates": label_len,
                "input_text": example.text,
                "input_rows": candidate_feature_rows(example.case),
                "gold_labels": example.labels,
                "predicted_labels": predictions,
                "probabilities": probabilities,
                "gold_change_steps": label_positions(example.labels),
                "predicted_change_steps": label_positions(predictions),
                "strict_counts": strict,
                "tolerant_0_5_beat_counts": tolerant,
                "total_harmony_changes": example.case.total_harmony_changes,
                "covered_harmony_changes": example.case.covered_harmony_changes,
                "uncovered_harmony_events": example.case.uncovered_harmony_events,
            }
        )
    return rows


def score_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict_merged = merge_counts([row["strict_counts"] for row in rows])
    tolerant_merged = merge_counts([row["tolerant_0_5_beat_counts"] for row in rows])
    total_gold_positive = sum(sum(row["gold_labels"]) for row in rows)
    total_pred_positive = sum(sum(row["predicted_labels"]) for row in rows)
    total_candidates = sum(row["num_candidates"] for row in rows)
    total_harmony_changes = sum(row["total_harmony_changes"] for row in rows)
    covered_harmony_changes = sum(row["covered_harmony_changes"] for row in rows)
    uncovered_events = [
        {"case_id": row["case_id"], **event}
        for row in rows
        for event in row["uncovered_harmony_events"]
    ]
    return {
        "num_examples": len(rows),
        "strict_counts_all_samples": strict_merged,
        "strict_all_samples": rates(strict_merged),
        "tolerant_0_5_beat_counts_all_samples": tolerant_merged,
        "tolerant_0_5_beat_all_samples": rates(tolerant_merged),
        "structural_alignment": {
            "passed": len(rows),
            "failed": 0,
            "pass_rate": 1.0 if rows else 0.0,
        },
        "label_distribution": {
            "gold_positive": total_gold_positive,
            "pred_positive": total_pred_positive,
            "total_candidates": total_candidates,
            "gold_positive_rate": total_gold_positive / total_candidates if total_candidates else 0.0,
            "pred_positive_rate": total_pred_positive / total_candidates if total_candidates else 0.0,
        },
        "candidate_coverage": {
            "total_harmony_changes": total_harmony_changes,
            "covered_harmony_changes": covered_harmony_changes,
            "uncovered_harmony_changes": total_harmony_changes - covered_harmony_changes,
            "coverage_rate": covered_harmony_changes / total_harmony_changes
            if total_harmony_changes
            else 0.0,
            "uncovered_examples": uncovered_events[:30],
        },
    }


def evaluate_classifier(
    classifier: QwenHarmonyRhythmClassifier,
    loader: DataLoader[dict[str, Any]],
    loss_fn: HarmonicRhythmLoss,
    threshold: float,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    classifier.eval()
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_tensor_batch(batch, device)
            logits = classifier(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                mask_positions=batch["mask_positions"],
            )
            loss = loss_fn(logits, batch["labels"], batch["label_mask"])
            losses.append(float(loss.detach().cpu()))
            probs = torch.sigmoid(logits)
            rows.extend(prediction_rows_from_batch(batch, probs, threshold=threshold))
    scores = score_prediction_rows(rows)
    scores["loss"] = sum(losses) / len(losses) if losses else 0.0
    return scores, rows


def write_report(
    path: Path,
    metadata: dict[str, Any],
    scores: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    strict = scores["strict_all_samples"]
    tolerant = scores["tolerant_0_5_beat_all_samples"]
    coverage = scores["candidate_coverage"]
    alignment = scores["structural_alignment"]
    worst = sorted(
        rows,
        key=lambda row: row["strict_counts"]["fn"] + row["strict_counts"]["fp"],
        reverse=True,
    )[:8]
    lines = [
        "# Qwen3 Harmony-Rhythm Token Classifier",
        "",
        "## Summary",
        "",
        "This run predicts chord-change timing by gathering LM logits at fixed mask-token label slots. It does not autoregressively generate a JSON array.",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Run type | `{metadata['run_type']}` |",
        f"| Model path | `{metadata['model_path']}` |",
        f"| Adapter path | `{metadata.get('adapter_path')}` |",
        f"| Canonical source | `{metadata['canonical_dir']}` |",
        f"| Examples | `{scores['num_examples']}` |",
        f"| Eval split | `{metadata['active_eval_split']}` |",
        f"| Window size | `{metadata['window_bars']} bars` |",
        f"| Slot layout | `{metadata['slot_layout']}` |",
        f"| Mask token | `{metadata['tokenizer']['mask_token']}` |",
        f"| Structural alignment | `{alignment['pass_rate']:.1%}` |",
        f"| Candidate coverage | `{coverage['coverage_rate']:.1%}` |",
        f"| Threshold | `{metadata['threshold']}` |",
        f"| Loss | `{scores['loss']:.4f}` |",
        "",
        "## Metrics",
        "",
        "| Metric | Precision | Recall | F1 | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Strict candidate match | {strict['precision']:.3f} | {strict['recall']:.3f} | {strict['f1']:.3f} | {strict['accuracy']:.3f} |",
        f"| ±0.5 beat tolerant | {tolerant['precision']:.3f} | {tolerant['recall']:.3f} | {tolerant['f1']:.3f} | n/a |",
        "",
        "## Counts",
        "",
        "| Count | Value |",
        "| --- | ---: |",
        f"| Gold chord-change candidates | {scores['label_distribution']['gold_positive']} |",
        f"| Predicted chord-change candidates | {scores['label_distribution']['pred_positive']} |",
        f"| Total candidates | {scores['label_distribution']['total_candidates']} |",
        f"| Strict TP | {scores['strict_counts_all_samples']['tp']} |",
        f"| Strict FP | {scores['strict_counts_all_samples']['fp']} |",
        f"| Strict FN | {scores['strict_counts_all_samples']['fn']} |",
        f"| Strict TN | {scores['strict_counts_all_samples']['tn']} |",
        f"| Harmony changes in windows | {coverage['total_harmony_changes']} |",
        f"| Harmony changes covered by candidates | {coverage['covered_harmony_changes']} |",
        f"| Harmony changes uncovered by candidates | {coverage['uncovered_harmony_changes']} |",
        "",
        "## Highest-Error Samples",
        "",
        "| Case | Gold change steps | Predicted change steps | Probability preview |",
        "| --- | --- | --- | --- |",
    ]
    for row in worst:
        preview = [round(value, 3) for value in row["probabilities"][:24]]
        lines.append(
            "| "
            f"`{row['case_id']}` | "
            f"`{row['gold_change_steps']}` | "
            f"`{row['predicted_change_steps']}` | "
            f"`{preview}` |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metadata.json`: run configuration",
            "- `split_manifest.json`: tune-level train/validation/test split",
            "- `scores.json`: aggregate metrics",
            "- `predictions.jsonl`: one row per evaluated window, with probabilities",
            "- `train_metrics.jsonl`: one row per epoch when `--mode train` is used",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_classifier(
    classifier: QwenHarmonyRhythmClassifier,
    train_loader: DataLoader[dict[str, Any]],
    val_loader: DataLoader[dict[str, Any]],
    loss_fn: HarmonicRhythmLoss,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    classifier.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in classifier.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_metrics: list[dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        classifier.train()
        running_loss = 0.0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = move_tensor_batch(batch, device)
            logits = classifier(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                mask_positions=batch["mask_positions"],
            )
            loss = loss_fn(logits, batch["labels"], batch["label_mask"])
            (loss / args.grad_accum_steps).backward()
            running_loss += float(loss.detach().cpu())
            steps += 1
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
        val_scores, _rows = evaluate_classifier(
            classifier,
            val_loader,
            loss_fn=loss_fn,
            threshold=args.threshold,
            device=device,
        )
        metric_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": running_loss / steps if steps else 0.0,
            "val_loss": val_scores["loss"],
            "val_strict_f1": val_scores["strict_all_samples"]["f1"],
            "val_strict_precision": val_scores["strict_all_samples"]["precision"],
            "val_strict_recall": val_scores["strict_all_samples"]["recall"],
            "val_pred_positive_rate": val_scores["label_distribution"]["pred_positive_rate"],
        }
        train_metrics.append(metric_row)
        print(
            f"epoch={epoch} train_loss={metric_row['train_loss']:.4f} "
            f"val_f1={metric_row['val_strict_f1']:.3f} "
            f"val_p={metric_row['val_strict_precision']:.3f} "
            f"val_r={metric_row['val_strict_recall']:.3f}",
            flush=True,
        )
    return train_metrics


def save_model_artifacts(
    output_dir: Path,
    tokenizer: Any,
    model: nn.Module,
    adapter_only: bool,
) -> None:
    artifact_dir = output_dir / ("adapter" if adapter_only else "model")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_dir)
    tokenizer.save_pretrained(output_dir / "tokenizer")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--window-bars", type=int, default=4)
    parser.add_argument("--stride-bars", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--eval-split", choices=["auto", "val", "test", "all"], default="auto")
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-eval-examples", type=int, default=120)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--slot-layout", choices=["target_tail", "inline"], default="target_tail")
    parser.add_argument("--mask-token", default="[MASK]")
    parser.add_argument("--mask-init-token", default="M")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--pos-weight", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--save-artifacts", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.mode == "train" and not args.use_lora and args.adapter_path is None and not args.full_finetune:
        print(
            "Refusing full 8B fine-tune without --full-finetune. "
            "Use --use-lora for the intended single-GPU path.",
            file=sys.stderr,
        )
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)

    cases = load_cases(
        args.canonical_dir,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        max_examples=None,
        seed=args.seed,
    )
    if not cases:
        print("No cases were generated.", file=sys.stderr)
        return 2

    splits = split_cases_by_tune(
        cases,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    split_manifest = split_manifest_from_cases(splits)
    train_cases = splits["train"]
    active_eval_split = args.eval_split
    if active_eval_split == "auto":
        active_eval_split = "val" if args.mode == "train" else "test"
    eval_cases = cases if active_eval_split == "all" else splits[active_eval_split]
    if args.max_train_examples is not None:
        train_cases = train_cases[: args.max_train_examples]
    if args.max_eval_examples is not None:
        eval_cases = eval_cases[: args.max_eval_examples]
    if not eval_cases:
        print("No evaluation cases were generated.", file=sys.stderr)
        return 2

    tokenizer, model, tokenizer_metadata = load_tokenizer_and_model(args)
    mask_token_id = tokenizer_metadata["mask_token_id"]
    train_dataset = build_dataset(train_cases, tokenizer, mask_token_id, args)
    eval_dataset = build_dataset(eval_cases, tokenizer, mask_token_id, args)
    if args.mode == "train" and not train_dataset:
        print("No training examples remain after tokenization.", file=sys.stderr)
        return 2
    if not eval_dataset:
        print("No evaluation examples remain after tokenization.", file=sys.stderr)
        return 2

    train_loader = make_loader(train_dataset, tokenizer, args.batch_size, shuffle=True)
    eval_loader = make_loader(eval_dataset, tokenizer, args.eval_batch_size, shuffle=False)
    classifier = QwenHarmonyRhythmClassifier(
        model,
        zero_token_id=tokenizer_metadata["zero_token_id"],
        one_token_id=tokenizer_metadata["one_token_id"],
    )
    device = primary_device(model)
    loss_fn = HarmonicRhythmLoss(
        pos_weight=args.pos_weight,
        alpha=args.alpha,
        beta=args.beta,
    )

    train_metrics: list[dict[str, Any]] = []
    if args.mode == "train":
        train_metrics = train_classifier(
            classifier,
            train_loader=train_loader,
            val_loader=eval_loader,
            loss_fn=loss_fn,
            args=args,
            device=device,
        )
        if args.save_artifacts:
            save_model_artifacts(
                args.output_dir,
                tokenizer,
                model,
                adapter_only=args.use_lora or args.adapter_path is not None,
            )

    scores, prediction_rows = evaluate_classifier(
        classifier,
        eval_loader,
        loss_fn=loss_fn,
        threshold=args.threshold,
        device=device,
    )
    ended_at = datetime.now(timezone.utc)
    metadata = {
        "run_type": f"qwen3_harmony_rhythm_token_classifier_{args.mode}",
        "created_at_utc": started_at.isoformat(),
        "completed_at_utc": ended_at.isoformat(),
        "duration_seconds": (ended_at - started_at).total_seconds(),
        "canonical_dir": args.canonical_dir.as_posix(),
        "model_path": args.model_path.as_posix(),
        "adapter_path": args.adapter_path.as_posix() if args.adapter_path else None,
        "window_bars": args.window_bars,
        "stride_bars": args.stride_bars,
        "slot_layout": args.slot_layout,
        "threshold": args.threshold,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "test_fraction": args.test_fraction,
        "eval_split": args.eval_split,
        "active_eval_split": active_eval_split,
        "max_train_examples": args.max_train_examples,
        "max_eval_examples": args.max_eval_examples,
        "max_seq_length": args.max_seq_length,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "split_manifest": {
            split_name: {
                "num_tunes": split_manifest["splits"][split_name]["num_tunes"],
                "num_windows": split_manifest["splits"][split_name]["num_windows"],
            }
            for split_name in ("train", "val", "test")
        },
        "skipped_too_long": {
            "train": train_dataset.skipped_too_long,
            "eval": eval_dataset.skipped_too_long,
        },
        "loss": {
            "pos_weight": args.pos_weight,
            "alpha": args.alpha,
            "beta": args.beta,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "use_lora": args.use_lora,
            "full_finetune": args.full_finetune,
            "load_in_4bit": args.load_in_4bit,
            "gradient_checkpointing": args.gradient_checkpointing,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_target_modules": [item.strip() for item in args.lora_target_modules.split(",")],
            "save_artifacts": args.save_artifacts,
        },
        "candidate_policy": {
            "name": "elastic_melody_backbone_v1",
            "row_schema": ["bar", "beat", "type", "pitch", "duration_beats"],
            "label_slot_schema": ["bar", "beat", "mask"],
        },
        "tokenizer": tokenizer_metadata,
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "split_manifest.json", split_manifest)
    write_json(args.output_dir / "scores.json", scores)
    write_jsonl(args.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(args.output_dir / "train_metrics.jsonl", train_metrics)
    write_report(args.output_dir / "TEST_REPORT.md", metadata, scores, prediction_rows)
    print(f"Wrote token-classifier log to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
