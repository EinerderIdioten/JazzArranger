"""Teacher-forced evaluation for masked harmony completion stages."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.harmony_tokens import QUALITY_TOKENS, ROOT_TOKENS
from src.train.masked_completion_dataset import (
    MaskedCompletionCollator,
    Stage1MaskedCompletionDataset,
    dataset_names_from_csv,
    describe_masked_dataset,
)
from src.train.tokenizer_setup import configure_tokenizer, resize_model_for_tokenizer


def token_ids(tokenizer, tokens: list[str]) -> set[int]:
    ids: set[int] = set()
    for token in tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"not a single token: {token} -> {encoded}")
        ids.add(encoded[0])
    return ids


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def empty_counter() -> Counter:
    return Counter(
        {
            "examples": 0,
            "exact": 0,
            "answer_token_total": 0,
            "answer_token_correct": 0,
            "weighted_token_total": 0.0,
            "weighted_token_correct": 0.0,
            "root_total": 0,
            "root_correct": 0,
            "quality_total": 0,
            "quality_correct": 0,
            "joint_total": 0,
            "joint_correct": 0,
        }
    )


def finalize_counter(counter: Counter) -> dict:
    return {
        "examples": int(counter["examples"]),
        "answer_token_accuracy": safe_div(counter["answer_token_correct"], counter["answer_token_total"]),
        "weighted_answer_token_accuracy": safe_div(
            counter["weighted_token_correct"], counter["weighted_token_total"]
        ),
        "exact_match_accuracy": safe_div(counter["exact"], counter["examples"]),
        "root_token_accuracy": safe_div(counter["root_correct"], counter["root_total"]),
        "quality_token_accuracy": safe_div(counter["quality_correct"], counter["quality_total"]),
        "root_quality_joint_accuracy": safe_div(counter["joint_correct"], counter["joint_total"]),
        "counts": {
            "answer_tokens": int(counter["answer_token_total"]),
            "answer_tokens_correct": int(counter["answer_token_correct"]),
            "root_tokens": int(counter["root_total"]),
            "root_tokens_correct": int(counter["root_correct"]),
            "quality_tokens": int(counter["quality_total"]),
            "quality_tokens_correct": int(counter["quality_correct"]),
            "joint_pairs": int(counter["joint_total"]),
            "joint_pairs_correct": int(counter["joint_correct"]),
        },
    }


def apply_single_root_balance(dataset, *, examples_per_token: int, seed: int) -> dict:
    if examples_per_token <= 0:
        return {}
    rng = random.Random(seed)
    groups = {token: [] for token in ROOT_TOKENS}
    for example in dataset.examples:
        answer = example.answer_text.strip()
        if "\n" not in answer and answer in groups:
            groups[answer].append(example)
    selected = []
    counts = {}
    for token in ROOT_TOKENS:
        examples = groups[token]
        rng.shuffle(examples)
        take = examples[:examples_per_token]
        counts[token] = len(take)
        selected.extend(take)
    rng.shuffle(selected)
    original_count = len(dataset.examples)
    dataset.examples = selected
    dataset.skipped["single_root_balance_excluded"] += original_count - len(selected)
    return {
        "mode": "single_root_balance",
        "examples_per_token": examples_per_token,
        "requested_total": len(ROOT_TOKENS) * examples_per_token,
        "selected_total": len(selected),
        "selected_by_root": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["root", "quality", "chord"], required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--examples-per-row", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--single-root-examples-per-token", type=int, default=0)
    parser.add_argument("--include-zero-weight", action="store_true")
    parser.add_argument("--datasets", help="Comma-separated dataset names to include")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    configure_tokenizer(tokenizer)

    dtype = torch.bfloat16 if args.bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        device_map="auto",
    )
    resize_model_for_tokenizer(model, tokenizer)
    model.eval()

    dataset = Stage1MaskedCompletionDataset(
        data_dir=args.data_dir,
        split=args.split,
        tokenizer=tokenizer,
        max_length=args.max_length,
        stage=args.stage,
        seed=args.seed,
        examples_per_row=args.examples_per_row,
        include_zero_weight=args.include_zero_weight,
        dataset_names=dataset_names_from_csv(args.datasets),
    )
    filters = []
    single_root_filter = apply_single_root_balance(
        dataset,
        examples_per_token=args.single_root_examples_per_token,
        seed=args.seed,
    )
    if single_root_filter:
        filters.append(single_root_filter)
    if args.limit is not None:
        original_count = len(dataset.examples)
        dataset.examples = dataset.examples[: args.limit]
        dataset.skipped["limit_excluded"] += original_count - len(dataset.examples)
        filters.append({"mode": "limit", "limit": args.limit, "selected_total": len(dataset.examples)})

    collator = MaskedCompletionCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    root_ids = token_ids(tokenizer, ROOT_TOKENS)
    quality_ids = token_ids(tokenizer, QUALITY_TOKENS)
    id_to_harmony_token = {}
    for token in ROOT_TOKENS + QUALITY_TOKENS:
        id_to_harmony_token[tokenizer.encode(token, add_special_tokens=False)[0]] = token

    total_counter = empty_counter()
    by_dataset: dict[str, Counter] = defaultdict(empty_counter)
    root_confusion: Counter = Counter()
    quality_confusion: Counter = Counter()

    with torch.no_grad():
        seen = 0
        for batch in loader:
            weights = batch.pop("example_weights")
            labels = batch["labels"]
            batch = {key: value.to(model.device) for key, value in batch.items()}
            labels_device = labels.to(model.device)
            outputs = model(**batch)
            pred = outputs.logits[..., :-1, :].argmax(dim=-1).cpu()
            gold = labels[:, 1:].cpu()
            mask = gold.ne(-100)
            correct = pred.eq(gold) & mask

            for row_idx in range(gold.shape[0]):
                meta = dataset.examples[seen + row_idx]
                counters = [total_counter, by_dataset[meta.dataset]]
                row_mask = mask[row_idx]
                row_correct = correct[row_idx]
                row_gold = gold[row_idx]
                row_pred = pred[row_idx]
                weight = float(weights[row_idx].item())

                token_count = int(row_mask.sum().item())
                correct_count = int(row_correct.sum().item())
                is_exact = bool(token_count and correct_count == token_count)
                for counter in counters:
                    counter["examples"] += 1
                    counter["answer_token_total"] += token_count
                    counter["answer_token_correct"] += correct_count
                    counter["weighted_token_total"] += token_count * weight
                    counter["weighted_token_correct"] += correct_count * weight
                    if is_exact:
                        counter["exact"] += 1

                root_positions = [idx for idx, value in enumerate(row_gold.tolist()) if value in root_ids]
                quality_positions = [idx for idx, value in enumerate(row_gold.tolist()) if value in quality_ids]

                for idx in root_positions:
                    gold_id = row_gold[idx].item()
                    pred_id = row_pred[idx].item()
                    root_confusion[(id_to_harmony_token.get(gold_id, str(gold_id)), id_to_harmony_token.get(pred_id, str(pred_id)))] += 1
                    for counter in counters:
                        counter["root_total"] += 1
                        if pred_id == gold_id:
                            counter["root_correct"] += 1
                for idx in quality_positions:
                    gold_id = row_gold[idx].item()
                    pred_id = row_pred[idx].item()
                    quality_confusion[
                        (id_to_harmony_token.get(gold_id, str(gold_id)), id_to_harmony_token.get(pred_id, str(pred_id)))
                    ] += 1
                    for counter in counters:
                        counter["quality_total"] += 1
                        if pred_id == gold_id:
                            counter["quality_correct"] += 1
                for root_idx, quality_idx in zip(root_positions, quality_positions):
                    is_joint_correct = (
                        row_pred[root_idx].item() == row_gold[root_idx].item()
                        and row_pred[quality_idx].item() == row_gold[quality_idx].item()
                    )
                    for counter in counters:
                        counter["joint_total"] += 1
                        if is_joint_correct:
                            counter["joint_correct"] += 1
            seen += gold.shape[0]

    metrics = {
        "stage": args.stage,
        "split": args.split,
        "dataset": describe_masked_dataset(dataset),
        "filters": filters,
        **finalize_counter(total_counter),
        "by_dataset": {name: finalize_counter(counter) for name, counter in sorted(by_dataset.items())},
        "root_confusion_top": [
            {"gold": gold, "pred": pred, "count": count}
            for (gold, pred), count in root_confusion.most_common(24)
        ],
        "quality_confusion_top": [
            {"gold": gold, "pred": pred, "count": count}
            for (gold, pred), count in quality_confusion.most_common(24)
        ],
    }

    text = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
