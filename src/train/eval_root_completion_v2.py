"""Diagnostics for stage 1A v2 root-completion checkpoints."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.harmony_tokens import ROOT_TOKENS
from src.experiment_manifest import (
    experiment_manifest_summary,
    load_experiment_manifest,
    manifest_allowed_qualities,
    manifest_dataset_names,
    resolve_experiment_manifest_path,
)
from src.train.masked_completion_dataset import MaskedCompletionCollator
from src.train.root_completion_v2_dataset import (
    MASK_RECIPE_WEIGHTS,
    Stage1RootCompletionV2Dataset,
    dataset_names_from_csv,
    describe_root_completion_v2_dataset,
    SIMPLE_ALLOWED_QUALITIES,
)
from src.train.tokenizer_setup import configure_tokenizer, resize_model_for_tokenizer


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def token_ids(tokenizer, tokens: list[str]) -> dict[int, str]:
    mapping = {}
    for token in tokens:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"not a single token: {token} -> {encoded}")
        mapping[encoded[0]] = token
    return mapping


def entropy_from_counter(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy


def qualities_from_csv(text: str | None) -> set[str] | None:
    if not text:
        return None
    values = {item.strip() for item in text.split(",") if item.strip()}
    return values or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--mode", choices=["natural_eval", "balanced_eval"], default="natural_eval")
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--balanced-examples-per-root", type=int, default=24)
    parser.add_argument("--natural-limit", type=int, default=5000)
    parser.add_argument("--natural-shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-chords", type=int)
    parser.add_argument("--diversity-candidate-pool", type=int, default=64)
    parser.add_argument("--include-zero-weight", action="store_true")
    parser.add_argument("--datasets", help="Comma-separated dataset names to include")
    parser.add_argument("--experiment-manifest", type=Path)
    parser.add_argument(
        "--allowed-qualities",
        default=None,
        help="Comma-separated allowed quality families; defaults to the frozen manifest or the built-in simple set.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest_path = resolve_experiment_manifest_path(args.experiment_manifest)
    manifest = load_experiment_manifest(manifest_path)

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

    dataset_filter = dataset_names_from_csv(args.datasets) or manifest_dataset_names(manifest)
    quality_filter = qualities_from_csv(args.allowed_qualities) or manifest_allowed_qualities(manifest)
    if quality_filter is None:
        quality_filter = set(SIMPLE_ALLOWED_QUALITIES)
    dataset = Stage1RootCompletionV2Dataset(
        data_dir=args.data_dir,
        split=args.split,
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
        include_zero_weight=args.include_zero_weight,
        dataset_names=dataset_filter,
        allowed_qualities=quality_filter,
        mode=args.mode,
        balanced_examples_per_root=args.balanced_examples_per_root,
        natural_limit=args.natural_limit,
        natural_shuffle=args.natural_shuffle,
        mask_recipe_weights=MASK_RECIPE_WEIGHTS,
        max_chords=args.max_chords,
        diversity_candidate_pool=args.diversity_candidate_pool,
    )
    collator = MaskedCompletionCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    id_to_root = token_ids(tokenizer, ROOT_TOKENS)
    root_id_list = sorted(id_to_root)
    root_id_tensor = torch.tensor(root_id_list, device=model.device)

    total = 0
    correct = 0
    top3_correct = 0
    exact = 0
    by_gold = {token: Counter() for token in ROOT_TOKENS}
    by_quality: dict[str, Counter] = defaultdict(Counter)
    by_mask_type: dict[str, Counter] = defaultdict(Counter)
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    confusion: Counter = Counter()
    gold_distribution: Counter = Counter()
    pred_distribution: Counter = Counter()

    seen_examples = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"]
            model_inputs = {
                key: value.to(model.device)
                for key, value in batch.items()
                if key not in {"example_weights", "labels"}
            }
            outputs = model(**model_inputs)
            root_logits = outputs.logits[..., :-1, :].index_select(-1, root_id_tensor).float().cpu()
            gold = labels[:, 1:].cpu()
            root_mask = gold.ne(-100)
            for row_idx in range(gold.shape[0]):
                example = dataset.examples[seen_examples + row_idx]
                row_total = 0
                row_correct = 0
                positions = torch.nonzero(root_mask[row_idx], as_tuple=False).flatten().tolist()
                for pos in positions:
                    gold_id = int(gold[row_idx, pos].item())
                    scores = root_logits[row_idx, pos]
                    pred_root_offset = int(scores.argmax(dim=-1).item())
                    pred_id = root_id_list[pred_root_offset]
                    gold_token = id_to_root.get(gold_id, str(gold_id))
                    pred_token = id_to_root.get(pred_id, str(pred_id))
                    is_correct = pred_id == gold_id
                    top_root_offsets = scores.topk(k=min(3, len(root_id_list)), dim=-1).indices.tolist()
                    is_top3 = gold_id in {root_id_list[int(value)] for value in top_root_offsets}
                    total += 1
                    correct += int(is_correct)
                    top3_correct += int(is_top3)
                    row_total += 1
                    row_correct += int(is_correct)
                    by_gold.setdefault(gold_token, Counter())["total"] += 1
                    by_gold[gold_token]["correct"] += int(is_correct)
                    by_quality[example.primary_quality]["total"] += 1
                    by_quality[example.primary_quality]["correct"] += int(is_correct)
                    by_mask_type[example.mask_type]["total"] += 1
                    by_mask_type[example.mask_type]["correct"] += int(is_correct)
                    by_dataset[example.dataset]["total"] += 1
                    by_dataset[example.dataset]["correct"] += int(is_correct)
                    confusion[(gold_token, pred_token)] += 1
                    gold_distribution[gold_token] += 1
                    pred_distribution[pred_token] += 1
                exact += int(row_total > 0 and row_correct == row_total)
            seen_examples += gold.shape[0]

    per_root = {
        root: {
            "total": int(counter["total"]),
            "correct": int(counter["correct"]),
            "accuracy": safe_div(counter["correct"], counter["total"]),
        }
        for root, counter in by_gold.items()
    }
    root_accuracies = [value["accuracy"] for value in per_root.values() if value["total"] > 0]

    metrics = {
        "split": args.split,
        "mode": args.mode,
        "experiment_manifest": experiment_manifest_summary(manifest_path, manifest),
        "effective_filters": {
            "datasets": sorted(dataset_filter) if dataset_filter else None,
            "allowed_qualities": sorted(quality_filter) if quality_filter else None,
        },
        "dataset": describe_root_completion_v2_dataset(dataset),
        "examples": len(dataset),
        "root_tokens": total,
        "root_accuracy": safe_div(correct, total),
        "root_top3_accuracy": safe_div(top3_correct, total),
        "example_exact_match_accuracy": safe_div(exact, len(dataset)),
        "root_macro_accuracy": sum(root_accuracies) / len(root_accuracies) if root_accuracies else 0.0,
        "root_min_accuracy": min(root_accuracies) if root_accuracies else 0.0,
        "gold_root_distribution": dict(sorted(gold_distribution.items())),
        "pred_root_distribution": dict(sorted(pred_distribution.items())),
        "gold_root_entropy": entropy_from_counter(gold_distribution),
        "pred_root_entropy": entropy_from_counter(pred_distribution),
        "gold_R0_share": safe_div(gold_distribution.get("<R_C>", 0), total),
        "pred_R0_share": safe_div(pred_distribution.get("<R_C>", 0), total),
        "per_root": per_root,
        "by_quality": {
            name: {
                "total": int(counter["total"]),
                "correct": int(counter["correct"]),
                "accuracy": safe_div(counter["correct"], counter["total"]),
            }
            for name, counter in sorted(by_quality.items())
        },
        "by_mask_type": {
            name: {
                "total": int(counter["total"]),
                "correct": int(counter["correct"]),
                "accuracy": safe_div(counter["correct"], counter["total"]),
            }
            for name, counter in sorted(by_mask_type.items())
        },
        "by_dataset": {
            name: {
                "total": int(counter["total"]),
                "correct": int(counter["correct"]),
                "accuracy": safe_div(counter["correct"], counter["total"]),
            }
            for name, counter in sorted(by_dataset.items())
        },
        "confusion_top": [
            {"gold": gold, "pred": pred, "count": count}
            for (gold, pred), count in confusion.most_common(60)
        ],
    }
    text = json.dumps(metrics, indent=2, ensure_ascii=False) + "\n"
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
