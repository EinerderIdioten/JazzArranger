"""Generate and validate stage-one tokenized harmony predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.data.harmony_tokens import parse_tokenized_harmony, validate_tokenized_harmony_text
from src.train.stage1_dataset import build_prompt, split_files
from src.train.tokenizer_setup import configure_tokenizer, resize_model_for_tokenizer


def load_rows(data_dir: Path, split: str, include_zero_weight: bool) -> list[dict]:
    rows: list[dict] = []
    for path in split_files(data_dir, split):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                if float(row.get("dataset_weight", 1.0)) <= 0.0 and not include_zero_weight:
                    continue
                rows.append(row)
    return rows


def boundary_metrics(gold: list, pred: list) -> dict:
    pred_by_span = {(span.start, span.end): span for span in pred}
    matched = 0
    root_correct = 0
    quality_correct = 0
    joint_correct = 0
    for span in gold:
        candidate = pred_by_span.get((span.start, span.end))
        if candidate is None:
            continue
        matched += 1
        if candidate.root == span.root:
            root_correct += 1
        if candidate.quality == span.quality:
            quality_correct += 1
        if candidate.root == span.root and candidate.quality == span.quality:
            joint_correct += 1
    denom = max(1, len(gold))
    return {
        "boundary_match_rate": matched / denom,
        "root_accuracy_on_gold_boundaries": root_correct / denom,
        "quality_accuracy_on_gold_boundaries": quality_correct / denom,
        "joint_accuracy_on_gold_boundaries": joint_correct / denom,
        "span_count_delta": len(pred) - len(gold),
    }


def extract_harmony(text: str) -> str:
    start = text.find("<HARMONY>")
    end = text.find("</HARMONY>")
    if start < 0:
        return text.strip()
    if end < 0:
        return text[start:].strip()
    return text[start : end + len("</HARMONY>")].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1-eval"))
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--include-zero-weight", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    configure_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    resize_model_for_tokenizer(model, tokenizer)
    model.eval()

    rows = load_rows(args.data_dir, args.split, args.include_zero_weight)
    if args.limit:
        rows = rows[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / f"{args.split}_predictions.jsonl"

    parse_ok = 0
    total = 0
    metric_sums = {
        "boundary_match_rate": 0.0,
        "root_accuracy_on_gold_boundaries": 0.0,
        "quality_accuracy_on_gold_boundaries": 0.0,
        "joint_accuracy_on_gold_boundaries": 0.0,
        "span_count_delta": 0.0,
    }

    with pred_path.open("w", encoding="utf-8") as out:
        for row in rows:
            prompt = build_prompt(row)
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_input_length,
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated_text = tokenizer.decode(generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=False)
            prediction = extract_harmony(generated_text)
            pred_spans, pred_errors = parse_tokenized_harmony(prediction, total_grid=row["total_grid"])
            gold_spans, gold_errors = parse_tokenized_harmony(row["harmony_tokens"], total_grid=row["total_grid"])
            format_errors = validate_tokenized_harmony_text(prediction) + pred_errors
            total += 1
            sample_metrics = {}
            if not format_errors and not gold_errors:
                parse_ok += 1
                sample_metrics = boundary_metrics(gold_spans, pred_spans)
                for key, value in sample_metrics.items():
                    metric_sums[key] += value
            out.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "dataset": row["dataset"],
                        "gold": row["harmony_tokens"],
                        "prediction": prediction,
                        "errors": format_errors,
                        "metrics": sample_metrics,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    denom = max(1, parse_ok)
    metrics = {
        "split": args.split,
        "total": total,
        "parse_success_rate": parse_ok / max(1, total),
        **{key: value / denom for key, value in metric_sums.items()},
    }
    metrics_path = args.output_dir / f"{args.split}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
