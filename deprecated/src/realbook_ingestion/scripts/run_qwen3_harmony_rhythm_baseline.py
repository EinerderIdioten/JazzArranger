#!/usr/bin/env python3
"""Run a zero-shot Qwen3 baseline for harmony-rhythm candidate prediction."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WindowCase:
    case_id: str
    source_file: str
    title: str
    context: dict[str, Any]
    bar_start: int
    bar_end: int
    candidates: list[dict[str, Any]]
    labels: list[int]
    total_harmony_changes: int
    covered_harmony_changes: int
    uncovered_harmony_events: list[dict[str, Any]]


def meter_beats(meter: str) -> float:
    numerator, denominator = [int(part) for part in meter.split("/")]
    return numerator * 4 / denominator


def normalize_beat(value: float) -> int | float:
    rounded = round(float(value), 6)
    return int(rounded) if float(rounded).is_integer() else rounded


def time_key(bar: int, beat: float) -> tuple[int, float]:
    return int(bar), round(float(beat), 6)


def absolute_position(bar: int, beat: float, beats_per_bar: float) -> float:
    return (int(bar) - 1) * beats_per_bar + (float(beat) - 1)


def backbone_beats_for_meter(meter: str) -> list[float]:
    if meter in {"4/4", "2/2"}:
        return [1.0, 3.0, 4.0]
    if meter == "3/4":
        return [1.0]
    beats = meter_beats(meter)
    if beats >= 4:
        return [1.0, 3.0, 4.0]
    return [1.0]


def active_pitch_at(
    melody_events: list[dict[str, Any]],
    position: float,
) -> str:
    active = [
        event
        for event in melody_events
        if event["absolute_start"] <= position < event["absolute_end"]
    ]
    if not active:
        return "R"
    return str(max(active, key=lambda event: event["absolute_start"])["pitch"])


def build_candidate_grid(
    context: dict[str, Any],
    melody_stream: list[list[Any]],
    harmony_stream: list[list[Any]],
    bar_start: int,
    window_bars: int,
) -> tuple[list[dict[str, Any]], list[int], int, int, list[dict[str, Any]]]:
    beats_per_bar = meter_beats(context["meter"])
    bar_end = bar_start + window_bars - 1
    window_start = absolute_position(bar_start, 1.0, beats_per_bar)
    window_end = absolute_position(bar_end, beats_per_bar, beats_per_bar) + 1
    melody_events: list[dict[str, Any]] = []
    for bar, beat, pitch, duration, *_rest in melody_stream:
        if int(bar) < 1:
            continue
        start = absolute_position(int(bar), float(beat), beats_per_bar)
        end = start + float(duration)
        if end <= window_start or start >= window_end:
            continue
        melody_events.append(
            {
                "bar": int(bar),
                "beat": normalize_beat(float(beat)),
                "pitch": pitch,
                "note_duration": duration,
                "absolute_start": start,
                "absolute_end": end,
            }
        )

    candidate_by_key: dict[tuple[int, float], dict[str, Any]] = {}
    for event in melody_events:
        bar = event["bar"]
        beat = float(event["beat"])
        if bar < bar_start or bar > bar_end:
            continue
        pitch = str(event["pitch"])
        candidate_by_key[time_key(bar, beat)] = {
            "bar": bar,
            "beat": normalize_beat(beat),
            "type": "R" if pitch == "R" else "N",
            "pitch": pitch,
            "note_duration": event["note_duration"],
            "absolute_position": event["absolute_start"],
        }

    for bar in range(bar_start, bar_end + 1):
        for beat in backbone_beats_for_meter(context["meter"]):
            if beat > beats_per_bar:
                continue
            key = time_key(bar, beat)
            if key in candidate_by_key:
                continue
            position = absolute_position(bar, beat, beats_per_bar)
            candidate_by_key[key] = {
                "bar": bar,
                "beat": normalize_beat(beat),
                "type": "C",
                "pitch": active_pitch_at(melody_events, position),
                "note_duration": 0,
                "absolute_position": position,
            }

    candidates = sorted(candidate_by_key.values(), key=lambda item: item["absolute_position"])
    for index, candidate in enumerate(candidates, start=1):
        candidate["step"] = index
        candidate["row"] = [
            candidate["bar"],
            candidate["beat"],
            candidate["type"],
            candidate["pitch"],
            candidate["note_duration"],
            "M",
        ]

    labels = [0 for _ in candidates]
    candidate_index = {
        time_key(candidate["bar"], float(candidate["beat"])): index
        for index, candidate in enumerate(candidates)
    }
    total_harmony_changes = 0
    covered_harmony_changes = 0
    uncovered_harmony_events: list[dict[str, Any]] = []
    for bar, beat, symbol, _duration in harmony_stream:
        bar = int(bar)
        beat = float(beat)
        if bar < bar_start or bar > bar_end:
            continue
        total_harmony_changes += 1
        index = candidate_index.get(time_key(bar, beat))
        if index is None:
            uncovered_harmony_events.append(
                {
                    "bar": bar,
                    "beat": normalize_beat(beat),
                    "symbol": symbol,
                }
            )
            continue
        labels[index] = 1
        covered_harmony_changes += 1

    return candidates, labels, total_harmony_changes, covered_harmony_changes, uncovered_harmony_events


def max_bar(data: dict[str, Any]) -> int:
    bars = [
        int(event[0])
        for stream_name in ("melody_stream", "harmony_stream")
        for event in data.get(stream_name, [])
        if int(event[0]) >= 1
    ]
    return max(bars) if bars else 0


def load_cases(
    canonical_dir: Path,
    window_bars: int,
    stride_bars: int,
    max_examples: int | None,
    seed: int,
) -> list[WindowCase]:
    candidates: list[WindowCase] = []
    for path in sorted(canonical_dir.glob("*.canonical.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        context = data["context"]
        tune_max_bar = max_bar(data)
        if tune_max_bar < window_bars:
            continue
        for bar_start in range(1, tune_max_bar - window_bars + 2, stride_bars):
            candidates_grid, labels, total_changes, covered_changes, uncovered_events = build_candidate_grid(
                context,
                data["melody_stream"],
                data["harmony_stream"],
                bar_start,
                window_bars,
            )
            if not any(candidate["type"] in {"N", "R"} for candidate in candidates_grid):
                continue
            case_id = f"{path.stem}:bars_{bar_start}_{bar_start + window_bars - 1}"
            candidates.append(
                WindowCase(
                    case_id=case_id,
                    source_file=path.as_posix(),
                    title=str(context.get("title", path.stem)),
                    context=context,
                    bar_start=bar_start,
                    bar_end=bar_start + window_bars - 1,
                    candidates=candidates_grid,
                    labels=labels,
                    total_harmony_changes=total_changes,
                    covered_harmony_changes=covered_changes,
                    uncovered_harmony_events=uncovered_events,
                )
            )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if max_examples is not None:
        return candidates[:max_examples]
    return candidates


def render_messages(case: WindowCase) -> list[dict[str, str]]:
    n_steps = len(case.labels)
    rows = [candidate["row"] for candidate in case.candidates]
    system = (
        "你是一个爵士乐编曲器。"
        "输入是一个二维数组，每行代表一个旋律时间检查点 "
        "[小节, 拍数, 类型, 音高, 时值, \"M\"]。"
        "请直接输出一个相同长度的一维 JSON 数组，用 1 或 0 替换 \"M\"。"
        "1 代表该检查点换新和弦，0 代表不换。"
        "不要输出任何多余的解释。"
    )
    user = (
        "类型说明：N=新音符开始，R=休止开始，C=骨架延续检查点。\n"
        f"输入长度: {n_steps}\n"
        "输入:\n"
        f"{json.dumps(rows, ensure_ascii=False)}\n"
        f"仅输出包含 {n_steps} 个数字的 JSON 数组。"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def first_json_array(text: str) -> list[Any] | None:
    start = text.find("[")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if char == "\\" and not escaped:
                    escaped = True
                    continue
                if char == '"' and not escaped:
                    in_string = False
                escaped = False
                continue
            if char == '"':
                in_string = True
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, list):
                        return parsed
                    break
        start = text.find("[", start + 1)
    return None


def normalize_prediction(raw_text: str, expected_len: int) -> tuple[list[int] | None, str]:
    parsed = first_json_array(raw_text)
    if parsed is None:
        return None, "no_json_array"
    if len(parsed) != expected_len:
        return None, f"wrong_length:{len(parsed)}"
    normalized: list[int] = []
    for value in parsed:
        if value not in (0, 1):
            return None, "non_binary_value"
        normalized.append(int(value))
    return normalized, "ok"


def strict_counts(labels: list[int], predictions: list[int]) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for gold, pred in zip(labels, predictions, strict=True):
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def tolerant_counts(
    labels: list[int],
    predictions: list[int],
    positions: list[float],
    tolerance_beats: float,
) -> dict[str, int]:
    gold_positions = [positions[index] for index, value in enumerate(labels) if value == 1]
    pred_positions = [positions[index] for index, value in enumerate(predictions) if value == 1]
    used_gold: set[int] = set()
    tp = 0
    for pred in pred_positions:
        matches = [
            gold_index
            for gold_index, gold in enumerate(gold_positions)
            if gold_index not in used_gold and abs(gold - pred) <= tolerance_beats
        ]
        if matches:
            gold_index = min(matches, key=lambda item: abs(gold_positions[item] - pred))
            used_gold.add(gold_index)
            tp += 1
    fp = len(pred_positions) - tp
    fn = len(gold_positions) - tp
    return {"tp": tp, "fp": fp, "fn": fn}


def rates(counts: dict[str, int]) -> dict[str, float]:
    tp = counts.get("tp", 0)
    fp = counts.get("fp", 0)
    fn = counts.get("fn", 0)
    tn = counts.get("tn", 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def merge_counts(items: list[dict[str, int]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def label_positions(values: list[int]) -> list[int]:
    return [index + 1 for index, value in enumerate(values) if value == 1]


def generate_response(
    tokenizer: Any,
    model: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    import torch

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
    if hasattr(inputs, "keys") and "input_ids" in inputs:
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
    else:
        inputs = inputs.to(model.device)
        input_length = inputs.shape[-1]
    with torch.inference_mode():
        generation_kwargs: dict[str, Any]
        if isinstance(inputs, dict):
            generation_kwargs = dict(inputs)
        else:
            generation_kwargs = {"input_ids": inputs}
        output_ids = model.generate(
            **generation_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, input_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_model(model_path: Path, load_in_4bit: bool) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return tokenizer, model


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_report(
    path: Path,
    metadata: dict[str, Any],
    scores: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    format_pass_pct = 100 * scores["format"]["pass_rate"]
    strict = scores["strict_all_samples"]
    tolerant = scores["tolerant_0_5_beat_all_samples"]
    worst = sorted(
        rows,
        key=lambda row: (
            row["format_status"] != "ok",
            row["strict_counts"]["fn"] + row["strict_counts"]["fp"],
        ),
        reverse=True,
    )[:8]
    coverage = scores["candidate_coverage"]
    lines = [
        "# Qwen3 Harmony-Rhythm Baseline",
        "",
        "## Summary",
        "",
        "This is a zero-shot, pre-finetuning baseline for predicting chord-change timing from elastic melody-checkpoint prompts. The prompt does not include chord names or gold harmony labels.",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Model path | `{metadata['model_path']}` |",
        f"| Canonical source | `{metadata['canonical_dir']}` |",
        f"| Examples | `{scores['num_examples']}` |",
        f"| Window size | `{metadata['window_bars']} bars` |",
        f"| Candidate policy | `{metadata['candidate_policy']['name']}` |",
        f"| Sampling seed | `{metadata['seed']}` |",
        f"| Format pass rate | `{format_pass_pct:.1f}%` |",
        f"| Candidate coverage | `{coverage['coverage_rate']:.1%}` |",
        "",
        "## Metrics",
        "",
        "Primary metrics treat chord-change positions as label `1` over generated candidate checkpoints. Invalid model outputs are scored as all-zero predictions in `all_samples` metrics, so format failures are penalized.",
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
        "## Format Failures",
        "",
        f"- Passed: `{scores['format']['passed']}`",
        f"- Failed: `{scores['format']['failed']}`",
        "",
        "## Highest-Error Samples",
        "",
        "| Case | Format | Gold change steps | Predicted change steps | Raw output preview |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in worst:
        preview = row["raw_response"].replace("\n", " ")[:120]
        lines.append(
            "| "
            f"`{row['case_id']}` | "
            f"`{row['format_status']}` | "
            f"`{row['gold_change_steps']}` | "
            f"`{row['predicted_change_steps']}` | "
            f"{preview} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metadata.json`: run configuration",
            "- `scores.json`: aggregate metrics",
            "- `predictions.jsonl`: one row per evaluated window, with raw model output",
            "- `sample_prompts.jsonl`: prompt messages for audit",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=40)
    parser.add_argument("--window-bars", type=int, default=4)
    parser.add_argument("--stride-bars", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(
        args.canonical_dir,
        window_bars=args.window_bars,
        stride_bars=args.stride_bars,
        max_examples=args.max_examples,
        seed=args.seed,
    )
    if not cases:
        print("No evaluation cases were generated.", file=sys.stderr)
        return 2

    tokenizer, model = load_model(args.model_path, args.load_in_4bit)

    rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    strict_items: list[dict[str, int]] = []
    tolerant_items: list[dict[str, int]] = []
    format_passed = 0
    total_pred_positive = 0
    total_gold_positive = 0
    total_candidates = 0
    total_harmony_changes = 0
    covered_harmony_changes = 0
    uncovered_harmony_events: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc)
    for index, case in enumerate(cases, start=1):
        messages = render_messages(case)
        raw_response = generate_response(tokenizer, model, messages, args.max_new_tokens)
        prediction, format_status = normalize_prediction(raw_response, len(case.labels))
        if prediction is None:
            scored_prediction = [0 for _ in case.labels]
        else:
            scored_prediction = prediction
            format_passed += 1
        strict = strict_counts(case.labels, scored_prediction)
        positions = [candidate["absolute_position"] for candidate in case.candidates]
        tolerant = tolerant_counts(case.labels, scored_prediction, positions, tolerance_beats=0.5)
        strict_items.append(strict)
        tolerant_items.append(tolerant)
        total_pred_positive += sum(scored_prediction)
        total_gold_positive += sum(case.labels)
        total_candidates += len(case.labels)
        total_harmony_changes += case.total_harmony_changes
        covered_harmony_changes += case.covered_harmony_changes
        uncovered_harmony_events.extend(
            {
                "case_id": case.case_id,
                **event,
            }
            for event in case.uncovered_harmony_events
        )
        row = {
            "case_index": index,
            "case_id": case.case_id,
            "source_file": case.source_file,
            "title": case.title,
            "meter": case.context.get("meter"),
            "key": case.context.get("key"),
            "bar_start": case.bar_start,
            "bar_end": case.bar_end,
            "num_candidates": len(case.labels),
            "format_status": format_status,
            "input_rows": [candidate["row"] for candidate in case.candidates],
            "gold_labels": case.labels,
            "predicted_labels": scored_prediction,
            "gold_change_steps": label_positions(case.labels),
            "predicted_change_steps": label_positions(scored_prediction),
            "strict_counts": strict,
            "tolerant_0_5_beat_counts": tolerant,
            "total_harmony_changes": case.total_harmony_changes,
            "covered_harmony_changes": case.covered_harmony_changes,
            "uncovered_harmony_events": case.uncovered_harmony_events,
            "raw_response": raw_response,
        }
        rows.append(row)
        prompt_rows.append({"case_id": case.case_id, "messages": messages})
        print(
            f"[{index}/{len(cases)}] {case.case_id} "
            f"format={format_status} gold={sum(case.labels)} pred={sum(scored_prediction)}",
            flush=True,
        )

    strict_merged = merge_counts(strict_items)
    tolerant_merged = merge_counts(tolerant_items)
    ended_at = datetime.now(timezone.utc)
    metadata = {
        "run_type": "zero_shot_pre_finetune_harmony_rhythm_baseline",
        "created_at_utc": started_at.isoformat(),
        "completed_at_utc": ended_at.isoformat(),
        "duration_seconds": (ended_at - started_at).total_seconds(),
        "canonical_dir": args.canonical_dir.as_posix(),
        "model_path": args.model_path.as_posix(),
        "window_bars": args.window_bars,
        "stride_bars": args.stride_bars,
        "candidate_policy": {
            "name": "elastic_melody_backbone_v1",
            "melody_condition": "candidate at every melody onset or rest start",
            "backbone_condition": "candidate at beat 1 and beat 3, plus beat 4 for 4/4 and 2/2",
            "row_schema": ["bar", "beat", "type", "pitch", "duration_beats", "mask"],
            "types": {
                "N": "new pitched melody onset",
                "R": "rest onset",
                "C": "continuation backbone checkpoint",
            },
        },
        "seed": args.seed,
        "max_examples": args.max_examples,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "prompt_policy": {
            "few_shot": False,
            "include_chord_symbols": False,
            "include_title_in_prompt": False,
            "output_contract": "strict JSON array of 0/1 labels matching input candidate count",
        },
    }
    scores = {
        "num_examples": len(cases),
        "strict_counts_all_samples": strict_merged,
        "strict_all_samples": rates(strict_merged),
        "tolerant_0_5_beat_counts_all_samples": tolerant_merged,
        "tolerant_0_5_beat_all_samples": rates(tolerant_merged),
        "format": {
            "passed": format_passed,
            "failed": len(cases) - format_passed,
            "pass_rate": format_passed / len(cases),
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
            "coverage_rate": covered_harmony_changes / total_harmony_changes if total_harmony_changes else 0.0,
            "uncovered_examples": uncovered_harmony_events[:30],
        },
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "scores.json", scores)
    write_jsonl(args.output_dir / "predictions.jsonl", rows)
    write_jsonl(args.output_dir / "sample_prompts.jsonl", prompt_rows)
    write_report(args.output_dir / "TEST_REPORT.md", metadata, scores, rows)
    print(f"Wrote baseline log to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
