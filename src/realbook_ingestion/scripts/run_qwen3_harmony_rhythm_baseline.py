#!/usr/bin/env python3
"""Run a zero-shot Qwen3 baseline for harmony-rhythm slot prediction."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SLOT_BEATS = 0.5


@dataclass(frozen=True)
class WindowCase:
    case_id: str
    source_file: str
    title: str
    context: dict[str, Any]
    bar_start: int
    bar_end: int
    slots_per_bar: int
    slots: list[dict[str, Any]]
    labels: list[int]


def meter_beats(meter: str) -> float:
    numerator, denominator = [int(part) for part in meter.split("/")]
    return numerator * 4 / denominator


def beat_to_slot(beat: float) -> int:
    return round((beat - 1) / SLOT_BEATS)


def pitch_to_midi(pitch: str | None) -> int | None:
    if not pitch or pitch == "R":
        return None
    match = re.fullmatch(r"([A-G])([#b]*)(-?\d+)", pitch)
    if not match:
        return None
    base = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }[match.group(1)]
    accidental = match.group(2).count("#") - match.group(2).count("b")
    octave = int(match.group(3))
    return 12 * (octave + 1) + base + accidental


def pitch_change_label(pitch: str, previous_midi: int | None) -> tuple[str, int | None]:
    midi = pitch_to_midi(pitch)
    if pitch == "R":
        return "rest", previous_midi
    if midi is None:
        return "same", previous_midi
    if previous_midi is None:
        return "same", midi
    if midi > previous_midi:
        return "up", midi
    if midi < previous_midi:
        return "down", midi
    return "same", midi


def beat_strength(beat: float, beats_per_bar: float) -> str:
    if math.isclose(beat, 1.0):
        return "强拍"
    if beats_per_bar >= 4 and math.isclose(beat, 3.0):
        return "次强拍"
    if float(beat).is_integer():
        return "正拍"
    return "弱拍"


def build_grid(
    context: dict[str, Any],
    melody_stream: list[list[Any]],
    harmony_stream: list[list[Any]],
    bar_start: int,
    window_bars: int,
) -> tuple[list[dict[str, Any]], list[int], int]:
    beats_per_bar = meter_beats(context["meter"])
    slots_per_bar = int(round(beats_per_bar / SLOT_BEATS))
    bar_end = bar_start + window_bars - 1
    total_slots = window_bars * slots_per_bar
    slots: list[dict[str, Any]] = []
    for index in range(total_slots):
        bar = bar_start + index // slots_per_bar
        slot_in_bar = index % slots_per_bar
        beat = 1 + slot_in_bar * SLOT_BEATS
        slots.append(
            {
                "step": index + 1,
                "bar": bar,
                "beat": int(beat) if float(beat).is_integer() else beat,
                "bar_pos": slot_in_bar + 1,
                "beat_strength": beat_strength(beat, beats_per_bar),
                "is_note_onset": 0,
                "pitch": None,
                "note_duration": 0,
                "pitch_change": "silence",
            }
        )

    previous_midi: int | None = None
    for bar, beat, pitch, duration, *_rest in melody_stream:
        if bar < bar_start or bar > bar_end:
            continue
        slot_index = (bar - bar_start) * slots_per_bar + beat_to_slot(float(beat))
        if not 0 <= slot_index < total_slots:
            continue
        change, previous_midi = pitch_change_label(str(pitch), previous_midi)
        slots[slot_index].update(
            {
                "is_note_onset": 1,
                "pitch": pitch,
                "note_duration": duration,
                "pitch_change": change,
            }
        )

    labels = [0 for _ in range(total_slots)]
    for bar, beat, _symbol, _duration in harmony_stream:
        if bar < bar_start or bar > bar_end:
            continue
        slot_index = (bar - bar_start) * slots_per_bar + beat_to_slot(float(beat))
        if 0 <= slot_index < total_slots:
            labels[slot_index] = 1

    return slots, labels, slots_per_bar


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
            slots, labels, slots_per_bar = build_grid(
                context,
                data["melody_stream"],
                data["harmony_stream"],
                bar_start,
                window_bars,
            )
            if not any(slot["is_note_onset"] for slot in slots):
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
                    slots_per_bar=slots_per_bar,
                    slots=slots,
                    labels=labels,
                )
            )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    if max_examples is not None:
        return candidates[:max_examples]
    return candidates


def render_slot(slot: dict[str, Any]) -> str:
    prefix = (
        f"[Step {slot['step']} (小节{slot['bar']}-拍{slot['beat']}, "
        f"小节位置{slot['bar_pos']})]: {slot['beat_strength']}, "
    )
    if slot["is_note_onset"]:
        return (
            f"{prefix}有新音符 {slot['pitch']}, "
            f"时值 {slot['note_duration']} 拍, 音高变化 {slot['pitch_change']}"
        )
    return f"{prefix}延续上个音符或无新音符"


def render_messages(case: WindowCase) -> list[dict[str, str]]:
    n_steps = len(case.labels)
    context = case.context
    step_lines = "\n".join(render_slot(slot) for slot in case.slots)
    system = (
        "你是爵士 Lead Sheet 和声节奏标注器。"
        "任务是判断每个 0.5 拍时间格是否应该更换新和弦。"
        "只能输出合法 JSON 数组。数组长度必须严格等于输入 Step 数。"
        "数组元素只能是 0 或 1：1=该 Step 换新和弦，0=延续前一个和弦。"
        "不要输出解释、Markdown、文字或和弦名。"
    )
    user = (
        f"调性: {context.get('key', 'unknown')} | "
        f"拍号: {context.get('meter', 'unknown')} | "
        f"风格: {context.get('style', 'unknown')}\n"
        f"范围: 小节 {case.bar_start}-{case.bar_end}\n"
        "每个 Step 代表 0.5 拍。输入没有和弦名，只包含旋律与节拍骨架。\n\n"
        "旋律时间网格流：\n"
        f"{step_lines}\n\n"
        "请预测每一步是否需要更换新和弦。\n"
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


def tolerant_counts(labels: list[int], predictions: list[int], tolerance_slots: int) -> dict[str, int]:
    gold_positions = [index for index, value in enumerate(labels) if value == 1]
    pred_positions = [index for index, value in enumerate(predictions) if value == 1]
    used_gold: set[int] = set()
    tp = 0
    for pred in pred_positions:
        matches = [
            gold
            for gold in gold_positions
            if gold not in used_gold and abs(gold - pred) <= tolerance_slots
        ]
        if matches:
            gold = min(matches, key=lambda item: abs(item - pred))
            used_gold.add(gold)
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
    lines = [
        "# Qwen3 Harmony-Rhythm Baseline",
        "",
        "## Summary",
        "",
        "This is a zero-shot, pre-finetuning baseline for predicting chord-change timing from melody-grid prompts. The prompt does not include chord names or gold harmony labels.",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Model path | `{metadata['model_path']}` |",
        f"| Canonical source | `{metadata['canonical_dir']}` |",
        f"| Examples | `{scores['num_examples']}` |",
        f"| Window size | `{metadata['window_bars']} bars` |",
        f"| Slot size | `{metadata['slot_beats']} beat` |",
        f"| Sampling seed | `{metadata['seed']}` |",
        f"| Format pass rate | `{format_pass_pct:.1f}%` |",
        "",
        "## Metrics",
        "",
        "Primary metrics treat chord-change positions as label `1` in a binary sequence. Invalid model outputs are scored as all-zero predictions in `all_samples` metrics, so format failures are penalized.",
        "",
        "| Metric | Precision | Recall | F1 | Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| Strict slot match | {strict['precision']:.3f} | {strict['recall']:.3f} | {strict['f1']:.3f} | {strict['accuracy']:.3f} |",
        f"| ±0.5 beat tolerant | {tolerant['precision']:.3f} | {tolerant['recall']:.3f} | {tolerant['f1']:.3f} | n/a |",
        "",
        "## Counts",
        "",
        "| Count | Value |",
        "| --- | ---: |",
        f"| Gold chord-change slots | {scores['label_distribution']['gold_positive']} |",
        f"| Predicted chord-change slots | {scores['label_distribution']['pred_positive']} |",
        f"| Total slots | {scores['label_distribution']['total_slots']} |",
        f"| Strict TP | {scores['strict_counts_all_samples']['tp']} |",
        f"| Strict FP | {scores['strict_counts_all_samples']['fp']} |",
        f"| Strict FN | {scores['strict_counts_all_samples']['fn']} |",
        f"| Strict TN | {scores['strict_counts_all_samples']['tn']} |",
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
    total_slots = 0
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
        tolerant = tolerant_counts(case.labels, scored_prediction, tolerance_slots=1)
        strict_items.append(strict)
        tolerant_items.append(tolerant)
        total_pred_positive += sum(scored_prediction)
        total_gold_positive += sum(case.labels)
        total_slots += len(case.labels)
        row = {
            "case_index": index,
            "case_id": case.case_id,
            "source_file": case.source_file,
            "title": case.title,
            "meter": case.context.get("meter"),
            "key": case.context.get("key"),
            "bar_start": case.bar_start,
            "bar_end": case.bar_end,
            "slots_per_bar": case.slots_per_bar,
            "num_steps": len(case.labels),
            "format_status": format_status,
            "gold_labels": case.labels,
            "predicted_labels": scored_prediction,
            "gold_change_steps": label_positions(case.labels),
            "predicted_change_steps": label_positions(scored_prediction),
            "strict_counts": strict,
            "tolerant_0_5_beat_counts": tolerant,
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
        "slot_beats": SLOT_BEATS,
        "seed": args.seed,
        "max_examples": args.max_examples,
        "max_new_tokens": args.max_new_tokens,
        "load_in_4bit": args.load_in_4bit,
        "prompt_policy": {
            "few_shot": False,
            "include_chord_symbols": False,
            "include_title_in_prompt": False,
            "output_contract": "strict JSON array of 0/1 labels matching input step count",
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
            "total_slots": total_slots,
            "gold_positive_rate": total_gold_positive / total_slots if total_slots else 0.0,
            "pred_positive_rate": total_pred_positive / total_slots if total_slots else 0.0,
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
