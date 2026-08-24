#!/usr/bin/env python3
"""Compare blind harmony output against reference chord spans.

The reference JSON can be generated with:

    python3 scripts/pop909_blind_harmony_cases.py reference-json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}
INTERVAL_NAMES = {
    0: "P1",
    1: "m2",
    2: "M2",
    3: "m3",
    4: "M3",
    5: "P4",
    6: "TT",
    7: "P5",
    8: "m6",
    9: "M6",
    10: "m7",
    11: "M7",
}
CHORD_RE = re.compile(r"^(?P<root>[A-G](?:#|b)?):(?P<quality>[^/]+)(?:/.*)?$")
RICH_QUALITY_HINTS = (
    "6",
    "9",
    "11",
    "13",
    "alt",
    "#",
    "b5",
    "b9",
    "#9",
    "#5",
    "dim",
    "m7b5",
    "hdim",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model harmony_spans against reference spans."
    )
    parser.add_argument("reference_json", type=Path)
    parser.add_argument("model_json", type=Path)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_chord(chord: str) -> tuple[int | None, str, str]:
    match = CHORD_RE.match(chord.strip())
    if not match:
        return None, "invalid", "invalid"
    root = match.group("root")
    quality = match.group("quality")
    return ROOT_PC.get(root), normalize_quality(quality), quality


def normalize_quality(quality: str) -> str:
    q = quality.lower().replace(":", "")
    if "m7b5" in q or "hdim" in q:
        return "half-dim"
    if "dim" in q:
        return "dim"
    if "sus" in q:
        return "sus"
    if q.startswith("maj") or q in {"6", "6/9"}:
        return "maj"
    if q.startswith("min") or (q.startswith("m") and not q.startswith("maj")):
        return "min"
    if "7" in q or "9" in q or "11" in q or "13" in q or "alt" in q:
        return "dom"
    return q


def is_rich_quality(quality: str) -> bool:
    q = quality.lower()
    if q in {"maj", "min", "maj7", "min7", "7", "dom7", "sus2", "sus4"}:
        return False
    return any(hint in q for hint in RICH_QUALITY_HINTS)


def case_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["case_id"]: case for case in payload.get("cases", [])}


def get_reference_spans(case: dict[str, Any]) -> list[dict[str, Any]]:
    if "reference_spans" in case:
        return case["reference_spans"]
    if "harmony_spans" in case:
        return case["harmony_spans"]
    if "chords" in case:
        return [
            {
                "start": int(chord["start"]),
                "end": int(chord["end"]),
                "chord": f"{chord['root']}:{chord['quality']}",
            }
            for chord in case["chords"]
        ]
    raise KeyError(f"No reference spans in case {case.get('case_id')}")


def get_model_spans(case: dict[str, Any]) -> list[dict[str, Any]]:
    return case.get("harmony_spans", [])


def validate_coverage(spans: list[dict[str, Any]], total_grid: int) -> list[str]:
    errors: list[str] = []
    if not spans:
        return ["missing harmony_spans"]
    previous_end = 0
    for index, span in enumerate(sorted(spans, key=lambda item: item["start"])):
        start = int(span["start"])
        end = int(span["end"])
        if start != previous_end:
            errors.append(f"gap/overlap before span {index}: expected {previous_end}, got {start}")
        if end <= start:
            errors.append(f"non-positive span {index}: {start}-{end}")
        previous_end = end
        root_pc, _, _ = parse_chord(str(span.get("chord", "")))
        if root_pc is None:
            errors.append(f"invalid chord at span {index}: {span.get('chord')}")
    if previous_end != total_grid:
        errors.append(f"coverage ends at {previous_end}, expected {total_grid}")
    return errors


def overlaps(spans: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    return [
        span
        for span in spans
        if int(span["start"]) < end and int(span["end"]) > start
    ]


def split_bars(spans: list[dict[str, Any]], total_grid: int, bar_grid: int) -> int:
    count = 0
    for start in range(0, total_grid, bar_grid):
        end = min(start + bar_grid, total_grid)
        if len(overlaps(spans, start, end)) > 1:
            count += 1
    return count


def dominant_span(
    spans: list[dict[str, Any]], start: int, end: int
) -> dict[str, Any] | None:
    candidates = overlaps(spans, start, end)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda span: min(end, int(span["end"])) - max(start, int(span["start"])),
    )


def chord_features(span: dict[str, Any] | None) -> tuple[int | None, str, str]:
    if span is None:
        return None, "missing", "missing"
    return parse_chord(str(span["chord"]))


def bar_any_match(
    ref_spans: list[dict[str, Any]], pred_spans: list[dict[str, Any]]
) -> tuple[bool, bool, bool]:
    ref_features = [chord_features(span) for span in ref_spans]
    pred_features = [chord_features(span) for span in pred_spans]
    root_hit = any(r_pc is not None and r_pc == p_pc for r_pc, _, _ in ref_features for p_pc, _, _ in pred_features)
    coarse_hit = any(
        r_pc is not None and r_pc == p_pc and r_quality == p_quality
        for r_pc, r_quality, _ in ref_features
        for p_pc, p_quality, _ in pred_features
    )
    exact_hit = any(
        r_pc is not None and r_pc == p_pc and r_exact.lower() == p_exact.lower()
        for r_pc, _, r_exact in ref_features
        for p_pc, _, p_exact in pred_features
    )
    return root_hit, coarse_hit, exact_hit


def evaluate_case(ref_case: dict[str, Any], pred_case: dict[str, Any] | None) -> dict[str, Any]:
    case_id = ref_case["case_id"]
    total_grid = int(ref_case["total_grid"])
    bar_grid = int(ref_case["bar_grid"])
    ref_spans = get_reference_spans(ref_case)
    pred_spans = get_model_spans(pred_case or {})
    coverage_errors = validate_coverage(pred_spans, total_grid)

    total_bars = 0
    root_hits = 0
    coarse_hits = 0
    exact_hits = 0
    rich_ref_bars = 0
    rich_collapsed_bars = 0
    confusion: Counter[tuple[str, str]] = Counter()
    interval_errors: Counter[str] = Counter()
    ref_quality_counts: Counter[str] = Counter()
    pred_quality_counts: Counter[str] = Counter()

    for start in range(0, total_grid, bar_grid):
        end = min(start + bar_grid, total_grid)
        total_bars += 1
        ref_bar = overlaps(ref_spans, start, end)
        pred_bar = overlaps(pred_spans, start, end)
        root_hit, coarse_hit, exact_hit = bar_any_match(ref_bar, pred_bar)
        root_hits += int(root_hit)
        coarse_hits += int(coarse_hit)
        exact_hits += int(exact_hit)

        ref_dom = dominant_span(ref_spans, start, end)
        pred_dom = dominant_span(pred_spans, start, end)
        ref_pc, ref_quality, ref_exact = chord_features(ref_dom)
        pred_pc, pred_quality, pred_exact = chord_features(pred_dom)
        confusion[(ref_quality, pred_quality)] += 1
        ref_quality_counts[ref_exact] += 1
        pred_quality_counts[pred_exact] += 1
        if ref_pc is not None and pred_pc is not None and ref_pc != pred_pc:
            interval = (pred_pc - ref_pc) % 12
            interval_errors[INTERVAL_NAMES[interval]] += 1

        ref_rich = any(is_rich_quality(chord_features(span)[2]) for span in ref_bar)
        pred_rich = any(is_rich_quality(chord_features(span)[2]) for span in pred_bar)
        rich_ref_bars += int(ref_rich)
        rich_collapsed_bars += int(ref_rich and not pred_rich)

    ref_split = split_bars(ref_spans, total_grid, bar_grid)
    pred_split = split_bars(pred_spans, total_grid, bar_grid)
    return {
        "case_id": case_id,
        "title": ref_case.get("title", ""),
        "total_bars": total_bars,
        "root_hits": root_hits,
        "coarse_hits": coarse_hits,
        "exact_hits": exact_hits,
        "root_hit_rate": root_hits / total_bars if total_bars else 0.0,
        "coarse_hit_rate": coarse_hits / total_bars if total_bars else 0.0,
        "exact_hit_rate": exact_hits / total_bars if total_bars else 0.0,
        "ref_split_bars": ref_split,
        "pred_split_bars": pred_split,
        "rich_ref_bars": rich_ref_bars,
        "rich_collapsed_bars": rich_collapsed_bars,
        "coverage_errors": coverage_errors,
        "confusion": dict(confusion),
        "interval_errors": dict(interval_errors),
        "ref_quality_counts": dict(ref_quality_counts),
        "pred_quality_counts": dict(pred_quality_counts),
    }


def aggregate(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "total_bars": 0,
        "root_hits": 0,
        "coarse_hits": 0,
        "exact_hits": 0,
        "ref_split_bars": 0,
        "pred_split_bars": 0,
        "rich_ref_bars": 0,
        "rich_collapsed_bars": 0,
    }
    confusion: Counter[tuple[str, str]] = Counter()
    interval_errors: Counter[str] = Counter()
    ref_quality_counts: Counter[str] = Counter()
    pred_quality_counts: Counter[str] = Counter()
    coverage_error_cases = 0
    for result in case_results:
        for key in totals:
            totals[key] += result[key]
        confusion.update(result["confusion"])
        interval_errors.update(result["interval_errors"])
        ref_quality_counts.update(result["ref_quality_counts"])
        pred_quality_counts.update(result["pred_quality_counts"])
        coverage_error_cases += int(bool(result["coverage_errors"]))

    total_bars = totals["total_bars"]
    rich_ref_bars = totals["rich_ref_bars"]
    return {
        **totals,
        "root_hit_rate": totals["root_hits"] / total_bars if total_bars else 0.0,
        "coarse_hit_rate": totals["coarse_hits"] / total_bars if total_bars else 0.0,
        "exact_hit_rate": totals["exact_hits"] / total_bars if total_bars else 0.0,
        "ref_split_bar_rate": totals["ref_split_bars"] / total_bars if total_bars else 0.0,
        "pred_split_bar_rate": totals["pred_split_bars"] / total_bars if total_bars else 0.0,
        "rich_collapse_rate": totals["rich_collapsed_bars"] / rich_ref_bars if rich_ref_bars else 0.0,
        "coverage_error_cases": coverage_error_cases,
        "confusion": dict(confusion),
        "interval_errors": dict(interval_errors),
        "ref_quality_counts": dict(ref_quality_counts),
        "pred_quality_counts": dict(pred_quality_counts),
    }


def stringify_counter(mapping: dict[Any, int], limit: int = 12) -> str:
    items = sorted(mapping.items(), key=lambda item: (-item[1], str(item[0])))[:limit]
    return ", ".join(f"{key}:{value}" for key, value in items) if items else "-"


def jsonable(result: Any) -> Any:
    if isinstance(result, dict):
        return {str(key): jsonable(value) for key, value in result.items()}
    if isinstance(result, list):
        return [jsonable(item) for item in result]
    return result


def render_markdown(case_results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Blind Harmony Evaluation",
        "",
        "## Summary",
        "",
        f"- Total bars: {summary['total_bars']}",
        f"- Root hit rate: {summary['root_hit_rate']:.1%}",
        f"- Root + coarse quality hit rate: {summary['coarse_hit_rate']:.1%}",
        f"- Root + exact quality hit rate: {summary['exact_hit_rate']:.1%}",
        f"- Reference split-bar rate: {summary['ref_split_bar_rate']:.1%}",
        f"- Prediction split-bar rate: {summary['pred_split_bar_rate']:.1%}",
        f"- Rich-quality collapse rate: {summary['rich_collapse_rate']:.1%}",
        f"- Coverage error cases: {summary['coverage_error_cases']}",
        f"- Reference quality counts: {stringify_counter(summary['ref_quality_counts'])}",
        f"- Prediction quality counts: {stringify_counter(summary['pred_quality_counts'])}",
        f"- Root error intervals: {stringify_counter(summary['interval_errors'])}",
        "",
        "## Cases",
        "",
        "| Case | Title | Bars | Root hit | Coarse hit | Exact hit | Ref split | Pred split | Rich collapse | Coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in case_results:
        coverage = "ok" if not result["coverage_errors"] else "; ".join(result["coverage_errors"][:2])
        rich_rate = (
            result["rich_collapsed_bars"] / result["rich_ref_bars"]
            if result["rich_ref_bars"]
            else 0.0
        )
        lines.append(
            "| {case_id} | {title} | {bars} | {root:.1%} | {coarse:.1%} | {exact:.1%} | "
            "{ref_split} | {pred_split} | {rich:.1%} | {coverage} |".format(
                case_id=result["case_id"],
                title=result["title"],
                bars=result["total_bars"],
                root=result["root_hit_rate"],
                coarse=result["coarse_hit_rate"],
                exact=result["exact_hit_rate"],
                ref_split=result["ref_split_bars"],
                pred_split=result["pred_split_bars"],
                rich=rich_rate,
                coverage=coverage,
            )
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    reference = load_json(args.reference_json)
    model = load_json(args.model_json)
    ref_cases = case_map(reference)
    model_cases = case_map(model)

    results = [
        evaluate_case(ref_case, model_cases.get(case_id))
        for case_id, ref_case in ref_cases.items()
    ]
    summary = aggregate(results)
    payload = {"summary": summary, "cases": results}
    if args.json:
        print(json.dumps(jsonable(payload), ensure_ascii=False, indent=2))
    else:
        print(render_markdown(results, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
