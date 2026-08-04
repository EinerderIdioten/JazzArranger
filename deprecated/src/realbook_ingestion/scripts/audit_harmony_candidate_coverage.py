#!/usr/bin/env python3
"""Audit rule-based harmony-rhythm candidate coverage against harmony_stream."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from realbook_ingestion.scripts.run_qwen3_harmony_rhythm_baseline import (  # noqa: E402
    absolute_position,
    backbone_beats_for_meter,
    build_candidate_grid,
    max_bar,
    meter_beats,
    normalize_beat,
    time_key,
)


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")


def candidate_policy_metadata() -> dict[str, Any]:
    return {
        "name": "elastic_melody_rest_onset_plus_meter_backbone",
        "implementation": "run_qwen3_harmony_rhythm_baseline.build_candidate_grid",
        "rules": [
            "melody note onset creates a candidate",
            "rest onset creates a candidate",
            "meter-specific backbone beats create continuation candidates when no onset exists",
        ],
        "backbone_beats": {
            "4/4": backbone_beats_for_meter("4/4"),
            "2/2": backbone_beats_for_meter("2/2"),
            "3/4": backbone_beats_for_meter("3/4"),
            "5/4": backbone_beats_for_meter("5/4"),
            "6/8": backbone_beats_for_meter("6/8"),
        },
    }


def nearest_candidate(
    candidates: list[dict[str, Any]],
    bar: int,
    beat: float,
    beats_per_bar: float,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    target = absolute_position(bar, beat, beats_per_bar)
    nearest = min(candidates, key=lambda item: abs(float(item["absolute_position"]) - target))
    return {
        "step": nearest["step"],
        "bar": nearest["bar"],
        "beat": nearest["beat"],
        "type": nearest["type"],
        "pitch": nearest["pitch"],
        "note_duration": nearest["note_duration"],
        "distance_beats": round(float(nearest["absolute_position"]) - target, 6),
    }


def load_tune(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_tune(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    data = load_tune(path)
    context = data["context"]
    tune_max_bar = max_bar(data)
    if tune_max_bar < 1:
        raise ValueError(f"{path}: no positive bars")
    candidates, labels, total, covered, uncovered = build_candidate_grid(
        context,
        data["melody_stream"],
        data["harmony_stream"],
        bar_start=1,
        window_bars=tune_max_bar,
    )
    beats_per_bar = meter_beats(context["meter"])
    candidate_by_key = {
        time_key(candidate["bar"], float(candidate["beat"])): candidate
        for candidate in candidates
    }

    harmony_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    for index, event in enumerate(data.get("harmony_stream", [])):
        if len(event) < 4 or int(event[0]) < 1:
            continue
        bar = int(event[0])
        beat = float(event[1])
        symbol = str(event[2])
        duration = event[3]
        candidate = candidate_by_key.get(time_key(bar, beat))
        row = {
            "source_file": path.name,
            "title": context.get("title", path.stem),
            "key": context.get("key"),
            "meter": context.get("meter"),
            "harmony_index": index,
            "bar": bar,
            "beat": normalize_beat(beat),
            "symbol": symbol,
            "duration_beats": duration,
            "covered": candidate is not None,
            "candidate": None,
        }
        if candidate is not None:
            row["candidate"] = {
                "step": candidate["step"],
                "type": candidate["type"],
                "pitch": candidate["pitch"],
                "note_duration": candidate["note_duration"],
            }
        else:
            row["nearest_candidate"] = nearest_candidate(candidates, bar, beat, beats_per_bar)
            missed_rows.append(row)
        harmony_rows.append(row)

    summary = {
        "source_file": path.name,
        "title": context.get("title", path.stem),
        "key": context.get("key"),
        "meter": context.get("meter"),
        "max_bar": tune_max_bar,
        "num_candidates": len(candidates),
        "candidate_type_counts": Counter(candidate["type"] for candidate in candidates),
        "num_harmony_events": total,
        "covered_harmony_events": covered,
        "missed_harmony_events": total - covered,
        "coverage_rate": covered / total if total else 0.0,
    }
    return summary, harmony_rows, missed_rows


def aggregate(
    tune_summaries: list[dict[str, Any]],
    harmony_rows: list[dict[str, Any]],
    missed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_harmony = sum(row["num_harmony_events"] for row in tune_summaries)
    covered = sum(row["covered_harmony_events"] for row in tune_summaries)
    candidate_type_counts: Counter[str] = Counter()
    for row in tune_summaries:
        candidate_type_counts.update(row["candidate_type_counts"])
    missed_by_meter = Counter(row["meter"] for row in missed_rows)
    missed_by_beat = Counter(str(row["beat"]) for row in missed_rows)
    missed_by_symbol = Counter(row["symbol"] for row in missed_rows)
    coverage_by_meter: dict[str, dict[str, Any]] = {}
    meter_totals: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "covered": 0})
    for row in harmony_rows:
        bucket = meter_totals[row["meter"]]
        bucket["total"] += 1
        bucket["covered"] += 1 if row["covered"] else 0
    for meter, counts in sorted(meter_totals.items()):
        coverage_by_meter[meter] = {
            **counts,
            "coverage_rate": counts["covered"] / counts["total"] if counts["total"] else 0.0,
        }
    return {
        "num_tunes": len(tune_summaries),
        "num_harmony_events": total_harmony,
        "covered_harmony_events": covered,
        "missed_harmony_events": total_harmony - covered,
        "coverage_rate": covered / total_harmony if total_harmony else 0.0,
        "candidate_type_counts": candidate_type_counts.most_common(),
        "coverage_by_meter": coverage_by_meter,
        "missed_by_meter": missed_by_meter.most_common(),
        "missed_by_beat": missed_by_beat.most_common(),
        "missed_by_symbol": missed_by_symbol.most_common(30),
    }


def markdown_table(rows: list[list[Any]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def write_report(
    path: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
    tune_summaries: list[dict[str, Any]],
    harmony_rows: list[dict[str, Any]],
    missed_rows: list[dict[str, Any]],
) -> None:
    worst_tunes = sorted(tune_summaries, key=lambda row: row["coverage_rate"])[:15]
    missed_preview = missed_rows[:40]
    covered_preview = [row for row in harmony_rows if row["covered"]][:20]
    lines = [
        "# Harmony Candidate Coverage Audit",
        "",
        "## Summary",
        "",
        "This report compares rule-based candidate mask slots against canonical `harmony_stream` chord onsets. It audits whether the model has a legal prediction slot at each gold chord-change position.",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Canonical source | `{metadata['canonical_dir']}` |",
        f"| Key filter | `{metadata['key_filter']}` |",
        f"| Candidate policy | `{metadata['candidate_policy']['name']}` |",
        f"| Tunes | `{summary['num_tunes']}` |",
        f"| Harmony events | `{summary['num_harmony_events']}` |",
        f"| Covered events | `{summary['covered_harmony_events']}` |",
        f"| Missed events | `{summary['missed_harmony_events']}` |",
        f"| Coverage | `{summary['coverage_rate']:.2%}` |",
        "",
        "## Candidate Policy",
        "",
        "Backbone beats by meter:",
        "",
    ]
    lines.extend(
        markdown_table(
            [
                [meter, beats]
                for meter, beats in metadata["candidate_policy"]["backbone_beats"].items()
            ],
            ["Meter", "Backbone beats"],
        )
    )
    lines.extend(
        [
            "",
            "Candidate type counts:",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            [[name, count] for name, count in summary["candidate_type_counts"]],
            ["Candidate type", "Count"],
        )
    )
    lines.extend(["", "## Coverage By Meter", ""])
    lines.extend(
        markdown_table(
            [
                [
                    meter,
                    values["total"],
                    values["covered"],
                    values["total"] - values["covered"],
                    f"{values['coverage_rate']:.2%}",
                ]
                for meter, values in summary["coverage_by_meter"].items()
            ],
            ["Meter", "Harmony events", "Covered", "Missed", "Coverage"],
        )
    )
    lines.extend(["", "## Worst Tunes", ""])
    lines.extend(
        markdown_table(
            [
                [
                    row["title"],
                    row["source_file"],
                    row["meter"],
                    row["num_harmony_events"],
                    row["missed_harmony_events"],
                    f"{row['coverage_rate']:.2%}",
                ]
                for row in worst_tunes
            ],
            ["Title", "File", "Meter", "Harmony events", "Missed", "Coverage"],
        )
    )
    lines.extend(["", "## Missed Onset Distribution", ""])
    lines.extend(
        markdown_table(
            [[beat, count] for beat, count in summary["missed_by_beat"]],
            ["Beat", "Missed count"],
        )
    )
    lines.extend(["", "## Missed Examples", ""])
    if missed_preview:
        lines.extend(
            markdown_table(
                [
                    [
                        row["title"],
                        row["bar"],
                        row["beat"],
                        row["symbol"],
                        row.get("nearest_candidate"),
                    ]
                    for row in missed_preview
                ],
                ["Title", "Bar", "Beat", "Chord", "Nearest candidate"],
            )
        )
    else:
        lines.append("No missed harmony onsets for this key filter.")
    lines.extend(["", "## Covered Examples", ""])
    lines.extend(
        markdown_table(
            [
                [
                    row["title"],
                    row["bar"],
                    row["beat"],
                    row["symbol"],
                    row["candidate"],
                ]
                for row in covered_preview
            ],
            ["Title", "Bar", "Beat", "Chord", "Candidate slot"],
        )
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metadata.json`: audit configuration",
            "- `summary.json`: aggregate coverage metrics",
            "- `tune_summaries.jsonl`: one row per tune",
            "- `harmony_event_comparison.jsonl`: one row per harmony event",
            "- `missed_harmony_events.jsonl`: only uncovered harmony events",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--key", default="C major", help="Filter by context.key; use ALL for all keys.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    key_filter = None if args.key == "ALL" else args.key
    tune_summaries: list[dict[str, Any]] = []
    harmony_rows: list[dict[str, Any]] = []
    missed_rows: list[dict[str, Any]] = []
    for path in sorted(args.canonical_dir.glob("*.canonical.json")):
        data = load_tune(path)
        if key_filter is not None and data.get("context", {}).get("key") != key_filter:
            continue
        tune_summary, tune_harmony_rows, tune_missed_rows = audit_tune(path)
        tune_summary["candidate_type_counts"] = dict(tune_summary["candidate_type_counts"])
        tune_summaries.append(tune_summary)
        harmony_rows.extend(tune_harmony_rows)
        missed_rows.extend(tune_missed_rows)

    summary = aggregate(tune_summaries, harmony_rows, missed_rows)
    metadata = {
        "canonical_dir": str(args.canonical_dir),
        "key_filter": args.key,
        "candidate_policy": candidate_policy_metadata(),
    }
    write_json(args.output_dir / "metadata.json", metadata)
    write_json(args.output_dir / "summary.json", summary)
    write_jsonl(args.output_dir / "tune_summaries.jsonl", tune_summaries)
    write_jsonl(args.output_dir / "harmony_event_comparison.jsonl", harmony_rows)
    write_jsonl(args.output_dir / "missed_harmony_events.jsonl", missed_rows)
    write_report(
        args.output_dir / "COVERAGE_AUDIT.md",
        metadata=metadata,
        summary=summary,
        tune_summaries=tune_summaries,
        harmony_rows=harmony_rows,
        missed_rows=missed_rows,
    )
    print(
        f"audited key={args.key!r} tunes={summary['num_tunes']} "
        f"coverage={summary['coverage_rate']:.2%} missed={summary['missed_harmony_events']} "
        f"output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
