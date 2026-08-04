"""Convert HLSD event JSON files into ABC melody + harmony spans."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data.common import (
    ChordSpan,
    NoteEvent,
    ROOTS_SHARP,
    chord_spans_to_harmony,
    counter_to_sorted_dict,
    ensure_dirs,
    melody_notes_to_abc,
    validate_chord_spans,
    validate_harmony_text,
    write_json,
    write_jsonl,
)
from src.data.harmony_tokens import chord_spans_to_tokenized_harmony, validate_tokenized_harmony_text


DATASET_NAME = "HLSD"
DATASET_WEIGHT = 1.0
GRIDS_PER_BEAT = 4


def beat_to_grid(value: object) -> int:
    return int(round(float(value) * GRIDS_PER_BEAT))


def hlsd_key(metadata: dict) -> str:
    key = str(metadata.get("key") or "C")
    return f"{key}:maj"


def canonicalize_hlsd_chord(chord: dict) -> tuple[str, str] | None:
    if chord.get("isRest"):
        return None
    try:
        root = ROOTS_SHARP[int(chord["root"]) % 12]
    except (KeyError, TypeError, ValueError):
        return None

    sus = chord.get("sus")
    if sus in {"sus2", "sus4"}:
        return root, str(sus)

    quality = str(chord.get("quality") or "")
    try:
        chord_type = int(chord.get("chord_type") or 5)
    except (TypeError, ValueError):
        chord_type = 5

    if quality == "":
        return root, "dom7" if chord_type >= 7 else "maj"
    if quality == "m":
        return root, "min7" if chord_type >= 7 else "min"
    if quality == "maj":
        return root, "maj7" if chord_type >= 7 else "maj"
    if quality == "o":
        return root, "dim"
    if quality in {"ø", "/o7"}:
        return root, "hdim7"
    return None


def parse_hlsd_file(path: Path) -> tuple[dict | None, list[str]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"json_error:{type(exc).__name__}"]
    if not isinstance(obj, dict):
        return None, ["unexpected_json_shape"]
    tracks = obj.get("tracks") or {}
    metadata = obj.get("metadata") or {}
    melody = [event for event in tracks.get("melody", []) if event]
    chords = [event for event in tracks.get("chord", []) if event]
    if not melody:
        return None, ["missing_melody"]
    if not chords:
        return None, ["missing_chords"]

    try:
        beats_in_measure = int(float(metadata.get("beats_in_measure") or 4))
        num_measures = float(obj.get("num_measures") or 0)
    except (TypeError, ValueError):
        return None, ["bad_meter"]
    bar_grid = beats_in_measure * GRIDS_PER_BEAT
    total_grid = int(round(num_measures * bar_grid))
    if total_grid <= 0:
        return None, ["bad_total_grid"]

    note_events: list[NoteEvent] = []
    for event in melody:
        if event.get("isRest"):
            continue
        try:
            start = beat_to_grid(event["event_on"])
            end = beat_to_grid(event["event_off"])
            pitch = 60 + int(round(float(event["pitch"])))
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            note_events.append(NoteEvent(start=start, end=end, pitch=pitch))

    chord_spans: list[ChordSpan] = []
    errors: list[str] = []
    for event in sorted(chords, key=lambda item: float(item.get("event_on", 0.0))):
        canonical = canonicalize_hlsd_chord(event)
        if canonical is None:
            errors.append(f"uncanonical_chord:{event.get('symbol')}")
            continue
        start = beat_to_grid(event.get("event_on"))
        end = beat_to_grid(event.get("event_off"))
        if end <= start:
            errors.append(f"non_positive_chord_duration:{event.get('symbol')}")
            continue
        root, quality = canonical
        raw = str(event.get("symbol") or f"{root}:{quality}")
        if chord_spans and chord_spans[-1].root == root and chord_spans[-1].quality == quality and chord_spans[-1].end == start:
            prev = chord_spans[-1]
            chord_spans[-1] = ChordSpan(prev.start, end, prev.root, prev.quality, prev.raw)
        else:
            chord_spans.append(ChordSpan(start, end, root, quality, raw))

    errors.extend(validate_chord_spans(chord_spans, total_grid))
    if not note_events:
        errors.append("missing_notes")
    if errors:
        return None, errors

    source_id = str(path.relative_to(path.parents[5])).replace("/", ":").replace("_symbol_key.json", "")
    key = hlsd_key(metadata)
    abc_melody, abc_stats = melody_notes_to_abc(
        note_events,
        total_grid=total_grid,
        title=source_id.replace(":", "_"),
        key=key,
        time_signature=f"{beats_in_measure}/4",
        unit_length="1/16",
        bar_grid=bar_grid,
    )
    harmony = chord_spans_to_harmony(chord_spans)
    harmony_tokens = chord_spans_to_tokenized_harmony(chord_spans)
    harmony_errors = validate_harmony_text(harmony) + validate_tokenized_harmony_text(harmony_tokens)
    if harmony_errors:
        return None, harmony_errors

    row = {
        "id": f"hlsd:{source_id}",
        "dataset": DATASET_NAME,
        "dataset_weight": DATASET_WEIGHT,
        "split": "train",
        "source_path": str(path),
        "abc_melody": abc_melody,
        "grid_resolution": "1/16",
        "time_signature": f"{beats_in_measure}/4",
        "key": key,
        "total_grid": total_grid,
        "chords": [span.to_json() for span in chord_spans],
        "harmony": harmony,
        "harmony_tokens": harmony_tokens,
        "metadata": metadata,
        "conversion": abc_stats,
    }
    return row, []


def event_paths(raw_root: Path) -> list[Path]:
    dataset_dir = raw_root / "datasets"
    list_path = dataset_dir / "event_list.json"
    if list_path.exists():
        rels = json.loads(list_path.read_text(encoding="utf-8"))
        return sorted((dataset_dir / rel).resolve() for rel in rels)
    return sorted((dataset_dir / "event").rglob("*_symbol_key.json"))


def convert(raw_root: Path, output_dir: Path, log_dir: Path) -> dict:
    ensure_dirs(output_dir, log_dir)
    rows: list[dict] = []
    failures: list[dict] = []
    quality_counter: Counter = Counter()
    root_counter: Counter = Counter()
    conversion_counter: Counter = Counter()

    paths = event_paths(raw_root)
    for path in paths:
        row, errors = parse_hlsd_file(path)
        if row is None:
            failures.append({"source_path": str(path), "errors": errors})
            for error in errors:
                conversion_counter[error.split(":", 1)[0]] += 1
            continue
        rows.append(row)
        conversion_counter["converted"] += 1
        conversion_counter["dropped_same_start_notes"] += row["conversion"]["dropped_same_start_notes"]
        conversion_counter["truncated_overlapping_notes"] += row["conversion"]["truncated_overlapping_notes"]
        for chord in row["chords"]:
            quality_counter[chord["quality"]] += 1
            root_counter[chord["root"]] += 1

    write_jsonl(output_dir / "hlsd_train.jsonl", rows)
    report = {
        "dataset": DATASET_NAME,
        "dataset_weight": DATASET_WEIGHT,
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "total_files": len(paths),
        "converted_files": len(rows),
        "failed_files": len(failures),
        "split_counts": {"train": len(rows)},
        "quality_distribution": counter_to_sorted_dict(quality_counter),
        "root_distribution": counter_to_sorted_dict(root_counter),
        "conversion_stats": counter_to_sorted_dict(conversion_counter),
        "failures": failures[:200],
    }
    write_json(log_dir / "hlsd_conversion_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/hlsd/lead-sheet-dataset-master"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    args = parser.parse_args()
    report = convert(args.raw_root, args.output_dir, args.log_dir)
    print(f"HLSD converted={report['converted_files']} failed={report['failed_files']}")


if __name__ == "__main__":
    main()
