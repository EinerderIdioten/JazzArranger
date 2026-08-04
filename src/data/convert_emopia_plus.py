"""Convert EMOPIA+ REMI lead-sheet events into ABC melody + harmony spans."""

from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from src.data.common import (
    ChordSpan,
    NoteEvent,
    canonicalize_emopia_chord,
    chord_spans_to_harmony,
    counter_to_sorted_dict,
    ensure_dirs,
    key_to_schema_key,
    melody_notes_to_abc,
    validate_chord_spans,
    validate_harmony_text,
    write_json,
    write_jsonl,
)


DATASET_NAME = "EMOPIA+"
DATASET_WEIGHT = 1.0
GRID_PER_BAR = 16
TICKS_PER_GRID = 120


def load_split_map(split_dir: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}_SL.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                song_id = row.get("songID")
                if song_id:
                    split_map[song_id] = split
    return split_map


def emopia_song_id(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) < 3:
        return stem
    return "_".join(parts[1:-1])


def key_from_functional(functional_path: Path) -> str:
    if not functional_path.exists():
        return "C:maj"
    try:
        events = pickle.load(functional_path.open("rb"))[1]
    except Exception:
        return "C:maj"
    for event in events:
        if event.get("name") == "Key":
            return key_to_schema_key(str(event.get("value")))
    return "C:maj"


def parse_remi_events(path: Path) -> tuple[list[NoteEvent], list[tuple[int, str]], int, list[str]]:
    errors: list[str] = []
    try:
        obj = pickle.load(path.open("rb"))
    except Exception as exc:
        return [], [], 0, [f"pickle_error:{type(exc).__name__}"]
    if not isinstance(obj, tuple) or len(obj) < 2 or not isinstance(obj[1], list):
        return [], [], 0, ["unexpected_pickle_shape"]

    events = obj[1]
    notes: list[NoteEvent] = []
    chord_points: list[tuple[int, str]] = []
    bar = -1
    beat = 0
    pending_pitch: tuple[int, int] | None = None
    bar_count = 0

    for event in events:
        name = event.get("name")
        value = event.get("value")
        if name == "Bar":
            bar += 1
            bar_count += 1
            beat = 0
        elif name == "Beat":
            try:
                beat = int(value)
            except (TypeError, ValueError):
                errors.append(f"bad_beat:{value}")
        elif name == "Chord":
            if bar < 0:
                errors.append("chord_before_bar")
                continue
            chord_points.append((bar * GRID_PER_BAR + beat, str(value)))
        elif name == "Note_Pitch":
            if bar < 0:
                errors.append("note_before_bar")
                continue
            try:
                pending_pitch = (bar * GRID_PER_BAR + beat, int(value))
            except (TypeError, ValueError):
                errors.append(f"bad_pitch:{value}")
                pending_pitch = None
        elif name == "Note_Duration" and pending_pitch is not None:
            try:
                duration = int(round(int(value) / TICKS_PER_GRID))
            except (TypeError, ValueError):
                errors.append(f"bad_duration:{value}")
                pending_pitch = None
                continue
            start, pitch = pending_pitch
            if duration > 0:
                notes.append(NoteEvent(start=start, end=start + duration, pitch=pitch))
            else:
                errors.append(f"non_positive_note_duration:{value}")
            pending_pitch = None

    total_grid = max(bar_count * GRID_PER_BAR, max((note.end for note in notes), default=0))
    return notes, sorted(chord_points), total_grid, errors


def build_chord_spans(chord_points: list[tuple[int, str]], total_grid: int) -> tuple[list[ChordSpan], list[str]]:
    errors: list[str] = []
    if not chord_points:
        return [], ["missing_chords"]
    if any(raw == "None_None" for _, raw in chord_points):
        return [], ["contains_no_chord"]

    spans: list[ChordSpan] = []
    for idx, (start, raw) in enumerate(chord_points):
        end = chord_points[idx + 1][0] if idx + 1 < len(chord_points) else total_grid
        canonical = canonicalize_emopia_chord(raw)
        if canonical is None:
            errors.append(f"uncanonical_chord:{raw}")
            continue
        if end <= start:
            errors.append(f"non_positive_chord_duration:{start}-{end}:{raw}")
            continue
        root, quality = canonical
        candidate = ChordSpan(start=start, end=end, root=root, quality=quality, raw=raw)
        if spans and spans[-1].root == root and spans[-1].quality == quality and spans[-1].end == start:
            prev = spans[-1]
            spans[-1] = ChordSpan(prev.start, end, prev.root, prev.quality, prev.raw)
        else:
            spans.append(candidate)

    errors.extend(validate_chord_spans(spans, total_grid))
    return spans, errors


def convert_one(path: Path, raw_root: Path, split_map: dict[str, str]) -> tuple[dict | None, dict]:
    notes, chord_points, total_grid, parse_errors = parse_remi_events(path)
    functional_path = raw_root / "functional" / "lead_sheet" / path.name
    key = key_from_functional(functional_path)
    song_id = emopia_song_id(path.stem)
    split = split_map.get(song_id, "unknown")
    spans, chord_errors = build_chord_spans(chord_points, total_grid)

    errors = parse_errors + chord_errors
    if not notes:
        errors.append("missing_notes")
    if total_grid <= 0:
        errors.append("empty_total_grid")
    if errors:
        return None, {
            "id": path.stem,
            "split": split,
            "song_id": song_id,
            "errors": errors,
        }

    abc_melody, abc_stats = melody_notes_to_abc(
        notes,
        total_grid=total_grid,
        title=path.stem,
        key=key,
        time_signature="4/4",
        unit_length="1/16",
        bar_grid=GRID_PER_BAR,
    )
    harmony = chord_spans_to_harmony(spans)
    harmony_errors = validate_harmony_text(harmony)
    if harmony_errors:
        return None, {
            "id": path.stem,
            "split": split,
            "song_id": song_id,
            "errors": harmony_errors,
        }

    row = {
        "id": f"emopia_plus:{path.stem}",
        "dataset": DATASET_NAME,
        "dataset_weight": DATASET_WEIGHT,
        "split": split,
        "source_path": str(path),
        "abc_melody": abc_melody,
        "grid_resolution": "1/16",
        "time_signature": "4/4",
        "key": key,
        "total_grid": total_grid,
        "chords": [span.to_json() for span in spans],
        "harmony": harmony,
        "conversion": abc_stats,
    }
    return row, {"id": path.stem, "split": split, "song_id": song_id, "errors": []}


def convert(raw_root: Path, output_dir: Path, log_dir: Path) -> dict:
    remi_dir = raw_root / "REMI" / "lead_sheet"
    split_map = load_split_map(raw_root / "split")
    ensure_dirs(output_dir, log_dir)

    rows_by_split: dict[str, list[dict]] = defaultdict(list)
    failures: list[dict] = []
    quality_counter: Counter = Counter()
    root_counter: Counter = Counter()
    split_counter: Counter = Counter()
    conversion_counter: Counter = Counter()

    for path in sorted(remi_dir.glob("*.pkl")):
        row, audit = convert_one(path, raw_root, split_map)
        if row is None:
            failures.append(audit)
            for error in audit["errors"]:
                conversion_counter[error.split(":", 1)[0]] += 1
            continue
        rows_by_split[row["split"]].append(row)
        split_counter[row["split"]] += 1
        conversion_counter["converted"] += 1
        conversion_counter["dropped_same_start_notes"] += row["conversion"]["dropped_same_start_notes"]
        conversion_counter["truncated_overlapping_notes"] += row["conversion"]["truncated_overlapping_notes"]
        for chord in row["chords"]:
            quality_counter[chord["quality"]] += 1
            root_counter[chord["root"]] += 1

    for split in ("train", "val", "test", "unknown"):
        rows = rows_by_split.get(split, [])
        if rows:
            write_jsonl(output_dir / f"emopia_plus_{split}.jsonl", rows)

    report = {
        "dataset": DATASET_NAME,
        "dataset_weight": DATASET_WEIGHT,
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "total_files": len(list(remi_dir.glob("*.pkl"))),
        "converted_files": conversion_counter["converted"],
        "failed_files": len(failures),
        "split_counts": counter_to_sorted_dict(split_counter),
        "quality_distribution": counter_to_sorted_dict(quality_counter),
        "root_distribution": counter_to_sorted_dict(root_counter),
        "conversion_stats": counter_to_sorted_dict(conversion_counter),
        "failures": failures[:200],
    }
    write_json(log_dir / "emopia_plus_conversion_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/emopia_plus/EMOPIA+"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    args = parser.parse_args()
    report = convert(args.raw_root, args.output_dir, args.log_dir)
    print(
        f"EMOPIA+ converted={report['converted_files']} failed={report['failed_files']} "
        f"splits={report['split_counts']}"
    )


if __name__ == "__main__":
    main()
