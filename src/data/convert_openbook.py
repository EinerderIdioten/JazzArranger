"""Conservatively audit/convert OpenBook LilyPond lead sheets to stage-one JSONL."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from src.data.common import (
    ChordSpan,
    NoteEvent,
    ROOTS_SHARP,
    canonicalize_root_quality,
    chord_spans_to_harmony,
    counter_to_sorted_dict,
    ensure_dirs,
    melody_notes_to_abc,
    normalize_root_name,
    validate_chord_spans,
    validate_harmony_text,
    write_json,
    write_jsonl,
)


DATASET_NAME = "OpenBook"
DATASET_WEIGHT = 0.0
GRIDS_PER_WHOLE = 16

LILY_ROOT_TO_SCHEMA = {
    "c": "C",
    "cis": "C#",
    "ces": "B",
    "cisis": "D",
    "ceses": "A#",
    "d": "D",
    "dis": "D#",
    "des": "C#",
    "disis": "E",
    "deses": "C",
    "e": "E",
    "eis": "F",
    "ees": "D#",
    "eisis": "F#",
    "eeses": "D",
    "f": "F",
    "fis": "F#",
    "fes": "E",
    "fisis": "G",
    "feses": "D#",
    "g": "G",
    "gis": "G#",
    "ges": "F#",
    "gisis": "A",
    "geses": "F",
    "a": "A",
    "ais": "A#",
    "aes": "G#",
    "aisis": "B",
    "aeses": "G",
    "b": "B",
    "bis": "C",
    "bes": "A#",
    "bisis": "C#",
    "beses": "A",
}

NOTE_PC = {root: ROOTS_SHARP.index(schema) for root, schema in LILY_ROOT_TO_SCHEMA.items()}

NOTE_RE = re.compile(
    r"^(?P<name>[a-g](?:isis|eses|is|es)?)[!?]*(?P<oct>[,']*)"
    r"(?:(?P<dur>\d+)(?P<dots>\.*)(?:\*(?P<mul>\d+))?)?(?P<trail>[~._^>!-]*)$"
)
CHORD_RE = re.compile(
    r"^(?P<root>[a-g](?:isis|eses|is|es)?)(?P<dur>\d+)?(?P<dots>\.*)"
    r"(?:\*(?P<mul>\d+))?(?::(?P<quality>[^{}|()\\\s]+))?$"
)
REST_RE = re.compile(r"^[rs](?P<dur>\d+)?(?P<dots>\.*)(?:\*(?P<mul>\d+))?$")


def split_for_name(name: str) -> str:
    value = sum(ord(char) for char in name)
    bucket = value % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        lines.append(line.split("%", 1)[0])
    return "\n".join(lines)


def block_for_part(text: str, part: str) -> str | None:
    pattern = re.compile(rf"%\s*if\s+part==['\"]{re.escape(part)}['\"]:(.*?)%\s*endif", re.S)
    match = pattern.search(text)
    return match.group(1) if match else None


def brace_content_after(text: str, marker: str | None = None) -> str | None:
    start_search = 0 if marker is None else text.find(marker)
    if start_search < 0:
        return None
    open_idx = text.find("{", start_search)
    if open_idx < 0:
        return None
    depth = 0
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : idx]
    return None


def choose_version(text: str) -> tuple[str, str, str] | None:
    for version in ("Fake", "Real"):
        chords = block_for_part(text, f"Chords{version}")
        voice = block_for_part(text, f"Voice{version}")
        if chords and voice:
            chord_body = brace_content_after(chords, "\\chordmode")
            voice_body = brace_content_after(voice)
            if chord_body and voice_body:
                return version, chord_body, voice_body
    return None


def duration_units(duration: str | None, dots: str | None, multiplier: str | None, default: int) -> tuple[int, int] | None:
    if duration is None:
        return default, default
    denominator = int(duration)
    if denominator <= 0:
        return None
    value = Fraction(GRIDS_PER_WHOLE, denominator)
    add = value
    for _ in dots or "":
        add /= 2
        value += add
    value *= int(multiplier or "1")
    if value.denominator != 1:
        return None
    units = int(value)
    if units <= 0:
        return None
    return units, units


def prepare_music_text(text: str) -> str:
    text = strip_comments(text)
    text = re.sub(r'"[^"]*"', " ", text)
    text = re.sub(r"\\tempo\s+(?:\d+\s*=\s*)?\d+", " ", text)
    text = re.sub(r"\\time\s+\d+\s*/\s*\d+", " ", text)
    text = re.sub(r"\\key\s+[a-g](?:isis|eses|is|es)?\s+\\(?:major|minor)", " ", text)
    text = re.sub(r"\\[A-Za-z][A-Za-z0-9_-]*", " ", text)
    text = re.sub(r"#[^\s{}|()]+", " ", text)
    return text


def music_tokens(text: str) -> list[str]:
    text = prepare_music_text(text)
    return re.findall(r"<[^>]+>\d*\.*(?:\*\d+)?|[{}|()\[\]]|[^\s{}|()\[\]]+", text)


def lily_root_to_schema(root: str) -> str | None:
    return LILY_ROOT_TO_SCHEMA.get(root.lower())


def parse_time_signature(voice_block: str) -> tuple[str, int]:
    match = re.search(r"\\time\s+(\d+)\s*/\s*(\d+)", voice_block)
    if not match:
        return "4/4", 16
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    bar_units = Fraction(numerator * GRIDS_PER_WHOLE, denominator)
    if bar_units.denominator != 1 or bar_units <= 0:
        return "4/4", 16
    return f"{numerator}/{denominator}", int(bar_units)


def parse_key(voice_block: str) -> str:
    match = re.search(r"\\key\s+([a-g](?:isis|eses|is|es)?)\s+\\(major|minor)", voice_block)
    if not match:
        return "C:maj"
    root = lily_root_to_schema(match.group(1))
    mode = "min" if match.group(2) == "minor" else "maj"
    return f"{root or 'C'}:{mode}"


def canonicalize_lily_chord(root_raw: str, quality_raw: str | None) -> tuple[str, str] | None:
    root = normalize_root_name(lily_root_to_schema(root_raw) or root_raw)
    if root is None:
        return None
    quality = (quality_raw or "").split("/", 1)[0]
    if quality.startswith("maj9") or quality.startswith("maj13"):
        return root, "maj7"
    if quality.startswith("m7.5-"):
        return root, "hdim7"
    if quality.startswith("m9") or quality.startswith("m11") or quality.startswith("m13"):
        return root, "min7"
    if quality.startswith("m7"):
        return root, "min7"
    if quality.startswith("maj7") or quality.startswith("M7"):
        return root, "maj7"
    if quality.startswith("maj") or quality in {"", "6"} or quality.startswith("6."):
        return root, "maj"
    if quality.startswith("m6") or quality == "m" or quality.startswith("m."):
        return root, "min"
    if quality.startswith("m5"):
        return root, "dim"
    if quality.startswith("5.11"):
        return root, "sus4"
    if "sus2" in quality:
        return root, "sus2"
    if "sus4" in quality or "sus" in quality:
        return root, "sus4"
    if quality.startswith(("7", "9", "11", "13", "5+")):
        return root, "dom7"
    if quality.startswith("dim") or quality.startswith("o"):
        return root, "dim"
    if quality.startswith("aug") or quality == "+":
        return root, "aug"
    return canonicalize_root_quality(root, quality)


def parse_lily_chords(chord_block: str) -> tuple[list[ChordSpan], int, list[str]]:
    tokens = music_tokens(chord_block)
    position = 0
    last_duration = 16
    spans: list[ChordSpan] = []
    errors: list[str] = []

    for token in tokens:
        if token in {"{", "}", "|", "(", ")", "[", "]"}:
            continue
        rest = REST_RE.match(token)
        if rest:
            parsed_duration = duration_units(rest.group("dur"), rest.group("dots"), rest.group("mul"), last_duration)
            if parsed_duration is None:
                errors.append(f"bad_chord_duration:{token}")
                continue
            duration, last_duration = parsed_duration
            position += duration
            continue

        match = CHORD_RE.match(token)
        if not match:
            continue
        parsed_duration = duration_units(match.group("dur"), match.group("dots"), match.group("mul"), last_duration)
        if parsed_duration is None:
            errors.append(f"bad_chord_duration:{token}")
            continue
        duration, last_duration = parsed_duration
        canonical = canonicalize_lily_chord(match.group("root"), match.group("quality"))
        if canonical is None:
            errors.append(f"uncanonical_chord:{token}")
            position += duration
            continue
        root, quality = canonical
        if spans and spans[-1].root == root and spans[-1].quality == quality and spans[-1].end == position:
            prev = spans[-1]
            spans[-1] = ChordSpan(prev.start, position + duration, prev.root, prev.quality, prev.raw)
        else:
            spans.append(ChordSpan(position, position + duration, root, quality, token))
        position += duration

    return spans, position, errors


def closest_relative_midi(pitch_class: int, previous: int) -> int:
    base_octave = previous // 12
    candidates = [base_octave * 12 + pitch_class + shift * 12 for shift in (-1, 0, 1)]
    return min(candidates, key=lambda value: abs(value - previous))


def parse_note_name(token: str) -> tuple[str, str, str | None, str | None, str | None] | None:
    match = NOTE_RE.match(token)
    if not match:
        return None
    name = match.group("name").lower()
    if name not in NOTE_PC:
        return None
    return name, match.group("oct"), match.group("dur"), match.group("dots"), match.group("mul")


def note_token_to_event(
    token: str,
    *,
    position: int,
    previous_midi: int,
    last_duration: int,
) -> tuple[NoteEvent | None, int, int, int] | None:
    parsed = parse_note_name(token)
    if parsed is None:
        return None
    name, octave_marks, duration_raw, dots, multiplier = parsed
    parsed_duration = duration_units(duration_raw, dots, multiplier, last_duration)
    if parsed_duration is None:
        return None
    duration, new_last_duration = parsed_duration
    midi = closest_relative_midi(NOTE_PC[name], previous_midi)
    midi += 12 * octave_marks.count("'")
    midi -= 12 * octave_marks.count(",")
    return NoteEvent(position, position + duration, midi), position + duration, midi, new_last_duration


def chord_note_token_to_event(
    token: str,
    *,
    position: int,
    previous_midi: int,
    last_duration: int,
) -> tuple[NoteEvent | None, int, int, int] | None:
    match = re.match(r"^<(?P<body>[^>]+)>(?P<dur>\d+)?(?P<dots>\.*)(?:\*(?P<mul>\d+))?$", token)
    if not match:
        return None
    parsed_duration = duration_units(match.group("dur"), match.group("dots"), match.group("mul"), last_duration)
    if parsed_duration is None:
        return None
    duration, new_last_duration = parsed_duration

    pitches: list[int] = []
    for item in match.group("body").split():
        parsed_note = parse_note_name(item)
        if parsed_note is None:
            continue
        name, octave_marks, _, _, _ = parsed_note
        midi = closest_relative_midi(NOTE_PC[name], previous_midi)
        midi += 12 * octave_marks.count("'")
        midi -= 12 * octave_marks.count(",")
        pitches.append(midi)
    if not pitches:
        return None
    midi = max(pitches)
    return NoteEvent(position, position + duration, midi), position + duration, midi, new_last_duration


def parse_lily_voice(voice_block: str) -> tuple[list[NoteEvent], int, list[str]]:
    if "\\tuplet" in voice_block or "\\times" in voice_block:
        return [], 0, ["unsupported_tuplet"]
    if "\\grace" in voice_block or "\\acciaccatura" in voice_block or "\\appoggiatura" in voice_block:
        return [], 0, ["unsupported_grace"]

    tokens = music_tokens(voice_block)
    position = 0
    last_duration = 4
    previous_midi = 60
    notes: list[NoteEvent] = []
    errors: list[str] = []

    for token in tokens:
        if token in {"{", "}", "|", "(", ")", "[", "]"}:
            continue
        rest = REST_RE.match(token)
        if rest:
            parsed_duration = duration_units(rest.group("dur"), rest.group("dots"), rest.group("mul"), last_duration)
            if parsed_duration is None:
                errors.append(f"bad_note_duration:{token}")
                continue
            duration, last_duration = parsed_duration
            position += duration
            continue

        parsed_event = chord_note_token_to_event(
            token,
            position=position,
            previous_midi=previous_midi,
            last_duration=last_duration,
        )
        if parsed_event is None:
            parsed_event = note_token_to_event(
                token,
                position=position,
                previous_midi=previous_midi,
                last_duration=last_duration,
            )
        if parsed_event is None:
            continue
        event, position, previous_midi, last_duration = parsed_event
        if event is not None:
            notes.append(event)

    return notes, position, errors


def split_contiguous_spans(spans: list[ChordSpan]) -> list[list[ChordSpan]]:
    segments: list[list[ChordSpan]] = []
    current: list[ChordSpan] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if current and span.start != current[-1].end:
            segments.append(current)
            current = []
        current.append(span)
    if current:
        segments.append(current)
    return segments


def rebase_openbook_spans(segment: list[ChordSpan]) -> tuple[int, int, list[ChordSpan]]:
    source_start = segment[0].start
    source_end = segment[-1].end
    rebased = [
        ChordSpan(span.start - source_start, span.end - source_start, span.root, span.quality, span.raw)
        for span in segment
    ]
    return source_start, source_end, rebased


def clip_notes(notes: list[NoteEvent], start: int, end: int) -> list[NoteEvent]:
    clipped: list[NoteEvent] = []
    for note in notes:
        clipped_start = max(start, note.start)
        clipped_end = min(end, note.end)
        if clipped_end > clipped_start:
            clipped.append(
                NoteEvent(start=clipped_start - start, end=clipped_end - start, pitch=note.pitch)
            )
    return clipped


def convert_file(path: Path) -> tuple[list[dict], list[dict]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    selected = choose_version(text)
    if selected is None:
        return [], [{"id": path.stem, "errors": ["missing_chord_voice_pair"]}]
    version, chord_block, voice_block = selected

    time_signature, bar_grid = parse_time_signature(voice_block)
    key = parse_key(voice_block)
    chord_spans, chord_total, chord_errors = parse_lily_chords(chord_block)
    notes, note_total, note_errors = parse_lily_voice(voice_block)
    errors = chord_errors + note_errors
    if not chord_spans:
        errors.append("missing_chords")
    if not notes:
        errors.append("missing_notes")
    if errors:
        return [], [{"id": path.stem, "version": version, "errors": errors[:20]}]

    rows: list[dict] = []
    failures: list[dict] = []
    split = split_for_name(path.stem)
    segments = split_contiguous_spans(chord_spans)

    for idx, segment in enumerate(segments):
        source_start, source_end, rebased_spans = rebase_openbook_spans(segment)
        total_grid = source_end - source_start
        segment_notes = clip_notes(notes, source_start, source_end)
        segment_errors = validate_chord_spans(rebased_spans, total_grid)
        if not segment_notes:
            segment_errors.append("missing_notes")
        if total_grid <= 0:
            segment_errors.append("empty_total_grid")
        if segment_errors:
            failures.append(
                {
                    "id": path.stem,
                    "version": version,
                    "segment": idx,
                    "source_grid": [source_start, source_end],
                    "errors": segment_errors,
                }
            )
            continue

        title = f"openbook_{path.stem}_seg{idx}_{version.lower()}"
        abc_melody, abc_stats = melody_notes_to_abc(
            segment_notes,
            total_grid=total_grid,
            title=title,
            key=key,
            time_signature=time_signature,
            unit_length="1/16",
            bar_grid=bar_grid,
        )
        harmony = chord_spans_to_harmony(rebased_spans)
        harmony_errors = validate_harmony_text(harmony)
        if harmony_errors:
            failures.append(
                {
                    "id": path.stem,
                    "version": version,
                    "segment": idx,
                    "source_grid": [source_start, source_end],
                    "errors": harmony_errors,
                }
            )
            continue

        rows.append(
            {
                "id": f"openbook:{path.stem}:seg{idx}:{version.lower()}",
                "dataset": DATASET_NAME,
                "dataset_weight": DATASET_WEIGHT,
                "split": split,
                "source_path": str(path),
                "abc_melody": abc_melody,
                "grid_resolution": "1/16",
                "time_signature": time_signature,
                "key": key,
                "total_grid": total_grid,
                "chords": [span.to_json() for span in rebased_spans],
                "harmony": harmony,
                "metadata": {
                    "source": "src/openbook",
                    "version": version,
                    "source_grid_start": source_start,
                    "source_grid_end": source_end,
                    "source_chord_total_grid": chord_total,
                    "source_voice_total_grid": note_total,
                    "conversion_status": "audit_pending",
                },
                "conversion": abc_stats,
            }
        )
    return rows, failures


def convert(raw_root: Path, output_dir: Path, log_dir: Path) -> dict:
    ensure_dirs(output_dir, log_dir)
    source_dir = raw_root / "src" / "openbook"
    paths = sorted(source_dir.glob("*.ly.mako"))
    rows_by_split: dict[str, list[dict]] = defaultdict(list)
    failures: list[dict] = []
    quality_counter: Counter = Counter()
    root_counter: Counter = Counter()
    split_counter: Counter = Counter()
    conversion_counter: Counter = Counter()

    for path in paths:
        rows, file_failures = convert_file(path)
        failures.extend(file_failures)
        for failure in file_failures:
            for error in failure["errors"]:
                conversion_counter[error.split(":", 1)[0]] += 1
        for row in rows:
            rows_by_split[row["split"]].append(row)
            split_counter[row["split"]] += 1
            conversion_counter["converted"] += 1
            conversion_counter["dropped_same_start_notes"] += row["conversion"]["dropped_same_start_notes"]
            conversion_counter["truncated_overlapping_notes"] += row["conversion"]["truncated_overlapping_notes"]
            for chord in row["chords"]:
                quality_counter[chord["quality"]] += 1
                root_counter[chord["root"]] += 1

    for split in ("train", "val", "test"):
        rows = rows_by_split.get(split, [])
        if rows:
            write_jsonl(output_dir / f"openbook_{split}.jsonl", rows)

    report = {
        "dataset": DATASET_NAME,
        "dataset_weight": DATASET_WEIGHT,
        "raw_root": str(raw_root),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "total_files": len(paths),
        "converted_records": conversion_counter["converted"],
        "failed_files_or_segments": len(failures),
        "split_counts": counter_to_sorted_dict(split_counter),
        "quality_distribution": counter_to_sorted_dict(quality_counter),
        "root_distribution": counter_to_sorted_dict(root_counter),
        "conversion_stats": counter_to_sorted_dict(conversion_counter),
        "failures": failures[:300],
        "notes": [
            "OpenBook rows are emitted for manual audit with dataset_weight=0.0.",
            "The parser is conservative and skips tuplets/grace-note files because timing would need LilyPond expansion.",
        ],
    }
    write_json(log_dir / "openbook_conversion_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/openbook/openbook-master"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    args = parser.parse_args()
    report = convert(args.raw_root, args.output_dir, args.log_dir)
    print(
        f"OpenBook converted={report['converted_records']} "
        f"failed={report['failed_files_or_segments']} splits={report['split_counts']}"
    )


if __name__ == "__main__":
    main()
