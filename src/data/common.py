"""Shared data conversion helpers for melody-to-harmony labeling."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOTS_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ROOT_ALIASES = {
    "Cb": "B",
    "Db": "C#",
    "Eb": "D#",
    "Fb": "E",
    "Gb": "F#",
    "Ab": "G#",
    "Bb": "A#",
    "B#": "C",
    "E#": "F",
}

ABC_PITCHES_SHARP = {
    0: "C",
    1: "^C",
    2: "D",
    3: "^D",
    4: "E",
    5: "F",
    6: "^F",
    7: "G",
    8: "^G",
    9: "A",
    10: "^A",
    11: "B",
}

QUALITY_MAP = {
    "": "maj",
    "M": "maj",
    "maj": "maj",
    "major": "maj",
    "m": "min",
    "min": "min",
    "minor": "min",
    "7": "dom7",
    "dom7": "dom7",
    "M7": "maj7",
    "maj7": "maj7",
    "m7": "min7",
    "min7": "min7",
    "o": "dim",
    "dim": "dim",
    "o7": "dim",
    "dim7": "dim",
    "/o7": "hdim7",
    "ø7": "hdim7",
    "hdim7": "hdim7",
    "+": "aug",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
}

VALID_QUALITIES = {
    "maj",
    "min",
    "dom7",
    "maj7",
    "min7",
    "dim",
    "hdim7",
    "aug",
    "sus2",
    "sus4",
}

HARMONY_LINE_RE = re.compile(r"^@(?P<start>\d+)-(?P<end>\d+)\s+(?P<root>[A-G](?:#|b)?):(?P<quality>[A-Za-z0-9_+]+)$")


@dataclass(frozen=True)
class NoteEvent:
    start: int
    end: int
    pitch: int


@dataclass(frozen=True)
class ChordSpan:
    start: int
    end: int
    root: str
    quality: str
    raw: str

    def to_json(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "root": self.root,
            "quality": self.quality,
            "raw": self.raw,
        }


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: object) -> None:
    ensure_dirs(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    ensure_dirs(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def normalize_root_name(raw_root: str) -> str | None:
    root = str(raw_root).strip()
    if not root:
        return None
    root = root[:1].upper() + root[1:]
    root = ROOT_ALIASES.get(root, root)
    if root not in ROOTS_SHARP:
        return None
    return root


def key_to_schema_key(raw_key: str | None) -> str:
    if not raw_key:
        return "C:maj"
    raw_key = str(raw_key)
    mode = "min" if raw_key[:1].islower() else "maj"
    root = normalize_root_name(raw_key) or raw_key[:1].upper() + raw_key[1:]
    return f"{root}:{mode}"


def canonicalize_root_quality(root: str | int, quality: str) -> tuple[str, str] | None:
    if isinstance(root, int):
        if root < 0 or root >= len(ROOTS_SHARP):
            return None
        root_name = ROOTS_SHARP[root]
    else:
        root_name = normalize_root_name(str(root))
        if root_name is None:
            return None
    mapped_quality = QUALITY_MAP.get(str(quality).strip())
    if mapped_quality not in VALID_QUALITIES:
        return None
    return root_name, mapped_quality


def canonicalize_emopia_chord(raw: str) -> tuple[str, str] | None:
    if raw == "None_None":
        return None
    if "_" not in raw:
        return None
    root_raw, quality_raw = raw.split("_", 1)
    try:
        root = int(root_raw)
    except ValueError:
        return None
    return canonicalize_root_quality(root, quality_raw)


def chord_spans_to_harmony(chords: list[ChordSpan]) -> str:
    lines = ["<HARMONY>"]
    for chord in chords:
        lines.append(f"@{chord.start}-{chord.end} {chord.root}:{chord.quality}")
    lines.append("</HARMONY>")
    return "\n".join(lines)


def validate_chord_spans(chords: list[ChordSpan], total_grid: int) -> list[str]:
    errors: list[str] = []
    if not chords:
        return ["empty_chords"]
    expected = 0
    for idx, chord in enumerate(chords):
        if chord.end <= chord.start:
            errors.append(f"non_positive_span:{idx}")
        if chord.start != expected:
            if chord.start > expected:
                errors.append(f"gap:{expected}-{chord.start}")
            else:
                errors.append(f"overlap:{chord.start}<expected{expected}")
        if chord.root not in ROOTS_SHARP:
            errors.append(f"invalid_root:{idx}:{chord.root}")
        if chord.quality not in VALID_QUALITIES:
            errors.append(f"invalid_quality:{idx}:{chord.quality}")
        expected = chord.end
    if expected != total_grid:
        errors.append(f"bad_coverage:{expected}!={total_grid}")
    return errors


def validate_harmony_text(harmony: str) -> list[str]:
    lines = harmony.strip().splitlines()
    if len(lines) < 3 or lines[0] != "<HARMONY>" or lines[-1] != "</HARMONY>":
        return ["bad_harmony_wrapper"]
    errors: list[str] = []
    for line in lines[1:-1]:
        match = HARMONY_LINE_RE.match(line)
        if not match:
            errors.append(f"bad_harmony_line:{line}")
    return errors


def midi_pitch_to_abc(pitch: int) -> str:
    pc = pitch % 12
    octave_offset = pitch // 12 - 5
    spelling = ABC_PITCHES_SHARP[pc]
    accidental = ""
    letter = spelling
    if spelling.startswith("^"):
        accidental = "^"
        letter = spelling[1:]
    if octave_offset >= 1:
        body = accidental + letter.lower() + ("'" * (octave_offset - 1))
    elif octave_offset == 0:
        body = accidental + letter
    else:
        body = accidental + letter + ("," * (-octave_offset))
    return body


def abc_duration_suffix(units: int) -> str:
    return "" if units == 1 else str(units)


def melody_notes_to_abc(
    notes: list[NoteEvent],
    *,
    total_grid: int,
    title: str,
    key: str,
    time_signature: str = "4/4",
    unit_length: str = "1/16",
    bar_grid: int = 16,
) -> tuple[str, dict]:
    sorted_notes = sorted(notes, key=lambda note: (note.start, note.end, -note.pitch))
    per_start: dict[int, list[NoteEvent]] = {}
    for note in sorted_notes:
        if note.end > note.start:
            per_start.setdefault(note.start, []).append(note)

    selected: list[NoteEvent] = []
    dropped_same_start = 0
    for start, group in sorted(per_start.items()):
        group = sorted(group, key=lambda note: (note.pitch, note.end - note.start), reverse=True)
        selected.append(group[0])
        dropped_same_start += max(0, len(group) - 1)

    tokens: list[str] = []
    pos = 0
    truncated_overlaps = 0

    def append_segment(kind: str, pitch: int | None, start: int, end: int) -> None:
        cur = start
        while cur < end:
            next_bar = ((cur // bar_grid) + 1) * bar_grid
            seg_end = min(end, next_bar)
            units = seg_end - cur
            if kind == "rest":
                token = "z" + abc_duration_suffix(units)
            else:
                assert pitch is not None
                token = midi_pitch_to_abc(pitch) + abc_duration_suffix(units)
            tokens.append(token)
            cur = seg_end
            if cur % bar_grid == 0:
                tokens.append("|")

    for idx, note in enumerate(selected):
        if note.start > pos:
            append_segment("rest", None, pos, note.start)
            pos = note.start
        note_end = min(note.end, total_grid)
        next_start = selected[idx + 1].start if idx + 1 < len(selected) else total_grid
        if note_end > next_start:
            note_end = next_start
            truncated_overlaps += 1
        if note_end > pos:
            append_segment("note", note.pitch, pos, note_end)
            pos = note_end
    if pos < total_grid:
        append_segment("rest", None, pos, total_grid)
    if not tokens or tokens[-1] != "|":
        tokens.append("|")

    body_lines = []
    line: list[str] = []
    for token in tokens:
        line.append(token)
        if token == "|" and len(line) >= 16:
            body_lines.append(" ".join(line))
            line = []
    if line:
        body_lines.append(" ".join(line))

    abc = "\n".join(
        [
            f"X:{title}",
            f"T:{title}",
            f"M:{time_signature}",
            f"L:{unit_length}",
            f"K:{key.split(':', 1)[0]}",
            *body_lines,
        ]
    )
    stats = {
        "dropped_same_start_notes": dropped_same_start,
        "truncated_overlapping_notes": truncated_overlaps,
        "selected_note_count": len(selected),
    }
    return abc, stats


def counter_to_sorted_dict(counter: Counter) -> dict:
    return {key: counter[key] for key in sorted(counter)}
