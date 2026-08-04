#!/usr/bin/env python3
"""Transpose canonical lead-sheet JSON files for key augmentation."""

from __future__ import annotations

import argparse
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any


BASE_PC: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

SHARP_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

FLAT_NAMES: tuple[str, ...] = (
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "Gb",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
)

CONVENTIONAL_MAJOR_NAMES: tuple[str, ...] = FLAT_NAMES
CONVENTIONAL_MINOR_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "Eb",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "Bb",
    "B",
)

FLAT_KEY_TONICS: set[str] = {
    "F",
    "Bb",
    "Eb",
    "Ab",
    "Db",
    "Gb",
    "Cb",
    "D",
    "G",
    "C",
}

SHARP_KEY_TONICS: set[str] = {
    "G",
    "D",
    "A",
    "E",
    "B",
    "F#",
    "C#",
}

PITCH_RE = re.compile(r"^(?P<name>[A-G](?:bb|##|b|#)?)(?P<octave>-?\d+)$")
CHORD_ROOT_RE = re.compile(r"^(?P<root>[A-G](?:bb|##|b|#)?)(?P<body>.*)$")
SLASH_BASS_RE = re.compile(r"^(?P<main>.*)/(?P<bass>[A-G](?:bb|##|b|#)?)(?P<tail>.*)$")
KEY_RE = re.compile(r"^(?P<tonic>[A-G](?:bb|##|b|#)?)(?:\s+(?P<mode>major|minor))?$")


def note_pc(name: str) -> int:
    if not name:
        raise ValueError("empty note name")
    letter = name[0]
    if letter not in BASE_PC:
        raise ValueError(f"unsupported note letter: {name!r}")
    accidental = name[1:]
    offset = accidental.count("#") - accidental.count("b")
    if len(accidental) != accidental.count("#") + accidental.count("b"):
        raise ValueError(f"unsupported accidental spelling: {name!r}")
    return (BASE_PC[letter] + offset) % 12


def name_for_pc(pc: int, spelling_policy: str) -> str:
    names = FLAT_NAMES if spelling_policy == "flat" else SHARP_NAMES
    return names[pc % 12]


def spelling_policy_for_key(key: str | None, fallback: str = "flat") -> str:
    if not key:
        return fallback
    match = KEY_RE.match(key.strip())
    if not match:
        return fallback
    tonic = match.group("tonic")
    if "b" in tonic:
        return "flat"
    if "#" in tonic:
        return "sharp"
    mode = match.group("mode") or "major"
    if mode == "minor":
        # Natural minor signatures: D/G/C/F minor are flat-side keys, while
        # E/B/F#/C# minor are sharp-side keys.
        if tonic in {"D", "G", "C", "F"}:
            return "flat"
        if tonic in {"E", "B", "F#", "C#"}:
            return "sharp"
    if tonic in SHARP_KEY_TONICS:
        return "sharp"
    if tonic in FLAT_KEY_TONICS:
        return "flat"
    return fallback


def parse_key(key: str) -> tuple[str, str]:
    match = KEY_RE.match(key.strip())
    if not match:
        raise ValueError(f"Unsupported key format: {key!r}")
    return match.group("tonic"), match.group("mode") or "major"


def conventional_key_for_pc(pc: int, mode: str) -> str:
    names = CONVENTIONAL_MINOR_NAMES if mode == "minor" else CONVENTIONAL_MAJOR_NAMES
    return f"{names[pc % 12]} {mode}"


def transpose_key(key: str | None, semitones: int, spelling_policy: str) -> str | None:
    if key is None:
        return None
    try:
        tonic, mode = parse_key(key)
    except ValueError:
        return key
    transposed = name_for_pc(note_pc(tonic) + semitones, spelling_policy)
    return f"{transposed} {mode}"


def semitones_between_keys(source_key: str, target_key: str) -> int:
    source_tonic, _source_mode = parse_key(source_key)
    target_tonic, _target_mode = parse_key(target_key)
    distance = (note_pc(target_tonic) - note_pc(source_tonic)) % 12
    if distance > 6:
        distance -= 12
    return distance


def transpose_pitch(pitch: str, semitones: int, spelling_policy: str) -> str:
    if pitch == "R":
        return pitch
    match = PITCH_RE.match(pitch)
    if not match:
        return pitch
    octave = int(match.group("octave"))
    midi = 12 * (octave + 1) + note_pc(match.group("name"))
    transposed_midi = midi + semitones
    transposed_pc = transposed_midi % 12
    transposed_octave = transposed_midi // 12 - 1
    return f"{name_for_pc(transposed_pc, spelling_policy)}{transposed_octave}"


def transpose_chord_symbol(symbol: str, semitones: int, spelling_policy: str) -> str:
    slash_match = SLASH_BASS_RE.match(symbol)
    if slash_match:
        main = slash_match.group("main")
        bass = slash_match.group("bass")
        tail = slash_match.group("tail")
        return (
            transpose_chord_symbol(main, semitones, spelling_policy)
            + "/"
            + name_for_pc(note_pc(bass) + semitones, spelling_policy)
            + tail
        )

    match = CHORD_ROOT_RE.match(symbol)
    if not match:
        return symbol
    root = match.group("root")
    body = match.group("body")
    return name_for_pc(note_pc(root) + semitones, spelling_policy) + body


def transpose_canonical(
    data: dict[str, Any],
    semitones: int,
    target_key: str | None,
    spelling_policy: str | None,
    source_name: str,
) -> dict[str, Any]:
    result = deepcopy(data)
    context = result.setdefault("context", {})
    source_key = context.get("key")
    if target_key is None and source_key is not None and spelling_policy is None:
        try:
            source_tonic, source_mode = parse_key(source_key)
            new_key = conventional_key_for_pc(note_pc(source_tonic) + semitones, source_mode)
        except ValueError:
            new_key = transpose_key(source_key, semitones, "flat")
    else:
        new_key = target_key or transpose_key(source_key, semitones, spelling_policy or "flat")
    policy = spelling_policy or spelling_policy_for_key(new_key or target_key or source_key)

    if new_key is not None:
        context["key"] = new_key
    context["transposition"] = {
        "source_file": source_name,
        "source_key": source_key,
        "target_key": new_key,
        "semitones": semitones,
        "spelling_policy": policy,
    }

    harmony_stream = []
    for event in result.get("harmony_stream", []):
        if len(event) < 4:
            harmony_stream.append(event)
            continue
        row = list(event)
        row[2] = transpose_chord_symbol(str(row[2]), semitones, policy)
        harmony_stream.append(row)
    result["harmony_stream"] = harmony_stream

    melody_stream = []
    for event in result.get("melody_stream", []):
        if len(event) < 5:
            melody_stream.append(event)
            continue
        row = list(event)
        row[2] = transpose_pitch(str(row[2]), semitones, policy)
        melody_stream.append(row)
    result["melody_stream"] = melody_stream
    return result


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact_suffix(semitones: int, target_key: str | None) -> str:
    if target_key:
        key_part = re.sub(r"[^A-Za-z0-9#b]+", "_", target_key).strip("_")
        return f"to_{key_part}"
    return f"tr_{semitones:+03d}".replace("+", "p").replace("-", "m")


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.json"))
    return [input_path]


def output_path_for(input_file: Path, output_dir: Path, suffix: str) -> Path:
    stem = input_file.name
    if stem.endswith(".canonical.json"):
        stem = stem.removesuffix(".canonical.json")
        return output_dir / f"{stem}.{suffix}.canonical.json"
    return output_dir / f"{input_file.stem}.{suffix}{input_file.suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical JSON file or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--semitones", type=int)
    mode.add_argument("--target-key")
    mode.add_argument("--all-keys", action="store_true")
    parser.add_argument(
        "--spelling-policy",
        choices=["flat", "sharp"],
        help="Override enharmonic spelling policy for transposed notes/chords.",
    )
    parser.add_argument(
        "--skip-zero",
        action="store_true",
        help="With --all-keys, omit the untransposed semitone-0 copy.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_files = iter_input_files(args.input)
    if not input_files:
        raise SystemExit(f"No JSON files found in {args.input}")

    written: list[Path] = []
    for input_file in input_files:
        data = load_json(input_file)
        source_key = data.get("context", {}).get("key")
        jobs: list[tuple[int, str | None]] = []
        if args.all_keys:
            jobs = [(step, None) for step in range(12) if not (args.skip_zero and step == 0)]
        elif args.target_key:
            if not source_key:
                raise SystemExit(f"{input_file}: --target-key requires context.key")
            jobs = [(semitones_between_keys(source_key, args.target_key), args.target_key)]
        else:
            jobs = [(args.semitones, None)]

        for semitones, target_key in jobs:
            transposed = transpose_canonical(
                data,
                semitones=semitones,
                target_key=target_key,
                spelling_policy=args.spelling_policy,
                source_name=str(input_file),
            )
            suffix = compact_suffix(semitones, target_key)
            output_path = output_path_for(input_file, args.output_dir, suffix)
            write_json(output_path, transposed)
            written.append(output_path)

    print(f"wrote {len(written)} transposed canonical files to {args.output_dir}")
    for path in written[:20]:
        print(path)
    if len(written) > 20:
        print(f"... {len(written) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
