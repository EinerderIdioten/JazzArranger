#!/usr/bin/env python3
"""Prepare a chord-aware tokenizer extension for Qwen-style models."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOTS: tuple[str, ...] = (
    "C",
    "C#",
    "Db",
    "D",
    "D#",
    "Eb",
    "E",
    "E#",
    "Fb",
    "F",
    "F#",
    "Gb",
    "G",
    "G#",
    "Ab",
    "A",
    "A#",
    "Bb",
    "B",
    "B#",
    "Cb",
)

PITCH_CLASS: dict[str, int] = {
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

FIFTH_ORDER: tuple[int, ...] = (0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5)
FIFTH_INDEX: dict[int, int] = {pitch_class: index for index, pitch_class in enumerate(FIFTH_ORDER)}

QUALITIES: tuple[str, ...] = (
    "maj",
    "min",
    "maj7",
    "m7",
    "7",
    "6",
    "m6",
    "dim",
    "dim7",
    "m7b5",
    "aug",
    "sus",
    "sus7",
    "unknown",
)

TENSIONS: tuple[str, ...] = (
    "b5",
    "#5",
    "b9",
    "9",
    "#9",
    "11",
    "#11",
    "b13",
    "13",
    "alt",
)

CORE_TOKENS: tuple[str, ...] = (
    "<HOLD>",
    "<NEW>",
    "<CHORD>",
    "</CHORD>",
    "<CHANGE:0>",
    "<CHANGE:1>",
    "<ROOT:none>",
    "<QUAL:none>",
    "<TENS:none>",
    "<BASS:none>",
)

ROOT_RE = re.compile(r"^(?P<root>[A-G](?:bb|##|b|#)?)(?P<body>.*)$")
SLASH_BASS_RE = re.compile(r"^(?P<main>.*)/(?P<bass>[A-G](?:bb|##|b|#)?)$")
TENSION_RE = re.compile(r"(#11|b13|#9|b9|#5|b5|13|11|9|alt)")


@dataclass(frozen=True)
class ParsedChord:
    surface: str
    root: dict[str, Any] | None
    quality: str
    tensions: list[str]
    bass: dict[str, Any] | None
    unknown: bool
    normalized_body: str


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def pitch_class_for(spelling: str) -> int | None:
    return PITCH_CLASS.get(spelling)


def root_payload(spelling: str) -> dict[str, Any]:
    pitch_class = pitch_class_for(spelling)
    if pitch_class is None:
        return {"spelling": spelling, "pc": None, "fifths_index": None}
    fifths_index = FIFTH_INDEX[pitch_class]
    angle = 2.0 * math.pi * pitch_class / 12.0
    fifth_angle = 2.0 * math.pi * fifths_index / 12.0
    return {
        "spelling": spelling,
        "pc": pitch_class,
        "fifths_index": fifths_index,
        "features": {
            "pc_sin": round(math.sin(angle), 8),
            "pc_cos": round(math.cos(angle), 8),
            "fifths_sin": round(math.sin(fifth_angle), 8),
            "fifths_cos": round(math.cos(fifth_angle), 8),
        },
    }


def split_slash_bass(raw: str) -> tuple[str, str | None]:
    match = SLASH_BASS_RE.match(raw)
    if not match:
        return raw, None
    return match.group("main"), match.group("bass")


def normalize_body(raw: str) -> str:
    body = raw.strip()
    body = body.replace("−", "-")
    body = body.replace("Δ", "maj")
    body = body.replace("^", "maj")
    body = body.replace("%", "m7b5")
    body = body.replace("ø", "m7b5")
    body = body.replace("°", "dim")
    body = body.replace("(", "")
    body = body.replace(")", "")
    body = body.replace(",", "")
    body = body.replace("5+", "#5")
    body = body.replace("5-", "b5")
    if body.startswith("min"):
        body = "m" + body[3:]
    if body.startswith("-"):
        body = "m" + body[1:]
    if body.startswith("+"):
        body = "aug" + body[1:]
    if body.startswith("o"):
        body = "dim" + body[1:]
    if body.startswith("M"):
        body = "maj" + body[1:]
    return body


def append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def extract_tensions(rest: str) -> list[str]:
    tensions: list[str] = []
    for match in TENSION_RE.finditer(rest):
        append_unique(tensions, match.group(1))
    return tensions


def parse_quality_and_tensions(body: str) -> tuple[str, list[str], bool]:
    body = normalize_body(body)
    if not body:
        return "maj", [], False

    quality = "unknown"
    initial_tensions: list[str] = []
    rest = ""

    rules: tuple[tuple[str, str, tuple[str, ...]], ...] = (
        ("m7b5", "m7b5", ()),
        ("dim7", "dim7", ()),
        ("dim", "dim", ()),
        ("aug", "aug", ()),
        ("sus7", "sus7", ()),
        ("7sus", "sus7", ()),
        ("sus", "sus", ()),
        ("maj13", "maj7", ("13",)),
        ("maj11", "maj7", ("11",)),
        ("maj9", "maj7", ("9",)),
        ("maj7", "maj7", ()),
        ("maj", "maj", ()),
        ("m13", "m7", ("13",)),
        ("m11", "m7", ("11",)),
        ("m9", "m7", ("9",)),
        ("m7", "m7", ()),
        ("m6", "m6", ()),
        ("m", "min", ()),
        ("6/9", "6", ("9",)),
        ("#513", "7", ("#5", "13")),
        ("#511", "7", ("#5", "11")),
        ("#59", "7", ("#5", "9")),
        ("#57", "7", ("#5",)),
        ("#5", "aug", ()),
        ("b5", "dim", ()),
        ("13", "7", ("13",)),
        ("11", "7", ("11",)),
        ("9", "7", ("9",)),
        ("7", "7", ()),
        ("6", "6", ()),
        ("alt", "7", ("alt",)),
    )
    for prefix, candidate_quality, candidate_tensions in rules:
        if body.startswith(prefix):
            quality = candidate_quality
            initial_tensions = list(candidate_tensions)
            rest = body[len(prefix) :]
            break

    tensions = list(initial_tensions)
    for tension in extract_tensions(rest):
        append_unique(tensions, tension)
    if "alt" in body:
        append_unique(tensions, "alt")

    unknown = quality == "unknown"
    return quality, tensions, unknown


def parse_chord_symbol(symbol: str) -> ParsedChord:
    surface = symbol.strip()
    main, bass = split_slash_bass(surface)
    match = ROOT_RE.match(main)
    if not match:
        return ParsedChord(
            surface=surface,
            root=None,
            quality="unknown",
            tensions=[],
            bass=root_payload(bass) if bass else None,
            unknown=True,
            normalized_body=normalize_body(main),
        )

    root_spelling = match.group("root")
    body = match.group("body")
    quality, tensions, unknown = parse_quality_and_tensions(body)
    if root_spelling not in PITCH_CLASS:
        unknown = True
    return ParsedChord(
        surface=surface,
        root=root_payload(root_spelling),
        quality=quality,
        tensions=tensions,
        bass=root_payload(bass) if bass else None,
        unknown=unknown,
        normalized_body=normalize_body(body),
    )


def load_canonical_harmony(canonical_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(canonical_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for index, event in enumerate(data.get("harmony_stream", [])):
            if len(event) < 4:
                continue
            bar, beat, symbol, duration_beats = event[:4]
            rows.append(
                {
                    "source_file": path.name,
                    "title": data.get("context", {}).get("title"),
                    "event_index": index,
                    "bar": bar,
                    "beat": beat,
                    "symbol": str(symbol),
                    "duration_beats": duration_beats,
                }
            )
    return rows


def root_token(spelling: str) -> str:
    return f"<ROOT:{spelling}>"


def bass_token(spelling: str) -> str:
    return f"<BASS:{spelling}>"


def quality_token(quality: str) -> str:
    return f"<QUAL:{quality}>"


def tension_token(tension: str) -> str:
    return f"<TENS:{tension}>"


def surface_token(symbol: str) -> str:
    safe_symbol = symbol.replace(" ", "")
    return f"<SURFACE:{safe_symbol}>"


def build_token_list(
    include_surface_tokens: bool,
    surface_counts: Counter[str],
    surface_min_count: int,
) -> list[str]:
    tokens = list(CORE_TOKENS)
    tokens.extend(root_token(root) for root in ROOTS)
    tokens.extend(bass_token(root) for root in ROOTS)
    tokens.extend(quality_token(quality) for quality in QUALITIES)
    tokens.extend(tension_token(tension) for tension in TENSIONS)
    if include_surface_tokens:
        for symbol, count in sorted(surface_counts.items()):
            if count >= surface_min_count:
                tokens.append(surface_token(symbol))
    return list(dict.fromkeys(tokens))


def build_manifest(
    canonical_dir: Path,
    harmony_rows: list[dict[str, Any]],
    parsed_chords: list[ParsedChord],
    tokens: list[str],
    tokenizer_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    surface_counts = Counter(row["symbol"] for row in harmony_rows)
    root_counts = Counter(
        chord.root["spelling"] for chord in parsed_chords if chord.root is not None
    )
    quality_counts = Counter(chord.quality for chord in parsed_chords)
    tension_counts = Counter(tension for chord in parsed_chords for tension in chord.tensions)
    bass_counts = Counter(
        chord.bass["spelling"] for chord in parsed_chords if chord.bass is not None
    )
    unknown = [
        asdict(chord)
        for chord in parsed_chords
        if chord.unknown
    ]
    return {
        "schema": "qwen_chord_vocab.v0",
        "canonical_dir": str(canonical_dir),
        "design": {
            "canonical_storage_policy": (
                "Keep raw harmony_stream symbols unchanged; this manifest is a derived "
                "training/tokenizer layer."
            ),
            "chord_decomposition": ["surface", "root", "quality", "tensions", "bass"],
            "active_training_scope": ["root", "quality"],
            "reserved_future_scope": ["tensions", "bass", "surface_tokens"],
        },
        "tokens": tokens,
        "token_count": len(tokens),
        "roots": [root_payload(root) | {"token": root_token(root), "bass_token": bass_token(root)} for root in ROOTS],
        "qualities": [{"quality": quality, "token": quality_token(quality)} for quality in QUALITIES],
        "tensions": [{"tension": tension, "token": tension_token(tension)} for tension in TENSIONS],
        "observed": {
            "num_harmony_events": len(harmony_rows),
            "num_unique_surface_symbols": len(surface_counts),
            "num_unknown_events": len(unknown),
            "unknown_rate": len(unknown) / len(harmony_rows) if harmony_rows else 0.0,
            "surface_counts": surface_counts.most_common(),
            "root_counts": root_counts.most_common(),
            "quality_counts": quality_counts.most_common(),
            "tension_counts": tension_counts.most_common(),
            "bass_counts": bass_counts.most_common(),
            "unknown_examples": unknown[:50],
        },
        "tokenizer": tokenizer_summary,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row) + "\n")


def save_extended_tokenizer(
    tokenizer_path: Path,
    output_dir: Path,
    tokens: list[str],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    old_vocab_size = len(tokenizer)
    existing_special = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    merged_special = list(dict.fromkeys(existing_special + tokens))
    added_count = tokenizer.add_special_tokens({"additional_special_tokens": merged_special})
    tokenizer.save_pretrained(output_dir / "tokenizer")

    reloaded = AutoTokenizer.from_pretrained(output_dir / "tokenizer", trust_remote_code=True)
    single_token_failures = {
        token: reloaded.encode(token, add_special_tokens=False)
        for token in tokens
        if len(reloaded.encode(token, add_special_tokens=False)) != 1
    }
    return {
        "base_tokenizer_path": str(tokenizer_path),
        "output_tokenizer_dir": str(output_dir / "tokenizer"),
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": len(reloaded),
        "added_token_count": added_count,
        "single_token_pass": not single_token_failures,
        "single_token_failures": single_token_failures,
        "requires_model_resize_token_embeddings": added_count > 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-surface-tokens", action="store_true")
    parser.add_argument("--surface-min-count", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    harmony_rows = load_canonical_harmony(args.canonical_dir)
    parsed_chords = [parse_chord_symbol(row["symbol"]) for row in harmony_rows]
    surface_counts = Counter(row["symbol"] for row in harmony_rows)
    tokens = build_token_list(
        include_surface_tokens=args.include_surface_tokens,
        surface_counts=surface_counts,
        surface_min_count=args.surface_min_count,
    )
    tokenizer_summary = None
    if args.tokenizer_path is not None:
        tokenizer_summary = save_extended_tokenizer(args.tokenizer_path, args.output_dir, tokens)

    manifest = build_manifest(
        canonical_dir=args.canonical_dir,
        harmony_rows=harmony_rows,
        parsed_chords=parsed_chords,
        tokens=tokens,
        tokenizer_summary=tokenizer_summary,
    )
    write_json(args.output_dir / "chord_vocab_manifest.json", manifest)
    write_jsonl(
        args.output_dir / "parsed_harmony_events.jsonl",
        [
            {
                **row,
                "parsed": asdict(parsed),
            }
            for row, parsed in zip(harmony_rows, parsed_chords, strict=True)
        ],
    )

    print(
        "Prepared chord vocab: "
        f"events={len(harmony_rows)} tokens={len(tokens)} "
        f"unknown={manifest['observed']['num_unknown_events']} "
        f"output={args.output_dir}",
        flush=True,
    )
    if tokenizer_summary is not None:
        print(
            "Tokenizer: "
            f"old_vocab={tokenizer_summary['old_vocab_size']} "
            f"new_vocab={tokenizer_summary['new_vocab_size']} "
            f"single_token_pass={tokenizer_summary['single_token_pass']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
