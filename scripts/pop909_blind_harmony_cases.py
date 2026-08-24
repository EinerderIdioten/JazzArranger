#!/usr/bin/env python3
"""Build anonymized POP909 melody-only harmony test material.

The blind prompt produced by this script intentionally excludes song titles and
artists. Use the `list` and `review` subcommands only outside the tested model's
context.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_POP909_INDEX = REPO_ROOT / "data" / "raw" / "pop909" / "POP909" / "index.xlsx"

DEFAULT_SELECTION: list[tuple[str, str]] = [
    ("PP001", "516"),
    ("PP002", "063"),
    ("PP003", "292"),
    ("PP004", "220"),
    ("PP005", "640"),
    ("PP006", "526"),
    ("PP007", "823"),
    ("PP008", "330"),
    ("PP009", "346"),
    ("PP010", "684"),
]

QUALITY_SYMBOLS = {
    "maj": "maj",
    "maj7": "maj7",
    "min": "min",
    "min7": "min7",
    "dom7": "7",
    "dim": "dim",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "m7b5": "m7b5",
    "sus2": "sus2",
    "sus4": "sus4",
}

ABC_NOTE_RE = re.compile(
    r"(?P<acc>\^{1,2}|_{1,2}|=)?(?P<note>[A-Ga-gz])(?P<oct>[,']*)(?P<dur>\d*)"
)
BASE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
PC_TO_JIANPU = {
    0: "1",
    1: "#1",
    2: "2",
    3: "b3",
    4: "3",
    5: "4",
    6: "#4",
    7: "5",
    8: "b6",
    9: "6",
    10: "b7",
    11: "7",
}
XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create POP909 blind harmony cases, references, and review tables."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing pop909_*.jsonl processed files.",
    )
    parser.add_argument(
        "--index-xlsx",
        type=Path,
        default=DEFAULT_POP909_INDEX,
        help="POP909 index.xlsx path with song title metadata.",
    )
    parser.add_argument(
        "--song-ids",
        default="",
        help="Comma-separated POP909 song ids. Defaults to the curated PP001-PP010 set.",
    )
    parser.add_argument(
        "--case-prefix",
        default="PP",
        help="Case prefix used when --song-ids is provided.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list", help="Print the selected song table with title metadata."
    )
    list_parser.add_argument("--all", action="store_true", help="List every POP909 record.")
    list_parser.add_argument("--limit", type=int, default=0, help="Limit rows for --all.")
    list_parser.add_argument("--output", type=Path, help="Optional output markdown path.")

    prompt_parser = subparsers.add_parser(
        "prompt", help="Print a blind prompt with anonymized ABC melody cases."
    )
    prompt_parser.add_argument("--model-label", default="gpt-5.5")
    prompt_parser.add_argument("--output", type=Path, help="Optional output text path.")

    review_parser = subparsers.add_parser(
        "review", help="Print title-aware per-bar Jianpu/reference tables for human audit."
    )
    review_parser.add_argument("--max-bars", type=int, default=0)
    review_parser.add_argument("--output", type=Path, help="Optional output markdown path.")

    ref_parser = subparsers.add_parser(
        "reference-json", help="Print machine-readable reference spans for evaluation."
    )
    ref_parser.add_argument("--output", type=Path, help="Optional output JSON path.")

    return parser.parse_args()


def write_or_print(text: str, output: Path | None) -> None:
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text)


def load_processed_records(processed_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(processed_dir.glob("pop909_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                song_id = record.get("metadata", {}).get("song_id")
                if not song_id:
                    continue
                current = records.get(song_id)
                if current is None or record.get("total_grid", 0) > current.get("total_grid", 0):
                    records[song_id] = record
    return records


def cell_column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch.upper()) - ord("A") + 1)
    return index - 1


def xml_text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.findall(".//a:t", XLSX_NS))


def read_shared_strings(zf: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [xml_text(item) for item in root.findall(".//a:si", XLSX_NS)]


def read_xlsx_rows(path: Path) -> list[list[str]]:
    with ZipFile(path) as zf:
        shared_strings = read_shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", XLSX_NS):
            cells: dict[int, str] = {}
            for cell in row.findall("a:c", XLSX_NS):
                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "inlineStr":
                    inline = cell.find("a:is", XLSX_NS)
                    value = xml_text(inline) if inline is not None else ""
                else:
                    value_node = cell.find("a:v", XLSX_NS)
                    if value_node is not None and value_node.text is not None:
                        if cell_type == "s":
                            value = shared_strings[int(value_node.text)]
                        else:
                            value = value_node.text
                if cell.attrib.get("r"):
                    cells[cell_column_index(cell.attrib["r"])] = value
            if cells:
                max_index = max(cells)
                rows.append([cells.get(index, "") for index in range(max_index + 1)])
        return rows


def normalize_song_id(raw: Any) -> str:
    text = str(raw).strip()
    if not text:
        return ""
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        text = str(int(float(text)))
    digits = re.sub(r"\D", "", text)
    return digits.zfill(3) if digits else text


def load_pop909_index(path: Path) -> dict[str, dict[str, str]]:
    rows = read_xlsx_rows(path)
    if not rows:
        return {}
    headers = [cell.strip() for cell in rows[0]]
    metadata: dict[str, dict[str, str]] = {}
    for row in rows[1:]:
        item = {
            headers[index]: row[index].strip()
            for index in range(min(len(headers), len(row)))
            if headers[index]
        }
        song_id = normalize_song_id(item.get("song_id", ""))
        if song_id:
            metadata[song_id] = item
    return metadata


def parse_selection(args: argparse.Namespace) -> list[tuple[str, str]]:
    if not args.song_ids.strip():
        return DEFAULT_SELECTION
    song_ids = [normalize_song_id(item) for item in args.song_ids.split(",") if item.strip()]
    return [(f"{args.case_prefix}{index:03d}", song_id) for index, song_id in enumerate(song_ids, 1)]


def bar_grid(time_signature: str) -> int:
    numerator_text, denominator_text = time_signature.split("/", maxsplit=1)
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    grid = numerator * 16 / denominator
    if not grid.is_integer():
        raise ValueError(f"Unsupported time signature for 1/16 grid: {time_signature}")
    return int(grid)


def chord_symbol(chord: dict[str, Any]) -> str:
    quality = QUALITY_SYMBOLS.get(chord["quality"], chord["quality"])
    return f"{chord['root']}:{quality}"


def reference_spans(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"start": int(chord["start"]), "end": int(chord["end"]), "chord": chord_symbol(chord)}
        for chord in record["chords"]
    ]


def overlapping_spans(
    spans: list[dict[str, Any]], start: int, end: int
) -> list[dict[str, Any]]:
    return [span for span in spans if span["start"] < end and span["end"] > start]


def bar_harmony(spans: list[dict[str, Any]], start: int, end: int) -> str:
    parts: list[str] = []
    for span in overlapping_spans(spans, start, end):
        clipped_start = max(start, int(span["start"]))
        clipped_end = min(end, int(span["end"]))
        if clipped_start == start and clipped_end == end:
            parts.append(span["chord"])
        else:
            parts.append(f"{clipped_start - start}-{clipped_end - start}:{span['chord']}")
    return " / ".join(parts) if parts else "N"


def split_bar_count(record: dict[str, Any], grid: int) -> int:
    spans = reference_spans(record)
    total = int(record["total_grid"])
    count = 0
    for start in range(0, total, grid):
        end = min(start + grid, total)
        if len(overlapping_spans(spans, start, end)) > 1:
            count += 1
    return count


def record_stats(record: dict[str, Any]) -> dict[str, Any]:
    grid = bar_grid(record["time_signature"])
    total = int(record["total_grid"])
    bars = math.ceil(total / grid)
    qualities = Counter(chord["quality"] for chord in record["chords"])
    splits = split_bar_count(record, grid)
    off_bar_starts = sum(1 for chord in record["chords"] if int(chord["start"]) % grid != 0)
    return {
        "bar_grid": grid,
        "bars": bars,
        "chord_spans": len(record["chords"]),
        "spans_per_bar": len(record["chords"]) / bars if bars else 0.0,
        "split_bars": splits,
        "split_bar_pct": splits / bars if bars else 0.0,
        "off_bar_starts": off_bar_starts,
        "qualities": qualities,
    }


def complexity_label(stats: dict[str, Any]) -> str:
    split_pct = stats["split_bar_pct"]
    spans_per_bar = stats["spans_per_bar"]
    rare_qualities = set(stats["qualities"]) - {
        "maj",
        "min",
        "maj7",
        "min7",
        "dom7",
        "sus2",
        "sus4",
    }
    if split_pct <= 0.03 and spans_per_bar <= 0.60 and not rare_qualities:
        return "low"
    if split_pct <= 0.12 and not rare_qualities:
        return "mid-low"
    if split_pct <= 0.25:
        return "mid"
    return "high"


def selected_cases(
    args: argparse.Namespace,
    records: dict[str, dict[str, Any]],
    index: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for case_id, song_id in parse_selection(args):
        record = records.get(song_id)
        if record is None:
            raise SystemExit(f"Missing processed POP909 record for song id {song_id}")
        meta = index.get(song_id, {})
        cases.append(
            {
                "case_id": case_id,
                "song_id": song_id,
                "title": meta.get("name", ""),
                "artist": meta.get("artist", ""),
                "record": record,
                "stats": record_stats(record),
            }
        )
    return cases


def all_cases(
    records: dict[str, dict[str, Any]], index: dict[str, dict[str, str]], limit: int
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for song_id in sorted(records):
        record = records[song_id]
        meta = index.get(song_id, {})
        cases.append(
            {
                "case_id": song_id,
                "song_id": song_id,
                "title": meta.get("name", ""),
                "artist": meta.get("artist", ""),
                "record": record,
                "stats": record_stats(record),
            }
        )
    return cases[:limit] if limit else cases


def format_quality_counts(counter: Counter[str]) -> str:
    return ", ".join(f"{key}:{counter[key]}" for key in sorted(counter))


def render_list(cases: list[dict[str, Any]]) -> str:
    lines = [
        "| Case | POP909 ID | Title | Artist | key | meter | grid | bars | spans/bar | split bars | qualities | complexity |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for case in cases:
        record = case["record"]
        stats = case["stats"]
        lines.append(
            "| {case_id} | {song_id} | {title} | {artist} | {key} | {meter} | "
            "{bar_grid} | {bars} | {spans_per_bar:.2f} | {split_bars} ({split_pct:.1%}) | "
            "{qualities} | {complexity} |".format(
                case_id=case["case_id"],
                song_id=case["song_id"],
                title=case["title"],
                artist=case["artist"],
                key=record["key"],
                meter=record["time_signature"],
                bar_grid=stats["bar_grid"],
                bars=stats["bars"],
                spans_per_bar=stats["spans_per_bar"],
                split_bars=stats["split_bars"],
                split_pct=stats["split_bar_pct"],
                qualities=format_quality_counts(stats["qualities"]),
                complexity=complexity_label(stats),
            )
        )
    return "\n".join(lines)


def prompt_case(case: dict[str, Any]) -> dict[str, Any]:
    record = case["record"]
    stats = case["stats"]
    return {
        "case_id": case["case_id"],
        "task": "abc_melody_to_harmony",
        "meter": record["time_signature"],
        "key": record["key"],
        "grid_resolution": record["grid_resolution"],
        "bar_grid": stats["bar_grid"],
        "total_grid": record["total_grid"],
        "abc_melody": anonymize_abc_headers(record["abc_melody"], case["case_id"]),
    }


def anonymize_abc_headers(abc_melody: str, case_id: str) -> str:
    lines: list[str] = []
    for line in abc_melody.splitlines():
        if line.startswith("X:"):
            lines.append(f"X:{case_id}")
        elif line.startswith("T:"):
            lines.append(f"T:{case_id}")
        else:
            lines.append(line)
    return "\n".join(lines)


def render_prompt(cases: list[dict[str, Any]], model_label: str) -> str:
    input_cases = [prompt_case(case) for case in cases]
    example_json = {
        "task": "pop909_abc_melody_to_harmony_blind_v1",
        "model": model_label,
        "cases": [
            {
                "case_id": "PP001",
                "harmony_spans": [{"start": 0, "end": 8, "chord": "C:maj"}],
                "comments": ["optional, max 3 short musical comments for this case"],
            }
        ],
    }
    return "\n".join(
        [
            "You are being evaluated on pop-song harmonization from melody only.",
            "",
            "Task:",
            "Given anonymized, C-centered ABC melodies, produce one complete chord-span sequence per case.",
            "",
            "Important anti-leakage rules:",
            "1. Do not search the web, use external lead sheets, or ask for song titles.",
            "2. Do not identify or mention a song title even if the melody seems familiar.",
            "3. Do not copy memorized song changes. Treat each case as an unknown melody and harmonize from the supplied ABC, meter, key, and grid only.",
            "4. Do not output hidden reasoning or prose outside the JSON object.",
            "",
            "Output rules:",
            "1. Return strict JSON only.",
            "2. Preserve every `case_id`.",
            "3. For each case, output `harmony_spans`: a sorted list of objects with `start`, `end`, and `chord`.",
            "4. Grid units are sixteenth-note units because `L:1/16`. Bar 1 beat 1 is `start: 0`; each case provides `bar_grid`, the number of grid units per bar.",
            "5. Each case must cover `[0, total_grid)` exactly with no gaps or overlaps. Use each case's `bar_grid` for bar-length calculations.",
            "6. Use chord symbols in colon format: `C:maj`, `A:min`, `G:7`, `F#:m7b5`, `D:sus4`, `E:dim`.",
            "7. Slash bass is allowed as `C:maj/E`, but the chord root before `:` is the primary harmony.",
            "8. Prefer idiomatic pop harmony. Use half-bar changes where musically justified, but avoid changing chords on every melody note.",
            "",
            "Return this JSON shape:",
            json.dumps(example_json, ensure_ascii=False, indent=2),
            "",
            "Input cases:",
            json.dumps(input_cases, ensure_ascii=False, indent=2),
        ]
    )


def abc_body(abc_melody: str) -> str:
    lines = abc_melody.splitlines()
    body: list[str] = []
    in_body = False
    for line in lines:
        if in_body:
            body.append(line.strip())
        elif line.startswith("K:"):
            in_body = True
    return " ".join(body)


def accidental_delta(accidental: str | None) -> int:
    if accidental == "^":
        return 1
    if accidental == "^^":
        return 2
    if accidental == "_":
        return -1
    if accidental == "__":
        return -2
    return 0


def abc_token_to_jianpu(match: re.Match[str]) -> str:
    note = match.group("note")
    duration = int(match.group("dur") or "1")
    if note == "z":
        return f"0({duration})"

    base_note = note.upper()
    pitch_class = (BASE_PC[base_note] + accidental_delta(match.group("acc"))) % 12
    degree = PC_TO_JIANPU[pitch_class]
    octave = 1 if note.islower() else 0
    for mark in match.group("oct"):
        octave += 1 if mark == "'" else -1
    octave_marks = "'" * octave if octave > 0 else "," * (-octave)
    return f"{degree}{octave_marks}({duration})"


def abc_to_jianpu_bars(abc_melody: str) -> list[str]:
    bars: list[str] = []
    for raw_bar in abc_body(abc_melody).split("|"):
        tokens = [abc_token_to_jianpu(match) for match in ABC_NOTE_RE.finditer(raw_bar)]
        if tokens:
            bars.append(" ".join(tokens))
    return bars


def render_review(cases: list[dict[str, Any]], max_bars: int) -> str:
    lines = [
        "# POP909 Blind Harmony Review Material",
        "",
        "This file includes title metadata and reference harmony. Do not send it to the tested model.",
        "",
    ]
    for case in cases:
        record = case["record"]
        stats = case["stats"]
        spans = reference_spans(record)
        jianpu_bars = abc_to_jianpu_bars(record["abc_melody"])
        total_bars = min(len(jianpu_bars), stats["bars"])
        if max_bars:
            total_bars = min(total_bars, max_bars)
        lines.extend(
            [
                f"## {case['case_id']} / POP909 {case['song_id']} / {case['title']}",
                "",
                f"- Artist: {case['artist']}",
                f"- eval key: {record['key']}; meter: {record['time_signature']}; grid: {record['total_grid']}; bars: {stats['bars']}",
                "- Notation: `1(4)` means scale degree 1 lasting 4 sixteenth-grid units. `0` is rest.",
                "",
                "| Bar | Melody Jianpu | Reference Harmony |",
                "|---:|---|---|",
            ]
        )
        for bar_index in range(total_bars):
            start = bar_index * stats["bar_grid"]
            end = min(start + stats["bar_grid"], int(record["total_grid"]))
            lines.append(
                f"| {bar_index + 1} | {jianpu_bars[bar_index]} | {bar_harmony(spans, start, end)} |"
            )
        if max_bars and stats["bars"] > max_bars:
            lines.append(f"| ... | omitted after {max_bars} bars | ... |")
        lines.append("")
    return "\n".join(lines)


def render_reference_json(cases: list[dict[str, Any]]) -> str:
    payload = {
        "task": "pop909_abc_melody_to_harmony_blind_reference_v1",
        "cases": [],
    }
    for case in cases:
        record = case["record"]
        stats = case["stats"]
        payload["cases"].append(
            {
                "case_id": case["case_id"],
                "song_id": case["song_id"],
                "title": case["title"],
                "artist": case["artist"],
                "meter": record["time_signature"],
                "key": record["key"],
                "grid_resolution": record["grid_resolution"],
                "bar_grid": stats["bar_grid"],
                "total_grid": record["total_grid"],
                "reference_spans": reference_spans(record),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    records = load_processed_records(args.processed_dir)
    index = load_pop909_index(args.index_xlsx)

    if args.command == "list" and args.all:
        cases = all_cases(records, index, args.limit)
    else:
        cases = selected_cases(args, records, index)

    if args.command == "list":
        write_or_print(render_list(cases), args.output)
    elif args.command == "prompt":
        write_or_print(render_prompt(cases, args.model_label), args.output)
    elif args.command == "review":
        write_or_print(render_review(cases, args.max_bars), args.output)
    elif args.command == "reference-json":
        write_or_print(render_reference_json(cases), args.output)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
