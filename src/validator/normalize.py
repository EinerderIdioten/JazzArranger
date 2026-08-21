"""Normalization and record parsing helpers for the validator."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from src.data.common import ChordSpan, HARMONY_LINE_RE, NoteEvent, canonicalize_root_quality
from src.data.common import validate_harmony_text
from src.data.common import validate_chord_spans
from src.data.harmony_tokens import SPAN_TOKEN_LINE_RE, TOKEN_TO_QUALITY, TOKEN_TO_SHARP_ROOT

from .types import ChordEvent


ABC_NOTE_RE = re.compile(r"^(?P<acc>[\^_=]*)(?P<note>[A-Ga-gz])(?P<oct>[,']*)(?P<dur>\d+)?$")
BAR_TOKENS = {"|", "||", "|]", "[|", ":|", "|:", "[|:"}

DEFAULT_REFERENCE_FIELDS = (
    "harmony_tokens",
    "harmony",
    "chords",
)
DEFAULT_CANDIDATE_FIELDS = (
    "prediction",
    "candidate",
    "output",
    "model_output",
    "harmony_tokens",
    "harmony",
)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def get_in(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not _is_mapping(current) or part not in current:
            return None
        current = current[part]
    return current


def pick_first(record: Mapping[str, Any], candidates: Sequence[str]) -> tuple[str | None, Any]:
    for candidate in candidates:
        value = get_in(record, candidate)
        if value is not None:
            return candidate, value
    return None, None


def parse_fraction(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    text = str(value).strip()
    if not text:
        return Fraction(1, 1)
    if "/" in text:
        return Fraction(text)
    return Fraction(int(text), 1)


def parse_time_signature(value: Any) -> tuple[int, int]:
    text = str(value).strip() if value is not None else "4/4"
    if "/" not in text:
        return 4, 4
    top, bottom = text.split("/", 1)
    return int(top), int(bottom)


def infer_bar_grid(record: Mapping[str, Any]) -> int:
    if "bar_grid" in record and record["bar_grid"]:
        return int(record["bar_grid"])
    grid_resolution = parse_fraction(record.get("grid_resolution", "1/16"))
    numerator, denominator = parse_time_signature(record.get("time_signature", "4/4"))
    bar_fraction = Fraction(numerator, denominator)
    units = bar_fraction / grid_resolution
    if units.denominator != 1:
        raise ValueError(
            f"cannot infer integer bar grid from time_signature={record.get('time_signature')} "
            f"grid_resolution={record.get('grid_resolution')}"
        )
    return int(units)


def _validate_span_sequence(spans: Sequence[ChordSpan], total_grid: int | None) -> list[str]:
    errors: list[str] = []
    if not spans:
        return ["empty_chords"]
    expected = spans[0].start
    for idx, span in enumerate(spans):
        if span.end <= span.start:
            errors.append(f"non_positive_span:{idx}")
        if idx == 0 and span.start != 0:
            errors.append(f"unexpected_initial_offset:{span.start}")
        if span.start != expected:
            if span.start > expected:
                errors.append(f"gap:{expected}-{span.start}")
            else:
                errors.append(f"overlap:{span.start}<expected{expected}")
        expected = span.end
    if total_grid is not None and expected != total_grid:
        errors.append(f"bad_coverage:{expected}!={total_grid}")
    return errors


def _to_event(span: ChordSpan, *, source: str) -> ChordEvent:
    return ChordEvent(
        start=span.start,
        end=span.end,
        root=span.root,
        quality=span.quality,
        raw=span.raw or f"{span.root}:{span.quality}",
    )


def parse_chord_list(
    chords: Sequence[Any],
    *,
    total_grid: int | None = None,
) -> tuple[list[ChordEvent], list[str]]:
    spans: list[ChordSpan] = []
    errors: list[str] = []
    for idx, item in enumerate(chords):
        if isinstance(item, ChordSpan):
            spans.append(item)
            continue
        if hasattr(item, "start") and hasattr(item, "end") and hasattr(item, "root") and hasattr(item, "quality"):
            spans.append(
                ChordSpan(
                    start=int(getattr(item, "start")),
                    end=int(getattr(item, "end")),
                    root=str(getattr(item, "root")),
                    quality=str(getattr(item, "quality")),
                    raw=str(getattr(item, "raw", "")),
                )
            )
            continue
        if not _is_mapping(item):
            errors.append(f"unsupported_chord_item:{idx}:{type(item).__name__}")
            continue
        root_quality = canonicalize_root_quality(item.get("root"), item.get("quality"))
        if root_quality is None:
            errors.append(f"invalid_chord:{idx}:{item.get('root')}:{item.get('quality')}")
            continue
        root, quality = root_quality
        try:
            start = int(item["start"])
            end = int(item["end"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid_span:{idx}:{exc}")
            continue
        spans.append(
            ChordSpan(
                start=start,
                end=end,
                root=root,
                quality=quality,
                raw=str(item.get("raw", f"{root}:{quality}")),
            )
        )
    spans = sorted(spans, key=lambda span: (span.start, span.end, span.root, span.quality))
    errors.extend(_validate_span_sequence(spans, total_grid))
    events = [_to_event(span, source="chords") for span in spans]
    return events, errors


def parse_raw_harmony_text(
    harmony: str,
    *,
    total_grid: int | None = None,
) -> tuple[list[ChordEvent], list[str]]:
    text = str(harmony or "").strip()
    errors: list[str] = []
    if not text:
        return [], ["empty_harmony"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0] == "<HARMONY>" and lines[-1] == "</HARMONY>":
        body = lines[1:-1]
    else:
        body = lines
        errors.append("bad_harmony_wrapper")
    spans: list[ChordSpan] = []
    expected = 0
    for idx, line in enumerate(body):
        match = HARMONY_LINE_RE.match(line)
        if not match:
            errors.append(f"bad_harmony_line:{line}")
            continue
        root_quality = canonicalize_root_quality(match.group("root"), match.group("quality"))
        if root_quality is None:
            errors.append(f"invalid_harmony_symbol:{idx}:{line}")
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        root, quality = root_quality
        spans.append(ChordSpan(start=start, end=end, root=root, quality=quality, raw=line))
        if idx == 0 and start != 0:
            errors.append(f"unexpected_initial_offset:{start}")
        if start != expected:
            if start > expected:
                errors.append(f"gap:{expected}-{start}")
            else:
                errors.append(f"overlap:{start}<expected{expected}")
        expected = end
    spans = sorted(spans, key=lambda span: (span.start, span.end, span.root, span.quality))
    errors.extend(_validate_span_sequence(spans, total_grid))
    events = [_to_event(span, source="harmony") for span in spans]
    return events, errors


def parse_tokenized_harmony_text(
    harmony_tokens: str,
    *,
    total_grid: int | None = None,
) -> tuple[list[ChordEvent], list[str]]:
    text = str(harmony_tokens or "").strip()
    errors: list[str] = []
    if not text:
        return [], ["empty_harmony_tokens"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0] == "<HARMONY>" and lines[-1] == "</HARMONY>":
        body = lines[1:-1]
    else:
        body = lines
        errors.append("bad_harmony_token_wrapper")
    spans: list[ChordSpan] = []
    expected = 0
    for idx, line in enumerate(body):
        match = SPAN_TOKEN_LINE_RE.match(line)
        if not match:
            errors.append(f"bad_harmony_token_line:{line}")
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        root = TOKEN_TO_SHARP_ROOT[match.group("root")]
        quality = TOKEN_TO_QUALITY[match.group("quality")]
        spans.append(ChordSpan(start=start, end=end, root=root, quality=quality, raw=line))
        if idx == 0 and start != 0:
            errors.append(f"unexpected_initial_offset:{start}")
        if start != expected:
            if start > expected:
                errors.append(f"gap:{expected}-{start}")
            else:
                errors.append(f"overlap:{start}<expected{expected}")
        expected = end
    spans = sorted(spans, key=lambda span: (span.start, span.end, span.root, span.quality))
    errors.extend(_validate_span_sequence(spans, total_grid))
    events = [_to_event(span, source="harmony_tokens") for span in spans]
    return events, errors


def parse_harmony_value(
    value: Any,
    *,
    total_grid: int | None = None,
) -> tuple[list[ChordEvent], list[str], str]:
    if value is None:
        return [], ["missing_harmony"], "missing"
    if isinstance(value, str):
        if "<SPAN>" in value or "<R_" in value:
            events, errors = parse_tokenized_harmony_text(value, total_grid=total_grid)
            return events, errors, "harmony_tokens"
        events, errors = parse_raw_harmony_text(value, total_grid=total_grid)
        return events, errors, "harmony"
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        events, errors = parse_chord_list(value, total_grid=total_grid)
        return events, errors, "chords"
    if _is_mapping(value):
        for key in ("harmony_tokens", "harmony", "chords"):
            if key in value:
                events, errors, source = parse_harmony_value(value[key], total_grid=total_grid)
                return events, errors, key if source == "missing" else source
        if {"start", "end", "root", "quality"} <= set(value.keys()):
            events, errors = parse_chord_list([value], total_grid=total_grid)
            return events, errors, "chord"
    return [], [f"unsupported_harmony_type:{type(value).__name__}"], type(value).__name__


def parse_abc_melody(
    abc_text: str,
    *,
    total_grid: int | None = None,
) -> tuple[list[NoteEvent], list[str]]:
    text = str(abc_text or "").strip()
    if not text:
        return [], ["empty_abc_melody"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    body = [line for line in lines if not line.startswith(("X:", "T:", "M:", "L:", "K:"))]
    tokens: list[str] = []
    for line in body:
        tokens.extend(line.split())
    events: list[NoteEvent] = []
    errors: list[str] = []
    pos = 0
    for idx, token in enumerate(tokens):
        if token in BAR_TOKENS:
            continue
        cleaned = token
        while cleaned and cleaned[0] in "|[]:":
            cleaned = cleaned[1:]
        while cleaned and cleaned[-1] in "|[]:":
            cleaned = cleaned[:-1]
        if not cleaned:
            continue
        match = ABC_NOTE_RE.match(cleaned)
        if not match:
            errors.append(f"bad_abc_token:{idx}:{token}")
            continue
        note = match.group("note")
        duration = int(match.group("dur") or 1)
        if note.lower() == "z":
            pos += duration
            continue
        accidental = match.group("acc") or ""
        octave = match.group("oct") or ""
        semitone = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[note.upper()]
        semitone += accidental.count("^")
        semitone -= accidental.count("_")
        if note.isupper():
            midi = 60 + semitone
            midi -= 12 * octave.count(",")
        else:
            midi = 72 + semitone
            midi += 12 * octave.count("'")
        events.append(NoteEvent(start=pos, end=pos + duration, pitch=midi))
        pos += duration
    if total_grid is not None and pos != total_grid:
        errors.append(f"bad_abc_coverage:{pos}!={total_grid}")
    return events, errors


def chord_events_to_spans(events: Sequence[ChordEvent]) -> list[ChordSpan]:
    return [
        ChordSpan(start=event.start, end=event.end, root=event.root, quality=event.quality, raw=event.raw)
        for event in events
    ]


def events_to_harmony(events: Sequence[ChordEvent]) -> str:
    lines = ["<HARMONY>"]
    for event in events:
        lines.append(f"@{event.start}-{event.end} {event.root}:{event.quality}")
    lines.append("</HARMONY>")
    return "\n".join(lines)


def events_to_tokenized_harmony(events: Sequence[ChordEvent]) -> str:
    lines = ["<HARMONY>"]
    for event in events:
        flat_root = {
            "C": "C",
            "C#": "Db",
            "D": "D",
            "D#": "Eb",
            "E": "E",
            "F": "F",
            "F#": "Gb",
            "G": "G",
            "G#": "Ab",
            "A": "A",
            "A#": "Bb",
            "B": "B",
        }[event.root]
        token_root = f"<R_{flat_root}>"
        token_quality = {
            "maj": "<Q_MAJ>",
            "min": "<Q_MIN>",
            "dom7": "<Q_DOM7>",
            "maj7": "<Q_MAJ7>",
            "min7": "<Q_MIN7>",
            "dim": "<Q_DIM>",
            "hdim7": "<Q_HDIM7>",
            "aug": "<Q_AUG>",
            "sus2": "<Q_SUS2>",
            "sus4": "<Q_SUS4>",
        }[event.quality]
        lines.append(f"<SPAN> @{event.start}-{event.end} {token_root} {token_quality} </SPAN>")
    lines.append("</HARMONY>")
    return "\n".join(lines)


def validate_event_sequence(events: Sequence[ChordEvent], total_grid: int | None) -> list[str]:
    spans = chord_events_to_spans(events)
    return validate_chord_spans(spans, total_grid if total_grid is not None else (spans[-1].end if spans else 0))


class Normalizer:
    """Field resolution and raw parsing helper for validator inputs."""

    def __init__(
        self,
        *,
        reference_fields: Sequence[str] = DEFAULT_REFERENCE_FIELDS,
        candidate_fields: Sequence[str] = DEFAULT_CANDIDATE_FIELDS,
        melody_field: str = "abc_melody",
        key_fields: Sequence[str] = ("key", "original_key", "metadata.key"),
        total_grid_fields: Sequence[str] = ("total_grid", "metadata.total_grid"),
        bar_grid_fields: Sequence[str] = ("bar_grid", "grid_resolution"),
    ) -> None:
        self.reference_fields = tuple(reference_fields)
        self.candidate_fields = tuple(candidate_fields)
        self.melody_field = melody_field
        self.key_fields = tuple(key_fields)
        self.total_grid_fields = tuple(total_grid_fields)
        self.bar_grid_fields = tuple(bar_grid_fields)

    def resolve_sample_id(self, record: Mapping[str, Any]) -> str:
        for field in ("id", "case_id", "sample_id", "song_id"):
            value = get_in(record, field)
            if value is not None:
                return str(value)
        return "unknown"

    def resolve_dataset(self, record: Mapping[str, Any]) -> str | None:
        value = get_in(record, "dataset")
        return str(value) if value is not None else None

    def resolve_key(self, record: Mapping[str, Any]) -> str | None:
        for field in self.key_fields:
            value = get_in(record, field)
            if value is not None:
                return str(value)
        return None

    def resolve_total_grid(self, record: Mapping[str, Any]) -> int | None:
        for field in self.total_grid_fields:
            value = get_in(record, field)
            if value is not None:
                try:
                    return int(value)
                except Exception:  # noqa: BLE001
                    continue
        return None

    def resolve_bar_grid(self, record: Mapping[str, Any]) -> int:
        if get_in(record, "bar_grid") is not None:
            return int(get_in(record, "bar_grid"))
        return infer_bar_grid(record)

    def pick_reference_value(self, record: Mapping[str, Any]) -> tuple[str | None, Any]:
        return pick_first(record, self.reference_fields)

    def pick_candidate_value(self, record: Mapping[str, Any]) -> tuple[str | None, Any]:
        return pick_first(record, self.candidate_fields)

    def parse_harmony(self, value: Any, *, total_grid: int | None = None) -> tuple[list[ChordEvent], list[str], str]:
        return parse_harmony_value(value, total_grid=total_grid)

    def parse_melody(
        self,
        record: Mapping[str, Any],
        *,
        total_grid: int | None = None,
        field: str | None = None,
    ) -> tuple[list[NoteEvent], list[str], str | None]:
        melody_field = field or self.melody_field
        source = get_in(record, melody_field)
        if source is None:
            return [], [], None
        events, errors = parse_abc_melody(source, total_grid=total_grid)
        return events, errors, melody_field

    def record_metadata(self, record: Mapping[str, Any]) -> dict[str, Any]:
        total_grid = self.resolve_total_grid(record)
        metadata = {
            "id": self.resolve_sample_id(record),
            "dataset": self.resolve_dataset(record),
            "key": self.resolve_key(record),
            "total_grid": total_grid,
            "bar_grid": self.resolve_bar_grid(record),
        }
        if total_grid is None:
            metadata["total_grid"] = 0
        return metadata
