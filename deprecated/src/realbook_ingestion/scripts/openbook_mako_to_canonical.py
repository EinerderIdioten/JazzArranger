#!/usr/bin/env python3
"""Convert OpenBook .ly.mako lead sheets into canonical lead-sheet JSON.

This is a conservative static converter. It does not execute Mako or
LilyPond; it extracts the OpenBook sections and parses the LilyPond subset
used by the lead-sheet files in src/openbook.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable


NOTE_RE = r"[a-g](?:isis|eses|is|es)?"
PITCH_TOKEN_RE = re.compile(
    rf"^(?P<pitch>{NOTE_RE}|r|s)(?P<oct>[,']*)(?P<natural>!?)"
    r"(?P<duration>\d+\.*(?:\*\d+(?:/\d+)?)?)?"
    r"(?P<suffix>.*)$"
)
CHORD_TOKEN_RE = re.compile(
    rf"^(?P<root>{NOTE_RE}|r|s)(?P<oct>[,']*)(?P<duration>\d+\.*(?:\*\d+(?:/\d+)?)?)?"
    r"(?P<quality>:[^/\s\\{}|]+)?(?P<bass>/"
    rf"{NOTE_RE}(?:[,']*)?)?(?P<suffix>.*)$"
)
PART_START_RE = re.compile(r"^%\s*if\s+part==['\"]([^'\"]+)['\"]\s*:")
ATTRIBUTE_RE = re.compile(r"attributes\[['\"]([^'\"]+)['\"]\]\s*=\s*(.+)$")
DEFAULT_VERSION_RE = re.compile(r"attributes\.set_default_version_name\(['\"]([^'\"]+)['\"]\)")
ADD_VERSION_RE = re.compile(r"attributes\.add_version\(['\"]([^'\"]+)['\"]")


LILYPOND_TO_CANONICAL_PITCH = {
    "c": ("C", 0),
    "cis": ("C#", 1),
    "cisis": ("C##", 2),
    "ces": ("Cb", -1),
    "ceses": ("Cbb", -2),
    "d": ("D", 2),
    "dis": ("D#", 3),
    "disis": ("D##", 4),
    "des": ("Db", 1),
    "deses": ("Dbb", 0),
    "e": ("E", 4),
    "eis": ("E#", 5),
    "eisis": ("E##", 6),
    "ees": ("Eb", 3),
    "eeses": ("Ebb", 2),
    "f": ("F", 5),
    "fis": ("F#", 6),
    "fisis": ("F##", 7),
    "fes": ("Fb", 4),
    "feses": ("Fbb", 3),
    "g": ("G", 7),
    "gis": ("G#", 8),
    "gisis": ("G##", 9),
    "ges": ("Gb", 6),
    "geses": ("Gbb", 5),
    "a": ("A", 9),
    "ais": ("A#", 10),
    "aisis": ("A##", 11),
    "aes": ("Ab", 8),
    "aeses": ("Abb", 7),
    "b": ("B", 11),
    "bis": ("B#", 12),
    "bisis": ("B##", 13),
    "bes": ("Bb", 10),
    "beses": ("Bbb", 9),
}


IGNORED_COMMANDS = {
    r"\chordmode",
    r"\startChords",
    r"\endChords",
    r"\startSong",
    r"\endSong",
    r"\startPart",
    r"\endPart",
    r"\myEndLine",
    r"\myEndLineVolta",
    r"\myEndLineVoltaLast",
    r"\myEndLineVoltaNotLast",
    r"\myFakeEndLine",
    r"\partBar",
    r"\endBar",
    r"\startBar",
    r"\startRepeat",
    r"\endRepeat",
    r"\LPC",
    r"\RPC",
    r"\OPC",
    r"\repeatTie",
    r"\fermata",
    r"\numericTimeSignature",
    r"\mark",
    r"\markup",
    r"\italic",
    r"\parenthesize",
    r"\xNote",
    r"\myGrace",
    r"\grace",
}


@dataclass(frozen=True)
class ScaleStart:
    factor: Fraction


@dataclass(frozen=True)
class ScaleEnd:
    pass


class Warnings:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def add(self, message: str) -> None:
        self._counts[message] = self._counts.get(message, 0) + 1

    def notes(self, limit: int = 40) -> list[str]:
        items = sorted(self._counts.items(), key=lambda item: (-item[1], item[0]))
        rendered = [
            f"{message} (x{count})" if count > 1 else message
            for message, count in items[:limit]
        ]
        if len(items) > limit:
            rendered.append(f"{len(items) - limit} additional converter warnings omitted.")
        return rendered


class Cursor:
    def __init__(self, meter_beats: Fraction) -> None:
        self.meter_beats = meter_beats
        self.bar = 1
        self.beat = Fraction(1)

    def set_partial(self, duration: Fraction) -> None:
        self.bar = 0
        self.beat = self.meter_beats - duration + 1

    def position(self) -> tuple[int, Fraction]:
        return self.bar, self.beat

    def advance(self, duration: Fraction) -> None:
        self.beat += duration
        next_bar_beat = self.meter_beats + 1
        while self.beat >= next_bar_beat:
            self.beat -= self.meter_beats
            self.bar = 1 if self.bar == 0 else self.bar + 1


def fraction_to_json_number(value: Fraction) -> int | float:
    if value.denominator == 1:
        return value.numerator
    return round(float(value), 6)


def clean_scalar(value: object) -> object:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def strip_lilypond_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        escaped = False
        out = []
        for char in line:
            if char == "%" and not escaped:
                break
            out.append(char)
            escaped = char == "\\" and not escaped
        lines.append("".join(out))
    return "\n".join(lines)


def strip_metadata_directives(text: str) -> str:
    text = re.sub(r'\\tempo\s+(?:"[^"]*"\s*)?\d+\s*=\s*\d+', " ", text)
    text = re.sub(r"\\time\s+\d+/\d+", " ", text)
    text = re.sub(r"\\key\s+[a-g](?:isis|eses|is|es)?\s+\\(?:major|minor)", " ", text)
    return text


def tokenize_lilypond(text: str) -> list[str]:
    text = strip_lilypond_comments(text)
    tokens: list[str] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isspace():
            i += 1
            continue
        if char == '"':
            j = i + 1
            escaped = False
            while j < len(text):
                if text[j] == '"' and not escaped:
                    j += 1
                    break
                escaped = text[j] == "\\" and not escaped
                if text[j] != "\\":
                    escaped = False
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        if char in "{}|":
            tokens.append(char)
            i += 1
            continue
        if char == "<":
            j = i + 1
            while j < len(text) and text[j] != ">":
                j += 1
            if j < len(text):
                j += 1
                while j < len(text) and not text[j].isspace() and text[j] not in "{}|\\":
                    j += 1
            tokens.append(text[i:j])
            i = j
            continue
        if char == "\\":
            j = i + 1
            while j < len(text) and text[j].isalpha():
                j += 1
            if j == i + 1 and j < len(text):
                j += 1
            tokens.append(text[i:j])
            i = j
            continue
        j = i
        while j < len(text) and not text[j].isspace() and text[j] not in "{}|\\":
            j += 1
        tokens.append(text[i:j])
        i = j
    return [token for token in tokens if token]


def parse_duration_expr(raw: str, warnings: Warnings | None = None) -> Fraction | None:
    match = re.fullmatch(r"(?P<base>\d+)(?P<dots>\.*)(?:\*(?P<num>\d+)(?:/(?P<den>\d+))?)?", raw)
    if not match:
        if warnings is not None:
            warnings.add(f"Unsupported duration expression: {raw}")
        return None
    base = int(match.group("base"))
    if base == 0:
        if warnings is not None:
            warnings.add("Ignoring zero-valued duration.")
        return None
    value = Fraction(4, base)
    add = value
    for _dot in match.group("dots"):
        add /= 2
        value += add
    if match.group("num"):
        value *= int(match.group("num"))
        if match.group("den"):
            value /= int(match.group("den"))
    return value


def extract_duration(raw: str | None, inherited: Fraction | None, default: Fraction, warnings: Warnings) -> Fraction:
    if raw:
        parsed = parse_duration_expr(raw, warnings)
        if parsed is not None:
            return parsed
    if inherited is not None:
        return inherited
    warnings.add(f"Duration missing before any explicit value; using {fraction_to_json_number(default)} beat(s).")
    return default


def extract_parts(text: str) -> dict[str, str]:
    parts: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        start = PART_START_RE.match(line)
        if start:
            current = start.group(1)
            parts[current] = []
            continue
        if current and re.match(r"^%\s*endif\s*$", line):
            current = None
            continue
        if current:
            parts[current].append(line)
    return {name: "\n".join(lines) for name, lines in parts.items()}


def parse_vars(vars_text: str) -> tuple[dict[str, object], str | None, list[str]]:
    attrs: dict[str, object] = {}
    versions: list[str] = []
    default_version: str | None = None
    for raw_line in vars_text.splitlines():
        line = raw_line.strip()
        add_version = ADD_VERSION_RE.search(line)
        if add_version:
            versions.append(add_version.group(1))
        default = DEFAULT_VERSION_RE.search(line)
        if default:
            default_version = default.group(1)
        attr = ATTRIBUTE_RE.search(line)
        if attr:
            key, raw_value = attr.group(1), attr.group(2).strip()
            try:
                attrs[key] = ast.literal_eval(raw_value)
            except (ValueError, SyntaxError):
                attrs[key] = raw_value
    return attrs, default_version, versions


def available_versions(parts: dict[str, str]) -> list[str]:
    chords = {name.removeprefix("Chords") for name in parts if name.startswith("Chords")}
    voices = {name.removeprefix("Voice") for name in parts if name.startswith("Voice")}
    return sorted(chords & voices)


def choose_version(parts: dict[str, str], requested: str | None, default_version: str | None) -> str:
    available = available_versions(parts)
    if requested:
        if requested not in available:
            raise ValueError(f"Version {requested!r} is not available. Available: {', '.join(available)}")
        return requested
    if default_version in available:
        return str(default_version)
    if available:
        return available[0]
    raise ValueError("No matching Chords*/Voice* section pair found.")


def parse_meter(voice_text: str, warnings: Warnings) -> str:
    match = re.search(r"\\time\s+(\d+/\d+)", voice_text)
    if match:
        return match.group(1)
    warnings.add("No \\time directive found; using 4/4.")
    return "4/4"


def meter_to_beats(meter: str) -> Fraction:
    numerator, denominator = meter.split("/", 1)
    return Fraction(int(numerator) * 4, int(denominator))


def parse_tempo(voice_text: str) -> int | None:
    match = re.search(r'\\tempo\s+(?:"[^"]*"\s*)?\d+\s*=\s*(\d+)', voice_text)
    if match:
        return int(match.group(1))
    return None


def parse_key(voice_text: str, warnings: Warnings) -> str:
    match = re.search(r"\\key\s+([a-g](?:isis|eses|is|es)?)\s+\\(major|minor)", voice_text)
    if not match:
        warnings.add("No \\key directive found; using C major.")
        return "C major"
    tonic, mode = match.group(1), match.group(2)
    return f"{lilypond_pitch_name(tonic)} {mode}"


def lilypond_pitch_name(raw: str) -> str:
    return LILYPOND_TO_CANONICAL_PITCH.get(raw, (raw.upper(), 0))[0]


def quality_part_to_symbol(part: str) -> str:
    replacements = {
        "5-": "b5",
        "5+": "#5",
        "9-": "b9",
        "9+": "#9",
        "11-": "b11",
        "11+": "#11",
        "13-": "b13",
        "13+": "#13",
        "3-": "b3",
        "3+": "#3",
    }
    return replacements.get(part, part.replace("-", "b").replace("+", "#"))


def normalize_quality(raw: str | None) -> str:
    if not raw:
        return ""
    parts = raw.lstrip(":").split(".")
    if not parts:
        return ""
    base = parts[0]
    extensions = [quality_part_to_symbol(part) for part in parts[1:] if part]
    if base == "7" and len(extensions) == 1 and extensions[0] in {"9", "11", "13"}:
        return extensions[0]
    if base == "6" and extensions == ["9"]:
        return "6/9"
    return base + "".join(extensions)


def normalize_chord_token(token: str, inherited: Fraction | None, meter_beats: Fraction, warnings: Warnings) -> tuple[str | None, Fraction | None]:
    token = token.replace("!", "")
    match = CHORD_TOKEN_RE.match(token)
    if not match:
        return None, inherited
    root = match.group("root")
    duration = extract_duration(match.group("duration"), inherited, meter_beats, warnings)
    if root in {"r", "s"}:
        return None, duration
    root_name = lilypond_pitch_name(root)
    quality = normalize_quality(match.group("quality"))
    bass = match.group("bass")
    bass_name = ""
    if bass:
        bass_root = re.match(rf"/({NOTE_RE})", bass)
        if bass_root:
            bass_name = "/" + lilypond_pitch_name(bass_root.group(1))
    return f"{root_name}{quality}{bass_name}", duration


def parse_scale_ratio(raw: str, command: str, warnings: Warnings) -> Fraction:
    match = re.fullmatch(r"(\d+)/(\d+)", raw)
    if not match:
        warnings.add(f"Unsupported tuplet ratio {raw!r}; using unscaled durations.")
        return Fraction(1)
    left, right = int(match.group(1)), int(match.group(2))
    if command == r"\times":
        return Fraction(left, right)
    return Fraction(right, left)


def expand_repeats_and_tuplets(tokens: list[str], warnings: Warnings) -> list[str | ScaleStart | ScaleEnd]:
    def parse_sequence(pos: int, until_rbrace: bool = False) -> tuple[list[str | ScaleStart | ScaleEnd], int]:
        result: list[str | ScaleStart | ScaleEnd] = []
        while pos < len(tokens):
            token = tokens[pos]
            if token == "}":
                if until_rbrace:
                    return result, pos + 1
                warnings.add("Ignoring unmatched closing brace.")
                pos += 1
                continue
            if token == "{":
                inner, pos = parse_sequence(pos + 1, True)
                result.extend(inner)
                continue
            if token == r"\repeat":
                pos += 1
                repeat_kind = "volta"
                if pos < len(tokens) and tokens[pos] in {"volta", "unfold", "percent"}:
                    repeat_kind = tokens[pos]
                    pos += 1
                count = 2
                if pos < len(tokens) and re.fullmatch(r"\d+", tokens[pos]):
                    count = int(tokens[pos])
                    pos += 1
                else:
                    warnings.add("Repeat without numeric count; assuming 2.")
                if repeat_kind == "percent":
                    warnings.add("Expanding repeat percent as ordinary repeated music.")
                if pos >= len(tokens) or tokens[pos] != "{":
                    warnings.add("Repeat without braced body; skipping repeat command.")
                    continue
                body, pos = parse_sequence(pos + 1, True)
                alternatives: list[list[str | ScaleStart | ScaleEnd]] = []
                if pos < len(tokens) and tokens[pos] == r"\alternative":
                    pos += 1
                    if pos < len(tokens) and tokens[pos] == "{":
                        pos += 1
                        while pos < len(tokens) and tokens[pos] != "}":
                            if tokens[pos] == "{":
                                alt, pos = parse_sequence(pos + 1, True)
                                alternatives.append(alt)
                            else:
                                warnings.add(f"Ignoring token inside alternative wrapper: {tokens[pos]}")
                                pos += 1
                        if pos < len(tokens) and tokens[pos] == "}":
                            pos += 1
                    else:
                        warnings.add("Alternative without wrapper braces.")
                if alternatives:
                    for repeat_index in range(count):
                        result.extend(body)
                        alt_index = min(repeat_index, len(alternatives) - 1)
                        result.extend(alternatives[alt_index])
                else:
                    for _ in range(count):
                        result.extend(body)
                continue
            if token in {r"\tuplet", r"\times"}:
                command = token
                pos += 1
                if pos >= len(tokens):
                    warnings.add(f"{command} without ratio.")
                    continue
                scale = parse_scale_ratio(tokens[pos], command, warnings)
                pos += 1
                if pos >= len(tokens) or tokens[pos] != "{":
                    warnings.add(f"{command} without braced body.")
                    continue
                body, pos = parse_sequence(pos + 1, True)
                result.append(ScaleStart(scale))
                result.extend(body)
                result.append(ScaleEnd())
                continue
            result.append(token)
            pos += 1
        return result, pos

    expanded, _ = parse_sequence(0)
    return expanded


def parse_chord_stream(section_text: str, meter_beats: Fraction, warnings: Warnings) -> list[list[object]]:
    tokens = tokenize_lilypond(strip_metadata_directives(section_text))
    tokens = expand_repeats_and_tuplets(tokens, warnings)
    cursor = Cursor(meter_beats)
    stream: list[list[object]] = []
    inherited_duration: Fraction | None = None
    scale_stack = [Fraction(1)]
    skip_next_string_arg = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if isinstance(token, ScaleStart):
            scale_stack.append(scale_stack[-1] * token.factor)
            index += 1
            continue
        if isinstance(token, ScaleEnd):
            if len(scale_stack) > 1:
                scale_stack.pop()
            index += 1
            continue
        if token == "|":
            index += 1
            continue
        if token == r"\partial":
            if index + 1 < len(tokens) and isinstance(tokens[index + 1], str):
                duration = parse_duration_expr(tokens[index + 1], warnings)
                if duration is not None:
                    cursor.set_partial(duration)
                index += 2
                continue
            warnings.add("\\partial without duration.")
            index += 1
            continue
        if token == r"\myMark":
            skip_next_string_arg = True
            index += 1
            continue
        if skip_next_string_arg and token.startswith('"'):
            skip_next_string_arg = False
            index += 1
            continue
        if token.startswith('"'):
            index += 1
            continue
        if token.startswith("\\"):
            if token not in IGNORED_COMMANDS:
                warnings.add(f"Ignoring unsupported chord command: {token}")
            index += 1
            continue
        symbol, duration = normalize_chord_token(token, inherited_duration, meter_beats, warnings)
        if duration is None:
            warnings.add(f"Ignoring unsupported chord token: {token}")
            index += 1
            continue
        inherited_duration = duration
        scaled_duration = duration * scale_stack[-1]
        if symbol:
            bar, beat = cursor.position()
            stream.append(
                [
                    bar,
                    fraction_to_json_number(beat),
                    symbol,
                    fraction_to_json_number(scaled_duration),
                ]
            )
        cursor.advance(scaled_duration)
        index += 1
    return stream


def midi_to_spelled_pitch(spelling: str, midi: int) -> str:
    octave = (midi // 12) - 1
    return f"{spelling}{octave}"


def parse_pitch_from_group(token: str) -> str:
    inside = token[1 : token.find(">")]
    for part in tokenize_lilypond(inside):
        if part.startswith("\\"):
            continue
        match = PITCH_TOKEN_RE.match(part.replace("!", ""))
        if match and match.group("pitch") not in {"r", "s"}:
            return part
    return "r"


def relative_pitch_to_midi(raw_pitch: str, octave_marks: str, previous_midi: int) -> int:
    _spelling, pitch_class = LILYPOND_TO_CANONICAL_PITCH[raw_pitch]
    previous_octave = (previous_midi // 12) - 1
    candidates = []
    for octave in range(previous_octave - 2, previous_octave + 3):
        candidates.append(12 * (octave + 1) + pitch_class)
    base = min(candidates, key=lambda midi: (abs(midi - previous_midi), midi))
    base += 12 * octave_marks.count("'")
    base -= 12 * octave_marks.count(",")
    return base


def parse_melody_token(
    token: str,
    inherited: Fraction | None,
    default_duration: Fraction,
    previous_midi: int,
    warnings: Warnings,
) -> tuple[str | None, Fraction | None, int, bool]:
    token = token.replace(r"\repeatTie", "")
    tied = "~" in token
    token = token.replace("~", "")
    if token.startswith("<"):
        group_pitch = parse_pitch_from_group(token)
        suffix = token[token.find(">") + 1 :]
        token = group_pitch + suffix
    token = re.sub(r"[-_^][A-Za-z0-9]+$", "", token)
    token = token.rstrip("-.^_")
    token = token.replace("!", "")
    match = PITCH_TOKEN_RE.match(token)
    if not match:
        return None, inherited, previous_midi, tied
    raw_pitch = match.group("pitch")
    duration = extract_duration(match.group("duration"), inherited, default_duration, warnings)
    if raw_pitch == "s":
        return None, duration, previous_midi, tied
    if raw_pitch == "r":
        return "R", duration, previous_midi, tied
    if raw_pitch not in LILYPOND_TO_CANONICAL_PITCH:
        warnings.add(f"Unsupported pitch spelling: {raw_pitch}")
        return None, duration, previous_midi, tied
    midi = relative_pitch_to_midi(raw_pitch, match.group("oct"), previous_midi)
    spelling = lilypond_pitch_name(raw_pitch)
    return midi_to_spelled_pitch(spelling, midi), duration, midi, tied


def parse_melody_stream(section_text: str, meter_beats: Fraction, warnings: Warnings) -> list[list[object]]:
    tokens = tokenize_lilypond(strip_metadata_directives(section_text))
    tokens = expand_repeats_and_tuplets(tokens, warnings)
    cursor = Cursor(meter_beats)
    stream: list[list[object]] = []
    inherited_duration: Fraction | None = None
    default_duration = Fraction(1)
    previous_midi = 60
    previous_tied = False
    scale_stack = [Fraction(1)]
    skip_next_string_arg = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if isinstance(token, ScaleStart):
            scale_stack.append(scale_stack[-1] * token.factor)
            index += 1
            continue
        if isinstance(token, ScaleEnd):
            if len(scale_stack) > 1:
                scale_stack.pop()
            index += 1
            continue
        if token == "|":
            index += 1
            continue
        if token == r"\partial":
            if index + 1 < len(tokens) and isinstance(tokens[index + 1], str):
                duration = parse_duration_expr(tokens[index + 1], warnings)
                if duration is not None:
                    cursor.set_partial(duration)
                index += 2
                continue
            warnings.add("\\partial without duration.")
            index += 1
            continue
        if token == r"\myMark":
            skip_next_string_arg = True
            index += 1
            continue
        if skip_next_string_arg and token.startswith('"'):
            skip_next_string_arg = False
            index += 1
            continue
        if token.startswith('"'):
            index += 1
            continue
        if token.startswith("\\"):
            if token not in IGNORED_COMMANDS:
                warnings.add(f"Ignoring unsupported melody command: {token}")
            index += 1
            continue
        pitch, duration, previous_midi, tied = parse_melody_token(
            token, inherited_duration, default_duration, previous_midi, warnings
        )
        if duration is None:
            warnings.add(f"Ignoring unsupported melody token: {token}")
            index += 1
            continue
        inherited_duration = duration
        scaled_duration = duration * scale_stack[-1]
        if pitch:
            bar, beat = cursor.position()
            if previous_tied and stream and stream[-1][2] == pitch:
                prior_duration = Fraction(str(stream[-1][3]))
                stream[-1][3] = fraction_to_json_number(prior_duration + scaled_duration)
            else:
                tag = "rest" if pitch == "R" else ("pickup" if bar == 0 else "unknown")
                stream.append(
                    [
                        bar,
                        fraction_to_json_number(beat),
                        pitch,
                        fraction_to_json_number(scaled_duration),
                        tag,
                    ]
                )
        cursor.advance(scaled_duration)
        previous_tied = tied
        index += 1
    return stream


def build_context(
    attrs: dict[str, object],
    voice_text: str,
    source_file: Path,
    version: str,
    style: str,
    instrument: str,
    warnings: Warnings,
) -> dict[str, object]:
    meter = parse_meter(voice_text, warnings)
    context: dict[str, object] = {
        "title": str(attrs.get("title") or source_file.stem.removesuffix(".ly")),
        "key": parse_key(voice_text, warnings),
        "meter": meter,
        "style": style,
        "instrument": instrument,
        "source_status": "openbook_source_review_required",
        "source_format": "openbook.ly.mako",
        "source_file": str(source_file),
        "openbook_version": version,
        "phrase_scope": "full_tune_expanded_from_openbook_sections",
    }
    tempo = parse_tempo(voice_text)
    if tempo is not None:
        context["tempo"] = tempo
    passthrough = {
        "subtitle": "subtitle",
        "composer": "composer_credit",
        "poet": "lyricist_credit",
        "piece": "source_piece",
        "style": "source_style",
        "structure": "source_structure",
        "structureremark": "source_structure_note",
        "location": "source_location",
        "uuid": "source_uuid",
        "remark": "source_remark",
    }
    for source_key, target_key in passthrough.items():
        value = attrs.get(source_key)
        if value:
            context[target_key] = value
    return context


def parse_version_streams(
    parts: dict[str, str],
    version: str,
    warnings: Warnings,
) -> tuple[str, Fraction, list[list[object]], list[list[object]]]:
    voice_text = parts[f"Voice{version}"]
    meter = parse_meter(voice_text, warnings)
    meter_beats = meter_to_beats(meter)
    harmony_stream = parse_chord_stream(parts[f"Chords{version}"], meter_beats, warnings)
    melody_stream = parse_melody_stream(voice_text, meter_beats, warnings)
    return voice_text, meter_beats, harmony_stream, melody_stream


def convert_file(source_file: Path, requested_version: str | None, style: str, instrument: str) -> dict[str, object]:
    warnings = Warnings()
    text = source_file.read_text(encoding="utf-8")
    parts = extract_parts(text)
    attrs, default_version, _declared_versions = parse_vars(parts.get("Vars", ""))
    version = choose_version(parts, requested_version, default_version)
    voice_text, _meter_beats, harmony_stream, melody_stream = parse_version_streams(parts, version, warnings)
    if requested_version is None and (not harmony_stream or not melody_stream):
        original_version = version
        original_problem = []
        if not harmony_stream:
            original_problem.append("empty harmony_stream")
        if not melody_stream:
            original_problem.append("empty melody_stream")
        for candidate in available_versions(parts):
            if candidate == original_version:
                continue
            candidate_warnings = Warnings()
            candidate_voice_text, _candidate_meter_beats, candidate_harmony, candidate_melody = parse_version_streams(
                parts, candidate, candidate_warnings
            )
            if candidate_harmony and candidate_melody:
                candidate_warnings.add(
                    f"Default version {original_version} produced {', '.join(original_problem)}; "
                    f"using version {candidate} instead."
                )
                version = candidate
                warnings = candidate_warnings
                voice_text = candidate_voice_text
                harmony_stream = candidate_harmony
                melody_stream = candidate_melody
                break
    context = build_context(attrs, voice_text, source_file, version, style, instrument, warnings)
    review_notes = [
        "Generated mechanically from OpenBook .ly.mako; review before benchmark use.",
        "Repeat/alternative notation was expanded into a linear playback order.",
        "Voice sections were parsed as LilyPond relative notation with outer \\relative c'.",
        "Melody tags are mechanical: pickup/rest/unknown only.",
    ]
    review_notes.extend(warnings.notes())
    return {
        "schema": "lead_sheet.v0",
        "context": context,
        "stream_schema": {
            "harmony_stream": ["bar", "beat", "symbol", "duration_beats"],
            "melody_stream": ["bar", "beat", "pitch", "duration_beats", "tag"],
        },
        "harmony_stream": harmony_stream,
        "melody_stream": melody_stream,
        "review": {
            "status": "needs_review",
            "reviewer": None,
            "notes": review_notes,
        },
    }


def iter_input_files(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(path.glob("*.ly.mako"))
    else:
        yield path


def output_path_for(source: Path, output: Path | None, batch: bool) -> Path | None:
    if output is None:
        return None
    if batch or output.is_dir() or output.suffix == "":
        return output / source.name.replace(".ly.mako", ".canonical.json")
    return output


COMPACT_STREAM_KEYS = {"harmony_stream", "melody_stream"}


def render_stream_array(events: list[object], indent: int) -> str:
    if not events:
        return "[]"
    line_prefix = " " * (indent + 2)
    close_prefix = " " * indent
    lines = ["["]
    for index, event in enumerate(events):
        comma = "," if index < len(events) - 1 else ""
        lines.append(f"{line_prefix}{json.dumps(event, ensure_ascii=False)}{comma}")
    lines.append(f"{close_prefix}]")
    return "\n".join(lines)


def render_json(data: object, indent: int = 0, parent_key: str | None = None) -> str:
    if isinstance(data, dict):
        if not data:
            return "{}"
        item_prefix = " " * (indent + 2)
        close_prefix = " " * indent
        lines = ["{"]
        items = list(data.items())
        for index, (key, value) in enumerate(items):
            comma = "," if index < len(items) - 1 else ""
            rendered_value = render_json(value, indent + 2, key)
            rendered_key = json.dumps(key, ensure_ascii=False)
            lines.append(f"{item_prefix}{rendered_key}: {rendered_value}{comma}")
        lines.append(f"{close_prefix}}}")
        return "\n".join(lines)
    if isinstance(data, list):
        if parent_key in COMPACT_STREAM_KEYS:
            return render_stream_array(data, indent)
        if not data:
            return "[]"
        item_prefix = " " * (indent + 2)
        close_prefix = " " * indent
        lines = ["["]
        for index, value in enumerate(data):
            comma = "," if index < len(data) - 1 else ""
            lines.append(f"{item_prefix}{render_json(value, indent + 2)}{comma}")
        lines.append(f"{close_prefix}]")
        return "\n".join(lines)
    return json.dumps(data, ensure_ascii=False)


def write_json(data: dict[str, object], path: Path | None) -> None:
    rendered = render_json(data)
    if path is None:
        print(rendered)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OpenBook src/openbook .ly.mako files to JazzArranger canonical lead-sheet JSON."
    )
    parser.add_argument("input", type=Path, help="A .ly.mako file or a directory containing .ly.mako files.")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON file, or output directory in batch mode.")
    parser.add_argument(
        "--version",
        help="OpenBook version suffix, e.g. Real, Fake, Aebersold. Defaults to the file's default version.",
    )
    parser.add_argument("--style", default="jazz_ballad", help="Canonical context.style value.")
    parser.add_argument("--instrument", default="piano", help="Canonical context.instrument value.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser()
    output_path = args.output.expanduser() if args.output else None
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 2
    files = list(iter_input_files(input_path))
    if not files:
        print(f"No .ly.mako files found in: {input_path}", file=sys.stderr)
        return 2
    batch = input_path.is_dir() or len(files) > 1
    if batch and output_path is None:
        print("Batch conversion requires --output to be a directory.", file=sys.stderr)
        return 2
    failures = 0
    for source_file in files:
        try:
            data = convert_file(source_file, args.version, args.style, args.instrument)
            write_json(data, output_path_for(source_file, output_path, batch))
        except Exception as exc:  # noqa: BLE001 - CLI should continue in batch mode.
            failures += 1
            print(f"{source_file}: {exc}", file=sys.stderr)
            if not batch:
                return 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
