"""Token vocabulary and serializers for structured harmony targets."""

from __future__ import annotations

import re

from src.data.common import ChordSpan


TASK_TOKENS = ["<MELODY>", "</MELODY>", "<HARMONY>", "</HARMONY>"]
SPAN_TOKENS = ["<SPAN>", "</SPAN>"]
ROOT_TOKENS = [
    "<R_C>",
    "<R_Db>",
    "<R_D>",
    "<R_Eb>",
    "<R_E>",
    "<R_F>",
    "<R_Gb>",
    "<R_G>",
    "<R_Ab>",
    "<R_A>",
    "<R_Bb>",
    "<R_B>",
]
QUALITY_TOKENS = [
    "<Q_MAJ>",
    "<Q_MIN>",
    "<Q_DOM7>",
    "<Q_MAJ7>",
    "<Q_MIN7>",
    "<Q_DIM>",
    "<Q_HDIM7>",
    "<Q_AUG>",
    "<Q_SUS2>",
    "<Q_SUS4>",
]
EXTENSION_TOKENS = ["<EXT_6>", "<EXT_7>", "<EXT_9>", "<EXT_11>", "<EXT_13>", "<EXT_ADD9>"]
ALTERATION_TOKENS = ["<ALT_b5>", "<ALT_s5>", "<ALT_b9>", "<ALT_s9>", "<ALT_s11>", "<ALT_b13>"]
BASS_TOKENS = [
    "<B_C>",
    "<B_Db>",
    "<B_D>",
    "<B_Eb>",
    "<B_E>",
    "<B_F>",
    "<B_Gb>",
    "<B_G>",
    "<B_Ab>",
    "<B_A>",
    "<B_Bb>",
    "<B_B>",
]
OMISSION_TOKENS = ["<OMIT_3>", "<OMIT_5>", "<OMIT_7>"]

NEW_TOKENS = [
    *TASK_TOKENS,
    *SPAN_TOKENS,
    *ROOT_TOKENS,
    *QUALITY_TOKENS,
    *EXTENSION_TOKENS,
    *ALTERATION_TOKENS,
    *BASS_TOKENS,
    *OMISSION_TOKENS,
]

SHARP_TO_FLAT_ROOT = {
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
}

QUALITY_TO_TOKEN = {
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
}

TOKEN_TO_SHARP_ROOT = {
    "<R_C>": "C",
    "<R_Db>": "C#",
    "<R_D>": "D",
    "<R_Eb>": "D#",
    "<R_E>": "E",
    "<R_F>": "F",
    "<R_Gb>": "F#",
    "<R_G>": "G",
    "<R_Ab>": "G#",
    "<R_A>": "A",
    "<R_Bb>": "A#",
    "<R_B>": "B",
}

TOKEN_TO_QUALITY = {value: key for key, value in QUALITY_TO_TOKEN.items()}

ROOT_TOKEN_RE = r"(?P<root><R_(?:C|Db|D|Eb|E|F|Gb|G|Ab|A|Bb|B)>)"
QUALITY_TOKEN_RE = r"(?P<quality><Q_(?:MAJ|MIN|DOM7|MAJ7|MIN7|DIM|HDIM7|AUG|SUS2|SUS4)>)"
SPAN_TOKEN_LINE_RE = re.compile(
    rf"^<SPAN>\s+@(?P<start>\d+)-(?P<end>\d+)\s+{ROOT_TOKEN_RE}\s+{QUALITY_TOKEN_RE}\s+</SPAN>$"
)


def root_to_token(root: str) -> str:
    flat_root = SHARP_TO_FLAT_ROOT[root]
    return f"<R_{flat_root}>"


def quality_to_token(quality: str) -> str:
    return QUALITY_TO_TOKEN[quality]


def chord_spans_to_tokenized_harmony(chords: list[ChordSpan]) -> str:
    lines = ["<HARMONY>"]
    for chord in chords:
        lines.append(
            f"<SPAN> @{chord.start}-{chord.end} {root_to_token(chord.root)} "
            f"{quality_to_token(chord.quality)} </SPAN>"
        )
    lines.append("</HARMONY>")
    return "\n".join(lines)


def validate_tokenized_harmony_text(harmony_tokens: str) -> list[str]:
    lines = harmony_tokens.strip().splitlines()
    if len(lines) < 3 or lines[0] != "<HARMONY>" or lines[-1] != "</HARMONY>":
        return ["bad_harmony_token_wrapper"]
    errors: list[str] = []
    for line in lines[1:-1]:
        if not SPAN_TOKEN_LINE_RE.match(line):
            errors.append(f"bad_harmony_token_line:{line}")
    return errors


def parse_tokenized_harmony(
    harmony_tokens: str,
    *,
    total_grid: int | None = None,
) -> tuple[list[ChordSpan], list[str]]:
    errors = validate_tokenized_harmony_text(harmony_tokens)
    if errors:
        return [], errors
    spans: list[ChordSpan] = []
    expected = 0
    for idx, line in enumerate(harmony_tokens.strip().splitlines()[1:-1]):
        match = SPAN_TOKEN_LINE_RE.match(line)
        if match is None:
            errors.append(f"bad_harmony_token_line:{line}")
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        root = TOKEN_TO_SHARP_ROOT[match.group("root")]
        quality = TOKEN_TO_QUALITY[match.group("quality")]
        if end <= start:
            errors.append(f"non_positive_span:{idx}")
        if start != expected:
            if start > expected:
                errors.append(f"gap:{expected}-{start}")
            else:
                errors.append(f"overlap:{start}<expected{expected}")
        expected = end
        spans.append(ChordSpan(start=start, end=end, root=root, quality=quality, raw=line))
    if total_grid is not None and expected != total_grid:
        errors.append(f"bad_coverage:{expected}!={total_grid}")
    return spans, errors
