"""Theory-aware distance helpers for the validator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.data.common import ROOTS_SHARP

from .types import ChordEvent


ROOT_TO_PC = {root: idx for idx, root in enumerate(ROOTS_SHARP)}

QUALITY_FAMILY = {
    "maj": "major",
    "maj7": "major",
    "min": "minor",
    "min7": "minor",
    "dom7": "dominant",
    "sus2": "suspended",
    "sus4": "suspended",
    "dim": "diminished",
    "hdim7": "diminished",
    "aug": "augmented",
}

QUALITY_CHORD_TONES = {
    "maj": {0, 4, 7},
    "maj7": {0, 4, 7, 11},
    "min": {0, 3, 7},
    "min7": {0, 3, 7, 10},
    "dom7": {0, 4, 7, 10},
    "sus2": {0, 2, 7},
    "sus4": {0, 5, 7},
    "dim": {0, 3, 6},
    "hdim7": {0, 3, 6, 10},
    "aug": {0, 4, 8},
}

TONIC_FAMILIES = {"major", "minor"}
PREDOMINANT_FAMILIES = {"major", "minor", "suspended"}
DOMINANT_FAMILIES = {"dominant", "diminished", "augmented"}


@dataclass(frozen=True)
class SegmentDistance:
    start: int
    end: int
    weight: int
    root_distance: float
    quality_distance: float
    function_distance: float
    melody_distance: float
    exact_match: bool

    @property
    def total(self) -> float:
        return self.root_distance * 0.35 + self.quality_distance * 0.25 + self.function_distance * 0.15 + self.melody_distance * 0.25


def root_pc(root: str) -> int:
    return ROOT_TO_PC[root]


def root_distance(root_a: str, root_b: str) -> float:
    a = root_pc(root_a)
    b = root_pc(root_b)
    delta = abs(a - b)
    delta = min(delta, 12 - delta)
    return delta / 6.0


def quality_family(quality: str) -> str:
    return QUALITY_FAMILY.get(quality, "other")


def quality_distance(quality_a: str, quality_b: str) -> float:
    if quality_a == quality_b:
        return 0.0
    family_a = quality_family(quality_a)
    family_b = quality_family(quality_b)
    if family_a == family_b:
        if {quality_a, quality_b} <= {"maj", "maj7"}:
            return 0.15
        if {quality_a, quality_b} <= {"min", "min7"}:
            return 0.15
        if {quality_a, quality_b} <= {"sus2", "sus4"}:
            return 0.2
        if {quality_a, quality_b} <= {"dim", "hdim7"}:
            return 0.2
        return 0.25
    if {family_a, family_b} <= {"major", "dominant"}:
        return 0.45
    if {family_a, family_b} <= {"minor", "diminished"}:
        return 0.45
    if "suspended" in {family_a, family_b}:
        return 0.4
    if "augmented" in {family_a, family_b}:
        return 0.7
    return 0.6


def _chromatic_degree_label(interval: int) -> str:
    return {
        0: "I",
        1: "bII",
        2: "II",
        3: "bIII",
        4: "III",
        5: "IV",
        6: "bV",
        7: "V",
        8: "bVI",
        9: "VI",
        10: "bVII",
        11: "VII",
    }[interval % 12]


def degree_label(root: str, key_root: str) -> str:
    return _chromatic_degree_label((root_pc(root) - root_pc(key_root)) % 12)


def _role_zone(function_role: str | None) -> str:
    if function_role is None:
        return "unknown"
    if function_role in {"T", "PD", "D"}:
        return function_role
    return "unknown"


def function_distance(role_a: str | None, role_b: str | None) -> float:
    a = _role_zone(role_a)
    b = _role_zone(role_b)
    if a == b:
        return 0.0
    if "unknown" in {a, b}:
        return 0.4 if a == b == "unknown" else 0.8
    zone_distance = {("T", "PD"): 0.35, ("PD", "D"): 0.35, ("T", "D"): 0.7}
    return zone_distance.get((a, b), zone_distance.get((b, a), 0.8))


def chord_tone_pitch_classes(root: str, quality: str) -> set[int]:
    root_index = root_pc(root)
    tones = QUALITY_CHORD_TONES.get(quality, {0, 4, 7})
    return {(root_index + interval) % 12 for interval in tones}


def melody_distance_for_segment(
    chord: ChordEvent,
    melody_events: Sequence,
    start: int,
    end: int,
) -> float:
    if not melody_events:
        return 0.0
    tones = chord_tone_pitch_classes(chord.root, chord.quality)
    total = 0
    mismatched = 0
    for note in melody_events:
        note_start = int(getattr(note, "start"))
        note_end = int(getattr(note, "end"))
        overlap = min(end, note_end) - max(start, note_start)
        if overlap <= 0:
            continue
        total += overlap
        pitch = int(getattr(note, "pitch"))
        if pitch % 12 not in tones:
            mismatched += overlap
    if total <= 0:
        return 0.0
    return mismatched / total


def melody_strong_beat_conflict(
    chord: ChordEvent,
    melody_events: Sequence,
    start: int,
    end: int,
    *,
    bar_grid: int,
) -> bool:
    if not melody_events:
        return False
    tones = chord_tone_pitch_classes(chord.root, chord.quality)
    strong_positions = {0}
    half_bar = bar_grid // 2
    if half_bar > 0:
        strong_positions.add(half_bar)
    for note in melody_events:
        note_start = int(getattr(note, "start"))
        if note_start < start or note_start >= end:
            continue
        if note_start % bar_grid not in strong_positions:
            continue
        pitch = int(getattr(note, "pitch"))
        if pitch % 12 not in tones:
            return True
    return False


def chord_signature(event: ChordEvent) -> tuple[int, int, str, str]:
    return event.start, event.end, event.root, event.quality


def compare_signature(a: ChordEvent, b: ChordEvent) -> bool:
    return chord_signature(a) == chord_signature(b)


def segment_distance(
    ref: ChordEvent | None,
    cand: ChordEvent | None,
    *,
    melody_events: Sequence | None = None,
    start: int,
    end: int,
) -> SegmentDistance:
    weight = max(0, end - start)
    if ref is None or cand is None:
        return SegmentDistance(
            start=start,
            end=end,
            weight=weight,
            root_distance=1.0,
            quality_distance=1.0,
            function_distance=1.0,
            melody_distance=1.0 if melody_events else 0.0,
            exact_match=False,
        )
    melody_distance = 0.0
    if melody_events is not None:
        melody_distance = melody_distance_for_segment(cand, melody_events, start, end)
    return SegmentDistance(
        start=start,
        end=end,
        weight=weight,
        root_distance=root_distance(ref.root, cand.root),
        quality_distance=quality_distance(ref.quality, cand.quality),
        function_distance=function_distance(ref.function_role, cand.function_role),
        melody_distance=melody_distance,
        exact_match=ref.root == cand.root and ref.quality == cand.quality,
    )

