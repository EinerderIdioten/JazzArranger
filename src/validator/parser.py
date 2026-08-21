"""Rule-based parser that annotates chord sequences with harmonic structure."""

from __future__ import annotations

from dataclasses import replace

from src.data.common import schema_key_parts

from .distance import degree_label, quality_family, root_pc
from .types import ChordEvent, Chunk, ParsedHarmony, TonicizationWindow, TreeNode


TONIC_QUALITIES = {"maj", "maj7", "min", "min7"}
PREDOMINANT_QUALITIES = {"min", "min7", "sus2", "sus4"}
DOMINANT_QUALITIES = {"dom7", "hdim7", "dim", "aug"}


def _key_label(root: str, mode: str) -> str:
    return f"{root}:{mode}"


def _mode_from_quality(quality: str, fallback: str) -> str:
    if quality in {"min", "min7", "hdim7", "dim"}:
        return "min"
    if quality in {"maj", "maj7", "dom7", "aug", "sus2", "sus4"}:
        return "maj"
    return fallback


def _role_for_event(event: ChordEvent, key_root: str, key_mode: str) -> tuple[str, str, float]:
    label = degree_label(event.root, key_root)
    family = quality_family(event.quality)
    confidence = 0.55

    if label in {"I", "III", "VI"} and event.quality in TONIC_QUALITIES:
        return "T", label, 0.95
    if label in {"ii", "II", "IV", "bII", "bVI"} and event.quality in PREDOMINANT_QUALITIES:
        return "PD", label, 0.85
    if event.quality in DOMINANT_QUALITIES:
        if label in {"V", "VII", "bII", "bV", "bVII", "II"}:
            return "D", label, 0.85
        return "D", label, 0.65
    if family == "suspended" and label in {"IV", "V", "II"}:
        return "PD", label, 0.7
    if family in {"major", "minor"} and label in {"I", "III", "VI"}:
        return "T", label, 0.7
    return "X", label, confidence


def _detect_tonicization_windows(
    events: list[ChordEvent],
    *,
    key_root: str,
    key_mode: str,
) -> list[TonicizationWindow]:
    windows: list[TonicizationWindow] = []
    if len(events) < 2:
        return windows
    for idx, event in enumerate(events[:-1]):
        next_event = events[idx + 1]
        if event.quality not in DOMINANT_QUALITIES:
            continue
        if event.root == next_event.root:
            continue
        if (root_pc(next_event.root) - root_pc(event.root)) % 12 != 5:
            continue
        target_root = next_event.root
        target_mode = _mode_from_quality(next_event.quality, key_mode)
        windows.append(
            TonicizationWindow(
                start=event.start,
                end=next_event.end,
                target_key=_key_label(target_root, target_mode),
                label=f"V/{target_root}",
                confidence=0.7,
            )
        )
    merged: list[TonicizationWindow] = []
    for window in sorted(windows, key=lambda item: (item.start, item.end)):
        if not merged:
            merged.append(window)
            continue
        last = merged[-1]
        if window.start <= last.end and window.target_key == last.target_key:
            merged[-1] = TonicizationWindow(
                start=last.start,
                end=max(last.end, window.end),
                target_key=last.target_key,
                label=last.label,
                confidence=max(last.confidence, window.confidence),
            )
        else:
            merged.append(window)
    return merged


def _assign_local_keys(
    events: list[ChordEvent],
    windows: list[TonicizationWindow],
    *,
    key_root: str,
    key_mode: str,
) -> list[str]:
    labels = [_key_label(key_root, key_mode) for _ in events]
    for window in windows:
        for idx, event in enumerate(events):
            if event.start >= window.start and event.end <= window.end:
                labels[idx] = window.target_key
    return labels


def _build_chunks(events: list[ChordEvent]) -> list[Chunk]:
    if not events:
        return []
    chunks: list[Chunk] = []
    chunk_start = 0
    pattern: list[str] = []
    current_key = events[0].local_key
    for idx, event in enumerate(events):
        pattern.append(event.function_role or "X")
        boundary = False
        if idx + 1 < len(events):
            nxt = events[idx + 1]
            if nxt.local_key != event.local_key:
                boundary = True
            if event.cadential_role in {"resolution", "terminal"}:
                boundary = True
        else:
            boundary = True
        if boundary:
            chunks.append(
                Chunk(
                    start=events[chunk_start].start,
                    end=event.end,
                    label=":".join(filter(None, [current_key, "-".join(pattern)])),
                    local_key=current_key,
                    function_pattern=tuple(pattern),
                )
            )
            if idx + 1 < len(events):
                chunk_start = idx + 1
                pattern = []
                current_key = nxt.local_key
    return chunks


def _build_tree(events: list[ChordEvent], chunks: list[Chunk]) -> TreeNode | None:
    if not events:
        return None
    root = TreeNode(label="song", start=events[0].start, end=events[-1].end)
    for chunk in chunks:
        chunk_node = TreeNode(
            label=chunk.label,
            start=chunk.start,
            end=chunk.end,
            metadata={
                "local_key": chunk.local_key,
                "function_pattern": list(chunk.function_pattern),
            },
        )
        for event in events:
            if event.start >= chunk.start and event.end <= chunk.end:
                chunk_node.children.append(
                    TreeNode(
                        label=f"{event.root}:{event.quality}",
                        start=event.start,
                        end=event.end,
                        metadata={
                            "function_role": event.function_role,
                            "local_key": event.local_key,
                            "cadential_role": event.cadential_role,
                        },
                    )
                )
        root.children.append(chunk_node)
    return root


class HarmonyParser:
    """Annotate a chord span sequence with weak harmonic structure."""

    def parse(
        self,
        events: list[ChordEvent],
        *,
        key: str | None,
        total_grid: int | None = None,
        bar_grid: int = 16,
        source: str = "harmony",
    ) -> ParsedHarmony:
        sorted_events = sorted(events, key=lambda item: (item.start, item.end, item.root, item.quality))
        parse_errors: list[str] = []
        if not sorted_events:
            parse_errors.append("empty_chords")
            return ParsedHarmony(
                source=source,
                key=key,
                bar_grid=bar_grid,
                total_grid=total_grid or 0,
                events=[],
                parse_errors=parse_errors,
            )

        key_root, key_mode = schema_key_parts(key)
        if total_grid is None:
            total_grid = sorted_events[-1].end

        windows = _detect_tonicization_windows(sorted_events, key_root=key_root, key_mode=key_mode)
        local_key_labels = _assign_local_keys(sorted_events, windows, key_root=key_root, key_mode=key_mode)

        annotated: list[ChordEvent] = []
        for event, local_key in zip(sorted_events, local_key_labels, strict=False):
            local_root, local_mode = local_key.split(":", 1)
            role, degree, confidence = _role_for_event(event, local_root, local_mode)
            annotated.append(
                replace(
                    event,
                    family=quality_family(event.quality),
                    degree=degree,
                    function_role=role,
                    local_key=local_key,
                    confidence=confidence,
                )
            )

        for idx, event in enumerate(annotated):
            cadential_role = None
            if idx + 1 < len(annotated):
                nxt = annotated[idx + 1]
                if event.function_role in {"D"} and nxt.function_role in {"T"} and event.local_key == nxt.local_key:
                    cadential_role = "dominant"
                    annotated[idx + 1] = replace(nxt, cadential_role="resolution", confidence=nxt.confidence)
                elif event.function_role == "PD" and nxt.function_role == "D" and event.local_key == nxt.local_key:
                    cadential_role = "preparation"
            else:
                if event.function_role == "T":
                    cadential_role = "terminal"
            if cadential_role is not None:
                annotated[idx] = replace(event, cadential_role=cadential_role, confidence=event.confidence)

        chunks = _build_chunks(annotated)
        tree = _build_tree(annotated, chunks)

        if total_grid is not None:
            expected = 0
            for idx, event in enumerate(annotated):
                if idx == 0 and event.start != 0:
                    parse_errors.append(f"unexpected_initial_offset:{event.start}")
                if event.start != expected:
                    if event.start > expected:
                        parse_errors.append(f"gap:{expected}-{event.start}")
                    else:
                        parse_errors.append(f"overlap:{event.start}<expected{expected}")
                expected = event.end
            if expected != total_grid:
                parse_errors.append(f"bad_coverage:{expected}!={total_grid}")

        return ParsedHarmony(
            source=source,
            key=key,
            bar_grid=bar_grid,
            total_grid=total_grid,
            events=annotated,
            parse_errors=parse_errors,
            tonicization_windows=windows,
            chunks=chunks,
            tree=tree,
        )
