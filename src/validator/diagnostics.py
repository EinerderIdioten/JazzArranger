"""Reference checking and candidate/reference comparison for harmony validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .distance import (
    chord_signature,
    compare_signature,
    melody_distance_for_segment,
    melody_strong_beat_conflict,
    segment_distance,
)
from .normalize import Normalizer, get_in
from .parser import HarmonyParser
from .types import ChordEvent, ParsedHarmony, ValidationIssue, ValidationResult


RICH_QUALITY_FAMILIES = {"maj7", "min7", "hdim7", "dim", "aug", "sus2", "sus4"}
SAFE_QUALITY_FAMILIES = {"maj", "min", "dom7"}
ROOT_MISMATCH_THRESHOLD = 0.5
LOCAL_TONAL_DRIFT_THRESHOLD = 0.55
HARMONIC_RHYTHM_DISTORTION_THRESHOLD = 0.45
HARMONIC_RHYTHM_DELTA_THRESHOLD = 2


def _weighted_mean(sum_value: float, weight: float) -> float:
    return sum_value / weight if weight else 0.0


def _parse_side(
    record: Mapping[str, Any],
    *,
    fields: Sequence[str],
    normalizer: Normalizer,
    parser: HarmonyParser,
    total_grid: int | None,
    key: str | None,
    bar_grid: int,
) -> tuple[ParsedHarmony | None, list[ValidationIssue], dict[str, ParsedHarmony]]:
    parsed_by_field: dict[str, ParsedHarmony] = {}
    issues: list[ValidationIssue] = []
    for field in fields:
        value = get_in(record, field)
        if value is None:
            continue
        events, parse_errors, _source = normalizer.parse_harmony(value, total_grid=total_grid)
        parsed = parser.parse(events, key=key, total_grid=total_grid, bar_grid=bar_grid, source=field)
        parsed_by_field[field] = parsed
        for error in parse_errors:
            issues.append(
                ValidationIssue(
                    code="PARSE_ERROR",
                    message=f"{field}: {error}",
                    severity="error",
                    details={"field": field, "error": error},
                )
            )
        for error in parsed.parse_errors:
            issues.append(
                ValidationIssue(
                    code="PARSE_ERROR",
                    message=f"{field}: {error}",
                    severity="error",
                    details={"field": field, "error": error},
                )
            )
    if len(parsed_by_field) > 1:
        signatures = {field: tuple(chord_signature(event) for event in parsed.events) for field, parsed in parsed_by_field.items()}
        reference_field = next(iter(parsed_by_field))
        reference_signature = signatures[reference_field]
        mismatch_fields = [field for field, signature in signatures.items() if signature != reference_signature]
        if mismatch_fields:
            issues.append(
                ValidationIssue(
                    code="REPRESENTATION_MISMATCH",
                    message=f"mismatch across {', '.join(sorted(signatures))}",
                    severity="error",
                    details={
                        "reference_field": reference_field,
                        "mismatch_fields": sorted(mismatch_fields),
                        "signatures": {field: list(signature) for field, signature in signatures.items()},
                    },
                )
            )
    primary = None
    for field in fields:
        if field in parsed_by_field:
            primary = parsed_by_field[field]
            break
    return primary, issues, parsed_by_field


def _internal_boundaries(events: Sequence[ChordEvent], total_grid: int) -> set[int]:
    return {
        event.start
        for event in events[1:]
        if 0 < event.start < total_grid
    }


def _segment_coverage(
    events: Sequence[ChordEvent],
    *,
    start: int,
    end: int,
    cursor: int,
) -> tuple[ChordEvent | None, int]:
    while cursor < len(events) and events[cursor].end <= start:
        cursor += 1
    if cursor >= len(events):
        return None, cursor
    event = events[cursor]
    if event.start <= start and event.end >= end:
        return event, cursor
    return None, cursor


def _bar_diagnostics(
    reference: ParsedHarmony,
    candidate: ParsedHarmony,
    *,
    melody_events: Sequence | None,
    aligned_segments: list[tuple[int, int, ChordEvent | None, ChordEvent | None, float]],
) -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    total_grid = max(reference.total_grid, candidate.total_grid)
    bar_grid = reference.bar_grid or candidate.bar_grid or 16
    for bar_index, start in enumerate(range(0, total_grid, bar_grid)):
        end = min(start + bar_grid, total_grid)
        ref_starts = sum(1 for event in reference.events if start <= event.start < end)
        cand_starts = sum(1 for event in candidate.events if start <= event.start < end)
        ref_overlaps = sum(1 for event in reference.events if event.start < end and event.end > start)
        cand_overlaps = sum(1 for event in candidate.events if event.start < end and event.end > start)
        bar_segments = [segment for segment in aligned_segments if segment[0] >= start and segment[1] <= end]
        if bar_segments:
            bar_distance = sum(segment[4] * max(1, segment[1] - segment[0]) for segment in bar_segments) / sum(
                max(1, segment[1] - segment[0]) for segment in bar_segments
            )
        else:
            bar_distance = 0.0
        bars.append(
            {
                "bar_index": bar_index,
                "start": start,
                "end": end,
                "reference_starts": ref_starts,
                "candidate_starts": cand_starts,
                "reference_overlaps": ref_overlaps,
                "candidate_overlaps": cand_overlaps,
                "split_delta": cand_starts - ref_starts,
                "distance": bar_distance,
                "melody_conflict": any(
                    segment[3] is not None
                    and melody_strong_beat_conflict(segment[3], melody_events or [], segment[0], segment[1], bar_grid=bar_grid)
                    for segment in bar_segments
                    if segment[3] is not None
                ),
            }
        )
    return bars


def _compare_parsed(
    reference: ParsedHarmony,
    candidate: ParsedHarmony,
    *,
    melody_events: Sequence | None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[ValidationIssue], list[str]]:
    total_grid = max(reference.total_grid, candidate.total_grid)
    boundaries = {0, total_grid}
    for event in reference.events:
        boundaries.add(event.start)
        boundaries.add(event.end)
    for event in candidate.events:
        boundaries.add(event.start)
        boundaries.add(event.end)
    ordered_boundaries = sorted(boundaries)
    segments: list[tuple[int, int, ChordEvent | None, ChordEvent | None, float]] = []
    ref_cursor = 0
    cand_cursor = 0
    root_sum = quality_sum = function_sum = melody_sum = 0.0
    ref_melody_sum = 0.0
    exact_weight = root_match_weight = quality_match_weight = function_match_weight = 0.0
    cadence_ref_weight = cadence_match_weight = 0.0
    local_key_mismatch_weight = 0.0
    content_weight = 0.0
    for start, end in zip(ordered_boundaries[:-1], ordered_boundaries[1:]):
        if end <= start:
            continue
        ref_event, ref_cursor = _segment_coverage(reference.events, start=start, end=end, cursor=ref_cursor)
        cand_event, cand_cursor = _segment_coverage(candidate.events, start=start, end=end, cursor=cand_cursor)
        seg = segment_distance(ref_event, cand_event, melody_events=melody_events, start=start, end=end)
        segments.append((start, end, ref_event, cand_event, seg.total))
        weight = max(1, end - start)
        if ref_event is not None and cand_event is not None:
            root_sum += seg.root_distance * weight
            quality_sum += seg.quality_distance * weight
            function_sum += seg.function_distance * weight
            melody_sum += seg.melody_distance * weight
            exact_weight += weight if seg.exact_match else 0.0
            root_match_weight += weight if ref_event.root == cand_event.root else 0.0
            quality_match_weight += weight if ref_event.quality == cand_event.quality else 0.0
            function_match_weight += weight if ref_event.function_role == cand_event.function_role else 0.0
            cadence_ref_weight += weight if ref_event.cadential_role in {"resolution", "terminal"} else 0.0
            cadence_match_weight += (
                weight
                if ref_event.cadential_role in {"resolution", "terminal"}
                and cand_event.cadential_role == ref_event.cadential_role
                and compare_signature(ref_event, cand_event)
                else 0.0
            )
            local_key_mismatch_weight += weight if ref_event.local_key != cand_event.local_key else 0.0
            content_weight += weight
            if melody_events:
                ref_melody_sum += melody_distance_for_segment(ref_event, melody_events, start, end) * weight
        elif cand_event is not None:
            root_sum += 1.0 * weight
            quality_sum += 1.0 * weight
            function_sum += 1.0 * weight
            melody_sum += 1.0 * weight if melody_events else 0.0
            content_weight += weight
        elif ref_event is not None:
            root_sum += 1.0 * weight
            quality_sum += 1.0 * weight
            function_sum += 1.0 * weight
            melody_sum += 1.0 * weight if melody_events else 0.0
            content_weight += weight

    ref_boundaries = _internal_boundaries(reference.events, total_grid)
    cand_boundaries = _internal_boundaries(candidate.events, total_grid)
    missing_boundaries = ref_boundaries - cand_boundaries
    extra_boundaries = cand_boundaries - ref_boundaries
    ref_intra = {boundary for boundary in missing_boundaries if boundary % reference.bar_grid != 0}
    cand_intra = {boundary for boundary in extra_boundaries if boundary % candidate.bar_grid != 0}
    boundary_total = len(ref_boundaries) + len(cand_boundaries)
    rhythm_distance = (len(missing_boundaries) + len(extra_boundaries)) / max(1, boundary_total)

    root_distance_value = _weighted_mean(root_sum, content_weight)
    quality_distance_value = _weighted_mean(quality_sum, content_weight)
    function_distance_value = _weighted_mean(function_sum, content_weight)
    melody_distance_value = _weighted_mean(melody_sum, content_weight)
    reference_melody_fit = _weighted_mean(ref_melody_sum, content_weight)
    content_distance = 0.35 * root_distance_value + 0.25 * quality_distance_value + 0.15 * function_distance_value + 0.25 * melody_distance_value
    total_distance = 0.8 * content_distance + 0.2 * rhythm_distance

    metrics = {
        "root_distance": root_distance_value,
        "quality_distance": quality_distance_value,
        "function_distance": function_distance_value,
        "melody_distance": melody_distance_value,
        "reference_melody_distance": reference_melody_fit,
        "melody_distance_delta": melody_distance_value - reference_melody_fit,
        "boundary_mismatch_rate": rhythm_distance,
        "content_distance": content_distance,
        "total_distance": total_distance,
        "score": 1.0 - total_distance,
        "root_match_rate": 1.0 - root_distance_value,
        "quality_match_rate": 1.0 - quality_distance_value,
        "function_match_rate": 1.0 - function_distance_value,
        "melody_fit_rate": 1.0 - melody_distance_value,
        "exact_match_rate": _weighted_mean(exact_weight, content_weight),
        "reference_boundary_count": float(len(ref_boundaries)),
        "candidate_boundary_count": float(len(cand_boundaries)),
        "boundary_missing_count": float(len(missing_boundaries)),
        "boundary_extra_count": float(len(extra_boundaries)),
        "intra_bar_boundary_missing_count": float(len(ref_intra)),
        "intra_bar_boundary_extra_count": float(len(cand_intra)),
        "reference_cadence_weight": _weighted_mean(cadence_ref_weight, content_weight),
        "cadence_match_rate": _weighted_mean(cadence_match_weight, cadence_ref_weight if cadence_ref_weight else 1.0),
        "local_key_mismatch_rate": _weighted_mean(local_key_mismatch_weight, content_weight),
        "reference_root_match_rate": _weighted_mean(root_match_weight, content_weight),
        "reference_quality_match_rate": _weighted_mean(quality_match_weight, content_weight),
        "reference_function_match_rate": _weighted_mean(function_match_weight, content_weight),
    }
    boundary_count_delta = abs(metrics["candidate_boundary_count"] - metrics["reference_boundary_count"])
    tonal_drift_index = 0.5 * metrics["root_distance"] + 0.3 * metrics["function_distance"] + 0.2 * metrics["boundary_mismatch_rate"]
    metrics["boundary_count_delta"] = boundary_count_delta
    metrics["tonal_drift_index"] = tonal_drift_index

    tags: list[str] = []
    issues: list[ValidationIssue] = []
    if len(missing_boundaries) and len(ref_intra):
        tags.append("INTRA_BAR_SPLIT_MISSING")
        issues.append(
            ValidationIssue(
                code="INTRA_BAR_SPLIT_MISSING",
                message=f"missing {len(ref_intra)} intra-bar boundaries",
                severity="warning",
                details={"missing": sorted(ref_intra)},
            )
        )
    if len(extra_boundaries) and len(cand_intra):
        tags.append("INTRA_BAR_SPLIT_EXTRA")
        issues.append(
            ValidationIssue(
                code="INTRA_BAR_SPLIT_EXTRA",
                message=f"extra {len(cand_intra)} intra-bar boundaries",
                severity="warning",
                details={"extra": sorted(cand_intra)},
            )
        )
    if metrics["boundary_mismatch_rate"] >= HARMONIC_RHYTHM_DISTORTION_THRESHOLD and boundary_count_delta >= HARMONIC_RHYTHM_DELTA_THRESHOLD:
        tags.append("HARMONIC_RHYTHM_DISTORTION")
        issues.append(
            ValidationIssue(
                code="HARMONIC_RHYTHM_DISTORTION",
                message="candidate boundary density departs sharply from reference",
                severity="warning",
                details={
                    "boundary_mismatch_rate": metrics["boundary_mismatch_rate"],
                    "boundary_count_delta": boundary_count_delta,
                },
            )
        )

    rich_to_safe_weight = 0.0
    rich_weight = 0.0
    cadence_erased = cadence_ref_weight > 0 and metrics["cadence_match_rate"] < 0.5
    for start, end, ref_event, cand_event, _distance in segments:
        if ref_event is None or cand_event is None:
            continue
        weight = max(1, end - start)
        if ref_event.quality in RICH_QUALITY_FAMILIES and cand_event.quality in SAFE_QUALITY_FAMILIES and ref_event.root == cand_event.root:
            rich_to_safe_weight += weight
        if ref_event.quality in RICH_QUALITY_FAMILIES:
            rich_weight += weight
    if rich_weight and rich_to_safe_weight / rich_weight > 0.2:
        tags.append("QUALITY_COLLAPSE_BASIC7")
        issues.append(
            ValidationIssue(
                code="QUALITY_COLLAPSE_BASIC7",
                message="rich qualities collapsed into safe family",
                severity="warning",
                details={"collapse_rate": rich_to_safe_weight / rich_weight},
            )
        )
    if metrics["function_distance"] > 0.45 and metrics["root_match_rate"] > 0.7:
        tags.append("FUNCTION_NEIGHBOR_CONFUSION")
        issues.append(
            ValidationIssue(
                code="FUNCTION_NEIGHBOR_CONFUSION",
                message="function drift with near-root agreement",
                severity="warning",
                details={"function_distance": metrics["function_distance"], "root_match_rate": metrics["root_match_rate"]},
            )
        )
    if metrics["local_key_mismatch_rate"] > 0.3 or metrics["tonal_drift_index"] > LOCAL_TONAL_DRIFT_THRESHOLD:
        tags.append("LOCAL_TONAL_DRIFT")
        issues.append(
            ValidationIssue(
                code="LOCAL_TONAL_DRIFT",
                message="local tonal plan diverges",
                severity="warning",
                details={
                    "local_key_mismatch_rate": metrics["local_key_mismatch_rate"],
                    "tonal_drift_index": metrics["tonal_drift_index"],
                },
            )
        )
    if cadence_erased:
        tags.append("CADENTIAL_PATTERN_ERASED")
        issues.append(
            ValidationIssue(
                code="CADENTIAL_PATTERN_ERASED",
                message="reference cadential material lost in candidate",
                severity="warning",
                details={"cadence_match_rate": metrics["cadence_match_rate"]},
            )
        )
    melody_conflicts = [
        segment
        for segment in segments
        if segment[2] is not None
        and segment[3] is not None
        and melody_events
        and melody_strong_beat_conflict(segment[3], melody_events, segment[0], segment[1], bar_grid=candidate.bar_grid)
    ]
    if melody_conflicts:
        tags.append("MELODY_STRONG_BEAT_CONFLICT")
        issues.append(
            ValidationIssue(
                code="MELODY_STRONG_BEAT_CONFLICT",
                message="candidate collides with melody on strong beats",
                severity="warning",
                details={"segment_count": len(melody_conflicts)},
            )
        )
    if metrics["root_distance"] > ROOT_MISMATCH_THRESHOLD:
        tags.append("ROOT_MISMATCH")
        issues.append(
            ValidationIssue(
                code="ROOT_MISMATCH",
                message="root mismatch dominates the comparison",
                severity="warning",
                details={"root_distance": metrics["root_distance"]},
            )
        )

    bar_diagnostics = _bar_diagnostics(reference, candidate, melody_events=melody_events, aligned_segments=segments)
    return metrics, bar_diagnostics, issues, tags


class ReferenceChecker:
    """Check reference material for parseability and internal consistency."""

    def __init__(
        self,
        *,
        normalizer: Normalizer | None = None,
        parser: HarmonyParser | None = None,
    ) -> None:
        self.normalizer = normalizer or Normalizer()
        self.parser = parser or HarmonyParser()

    def check_record(
        self,
        record: Mapping[str, Any],
        *,
        reference_fields: Sequence[str] | None = None,
    ) -> ValidationResult:
        metadata = self.normalizer.record_metadata(record)
        total_grid = metadata["total_grid"] or None
        key = metadata["key"]
        bar_grid = metadata["bar_grid"]
        fields = reference_fields or self.normalizer.reference_fields
        parsed, issues, parsed_by_field = _parse_side(
            record,
            fields=fields,
            normalizer=self.normalizer,
            parser=self.parser,
            total_grid=total_grid,
            key=key,
            bar_grid=bar_grid,
        )
        if parsed is None:
            issues.append(
                ValidationIssue(
                    code="REFERENCE_MISSING",
                    message="no reference harmony field found",
                    severity="error",
                )
            )
        return ValidationResult(
            sample_id=metadata["id"],
            dataset=metadata["dataset"],
            total_grid=total_grid,
            key=key,
            reference=parsed,
            metrics={
                "reference_available": 1.0 if parsed is not None else 0.0,
                "reference_source_count": float(len(parsed_by_field)),
                "reference_parse_error_count": float(sum(len(parsed.parse_errors) for parsed in parsed_by_field.values())) if parsed_by_field else 0.0,
            },
            issues=issues,
        )


class Comparator:
    """Compare a candidate harmony sequence against reference material."""

    def __init__(
        self,
        *,
        normalizer: Normalizer | None = None,
        parser: HarmonyParser | None = None,
    ) -> None:
        self.normalizer = normalizer or Normalizer()
        self.parser = parser or HarmonyParser()

    def compare_record(
        self,
        record: Mapping[str, Any],
        *,
        reference_fields: Sequence[str] | None = None,
        candidate_fields: Sequence[str] | None = None,
        melody_field: str | None = None,
    ) -> ValidationResult:
        metadata = self.normalizer.record_metadata(record)
        total_grid = metadata["total_grid"] or None
        key = metadata["key"]
        bar_grid = metadata["bar_grid"]
        reference_fields = reference_fields or self.normalizer.reference_fields
        candidate_fields = candidate_fields or self.normalizer.candidate_fields

        reference, ref_issues, ref_sources = _parse_side(
            record,
            fields=reference_fields,
            normalizer=self.normalizer,
            parser=self.parser,
            total_grid=total_grid,
            key=key,
            bar_grid=bar_grid,
        )
        candidate, cand_issues, cand_sources = _parse_side(
            record,
            fields=candidate_fields,
            normalizer=self.normalizer,
            parser=self.parser,
            total_grid=total_grid,
            key=key,
            bar_grid=bar_grid,
        )

        melody_events, melody_errors, melody_source = self.normalizer.parse_melody(
            record,
            total_grid=total_grid,
            field=melody_field,
        )

        issues = list(ref_issues) + list(cand_issues)
        if melody_errors:
            for error in melody_errors:
                issues.append(
                    ValidationIssue(
                        code="MELODY_PARSE_ERROR",
                        message=error,
                        severity="warning",
                        details={"field": melody_source},
                    )
                )

        if reference is None:
            issues.append(
                ValidationIssue(
                    code="REFERENCE_MISSING",
                    message="no reference harmony field found",
                    severity="error",
                )
            )
        if candidate is None:
            issues.append(
                ValidationIssue(
                    code="CANDIDATE_MISSING",
                    message="no candidate harmony field found",
                    severity="error",
                )
            )

        metrics: dict[str, float] = {
            "reference_available": 1.0 if reference is not None else 0.0,
            "candidate_available": 1.0 if candidate is not None else 0.0,
            "reference_source_count": float(len(ref_sources)),
            "candidate_source_count": float(len(cand_sources)),
            "melody_available": 1.0 if melody_events else 0.0,
        }
        bar_diagnostics: list[dict[str, Any]] = []
        tags: list[str] = []
        if reference is not None and candidate is not None:
            comparison_metrics, bar_diagnostics, comparison_issues, comparison_tags = _compare_parsed(
                reference,
                candidate,
                melody_events=melody_events,
            )
            metrics.update(comparison_metrics)
            issues.extend(comparison_issues)
            tags.extend(comparison_tags)
        return ValidationResult(
            sample_id=metadata["id"],
            dataset=metadata["dataset"],
            total_grid=total_grid,
            key=key,
            reference=reference,
            candidate=candidate,
            metrics=metrics,
            tags=tags,
            issues=issues,
            bar_diagnostics=bar_diagnostics,
        )
