"""Markdown review report generation for validator outputs."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .normalize import parse_time_signature


DEFAULT_REVIEW_TAGS = (
    "HARMONIC_RHYTHM_DISTORTION",
    "LOCAL_TONAL_DRIFT",
    "CADENTIAL_PATTERN_ERASED",
    "QUALITY_COLLAPSE_BASIC7",
    "INTRA_BAR_SPLIT_MISSING",
    "INTRA_BAR_SPLIT_EXTRA",
    "ROOT_MISMATCH",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_source_by_id(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    return {str(row.get("id")): row for row in load_jsonl(path)}


def _score(row: Mapping[str, Any]) -> float:
    score = row.get("metrics", {}).get("score")
    if score is None:
        return 0.0
    return float(score)


def _format_float(value: Any, digits: int = 3) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return "-"


def _chord_label(chord: Mapping[str, Any]) -> str:
    root = chord.get("root", "?")
    quality = chord.get("quality", "?")
    bass = chord.get("bass")
    label = f"{root}:{quality}"
    if bass:
        label += f"/{bass}"
    return label


def _render_chords(chords: list[Mapping[str, Any]], *, title: str) -> list[str]:
    lines = [f"**{title}**", "", "| span | chord | raw |", "|---|---|---|"]
    for chord in chords:
        span = f"{chord.get('start', '?')}-{chord.get('end', '?')}"
        raw = str(chord.get("raw", ""))
        lines.append(f"| `{span}` | `{_chord_label(chord)}` | `{raw}` |")
    if not chords:
        lines.append("| - | - | - |")
    lines.append("")
    return lines


def _render_metrics(metrics: Mapping[str, Any]) -> list[str]:
    keys = (
        "score",
        "total_distance",
        "root_distance",
        "quality_distance",
        "function_distance",
        "boundary_mismatch_rate",
        "exact_match_rate",
        "local_key_mismatch_rate",
        "cadence_match_rate",
    )
    lines = ["| metric | value |", "|---|---|"]
    for key in keys:
        lines.append(f"| `{key}` | `{_format_float(metrics.get(key))}` |")
    lines.append("")
    return lines


def _render_bar_diagnostics(bars: list[Mapping[str, Any]], *, limit: int = 4) -> list[str]:
    lines = ["**Worst Bars**", "", "| bar | span | distance | ref starts | cand starts | split delta |", "|---|---|---|---|---|---|"]
    selected = sorted(bars, key=lambda item: float(item.get("distance") or 0.0), reverse=True)[:limit]
    for bar in selected:
        span = f"{bar.get('start', '?')}-{bar.get('end', '?')}"
        lines.append(
            "| "
            f"`{bar.get('bar_index', '?')}` | `{span}` | `{_format_float(bar.get('distance'))}` | "
            f"`{bar.get('reference_starts', '-')}` | `{bar.get('candidate_starts', '-')}` | "
            f"`{bar.get('split_delta', '-')}` |"
        )
    if not selected:
        lines.append("| - | - | - | - | - | - |")
    lines.append("")
    return lines


def _title_key(row: Mapping[str, Any], source_by_id: Mapping[str, Mapping[str, Any]]) -> str:
    sample_id = str(row.get("id"))
    source = source_by_id.get(sample_id, {})
    return str(source.get("title") or row.get("title") or sample_id.split(":", 1)[0])


def _window_range(source: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[int, int] | None:
    conversion = source.get("conversion", {})
    start = conversion.get("bar_start")
    end = conversion.get("bar_end")
    if start is not None and end is not None:
        return int(start), int(end)
    sample_id = str(row.get("id") or "")
    match = re.search(r"bars_(\d+)_(\d+)", sample_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _window_label(source: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    window_range = _window_range(source, row)
    if window_range is not None:
        return f"{window_range[0]}-{window_range[1]}"
    sample_id = str(row.get("id") or "")
    return sample_id


def _window_sort_key(source: Mapping[str, Any], row: Mapping[str, Any], score: float) -> tuple[float, int, int, str]:
    window_range = _window_range(source, row)
    if window_range is None:
        return score, 10**9, 10**9, str(row.get("id") or "")
    return score, window_range[0], window_range[1], str(row.get("id") or "")


def _chord_at(chords: list[Mapping[str, Any]], position: int) -> Mapping[str, Any] | None:
    for chord in chords:
        start = int(chord.get("start", 0))
        end = int(chord.get("end", 0))
        if start <= position < end:
            return chord
    return chords[-1] if chords else None


def _render_bar_pattern(
    chords: list[Mapping[str, Any]],
    *,
    bar_start: int,
    bar_end: int,
    beats_per_bar: int,
) -> str:
    if not chords or bar_end <= bar_start:
        return " ".join("-" for _ in range(max(1, beats_per_bar)))

    bar_grid = bar_end - bar_start
    slot_size = max(1, round(bar_grid / max(1, beats_per_bar)))
    tokens: list[str] = []
    previous_label: str | None = None
    for beat_index in range(beats_per_bar):
        position = bar_start + beat_index * slot_size
        chord = _chord_at(chords, position)
        label = _chord_label(chord) if chord is not None else "-"
        if beat_index > 0 and label == previous_label:
            tokens.append("-")
        else:
            tokens.append(label)
        previous_label = label
    return " ".join(tokens)


def _build_review_cases(
    rows: list[dict[str, Any]],
    *,
    source_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    max_windows_per_case: int = 2,
) -> list[dict[str, Any]]:
    source_by_id = source_by_id or {}
    cases_by_title: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id"))
        source = source_by_id.get(sample_id, {})
        title = _title_key(row, source_by_id)
        case = cases_by_title.setdefault(
            title,
            {
                "title": title,
                "selected_by": [],
                "windows_by_key": {},
            },
        )
        for reason in row.get("selected_by", []):
            if reason not in case["selected_by"]:
                case["selected_by"].append(reason)
        window_key = _window_range(source, row)
        if window_key is None:
            window_key = (sample_id, sample_id)  # type: ignore[assignment]
        score = _score(row)
        window = {
            "row": row,
            "source": source,
            "score": score,
            "selected_by": list(row.get("selected_by", [])),
            "bar_label": _window_label(source, row),
            "window_key": window_key,
            "sort_key": _window_sort_key(source, row, score),
        }
        existing = case["windows_by_key"].get(window_key)
        if existing is None or score < existing["score"]:
            case["windows_by_key"][window_key] = window

    cases: list[dict[str, Any]] = []
    for case in cases_by_title.values():
        windows = sorted(case["windows_by_key"].values(), key=lambda item: item["sort_key"])[:max_windows_per_case]
        case["windows"] = windows
        case["window_count"] = len(windows)
        case["score"] = windows[0]["score"] if windows else 0.0
        case["tags"] = sorted({tag for window in windows for tag in window["row"].get("tags", [])})
        case["bars"] = ", ".join(window["bar_label"] for window in windows)
        del case["windows_by_key"]
        cases.append(case)
    return sorted(cases, key=lambda item: (item["score"], item["title"]))


def select_review_cases(
    results: list[dict[str, Any]],
    *,
    source_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    worst_count: int = 20,
    tag_count: int = 10,
    no_tag_count: int = 10,
    max_per_title: int = 2,
    tags: Iterable[str] = DEFAULT_REVIEW_TAGS,
) -> list[dict[str, Any]]:
    source_by_id = source_by_id or {}
    selected: dict[str, dict[str, Any]] = {}
    selected_by: defaultdict[str, list[str]] = defaultdict(list)
    title_counts: Counter[str] = Counter()

    def title_key(row: Mapping[str, Any]) -> str:
        return _title_key(row, source_by_id)

    def add(row: dict[str, Any], reason: str) -> bool:
        sample_id = str(row.get("id"))
        new_selection = sample_id not in selected
        if sample_id not in selected:
            title = title_key(row)
            if max_per_title > 0 and title_counts[title] >= max_per_title:
                return False
            selected[sample_id] = row
            title_counts[title] += 1
        if reason not in selected_by[sample_id]:
            selected_by[sample_id].append(reason)
        return new_selection

    ranked = sorted(results, key=_score)
    bucket_added = 0
    for row in ranked:
        if add(row, "worst_score"):
            bucket_added += 1
        if bucket_added >= worst_count:
            break
    for tag in tags:
        tagged = [row for row in ranked if tag in row.get("tags", [])]
        bucket_added = 0
        for row in tagged:
            if add(row, f"tag:{tag}"):
                bucket_added += 1
            if bucket_added >= tag_count:
                break
    no_tag = [row for row in ranked if not row.get("tags")]
    bucket_added = 0
    for row in no_tag:
        if add(row, "low_score_no_tag"):
            bucket_added += 1
        if bucket_added >= no_tag_count:
            break

    ordered = sorted(selected.values(), key=_score)
    for row in ordered:
        row["selected_by"] = selected_by[str(row.get("id"))]
    return ordered


def _render_window(window: dict[str, Any], *, index: int) -> list[str]:
    row = window["row"]
    source = window["source"]
    sample_id = str(row.get("id"))
    metrics = row.get("metrics", {})
    tags_text = ", ".join(f"`{tag}`" for tag in row.get("tags", [])) or "`none`"
    reasons = ", ".join(f"`{reason}`" for reason in window.get("selected_by", [])) or "`none`"
    key = row.get("key") or source.get("key") or "-"
    meter = source.get("time_signature", "-")
    conversion = source.get("conversion", {})
    bar_start = conversion.get("bar_start")
    bar_end = conversion.get("bar_end")
    reference_chords = list(source.get("reference_chords", []))
    candidate_chords = list(source.get("candidate_chords", []))
    beats_per_bar = parse_time_signature(meter)[0] if meter and meter != "-" else 4
    total_grid = int(row.get("total_grid") or source.get("total_grid") or 0)
    bar_count = 0
    bar_grid = 0
    if bar_start is not None and bar_end is not None:
        bar_count = int(bar_end) - int(bar_start) + 1
        if bar_count > 0 and total_grid > 0:
            bar_grid = total_grid // bar_count

    lines = [
        f"### Window {index} — bars {window['bar_label']}",
        "",
        f"- id: `{sample_id}`",
        f"- selected_by: {reasons}",
        f"- key: `{key}`",
        f"- meter: `{meter}`",
        f"- score: `{_format_float(metrics.get('score'))}`",
        f"- tags: {tags_text}",
        "",
    ]
    if bar_count <= 0 or bar_grid <= 0:
        lines.extend(["- unable to render bar grid", ""])
        return lines

    for bar_offset in range(bar_count):
        absolute_bar = int(bar_start) + bar_offset if bar_start is not None else bar_offset + 1
        local_start = bar_offset * bar_grid
        local_end = local_start + bar_grid
        ref_line = _render_bar_pattern(
            reference_chords,
            bar_start=local_start,
            bar_end=local_end,
            beats_per_bar=beats_per_bar,
        )
        cand_line = _render_bar_pattern(
            candidate_chords,
            bar_start=local_start,
            bar_end=local_end,
            beats_per_bar=beats_per_bar,
        )
        lines.extend(
            [
                f"#### Bar {absolute_bar}",
                f"reference: `{ref_line}`",
                f"candidate:  `{cand_line}`",
                "",
            ]
        )

    lines.extend(
        [
            "### Rating",
            "- Acceptability: [ ] acceptable [ ] debatable [ ] wrong",
            "- Severity: ``",
            "- Needs rule change: [ ] yes",
            "- Comment:",
            "",
        ]
    )
    return lines


def render_review_markdown(
    results: list[dict[str, Any]],
    *,
    source_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    result_path: Path | None = None,
    source_path: Path | None = None,
    worst_count: int = 20,
    tag_count: int = 10,
    no_tag_count: int = 10,
    max_per_title: int = 2,
    max_windows_per_case: int = 2,
    tags: Iterable[str] = DEFAULT_REVIEW_TAGS,
) -> str:
    source_by_id = source_by_id or {}
    selected = select_review_cases(
        results,
        source_by_id=source_by_id,
        worst_count=worst_count,
        tag_count=tag_count,
        no_tag_count=no_tag_count,
        max_per_title=max_per_title,
        tags=tags,
    )
    cases = _build_review_cases(selected, source_by_id=source_by_id, max_windows_per_case=max_windows_per_case)
    tag_counts = Counter(tag for row in results for tag in row.get("tags", []))
    issue_counts = Counter(issue.get("code") for row in results for issue in row.get("issues", []))
    scores = [_score(row) for row in results]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    selected_windows = sum(case["window_count"] for case in cases)

    lines = [
        "# Harmony Validator Review Cases",
        "",
        "This report is for human calibration of validator labels, not a final model scorecard.",
        "",
        "## Run Summary",
        "",
        f"- result_jsonl: `{result_path}`" if result_path else "- result_jsonl: `-`",
        f"- source_jsonl: `{source_path}`" if source_path else "- source_jsonl: `-`",
        f"- total_results: `{len(results)}`",
        f"- selected_cases: `{len(cases)}`",
        f"- selected_windows: `{selected_windows}`",
        f"- max_windows_per_case: `{max_windows_per_case}`",
        f"- mean_score: `{mean_score:.3f}`",
        "",
        "### Tag Counts",
        "",
        "| tag | count |",
        "|---|---|",
    ]
    for tag, count in sorted(tag_counts.items()):
        lines.append(f"| `{tag}` | `{count}` |")
    if not tag_counts:
        lines.append("| - | 0 |")
    lines.extend(["", "### Issue Counts", "", "| issue | count |", "|---|---|"])
    for issue, count in sorted(issue_counts.items()):
        lines.append(f"| `{issue}` | `{count}` |")
    if not issue_counts:
        lines.append("| - | 0 |")
    lines.extend(
        [
            "",
            "## Review Checklist",
            "",
            "Each case is one song. Review at most two windows per case.",
            "",
        ]
    )

    for index, case in enumerate(cases, start=1):
        selected_by = ", ".join(f"`{reason}`" for reason in case.get("selected_by", [])) or "`none`"
        tags_text = ", ".join(f"`{tag}`" for tag in case.get("tags", [])) or "`none`"
        lines.extend(
            [
                f"## Case {index}: {case['title']}",
                "",
                f"- title: `{case['title']}`",
                f"- selected_by: {selected_by}",
                f"- windows_shown: `{case['window_count']}`",
                f"- auto score: `{_format_float(case.get('score'))}`",
                f"- auto tags: {tags_text}",
                "",
            ]
        )
        for window_index, window in enumerate(case["windows"], start=1):
            lines.extend(_render_window(window, index=window_index))

    return "\n".join(lines).rstrip() + "\n"


def write_review_markdown(
    *,
    result_jsonl: Path,
    output_md: Path,
    source_jsonl: Path | None = None,
    worst_count: int = 20,
    tag_count: int = 10,
    no_tag_count: int = 10,
    max_per_title: int = 2,
    max_windows_per_case: int = 2,
    tags: Iterable[str] = DEFAULT_REVIEW_TAGS,
) -> dict[str, Any]:
    results = load_jsonl(result_jsonl)
    source_by_id = load_source_by_id(source_jsonl)
    selected = select_review_cases(
        results,
        source_by_id=source_by_id,
        worst_count=worst_count,
        tag_count=tag_count,
        no_tag_count=no_tag_count,
        max_per_title=max_per_title,
        tags=tags,
    )
    cases = _build_review_cases(selected, source_by_id=source_by_id, max_windows_per_case=max_windows_per_case)
    markdown = render_review_markdown(
        results,
        source_by_id=source_by_id,
        result_path=result_jsonl,
        source_path=source_jsonl,
        worst_count=worst_count,
        tag_count=tag_count,
        no_tag_count=no_tag_count,
        max_per_title=max_per_title,
        max_windows_per_case=max_windows_per_case,
        tags=tags,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    return {
        "result_jsonl": str(result_jsonl),
        "source_jsonl": str(source_jsonl) if source_jsonl else None,
        "output_md": str(output_md),
        "total_results": len(results),
        "selected_cases": len(cases),
        "selected_windows": sum(case["window_count"] for case in cases),
    }
