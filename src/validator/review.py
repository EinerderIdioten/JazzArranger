"""Markdown review report generation for validator outputs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_REVIEW_TAGS = (
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


def select_review_cases(
    results: list[dict[str, Any]],
    *,
    source_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    worst_count: int = 20,
    tag_count: int = 10,
    no_tag_count: int = 10,
    max_per_title: int = 6,
    tags: Iterable[str] = DEFAULT_REVIEW_TAGS,
) -> list[dict[str, Any]]:
    source_by_id = source_by_id or {}
    selected: dict[str, dict[str, Any]] = {}
    selected_by: defaultdict[str, list[str]] = defaultdict(list)
    title_counts: Counter[str] = Counter()

    def title_key(row: Mapping[str, Any]) -> str:
        sample_id = str(row.get("id"))
        source = source_by_id.get(sample_id, {})
        return str(source.get("title") or sample_id.split(":", 1)[0])

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


def render_review_markdown(
    results: list[dict[str, Any]],
    *,
    source_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    result_path: Path | None = None,
    source_path: Path | None = None,
    worst_count: int = 20,
    tag_count: int = 10,
    no_tag_count: int = 10,
    max_per_title: int = 6,
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
    tag_counts = Counter(tag for row in results for tag in row.get("tags", []))
    issue_counts = Counter(issue.get("code") for row in results for issue in row.get("issues", []))
    scores = [_score(row) for row in results]
    mean_score = sum(scores) / len(scores) if scores else 0.0

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
        f"- selected_cases: `{len(selected)}`",
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
            "For each case, mark whether the validator label is musically justified and whether the severity feels right.",
            "",
        ]
    )

    for index, row in enumerate(selected, start=1):
        sample_id = str(row.get("id"))
        source = source_by_id.get(sample_id, {})
        metrics = row.get("metrics", {})
        tags_text = ", ".join(f"`{tag}`" for tag in row.get("tags", [])) or "`none`"
        reasons = ", ".join(f"`{reason}`" for reason in row.get("selected_by", []))
        issues = row.get("issues", [])
        title = source.get("title") or sample_id
        conversion = source.get("conversion", {})
        bars = "-"
        if conversion.get("bar_start") is not None and conversion.get("bar_end") is not None:
            bars = f"{conversion.get('bar_start')}-{conversion.get('bar_end')}"
        lines.extend(
            [
                f"## Case {index}: {title}",
                "",
                f"- id: `{sample_id}`",
                f"- selected_by: {reasons}",
                f"- title: `{title}`",
                f"- bars: `{bars}`",
                f"- key: `{row.get('key')}`",
                f"- meter: `{source.get('time_signature', '-')}`",
                f"- score: `{_format_float(metrics.get('score'))}`",
                f"- tags: {tags_text}",
                "",
                "- [ ] labels are musically justified",
                "- [ ] severity/ranking is reasonable",
                "- [ ] needs validator rule change",
                "",
                "### Metrics",
                "",
            ]
        )
        lines.extend(_render_metrics(metrics))
        if issues:
            lines.extend(["### Issues", "", "| code | message |", "|---|---|"])
            for issue in issues:
                lines.append(f"| `{issue.get('code')}` | {issue.get('message')} |")
            lines.append("")
        lines.extend(_render_chords(list(source.get("reference_chords", [])), title="Reference Chords"))
        lines.extend(_render_chords(list(source.get("candidate_chords", [])), title="Candidate Chords"))
        lines.extend(_render_bar_diagnostics(list(row.get("bar_diagnostics", []))))

    return "\n".join(lines).rstrip() + "\n"


def write_review_markdown(
    *,
    result_jsonl: Path,
    output_md: Path,
    source_jsonl: Path | None = None,
    worst_count: int = 20,
    tag_count: int = 10,
    no_tag_count: int = 10,
    max_per_title: int = 6,
    tags: Iterable[str] = DEFAULT_REVIEW_TAGS,
) -> dict[str, Any]:
    results = load_jsonl(result_jsonl)
    source_by_id = load_source_by_id(source_jsonl)
    markdown = render_review_markdown(
        results,
        source_by_id=source_by_id,
        result_path=result_jsonl,
        source_path=source_jsonl,
        worst_count=worst_count,
        tag_count=tag_count,
        no_tag_count=no_tag_count,
        max_per_title=max_per_title,
        tags=tags,
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    selected_count = markdown.count("\n## Case ")
    return {
        "result_jsonl": str(result_jsonl),
        "source_jsonl": str(source_jsonl) if source_jsonl else None,
        "output_md": str(output_md),
        "total_results": len(results),
        "selected_cases": selected_count,
    }
