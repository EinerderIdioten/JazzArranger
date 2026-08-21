"""Batch aggregation and export helpers for harmony validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.data.common import counter_to_sorted_dict, write_json, write_jsonl

from .types import ValidationResult


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class BatchReporter:
    """Aggregate validation results into a regression-friendly report."""

    def summarize(self, results: list[ValidationResult], *, top_k: int = 20) -> dict[str, Any]:
        results = list(results)
        sample_count = len(results)
        dataset_counts = Counter(result.dataset or "unknown" for result in results)
        tag_counts = Counter(tag for result in results for tag in result.tags)
        issue_counts = Counter(issue.code for result in results for issue in result.issues)
        metric_sums: Counter[str] = Counter()
        metric_counts: Counter[str] = Counter()
        reference_available = 0
        candidate_available = 0
        melody_available = 0
        reference_valid = 0
        candidate_valid = 0
        for result in results:
            if result.reference is not None:
                reference_available += 1
                if result.reference.valid:
                    reference_valid += 1
            if result.candidate is not None:
                candidate_available += 1
                if result.candidate.valid:
                    candidate_valid += 1
            if result.metrics.get("melody_available", 0.0) > 0:
                melody_available += 1
            for key, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    metric_sums[key] += float(value)
                    metric_counts[key] += 1
        metric_averages = {
            key: metric_sums[key] / metric_counts[key]
            for key in metric_sums
            if metric_counts[key]
        }
        worst_samples = sorted(
            results,
            key=lambda result: result.metrics.get("total_distance", 1.0),
            reverse=True,
        )[:top_k]
        worst_bars: list[dict[str, Any]] = []
        for result in results:
            for bar in result.bar_diagnostics:
                entry = dict(bar)
                entry["id"] = result.sample_id
                entry["dataset"] = result.dataset
                worst_bars.append(entry)
        worst_bars = sorted(worst_bars, key=lambda item: item.get("distance", 0.0), reverse=True)[:top_k]
        return {
            "sample_count": sample_count,
            "dataset_counts": counter_to_sorted_dict(dataset_counts),
            "tag_counts": counter_to_sorted_dict(tag_counts),
            "issue_counts": counter_to_sorted_dict(issue_counts),
            "metric_averages": metric_averages,
            "availability": {
                "reference": reference_available / sample_count if sample_count else 0.0,
                "candidate": candidate_available / sample_count if sample_count else 0.0,
                "melody": melody_available / sample_count if sample_count else 0.0,
            },
            "parse_success": {
                "reference": reference_valid / reference_available if reference_available else 0.0,
                "candidate": candidate_valid / candidate_available if candidate_available else 0.0,
            },
            "worst_samples": [
                {
                    "id": result.sample_id,
                    "dataset": result.dataset,
                    "score": result.metrics.get("score"),
                    "distance": result.metrics.get("total_distance"),
                    "tags": list(result.tags),
                    "issues": [issue.code for issue in result.issues],
                    "reference_event_count": result.reference.event_count if result.reference is not None else None,
                    "candidate_event_count": result.candidate.event_count if result.candidate is not None else None,
                }
                for result in worst_samples
            ],
            "worst_bars": worst_bars,
        }

    def write_results(
        self,
        results: list[ValidationResult],
        path: Path,
        *,
        include_tree: bool = False,
    ) -> int:
        return write_jsonl(path, (result.to_dict(include_tree=include_tree) for result in results))

    def write_summary(self, summary: dict[str, Any], path: Path) -> None:
        write_json(path, summary)

