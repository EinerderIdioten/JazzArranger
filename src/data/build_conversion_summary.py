"""Build a compact summary from all dataset conversion reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.common import ensure_dirs, write_json


REPORT_FILES = {
    "EMOPIA+": "emopia_plus_conversion_report.json",
    "HLSD": "hlsd_conversion_report.json",
    "POP909": "pop909_conversion_report.json",
    "OpenBook": "openbook_conversion_report.json",
}


def summarize_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    converted = report.get("converted_files", report.get("converted_records", 0))
    failed = report.get("failed_files", report.get("failed_segments", report.get("failed_files_or_segments", 0)))
    return {
        "dataset": report.get("dataset"),
        "dataset_weight": report.get("dataset_weight"),
        "total_files": report.get("total_files"),
        "converted": converted,
        "failed": failed,
        "split_counts": report.get("split_counts", {}),
        "key_distribution": report.get("key_distribution", {}),
        "source_key_distribution": report.get("source_key_distribution", {}),
        "transpose_distribution": report.get("transpose_distribution", {}),
        "quality_distribution": report.get("quality_distribution", {}),
        "root_distribution": report.get("root_distribution", {}),
        "conversion_stats": report.get("conversion_stats", {}),
    }


def build_summary(log_dir: Path) -> dict:
    datasets: list[dict] = []
    for _, filename in REPORT_FILES.items():
        path = log_dir / filename
        if path.exists():
            datasets.append(summarize_report(path))
    total_converted = sum(int(dataset.get("converted") or 0) for dataset in datasets)
    return {
        "total_converted_records": total_converted,
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    args = parser.parse_args()
    ensure_dirs(args.log_dir)
    summary = build_summary(args.log_dir)
    write_json(args.log_dir / "conversion_summary.json", summary)
    print(f"summary datasets={len(summary['datasets'])} total_converted={summary['total_converted_records']}")


if __name__ == "__main__":
    main()
