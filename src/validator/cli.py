"""Command line interface for the harmony validator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .diagnostics import Comparator, ReferenceChecker
from .normalize import Normalizer
from .reporting import BatchReporter


def _split_fields(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    fields = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(fields) if fields else None


def _iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    paths = [path]
    if path.is_dir():
        paths = sorted(path.glob("*.jsonl"))
    for item in paths:
        with item.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate melody-to-harmony outputs with rule-based diagnostics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("input", type=Path, help="JSONL file or directory of JSONL files.")
        subparser.add_argument("--output-dir", type=Path, default=Path("outputs/validator"))
        subparser.add_argument("--output-jsonl", type=Path)
        subparser.add_argument("--output-json", type=Path)
        subparser.add_argument("--limit", type=int)
        subparser.add_argument("--top-k", type=int, default=20)
        subparser.add_argument("--include-tree", action="store_true")
        subparser.add_argument("--reference-fields")
        subparser.add_argument("--candidate-fields")
        subparser.add_argument("--melody-field")

    add_common(subparsers.add_parser("inspect", help="Check reference harmonies only."))
    add_common(subparsers.add_parser("compare", help="Compare candidate harmonies against reference."))
    return parser


def _resolve_output_paths(args: argparse.Namespace, default_name: str) -> tuple[Path, Path]:
    output_dir = args.output_dir or Path("outputs/validator")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_jsonl or (output_dir / f"{default_name}.jsonl")
    output_json = args.output_json or (output_dir / f"{default_name}.json")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    return output_jsonl, output_json


def _load_results(args: argparse.Namespace) -> list:
    default_normalizer = Normalizer()
    reference_fields = _split_fields(args.reference_fields) or default_normalizer.reference_fields
    candidate_fields = _split_fields(args.candidate_fields) or default_normalizer.candidate_fields
    normalizer = Normalizer(
        reference_fields=reference_fields,
        candidate_fields=candidate_fields,
        melody_field=args.melody_field or "abc_melody",
    )
    results = []
    rows = _iter_jsonl_rows(args.input)
    if args.limit is not None:
        rows = list(rows)[: args.limit]
    if args.command == "inspect":
        checker = ReferenceChecker(normalizer=normalizer)
        for row in rows:
            results.append(checker.check_record(row, reference_fields=normalizer.reference_fields))
    else:
        comparator = Comparator(normalizer=normalizer)
        for row in rows:
            results.append(
                comparator.compare_record(
                    row,
                    reference_fields=normalizer.reference_fields,
                    candidate_fields=normalizer.candidate_fields,
                    melody_field=normalizer.melody_field,
                )
            )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    results = _load_results(args)
    reporter = BatchReporter()
    summary = reporter.summarize(results, top_k=args.top_k)
    output_jsonl, output_json = _resolve_output_paths(args, args.command)
    reporter.write_results(results, output_jsonl, include_tree=args.include_tree)
    reporter.write_summary(summary, output_json)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
