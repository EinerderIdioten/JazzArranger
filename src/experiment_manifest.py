"""Helpers for frozen experiment manifests."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

DEFAULT_MANIFEST_PATH = Path("manifests/root_simple_v1.json")


def resolve_experiment_manifest_path(path: Path | None) -> Path | None:
    if path is not None:
        return path
    return DEFAULT_MANIFEST_PATH if DEFAULT_MANIFEST_PATH.exists() else None


def load_experiment_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def experiment_manifest_digest(path: Path | None) -> str | None:
    if path is None:
        return None
    return sha256(Path(path).read_bytes()).hexdigest()


def experiment_manifest_summary(path: Path | None, manifest: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    if path is None and manifest is None:
        return None
    if manifest is None and path is not None:
        manifest = load_experiment_manifest(path)
    if manifest is None:
        return None
    manifest_path = str(path) if path is not None else None
    return {
        "path": manifest_path,
        "sha256": experiment_manifest_digest(path),
        "name": manifest.get("manifest_name") or manifest.get("name"),
        "version": manifest.get("version"),
        "status": manifest.get("status"),
    }


def _section(manifest: Mapping[str, Any] | None, name: str) -> dict[str, Any]:
    if not manifest:
        return {}
    value = manifest.get(name)
    return dict(value) if isinstance(value, Mapping) else {}


def manifest_training_section(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    return _section(manifest, "training")


def manifest_validator_section(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    return _section(manifest, "validator")


def manifest_dataset_names(manifest: Mapping[str, Any] | None) -> set[str] | None:
    if not manifest:
        return None
    training = manifest_training_section(manifest)
    datasets = training.get("datasets") or []
    names: set[str] = set()
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            continue
        if not dataset.get("include", False):
            continue
        name = str(dataset.get("name") or "").strip()
        if name:
            names.add(name)
    return names or None


def manifest_allowed_qualities(manifest: Mapping[str, Any] | None) -> set[str] | None:
    if not manifest:
        return None
    training = manifest_training_section(manifest)
    values = training.get("allowed_qualities") or []
    qualities = {str(item).strip() for item in values if str(item).strip()}
    return qualities or None
