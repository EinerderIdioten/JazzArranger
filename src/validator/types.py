"""Core data structures for the harmony validator."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from src.data.common import counter_to_sorted_dict


@dataclass(frozen=True)
class ChordEvent:
    start: int
    end: int
    root: str
    quality: str
    raw: str = ""
    bass: str | None = None
    extensions: tuple[str, ...] = ()
    alterations: tuple[str, ...] = ()
    family: str | None = None
    degree: str | None = None
    function_role: str | None = None
    local_key: str | None = None
    cadential_role: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TonicizationWindow:
    start: int
    end: int
    target_key: str
    label: str
    confidence: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    label: str
    local_key: str | None = None
    function_pattern: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["function_pattern"] = list(self.function_pattern)
        return payload


@dataclass
class TreeNode:
    label: str
    start: int
    end: int
    children: list["TreeNode"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "metadata": self.metadata,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass
class ParsedHarmony:
    source: str
    key: str | None
    bar_grid: int
    total_grid: int
    events: list[ChordEvent]
    parse_errors: list[str] = field(default_factory=list)
    tonicization_windows: list[TonicizationWindow] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    tree: TreeNode | None = None

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def valid(self) -> bool:
        return not self.parse_errors

    def summary(self, *, include_tree: bool = False) -> dict[str, Any]:
        role_counts = counter_to_sorted_dict(Counter(event.function_role or "X" for event in self.events))
        quality_counts = counter_to_sorted_dict(Counter(event.quality for event in self.events))
        payload = {
            "source": self.source,
            "key": self.key,
            "bar_grid": self.bar_grid,
            "total_grid": self.total_grid,
            "event_count": len(self.events),
            "parse_errors": list(self.parse_errors),
            "chunk_count": len(self.chunks),
            "tonicization_count": len(self.tonicization_windows),
            "function_role_counts": role_counts,
            "quality_counts": quality_counts,
        }
        if include_tree and self.tree is not None:
            payload["tree"] = self.tree.to_dict()
        return payload


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "warning"
    span: tuple[int, int] | None = None
    bar_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.span is not None:
            payload["span"] = list(self.span)
        return payload


@dataclass
class ValidationResult:
    sample_id: str
    dataset: str | None
    total_grid: int | None
    key: str | None
    reference: ParsedHarmony | None = None
    candidate: ParsedHarmony | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)
    bar_diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(
        self,
        *,
        include_tree: bool = False,
        include_bar_diagnostics: bool = True,
    ) -> dict[str, Any]:
        payload = {
            "id": self.sample_id,
            "dataset": self.dataset,
            "total_grid": self.total_grid,
            "key": self.key,
            "metrics": self.metrics,
            "tags": list(self.tags),
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if self.reference is not None:
            payload["reference"] = self.reference.summary(include_tree=include_tree)
        if self.candidate is not None:
            payload["candidate"] = self.candidate.summary(include_tree=include_tree)
        if include_bar_diagnostics:
            payload["bar_diagnostics"] = list(self.bar_diagnostics)
        return payload
