"""Frozen dataclasses for cuecard's data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt

MAX_RULE_LENGTH = 500


@dataclass(frozen=True)
class Provenance:
    """Traces a rule back to its source file and location."""

    file: str
    line_start: int
    line_end: int
    section_path: tuple[str, ...] = ()
    chunk_type: str = "rule"


@dataclass(frozen=True)
class Rule:
    """A single rule with its source provenance."""

    text: str
    provenance: Provenance
    summary: str | None = None

    MAX_LENGTH: int = field(default=500, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class RankedResult:
    """A rule with its retrieval score."""

    rule: Rule
    score: float


@dataclass(frozen=True)
class SourceMeta:
    """Metadata for a tracked source file in the index."""

    mtime: float
    content_hash: str
    rule_count: int


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved and validated configuration."""

    source_paths: tuple[str, ...]
    model_name: str
    top_k: int
    threshold: float
    dedup_threshold: float
    query_max_length: int
    hook_events: tuple[str, ...]
    verbose: bool
    redact: bool
    max_log_size_mb: int
    global_cache_dir: str
    project_cache_dir: str | None
    allowed_dirs: tuple[str, ...]


@dataclass(frozen=True)
class StageTrace:
    """Provenance for one pipeline stage."""

    stage: str  # "embedding", "rerank", "llm"
    input_count: int
    output_count: int
    latency_ms: float
    error: str | None = None  # None if successful, error message if degraded


@dataclass(frozen=True)
class PipelineResult:
    """Full pipeline output with per-stage tracing."""

    results: list[RankedResult]
    stages: tuple[StageTrace, ...]
    mode: str


class Index:
    """Embedding index over rules for semantic retrieval.

    Not frozen because it holds a mutable numpy array,
    but should be treated as immutable after construction.
    """

    __slots__ = (
        "embeddings",
        "rules",
        "model_name",
        "dim",
        "sources",
    )

    def __init__(
        self,
        embeddings: npt.NDArray[Any],
        rules: tuple[Rule, ...],
        model_name: str,
        dim: int,
        sources: dict[str, SourceMeta],
    ) -> None:
        if embeddings.shape[0] != len(rules):
            msg = (
                f"Embedding rows ({embeddings.shape[0]}) "
                f"must match rule count ({len(rules)})"
            )
            raise ValueError(msg)
        self.embeddings = embeddings
        self.rules = rules
        self.model_name = model_name
        self.dim = dim
        self.sources = sources

    @property
    def size(self) -> int:
        """Number of rules in the index."""
        return len(self.rules)

    def __repr__(self) -> str:
        return (
            f"Index(model={self.model_name!r}, "
            f"rules={self.size}, dim={self.dim})"
        )
