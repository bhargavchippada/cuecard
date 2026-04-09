"""Frozen dataclasses for cuecard's data model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt

MAX_RULE_LENGTH = 500
MAX_EXPANSION_LENGTH = 200
MAX_EXPANSIONS_PER_RULE = 10

KNOWN_HOOK_EVENTS: frozenset[str] = frozenset({
    "PreToolUse", "PostToolUse", "UserPromptSubmit", "SubagentStart", "Stop",
})

AffinitySource = Literal["explicit", "inferred", "explicit+inferred", "default"]


def _hash_rule_text(text: str) -> str:
    """Canonical SHA-256 hash of rule text for affinity lookup.

    Uses UTF-8 encoding, lowercase hex, no prefix.
    Stored in affinity.json as bare hex string (NOT "sha256:..." prefixed).
    """
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RuleAffinity:
    """Event and tool affinity for a single rule."""

    events: frozenset[str]
    tools: frozenset[str]
    source: AffinitySource
    explicit_events: frozenset[str] = field(default_factory=frozenset)
    explicit_tools: frozenset[str] = field(default_factory=frozenset)
    reasoning: str = ""


class AffinityIndex:
    """Per-rule affinity metadata with O(1) lookup by rule text hash.

    Not a frozen dataclass — holds a dict for O(1) lookup.
    Treated as immutable after construction (same pattern as Index).
    """

    __slots__ = ("version", "mode", "model", "_lookup", "_items")
    __hash__ = None  # type: ignore[assignment]

    def __init__(
        self,
        version: int,
        mode: str,
        model: str,
        affinities: tuple[tuple[str, RuleAffinity], ...],
    ) -> None:
        self.version = version
        self.mode = mode
        self.model = model
        self._items = affinities
        self._lookup: dict[str, RuleAffinity] = dict(affinities)

    def get(self, rule: Rule) -> RuleAffinity | None:
        """Look up affinity by rule text hash. O(1)."""
        return self._lookup.get(_hash_rule_text(rule.text))

    def get_by_hash(self, text_hash: str) -> RuleAffinity | None:
        """Direct hash lookup (avoids re-hashing when hash is pre-computed)."""
        return self._lookup.get(text_hash)

    @property
    def items(self) -> tuple[tuple[str, RuleAffinity], ...]:
        """Serializable representation."""
        return self._items

    def __repr__(self) -> str:
        return f"AffinityIndex(mode={self.mode!r}, rules={len(self._lookup)})"


@dataclass(frozen=True)
class LoadedIndex:
    """Composite return from load_or_build() — index + optional affinity."""

    index: Index
    affinity: AffinityIndex | None = None


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
    expansions: tuple[str, ...] = ()
    events: frozenset[str] = field(default_factory=frozenset)
    tools: frozenset[str] = field(default_factory=frozenset)

    MAX_LENGTH: int = field(default=500, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class ExpandProgress:
    """Progress update from expand_rules()."""

    rule_index: int  # 0-based index of the rule being processed
    total_rules: int  # total number of rules
    rule_text: str  # text of the current rule (for display)
    expansions_generated: int  # total expansions generated so far
    skipped: bool  # True if this rule was skipped (missing_only)


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
class PipelineConfig:
    """Pipeline mode and LLM settings."""

    mode: str = "embedding"
    local_endpoint: str = "http://localhost:8081/v1"
    haiku_model: str = "claude-haiku-4-5"
    thinking: bool = False


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved and validated configuration."""

    source_paths: tuple[str, ...]
    global_source_paths: tuple[str, ...]
    project_source_paths: tuple[str, ...]
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
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    fusion_k: int = 60
    llm_candidates: int = 12
    sparse_enabled: bool = True
    expansion_max_per_rule: int = 10
    expansion_max_length: int = 200
    affinity_mode: str = "infer"


@dataclass(frozen=True)
class StageTrace:
    """Provenance for one pipeline stage."""

    stage: str  # "embedding", "rerank", "llm", "retrieval"
    input_count: int
    output_count: int
    latency_ms: float
    error: str | None = None  # None if successful, error message if degraded


@dataclass(frozen=True)
class RetrieverTrace:
    """Performance trace for a single retriever."""

    name: str  # "dense", "sparse"
    candidate_count: int  # results before fusion
    latency_ms: float
    unique_rules: int  # rules found by this retriever but NOT by others


@dataclass(frozen=True)
class RetrievalStageTrace(StageTrace):
    """Extended trace for the multi-retriever stage."""

    retrievers: tuple[RetrieverTrace, ...] = ()
    fusion_latency_ms: float = 0.0


@dataclass(frozen=True)
class PipelineResult:
    """Full pipeline output with per-stage tracing."""

    results: tuple[RankedResult, ...]
    stages: tuple[StageTrace, ...]
    mode: str
    event: str = ""
    event_mask_applied: bool = False
    rules_masked: int = 0


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
        "rule_map",
        "bm25_corpus",
    )

    def __init__(
        self,
        embeddings: npt.NDArray[Any],
        rules: tuple[Rule, ...],
        model_name: str,
        dim: int,
        sources: Mapping[str, SourceMeta],
        rule_map: tuple[int, ...] | None = None,
        bm25_corpus: tuple[str, ...] | None = None,
    ) -> None:
        # Default rule_map: identity mapping (one embedding per rule)
        if rule_map is None:
            rule_map = tuple(range(len(rules)))

        if embeddings.shape[0] != len(rule_map):
            msg = (
                f"Embedding rows ({embeddings.shape[0]}) "
                f"must match rule_map length ({len(rule_map)})"
            )
            raise ValueError(msg)
        if rule_map and not all(0 <= i < len(rules) for i in rule_map):
            msg = (
                f"All rule_map indices must be in range [0, {len(rules)})"
            )
            raise ValueError(msg)
        if bm25_corpus is not None and len(bm25_corpus) != len(rule_map):
            msg = (
                f"bm25_corpus length ({len(bm25_corpus)}) "
                f"must match rule_map length ({len(rule_map)})"
            )
            raise ValueError(msg)
        self.embeddings = embeddings
        self.rules = rules
        self.model_name = model_name
        self.dim = dim
        self.sources = sources
        self.rule_map = rule_map
        self.bm25_corpus = bm25_corpus

    @property
    def size(self) -> int:
        """Number of rules in the index."""
        return len(self.rules)

    def __repr__(self) -> str:
        return (
            f"Index(model={self.model_name!r}, "
            f"rules={self.size}, dim={self.dim})"
        )
