#!/usr/bin/env python3
"""Claude Code PreToolUse hook for cuecard.

Reads hook JSON from stdin, retrieves relevant rules,
injects them into hookSpecificOutput.additionalContext,
and writes the result to stdout.
"""

from __future__ import annotations

import json
import sys
import time

from cuecard.config import load_config
from cuecard.formatter import format_rules
from cuecard.indexer import load_index
from cuecard.logger import log_retrieval
from cuecard.retriever import retrieve

_MAX_STDIN = 1_000_000  # 1 MB guard
_MAX_TOOL_NAME = 200


def _sanitize_field(value: str, max_len: int) -> str:
    """Cap length and strip control characters."""
    return value[:max_len].replace("\n", " ").replace("\r", " ")


def _format_tool_input(tool_input: object) -> str:
    """Format tool_input for the query string.

    Dicts are serialized with json.dumps (not str(), which produces
    Python repr syntax).  Everything else is converted via str().
    """
    if isinstance(tool_input, dict):
        return json.dumps(tool_input)[:500]
    return str(tool_input)[:500]


def main() -> None:
    """Read hook JSON from stdin, retrieve rules, inject into context."""
    data: dict[str, object] = {}

    try:
        raw = sys.stdin.read(_MAX_STDIN)
        data = json.loads(raw)
        tool_name = _sanitize_field(
            str(data.get("tool_name", "")), _MAX_TOOL_NAME,
        )
        tool_input = _format_tool_input(data.get("tool_input", ""))
        query = f"{tool_name}: {tool_input}"

        start = time.monotonic()

        config = load_config()
        index = load_index(config.global_cache_dir)

        if index is not None and index.size > 0:
            from fastembed import TextEmbedding

            model = TextEmbedding(
                model_name=config.model_name,
            )

            results = retrieve(
                index,
                query,
                top_k=config.top_k,
                threshold=config.threshold,
                dedup_threshold=config.dedup_threshold,
                max_query_length=config.query_max_length,
                model=model,
            )

            latency_ms = (time.monotonic() - start) * 1000

            if results:
                context = format_rules(results)
                hook_output = data.get("hookSpecificOutput")
                if not isinstance(hook_output, dict):
                    hook_output = {}
                    data["hookSpecificOutput"] = hook_output
                hook_output["additionalContext"] = context

            log_retrieval(
                event="PreToolUse",
                tool_name=tool_name,
                query=query,
                results=results,
                total_rules=index.size,
                index_rebuilt=False,
                latency_ms=latency_ms,
                model=config.model_name,
                redact=config.redact,
                max_query_length=config.query_max_length,
                max_log_size_mb=config.max_log_size_mb,
                verbose=config.verbose,
            )
        else:
            latency_ms = (time.monotonic() - start) * 1000
            if index is not None:
                log_retrieval(
                    event="PreToolUse",
                    tool_name=tool_name,
                    query=query,
                    results=[],
                    total_rules=0,
                    index_rebuilt=False,
                    latency_ms=latency_ms,
                    model=config.model_name,
                    redact=config.redact,
                    max_query_length=config.query_max_length,
                    max_log_size_mb=config.max_log_size_mb,
                    verbose=config.verbose,
                )
    except Exception as exc:
        print(f"[cuecard] Error: {exc}", file=sys.stderr)

    print(json.dumps(data))


if __name__ == "__main__":
    main()
