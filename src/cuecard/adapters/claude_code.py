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


def main() -> None:
    """Read hook JSON from stdin, retrieve rules, inject into context."""
    raw = sys.stdin.read()
    data: dict[str, object] = json.loads(raw)
    tool_name = str(data.get("tool_name", ""))
    tool_input = str(data.get("tool_input", ""))[:500]
    query = f"{tool_name}: {tool_input}"

    start = time.monotonic()

    try:
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
