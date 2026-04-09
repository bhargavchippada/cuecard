#!/usr/bin/env python3
"""Claude Code hook adapter for cuecard.

Reads hook JSON from stdin, retrieves relevant rules,
injects them into hookSpecificOutput.additionalContext,
and writes the result to stdout.

Supports all 5 Claude Code hook events:
PreToolUse, PostToolUse, UserPromptSubmit, SubagentStart, Stop.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cuecard.models import RankedResult

from cuecard.config import load_config
from cuecard.indexing.loader import load_or_build
from cuecard.logger import log_retrieval
from cuecard.models import KNOWN_HOOK_EVENTS, MAX_REQUEST_BYTES, MAX_TOOL_NAME_LENGTH
from cuecard.retrieval.formatter import format_rules
from cuecard.security import scrub_secrets

_AGENT_TYPE_RE = re.compile(r"[^a-zA-Z0-9_-]")

# --- Per-event injection labels (Decision D13) ---

_LABEL_PREVENT = (
    "[cuecard \u2014 RULES you must follow for this action"
    " to avoid failures]"
)
_LABEL_VERIFY = (
    "[cuecard \u2014 VERIFY compliance for this completed action]"
)
_LABEL_PROPAGATE = (
    "[cuecard \u2014 RULES this agent must follow]"
)
_LABEL_AUDIT = (
    "[cuecard \u2014 AUDIT: verify these rules were followed this turn]"
)

_EVENT_LABELS: dict[str, str] = {
    "PreToolUse": _LABEL_PREVENT,
    "PostToolUse": _LABEL_VERIFY,
    "UserPromptSubmit": _LABEL_PREVENT,
    "SubagentStart": _LABEL_PROPAGATE,
    "Stop": _LABEL_AUDIT,
}


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


# --- Event handlers ---
# Each returns (query, tool_name, event).


def _handle_pre_tool_use(data: dict[str, object]) -> tuple[str, str, str]:
    """Extract query for PreToolUse events."""
    tool_name = _sanitize_field(
        str(data.get("tool_name", "")), MAX_TOOL_NAME_LENGTH,
    )
    tool_input = _format_tool_input(data.get("tool_input", ""))
    query = f"{tool_name}: {tool_input}"
    return query, tool_name, "PreToolUse"


def _handle_post_tool_use(data: dict[str, object]) -> tuple[str, str, str]:
    """Extract query for PostToolUse events.

    Security: raw output capped at 2000 chars BEFORE str(),
    then scrub_secrets() on full string BEFORE truncation to 500.
    """
    tool_name = _sanitize_field(
        str(data.get("tool_name", "")), MAX_TOOL_NAME_LENGTH,
    )
    tool_input = _format_tool_input(data.get("tool_input", ""))[:200]
    raw_output = data.get("tool_output", "")
    if isinstance(raw_output, str):
        raw_output = raw_output[:2000]
    else:
        raw_output = str(raw_output)[:2000]
    tool_output = _sanitize_field(scrub_secrets(raw_output), 500)
    query = f"PostToolUse:{tool_name}: {tool_input} \u2192 {tool_output}"
    return query, tool_name, "PostToolUse"


def _handle_user_prompt_submit(
    data: dict[str, object],
) -> tuple[str, str, str]:
    """Extract query for UserPromptSubmit events."""
    prompt_text = _sanitize_field(
        str(data.get("prompt", "")), 500,
    )
    return f"UserPromptSubmit: {prompt_text}", "", "UserPromptSubmit"


def _handle_subagent_start(
    data: dict[str, object],
) -> tuple[str, str, str]:
    """Extract query for SubagentStart events.

    agent_type validated: only alphanumeric, hyphen, underscore.
    """
    raw_type = str(data.get("agent_type", ""))
    agent_type = _AGENT_TYPE_RE.sub("", raw_type)[:100]
    raw_input = data.get("tool_input", {})
    prompt = ""
    if isinstance(raw_input, dict):
        raw_prompt = str(raw_input.get("prompt", ""))[:2000]
        prompt = _sanitize_field(scrub_secrets(raw_prompt), 500)
    query = f"SubagentStart:{agent_type}: {prompt}"
    return query, "", "SubagentStart"


def _handle_stop(data: dict[str, object]) -> tuple[str, str, str]:
    """Extract query for Stop events.

    Falls back to "turn completed" when no stop_reason is provided.
    """
    raw_reason = str(data.get("stop_reason", "turn completed"))[:2000]
    stop_reason = _sanitize_field(scrub_secrets(raw_reason), 500)
    query = f"Stop: {stop_reason}"
    return query, "", "Stop"


_EVENT_HANDLERS: dict[str, Callable[[dict[str, object]], tuple[str, str, str]]] = {
    "PreToolUse": _handle_pre_tool_use,
    "PostToolUse": _handle_post_tool_use,
    "UserPromptSubmit": _handle_user_prompt_submit,
    "SubagentStart": _handle_subagent_start,
    "Stop": _handle_stop,
}


def _try_daemon(data: dict[str, object]) -> dict[str, object] | None:
    """Try the daemon fast path.  Returns response dict or None."""
    from cuecard.serve import query_daemon

    return query_daemon(data)


def _detect_event(data: dict[str, object]) -> str:
    """Detect the hook event type from input JSON."""
    raw_event = str(
        data.get("hook_event_name", data.get("event", "PreToolUse")),
    )
    return raw_event if raw_event in KNOWN_HOOK_EVENTS else "PreToolUse"


def _run_pipeline_path(
    data: dict[str, object],
    event: str,
    query: str,
    tool_name: str,
) -> dict[str, object]:
    """Load config, run pipeline, return enriched output dict."""
    start = time.monotonic()

    config = load_config(project_dir=Path.cwd())

    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=config.model_name)
    loaded = load_or_build(config, model)  # type: ignore[arg-type]

    # Build hook output — separate from input data
    raw_hook_output = data.get("hookSpecificOutput")
    hook_output: dict[str, object] = (
        dict(raw_hook_output)
        if isinstance(raw_hook_output, dict) else {}
    )
    hook_output["hookEventName"] = event
    if event == "PreToolUse":
        hook_output["permissionDecision"] = "allow"

    results: Sequence[RankedResult] = []
    index = loaded.index if loaded is not None else None
    affinity = loaded.affinity if loaded is not None else None

    if index is not None and index.size > 0:
        from cuecard.retrieval.pipeline import run_pipeline

        label = _EVENT_LABELS.get(event, _LABEL_PREVENT)
        pipeline_result = run_pipeline(
            query, index, config,
            embedding_model=model, mode=config.pipeline.mode,
            event=event, tool_name=tool_name,
            affinity=affinity,
        )
        results = pipeline_result.results
        if results:
            hook_output["additionalContext"] = format_rules(
                results, label=label,
            )

    latency_ms = (time.monotonic() - start) * 1000
    if index is not None:
        log_retrieval(
            event=event,
            tool_name=tool_name,
            query=query,
            results=list(results),
            total_rules=index.size if index else 0,
            index_rebuilt=False,
            latency_ms=latency_ms,
            model=config.model_name,
            redact=config.redact,
            max_query_length=config.query_max_length,
            max_log_size_mb=config.max_log_size_mb,
            verbose=config.verbose,
        )

    return {**data, "hookSpecificOutput": hook_output}


def main() -> None:
    """Read hook JSON from stdin, retrieve rules, inject into context."""
    import logging as _logging

    _logger = _logging.getLogger(__name__)

    output: dict[str, object] = {}

    try:
        raw = sys.stdin.read(MAX_REQUEST_BYTES)
        output = json.loads(raw)

        # Fast path: try daemon first
        daemon_result = _try_daemon(output)
        if daemon_result is not None:
            print(json.dumps(daemon_result))
            return

        # Detect event and dispatch to handler
        event = _detect_event(output)
        handler = _EVENT_HANDLERS.get(event, _handle_pre_tool_use)
        query, tool_name, event = handler(output)

        # Slow path: full pipeline (output only updated on success)
        output = _run_pipeline_path(output, event, query, tool_name)

    except Exception as exc:
        _logger.exception("Hook failed")
        print(f"[cuecard] Error: {exc}", file=sys.stderr)

    print(json.dumps(output))


if __name__ == "__main__":
    main()
