"""Shared LLM helpers for local and Haiku backends."""

from __future__ import annotations

import logging

import httpx

from cuecard.security import ConfigError

logger = logging.getLogger(__name__)

_ALLOWED_LLM_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})
_ALLOWED_HAIKU_MODELS: frozenset[str] = frozenset({
    "claude-haiku-4-5",
    "claude-haiku-4-5-20251001",
})
_TIMEOUT = 60.0


def validate_endpoint(endpoint: str) -> None:
    """Validate that endpoint points to localhost only. Raises ConfigError."""
    from urllib.parse import urlparse

    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https"):
        msg = f"llm.local_endpoint must use http/https, got {parsed.scheme!r}"
        raise ConfigError(msg)
    if parsed.username or parsed.password:
        msg = "llm.local_endpoint must not contain userinfo (@ in URL)"
        raise ConfigError(msg)
    if parsed.hostname not in _ALLOWED_LLM_HOSTS:
        msg = f"llm.local_endpoint must be localhost, got {parsed.hostname!r}"
        raise ConfigError(msg)


def call_local(
    system_prompt: str,
    user_prompt: str,
    endpoint: str,
    thinking: bool,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Call a local OpenAI-compatible LLM endpoint."""
    validate_endpoint(endpoint)
    body: dict[str, object] = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    response = httpx.post(
        f"{endpoint.rstrip('/')}/chat/completions",
        json=body,
        timeout=_TIMEOUT,
    )
    response.raise_for_status()
    data: dict[str, object] = response.json()
    choices = data["choices"]
    if not isinstance(choices, list) or not choices:
        msg = "No choices in response"
        raise ValueError(msg)
    first = choices[0]
    if not isinstance(first, dict):
        msg = "Invalid choice format"
        raise ValueError(msg)
    message = first["message"]
    if not isinstance(message, dict):
        msg = "Invalid message format"
        raise ValueError(msg)
    content = message["content"]
    if not isinstance(content, str):
        msg = "Invalid content format"
        raise ValueError(msg)
    return content


def call_haiku(system_prompt: str, user_prompt: str, model: str) -> str:
    """Call Claude via claude-agent-sdk."""
    import asyncio

    from claude_agent_sdk import (  # type: ignore[import-not-found]
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        query,
    )

    options = ClaudeAgentOptions(
        model=model,
        max_turns=1,
        system_prompt=system_prompt,
        tools=[],
        permission_mode="bypassPermissions",
        setting_sources=[],
    )

    async def _run() -> str:
        parts: list[str] = []
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
        return "".join(parts)

    return asyncio.run(_run())
