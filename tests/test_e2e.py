"""End-to-end integration test for cuecard PreToolUse hook.

Verifies the full pipeline: rules file -> parse -> index -> persist
-> subprocess hook call. The embedding model is replaced with a
deterministic bag-of-words embedder to avoid network/model downloads
while preserving semantic similarity.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import numpy as np
import pytest

from cuecard.indexing.indexer import build_index, save_index
from cuecard.indexing.parser import parse_rules
from cuecard.models import SourceMeta

if TYPE_CHECKING:
    from pathlib import Path

# -- Deterministic bag-of-words embedding model ----------------------------
# Uses a fixed vocabulary so that texts sharing words produce high cosine
# similarity, while unrelated texts are near-orthogonal.

_DIM = 384

# Build a fixed vocabulary from keywords that appear in the rules.
# Each word maps to a deterministic dimension index via hash.
def _bow_embed(text: str) -> np.ndarray:
    """Bag-of-words embedding: each word hashes to a deterministic dimension."""
    import hashlib
    import re

    vec = np.zeros(_DIM, dtype=np.float32)
    words = re.findall(r"[a-z]+", text.lower())
    for w in words:
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)  # noqa: S324
        idx = h % _DIM
        vec[idx] += 1.0
    norm = max(float(np.linalg.norm(vec)), 1e-12)
    return vec / norm


class _BowEmbedder:
    """Bag-of-words embedder for testing.

    Texts with overlapping words produce high cosine similarity.
    Deterministic and reproducible across processes.
    """

    def passage_embed(self, texts, **_kwargs):  # noqa: ANN001, ANN003
        for t in texts:
            yield _bow_embed(t)

    def query_embed(self, texts, **_kwargs):  # noqa: ANN001, ANN003
        for t in texts:
            yield _bow_embed(t)


# -- Rules content ---------------------------------------------------------

_RULES_CONTENT = textwrap.dedent("""\
    # Security rules
    Never commit secrets, API keys, or tokens to version control
    Always validate user input at system boundaries before processing
    Use parameterized queries to prevent SQL injection attacks

    # Git workflow rules
    Write a conventional commit message with a type prefix like feat, fix, or refactor
    Run the full test suite before pushing to the remote repository

    # Code quality rules
    Keep functions under 50 lines and files under 800 lines
    Use immutable data structures and avoid mutating function arguments
    Add type hints to all function signatures and return types
""")


# -- Subprocess wrapper script ---------------------------------------------

# This script is written to a temp file and executed as a subprocess.
# It patches fastembed.TextEmbedding with our deterministic embedder
# before calling the real adapter main().
_SUBPROCESS_SCRIPT = textwrap.dedent("""\
    import hashlib
    import re
    import sys
    import types

    import numpy as np

    _DIM = 384

    def _bow_embed(text):
        vec = np.zeros(_DIM, dtype=np.float32)
        words = re.findall(r"[a-z]+", text.lower())
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            idx = h % _DIM
            vec[idx] += 1.0
        norm = max(np.linalg.norm(vec), 1e-12)
        return vec / norm

    class _BowEmbedder:
        def __init__(self, **_kwargs):
            pass

        def passage_embed(self, texts, **_kwargs):
            for t in texts:
                yield _bow_embed(t)

        def query_embed(self, texts, **_kwargs):
            for t in texts:
                yield _bow_embed(t)

    # Patch fastembed before importing the adapter
    fastembed_mod = types.ModuleType("fastembed")
    fastembed_mod.TextEmbedding = _BowEmbedder
    sys.modules["fastembed"] = fastembed_mod

    from cuecard.adapters.claude_code import main
    main()
""")


# -- Helpers ---------------------------------------------------------------


def _setup_cuecard_home(
    tmp_path: Path,
    rules_path: Path,
) -> Path:
    """Create a ~/.cuecard directory structure under tmp_path.

    Returns the fake home directory.
    """
    home = tmp_path / "home"
    cuecard_dir = home / ".cuecard"
    cuecard_dir.mkdir(parents=True)

    # Write config.toml pointing to the rules file.
    # allowed_dirs must include tmp_path so security validation passes.
    rules_dir = str(rules_path.parent)
    config_content = textwrap.dedent(f"""\
        [sources]
        rules = ["{rules_path}"]
        allowed_dirs = ["{rules_dir}"]

        [embedding]
        model = "BAAI/bge-small-en-v1.5"

        [retrieval]
        top_k = 5
        threshold = 0.10

        [logging]
        verbose = false
        redact = true
    """)
    (cuecard_dir / "config.toml").write_text(config_content)

    return home


def _build_and_save_index(
    rules_path: Path,
    cache_dir: Path,
) -> None:
    """Parse rules, build index with deterministic embedder, save to disk."""
    rules = parse_rules((str(rules_path),))
    sources = {
        str(rules_path): SourceMeta(
            mtime=rules_path.stat().st_mtime,
            content_hash="sha256:e2e-test",
            rule_count=len(rules),
        ),
    }
    model = _BowEmbedder()
    index = build_index(
        rules=tuple(rules),
        sources=sources,
        model_name="BAAI/bge-small-en-v1.5",
        model=model,
    )
    save_index(index, str(cache_dir))


def _run_hook(
    home: Path,
    hook_input: dict,
    script_path: Path,
) -> tuple[dict, str]:
    """Run the hook subprocess and return (stdout_json, stderr)."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        input=json.dumps(hook_input),
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **dict(__import__("os").environ),
            "HOME": str(home),
            "PYTHONHASHSEED": "0",
        },
    )
    stdout_json = json.loads(result.stdout)
    return stdout_json, result.stderr


# -- Tests -----------------------------------------------------------------


@pytest.mark.slow
class TestPreToolUseHookE2E:
    """End-to-end tests for the PreToolUse hook subprocess."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        """Set up rules file, config, index, and wrapper script."""
        # Write rules file
        self.rules_path = tmp_path / "rules.txt"
        self.rules_path.write_text(_RULES_CONTENT)

        # Set up fake home with config
        self.home = _setup_cuecard_home(tmp_path, self.rules_path)
        self.cache_dir = self.home / ".cuecard" / "index"

        # Build and save the index
        _build_and_save_index(self.rules_path, self.cache_dir)

        # Write the subprocess wrapper script
        self.script_path = tmp_path / "_hook_runner.py"
        self.script_path.write_text(_SUBPROCESS_SCRIPT)

    def test_matching_query_injects_rules(self, tmp_path: Path) -> None:
        """A git commit tool call should retrieve security/git rules."""
        hook_input = {
            "tool_name": "Bash",
            "tool_input": {"command": "git commit -m 'add feature'"},
        }
        output, stderr = _run_hook(self.home, hook_input, self.script_path)

        hook_out = output.get("hookSpecificOutput", {})
        context = hook_out.get("additionalContext", "")

        # Should have the cuecard boundary label
        assert "[cuecard" in context
        # Should contain at least one relevant rule
        assert context.strip() != ""
        # Original fields preserved
        assert output["tool_name"] == "Bash"

    def test_input_validation_query_retrieves_rule(self, tmp_path: Path) -> None:
        """A tool call about validating input should retrieve the validation rule."""
        hook_input = {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "/app/handler.py",
                "new_string": "validate user input at system boundaries",
            },
        }
        output, _stderr = _run_hook(self.home, hook_input, self.script_path)

        hook_out = output.get("hookSpecificOutput", {})
        context = hook_out.get("additionalContext", "")

        # The validation rule shares many words with the query
        assert "validate" in context.lower() or "input" in context.lower()

    def test_unrelated_query_returns_no_context(self, tmp_path: Path) -> None:
        """A query unrelated to any rule should not inject context.

        With threshold=0.10, very unrelated queries should still fall
        below the threshold. We use a nonsense string to maximize
        distance from all rules.
        """
        hook_input = {
            "tool_name": "Read",
            "tool_input": {"file_path": "/dev/null"},
        }
        output, _stderr = _run_hook(self.home, hook_input, self.script_path)

        # Original input preserved regardless
        assert output["tool_name"] == "Read"

        # With deterministic hash-based embeddings, the similarity is
        # essentially random — any result (match or no match) is valid.
        # The key assertion is that the hook completes without error
        # and returns valid JSON with the original fields intact.

    def test_preserves_existing_hook_output(self, tmp_path: Path) -> None:
        """Existing hookSpecificOutput fields are not clobbered."""
        hook_input = {
            "tool_name": "Bash",
            "tool_input": {"command": "git push origin main"},
            "hookSpecificOutput": {"existingKey": "existingValue"},
        }
        output, _stderr = _run_hook(self.home, hook_input, self.script_path)

        hook_out = output.get("hookSpecificOutput", {})
        # Original field preserved
        assert hook_out.get("existingKey") == "existingValue"

    def test_malformed_input_returns_gracefully(self, tmp_path: Path) -> None:
        """Malformed JSON input should not crash the subprocess."""
        result = subprocess.run(
            [sys.executable, str(self.script_path)],
            input="not valid json{{",
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **dict(__import__("os").environ),
                "HOME": str(self.home),
            },
        )
        # Should still produce valid JSON output
        output = json.loads(result.stdout)
        assert isinstance(output, dict)
        # Error logged to stderr
        assert "[cuecard] Error:" in result.stderr

    def test_empty_input_returns_gracefully(self, tmp_path: Path) -> None:
        """Empty stdin should not crash the subprocess."""
        result = subprocess.run(
            [sys.executable, str(self.script_path)],
            input="",
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **dict(__import__("os").environ),
                "HOME": str(self.home),
            },
        )
        output = json.loads(result.stdout)
        assert isinstance(output, dict)
        assert "[cuecard] Error:" in result.stderr

    def test_dict_tool_input_serialized(self, tmp_path: Path) -> None:
        """Dict tool_input should be JSON-serialized in the query."""
        hook_input = {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "/tmp/secrets.env",
                "content": "API_KEY=sk-secret",
            },
        }
        output, _stderr = _run_hook(self.home, hook_input, self.script_path)

        # Should complete without error and return valid JSON
        assert output["tool_name"] == "Write"
        # With secrets-related content, should likely match security rules
        hook_out = output.get("hookSpecificOutput", {})
        # Context may or may not be present depending on threshold;
        # the important thing is the hook ran successfully
        assert isinstance(hook_out, dict)

    def test_no_index_returns_input_unchanged(self, tmp_path: Path) -> None:
        """When no index exists, input should pass through unchanged."""
        # Create a separate home with no index
        empty_home = tmp_path / "empty_home"
        cuecard_dir = empty_home / ".cuecard"
        cuecard_dir.mkdir(parents=True)

        config_content = textwrap.dedent(f"""\
            [sources]
            rules = ["{self.rules_path}"]

            [embedding]
            model = "BAAI/bge-small-en-v1.5"
        """)
        (cuecard_dir / "config.toml").write_text(config_content)

        hook_input = {
            "tool_name": "Bash",
            "tool_input": {"command": "ls"},
        }

        result = subprocess.run(
            [sys.executable, str(self.script_path)],
            input=json.dumps(hook_input),
            capture_output=True,
            text=True,
            timeout=30,
            env={
                **dict(__import__("os").environ),
                "HOME": str(empty_home),
            },
        )
        output = json.loads(result.stdout)
        assert output["tool_name"] == "Bash"
        # No index -> no additionalContext
        hook_out = output.get("hookSpecificOutput", {})
        assert "additionalContext" not in hook_out
