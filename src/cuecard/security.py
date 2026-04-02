"""Path validation, secrets scrubbing, and file permission helpers."""

from __future__ import annotations

import logging
import os
import re
import stat
import warnings
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised for configuration or path validation errors."""


# --- Secrets scrubbing patterns ---

_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    # Stripe live/test keys
    (r"sk_live_[A-Za-z0-9]{24,}", "[REDACTED]"),
    (r"sk_test_[A-Za-z0-9]{24,}", "[REDACTED]"),
    # AWS access key IDs
    (r"AKIA[A-Z0-9]{16}", "[REDACTED]"),
    # GitHub personal access tokens
    (r"ghp_[A-Za-z0-9]{36,}", "[REDACTED]"),
    # Connection strings with embedded credentials
    (r"postgresql://[^\s]+:[^\s]+@[^\s]+", "[REDACTED]"),
    # Bearer JWT tokens
    (
        r"Bearer\s+eyJ[A-Za-z0-9_-]+"
        r"\.?[A-Za-z0-9_-]*\.?[A-Za-z0-9_-]*",
        "Bearer [REDACTED]",
    ),
    # Anthropic API keys
    (r"sk-ant-[A-Za-z0-9-]{20,}", "[REDACTED]"),
    # OpenAI project keys
    (r"sk-proj-[A-Za-z0-9]{20,}", "[REDACTED]"),
    # OpenAI legacy keys
    (r"sk-[A-Za-z0-9]{40,}", "[REDACTED]"),
    # Slack bot tokens
    (r"xoxb-[0-9]+-[0-9]+-[A-Za-z0-9]+", "[REDACTED]"),
    # Google OAuth tokens
    (r"ya29\.[A-Za-z0-9._-]{20,}", "[REDACTED]"),
    # Google API keys
    (r"AIzaSy[A-Za-z0-9_-]{33}", "[REDACTED]"),
    # Generic KEY=value where KEY contains secret-like words
    (
        r"(?i)([A-Z_]*(?:SECRET|TOKEN|PASSWORD|API_KEY)[A-Z_]*)=(\S+)",
        r"\1=[REDACTED]",
    ),
)

_COMPILED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat), repl) for pat, repl in _SECRET_PATTERNS
)


def scrub_secrets(text: str) -> str:
    """Replace secrets in text with [REDACTED]."""
    result = text
    for pattern, replacement in _COMPILED_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


# --- Path validation ---


def validate_source_path(
    raw_path: str,
    config_dir: Path,
    allowed_dirs: tuple[str, ...],
    *,
    is_global: bool,
) -> Path:
    """Validate and resolve a source path from config.

    Args:
        raw_path: The raw path string from config.
        config_dir: Directory containing the config file.
        allowed_dirs: Tuple of additional allowed directory paths.
        is_global: Whether this path comes from the global config.

    Returns:
        Resolved absolute path.

    Raises:
        ConfigError: If the path fails validation.
    """
    if ".." in Path(raw_path).parts:
        msg = f"Path contains '..': {raw_path!r} (rejected for security)"
        raise ConfigError(msg)

    expanded = Path(os.path.expanduser(raw_path))

    if not expanded.is_absolute():
        expanded = config_dir / expanded

    resolved = expanded.resolve()

    # Build the set of allowed root directories
    allowed_roots: list[Path] = []
    # config_dir is already the correct root:
    #   global: ~/.cuecard (or test override)
    #   project: the directory containing cuecard.toml
    allowed_roots.append(config_dir.resolve())

    for d in allowed_dirs:
        allowed_roots.append(Path(os.path.expanduser(d)).resolve())

    for root in allowed_roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue

    if is_global:
        scope = "~/.cuecard/ or allowed_dirs"
    else:
        scope = f"{config_dir} or allowed_dirs"

    msg = f"Path {resolved} is outside {scope}"
    raise ConfigError(msg)


# --- File permission helpers ---

_DIR_MODE = 0o700
_FILE_MODE = 0o600


def ensure_directory(path: Path) -> None:
    """Create directory with 0o700 permissions if it doesn't exist."""
    if not path.exists():
        path.mkdir(parents=True, mode=_DIR_MODE)
    elif not path.is_dir():
        msg = f"Expected directory, got file: {path}"
        raise ConfigError(msg)


def secure_open(path: Path, mode: str) -> IO[str]:
    """Open a file with 0o600 permissions (TOCTOU-safe for creation).

    For write modes on new files, uses os.open + os.fdopen to atomically
    create with the correct permissions.
    """
    if path.exists():
        # Warn if existing file is too permissive
        current = stat.S_IMODE(path.stat().st_mode)
        if current & (stat.S_IRWXG | stat.S_IRWXO):
            warnings.warn(
                f"File {path} has permissive mode {oct(current)}, "
                f"expected {oct(_FILE_MODE)}",
                stacklevel=2,
            )
        return open(path, mode)  # noqa: SIM115

    # New file: create atomically with correct permissions
    if "r" in mode and "+" not in mode:
        msg = f"File does not exist: {path}"
        raise FileNotFoundError(msg)

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if "a" in mode:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_EXCL
    fd = os.open(str(path), flags, _FILE_MODE)
    return os.fdopen(fd, mode)
