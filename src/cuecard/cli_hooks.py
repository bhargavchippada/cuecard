"""Hook install/uninstall, status, and log commands for cuecard CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

import cuecard.cli as _cli
from cuecard.cli import app, console, err_console

# --- hook helpers ---

_HOOK_COMMAND = "cuecard hook 2>/dev/null"
_HOOK_MARKER = "cuecard"


def _claude_settings_path() -> Path:
    return _cli._home_dir() / ".claude" / "settings.json"


def _load_claude_settings(path: Path) -> dict[str, object]:
    """Load Claude Code settings.json, returning empty dict if missing."""
    import json

    if not path.exists():
        return {}
    result: dict[str, object] = json.loads(path.read_text())
    return result


def _save_claude_settings(path: Path, data: dict[str, object]) -> None:
    """Write Claude Code settings.json atomically."""
    import json
    import os
    import tempfile

    from cuecard.security import ensure_directory

    ensure_directory(path.parent)
    with tempfile.NamedTemporaryFile(
        dir=str(path.parent), mode="w", suffix=".json", delete=False,
    ) as tmp:
        json.dump(data, tmp, indent=2)
        tmp.write("\n")
        tmp_path = tmp.name
    os.chmod(tmp_path, 0o600)
    os.replace(tmp_path, str(path))


def _entry_has_cuecard(entry: object) -> bool:
    """Check if a single hook entry contains a cuecard command."""
    if not isinstance(entry, dict):
        return False
    # Flat format: {"type": "command", "command": "cuecard hook"}
    if _HOOK_MARKER in str(entry.get("command", "")):
        return True
    # Nested format: {"matcher": "", "hooks": [{...command...}]}
    inner = entry.get("hooks", [])
    if isinstance(inner, list):
        return any(
            isinstance(h, dict) and _HOOK_MARKER in str(h.get("command", ""))
            for h in inner
        )
    return False


def _has_cuecard_hook(settings: dict[str, object]) -> bool:
    """Check if cuecard hook is already registered in any event."""
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, dict):
        return False
    for event in ("PreToolUse", "UserPromptSubmit"):
        event_hooks = hooks.get(event, [])
        if not isinstance(event_hooks, list):
            continue
        if any(_entry_has_cuecard(h) for h in event_hooks):
            return True
    return False


# --- install ---


@app.command()
def install(
    target: Annotated[
        str, typer.Argument(help="Target to install (claude-code)")
    ],
) -> None:
    """Install cuecard as a hook for an AI coding agent."""
    if target != "claude-code":
        err_console.print(
            f"[red]Unknown target: {target!r}.[/red]"
            " Supported: claude-code"
        )
        raise typer.Exit(1)

    settings_path = _claude_settings_path()
    settings = _load_claude_settings(settings_path)

    if _has_cuecard_hook(settings):
        console.print("[yellow]cuecard hook already installed.[/yellow]")
        return

    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        hooks = {}
        settings["hooks"] = hooks

    hook_entry = {
        "matcher": "",
        "hooks": [{"type": "command", "command": _HOOK_COMMAND}],
    }

    for event in ("PreToolUse", "UserPromptSubmit"):
        event_hooks = hooks.get(event)
        if not isinstance(event_hooks, list):
            event_hooks = []
            hooks[event] = event_hooks
        event_hooks.append(hook_entry)

    _save_claude_settings(settings_path, settings)

    console.print(
        "[green]Installed[/green] cuecard hooks"
        " (PreToolUse + UserPromptSubmit)"
        f" in {settings_path}"
    )


# --- uninstall ---


@app.command()
def uninstall(
    target: Annotated[
        str, typer.Argument(help="Target to uninstall (claude-code)")
    ],
) -> None:
    """Remove cuecard hook from an AI coding agent."""
    if target != "claude-code":
        err_console.print(
            f"[red]Unknown target: {target!r}.[/red]"
            " Supported: claude-code"
        )
        raise typer.Exit(1)

    settings_path = _claude_settings_path()
    settings = _load_claude_settings(settings_path)

    if not _has_cuecard_hook(settings):
        console.print("[yellow]cuecard hook not found.[/yellow]")
        return

    hooks = settings.get("hooks", {})
    if isinstance(hooks, dict):
        for event in ("PreToolUse", "UserPromptSubmit"):
            event_hooks = hooks.get(event, [])
            if isinstance(event_hooks, list):
                hooks[event] = [
                    h for h in event_hooks
                    if not _entry_has_cuecard(h)
                ]
                if not hooks[event]:
                    del hooks[event]
        if not hooks:
            del settings["hooks"]

    _save_claude_settings(settings_path, settings)
    console.print(
        "[green]Uninstalled[/green] cuecard hooks"
        f" from {settings_path}"
    )


# --- status ---


@app.command()
def status() -> None:
    """Show installed hooks, index health, and model info."""
    # Check hook installation
    settings_path = _claude_settings_path()
    settings = _load_claude_settings(settings_path)
    hook_installed = _has_cuecard_hook(settings)

    if hook_installed:
        console.print("[green]\u2713[/green] Claude Code hook installed")
    else:
        console.print("[yellow]\u2717[/yellow] Claude Code hook not installed")

    # Check index
    cfg = _cli._load_config_or_exit(
        project_dir=Path.cwd(), home_dir=_cli._home_dir(),
    )

    from cuecard.indexer import load_index

    found_any = False
    global_idx = load_index(cfg.global_cache_dir)
    if global_idx is not None:
        found_any = True
        idx = global_idx
        console.print(
            f"[green]\u2713[/green] Global index: {idx.size} rules, dim={idx.dim}",
        )
        console.print(f"  Model: {global_idx.model_name}")

    if cfg.project_cache_dir:
        project_idx = load_index(cfg.project_cache_dir)
        if project_idx is not None:
            found_any = True
            idx = project_idx
            console.print(
                f"[green]\u2713[/green] Project index: {idx.size} rules, dim={idx.dim}",
            )
            console.print(f"  Model: {project_idx.model_name}")

    if not found_any:
        console.print("[yellow]\u2717[/yellow] No valid index found")

    # Check daemon
    from cuecard.serve import daemon_status as _daemon_status

    running, pid = _daemon_status(_cli._home_dir())
    if running:
        console.print(f"[green]\u2713[/green] Daemon running (PID {pid})")
    else:
        console.print("[dim]-[/dim] Daemon not running")

    # Check log
    log_path = Path(cfg.global_cache_dir).parent / "log.jsonl"
    if log_path.exists():
        size_kb = log_path.stat().st_size / 1024
        console.print(f"[green]\u2713[/green] Log: {size_kb:.1f} KB")
    else:
        console.print("[dim]-[/dim] No log file yet")


# --- log ---


@app.command(name="log")
def log_cmd(
    stats: Annotated[
        bool, typer.Option("--stats", help="Show aggregate statistics")
    ] = False,
    limit: Annotated[
        int, typer.Option(help="Number of recent entries to show")
    ] = 20,
) -> None:
    """Show recent log entries or aggregate stats."""
    home = _cli._home_dir()
    log_dir = home / ".cuecard"

    from cuecard.logger import compute_stats, read_log

    entries = read_log(log_dir=log_dir, limit=limit)

    if not entries:
        console.print("[yellow]No log entries found.[/yellow]")
        return

    if stats:
        s = compute_stats(entries)
        table = Table(title="Log Statistics")
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Total events", str(s["total_events"]))
        table.add_row("Avg injected", str(s["avg_injected"]))
        table.add_row(
            "Coverage",
            f"{float(str(s['coverage'])) * 100:.1f}%",
        )
        table.add_row("Latency p50", f"{s['latency_p50']}ms")
        table.add_row("Latency p95", f"{s['latency_p95']}ms")
        table.add_row("Latency p99", f"{s['latency_p99']}ms")

        console.print(table)

        top_rules = s.get("top_rules")
        if isinstance(top_rules, list) and top_rules:
            console.print("\n[bold]Top Rules:[/bold]")
            for text, count in top_rules:
                console.print(f"  ({count}x) {text}")
        return

    # Show recent entries
    for entry in entries:
        ts = str(entry.get("timestamp", ""))[:19]
        event = entry.get("event", "")
        tool = entry.get("tool_name", "")
        count = entry.get("injected_count", 0)
        lat = entry.get("latency_ms", 0)
        console.print(
            f"[dim]{ts}[/dim] {event}:{tool}"
            f" \u2192 {count} rules ({lat}ms)"
        )
