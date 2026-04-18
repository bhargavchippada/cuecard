"""Typer CLI for cuecard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from cuecard.models import DEFAULT_EMBEDDING_MODEL

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cuecard.models import RankedResult, ResolvedConfig

app = typer.Typer(
    name="cuecard",
    help="The right rule, at the right moment.",
    no_args_is_help=True,
)

rules_app = typer.Typer(help="Manage rules.", no_args_is_help=False)
app.add_typer(rules_app, name="rules")

console = Console()
err_console = Console(stderr=True)

# --- Defaults ---

_DEFAULT_MODEL = DEFAULT_EMBEDDING_MODEL
_SETUP_NOT_DONE = "[cuecard] Not set up. Run 'cuecard setup'."


def _iter_cache_dirs(
    cfg: ResolvedConfig,
) -> list[tuple[str, str]]:
    """Return (label, cache_dir) pairs for all active scopes."""
    dirs: list[tuple[str, str]] = [("Global", cfg.global_cache_dir)]
    if cfg.project_cache_dir and cfg.project_source_paths:
        dirs.append(("Project", cfg.project_cache_dir))
    return dirs


def _iter_scoped_sources(
    cfg: ResolvedConfig,
) -> list[tuple[str, str, tuple[str, ...]]]:
    """Return (label, cache_dir, source_paths) for each scope."""
    scopes: list[tuple[str, str, tuple[str, ...]]] = [
        ("Global", cfg.global_cache_dir, cfg.global_source_paths),
    ]
    if cfg.project_cache_dir and cfg.project_source_paths:
        scopes.append((
            "Project", cfg.project_cache_dir, cfg.project_source_paths,
        ))
    return scopes


def _atomic_write_new_file(path: Path, content: str) -> None:
    """Create a new file atomically with 0o600 permissions (TOCTOU-safe)."""
    import os

    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, content.encode())
    finally:
        os.close(fd)


def _home_dir() -> Path:
    return Path.home()


def _load_config_or_exit(
    project_dir: Path | None = None,
    home_dir: Path | None = None,
    *,
    allow_custom_model: bool = False,
) -> ResolvedConfig:
    from cuecard.config import load_config
    from cuecard.security import ConfigError

    try:
        return load_config(
            project_dir=project_dir,
            home_dir=home_dir,
            allow_custom_model=allow_custom_model,
        )
    except ConfigError as e:
        err_console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(1) from None


# --- config ---


@app.command()
def config() -> None:
    """Show resolved config (global + project merged)."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    table = Table(title="Resolved Config")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("model", cfg.model_name)
    table.add_row("top_k", str(cfg.top_k))
    table.add_row("threshold", str(cfg.threshold))
    table.add_row("fusion_k", str(cfg.fusion_k))
    table.add_row("sparse_enabled", str(cfg.sparse_enabled))
    table.add_row("dedup_threshold", str(cfg.dedup_threshold))
    table.add_row("query_max_length", str(cfg.query_max_length))
    table.add_row("pipeline_mode", cfg.pipeline.mode)
    table.add_row("local_endpoint", cfg.pipeline.local_endpoint)
    table.add_row("thinking", str(cfg.pipeline.thinking))
    table.add_row("hook_events", ", ".join(cfg.hook_events))
    table.add_row("verbose", str(cfg.verbose))
    table.add_row("redact", str(cfg.redact))
    table.add_row("sources", "\n".join(cfg.source_paths) or "(none)")
    table.add_row("global_cache", cfg.global_cache_dir)
    table.add_row("project_cache", cfg.project_cache_dir or "(none)")

    console.print(table)


# --- parse ---


@app.command()
def parse() -> None:
    """Show all parsed rules with provenance."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from cuecard.indexing.parser import parse_rules

    rules = parse_rules(cfg.source_paths)
    if not rules:
        console.print("[yellow]No rules found.[/yellow]")
        return

    for i, rule in enumerate(rules, 1):
        prov = rule.provenance
        console.print(
            f"[bold]{i:3d}[/bold]  {rule.text}\n"
            f"     [dim]{prov.file}:{prov.line_start}[/dim]"
        )


# --- index ---


@app.command()
def index(
    status: Annotated[
        bool, typer.Option("--status", help="Show index status only")
    ] = False,
) -> None:
    """Rebuild index from scratch, or show status."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    if status:
        from cuecard.indexing.indexer import load_index

        for label, cache_dir in _iter_cache_dirs(cfg):
            idx = load_index(cache_dir)
            if idx is None:
                console.print(f"[yellow]{label}: no valid index found.[/yellow]")
                continue
            console.print(f"\n[bold]{label}[/bold]")
            console.print(f"  Model: {idx.model_name}")
            console.print(f"  Rules: {idx.size}")
            console.print(f"  Dim: {idx.dim}")
            console.print(f"  Sources: {len(idx.sources)}")
            for path, meta in idx.sources.items():
                short_hash = meta.content_hash[:20]
                console.print(
                    f"    {path}: {meta.rule_count} rules,"
                    f" hash={short_hash}..."
                )
        return

    # Full rebuild per scope
    from fastembed import TextEmbedding

    from cuecard.indexing.freshness import check_freshness
    from cuecard.indexing.indexer import (
        build_index,
        load_rules_json,
        merge_rules_json,
        save_index,
        save_rules_json,
    )
    from cuecard.indexing.parser import parse_rules

    model = TextEmbedding(model_name=cfg.model_name)
    total_rules = 0

    for label, cache_dir, source_paths in _iter_scoped_sources(cfg):
        rules = parse_rules(source_paths)
        if not rules:
            console.print(f"[yellow]{label}: no rules found.[/yellow]")
            continue

        # Merge with cached rules.json to preserve expansions
        cached_result = load_rules_json(cache_dir)
        if cached_result is not None:
            cached_rules, _cached_aff = cached_result
            rules = merge_rules_json(rules, cached_rules)
        save_rules_json(rules, cache_dir)

        freshness = check_freshness(source_paths, {})
        idx = build_index(
            tuple(rules), freshness.updated_sources, cfg.model_name,
            model=model,  # type: ignore[arg-type]
        )
        save_index(idx, cache_dir)
        console.print(
            f"[green]{label} index rebuilt:[/green]"
            f" {idx.size} rules, dim={idx.dim}"
        )
        total_rules += idx.size

    if total_rules == 0:
        console.print("[yellow]No rules found.[/yellow]")
    else:
        console.print(f"\n[green]Total:[/green] {total_rules} rules indexed")


# --- retrieve ---


@app.command()
def retrieve(
    query: Annotated[str, typer.Argument(help="Tool context query string")],
    top_k: Annotated[int, typer.Option(help="Max results")] = 0,
    threshold: Annotated[float, typer.Option(help="Min similarity")] = 0.0,
    mode: Annotated[str, typer.Option(help="Pipeline mode")] = "",
) -> None:
    """Retrieve top-k rules matching a query."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from fastembed import TextEmbedding

    from cuecard.indexing.loader import load_or_build
    from cuecard.retrieval.formatter import format_rules_verbose

    model = TextEmbedding(model_name=cfg.model_name)
    loaded = load_or_build(cfg, model)  # type: ignore[arg-type]
    if loaded is None:
        err_console.print(_SETUP_NOT_DONE)
        raise typer.Exit(1)

    # Determine effective mode: explicit flag > config > default
    effective_mode = mode if mode else cfg.pipeline.mode

    from cuecard.retrieval.pipeline import VALID_MODES
    if effective_mode not in VALID_MODES:
        err_console.print(
            f"[red]Invalid mode {effective_mode!r}. "
            f"Valid: {sorted(VALID_MODES)}[/red]",
        )
        raise typer.Exit(1)

    from cuecard.retrieval.pipeline import run_pipeline

    pipeline_result = run_pipeline(
        query, loaded.index, cfg,
        embedding_model=model, mode=effective_mode,
        affinity=loaded.affinity,
    )
    results: Sequence[RankedResult] = pipeline_result.results

    console.print(format_rules_verbose(results))


# --- format ---


@app.command(name="format")
def format_cmd(
    query: Annotated[str, typer.Argument(help="Tool context query string")],
) -> None:
    """Show final injectable text for a query."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from fastembed import TextEmbedding

    from cuecard.indexing.loader import load_or_build
    from cuecard.retrieval.formatter import format_rules
    from cuecard.retrieval.pipeline import run_pipeline

    model = TextEmbedding(model_name=cfg.model_name)
    loaded = load_or_build(cfg, model)  # type: ignore[arg-type]
    if loaded is None:
        err_console.print(_SETUP_NOT_DONE)
        raise typer.Exit(1)

    pipeline_result = run_pipeline(
        query, loaded.index, cfg,
        embedding_model=model, mode=cfg.pipeline.mode,
        affinity=loaded.affinity,
    )
    output = format_rules(pipeline_result.results)
    if output:
        console.print(output)
    else:
        console.print("[yellow]No matching rules.[/yellow]")


# --- embed ---


@app.command()
def embed() -> None:
    """Show embedding stats (model, dim, count)."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from cuecard.indexing.indexer import load_index

    found = False
    for label, cache_dir in _iter_cache_dirs(cfg):
        idx = load_index(cache_dir)
        if idx is None:
            continue
        found = True
        npz_path = Path(cache_dir) / "embeddings.npz"
        size_mb = npz_path.stat().st_size / (1024 * 1024) if npz_path.exists() else 0
        console.print(f"[bold]{label}[/bold]")
        console.print(f"  Model: {idx.model_name}")
        console.print(f"  Dimensions: {idx.dim}")
        console.print(f"  Rules embedded: {idx.size}")
        console.print(f"  Index size: {size_mb:.2f} MB")

    if not found:
        err_console.print(_SETUP_NOT_DONE)
        raise typer.Exit(1)


# --- hook ---


def _hook_main() -> None:
    """Wrapper so tests can patch this entry point."""
    from cuecard.adapters.claude_code import main

    main()


@app.command(hidden=True)
def hook() -> None:
    """Run the hook adapter (reads stdin, writes stdout). Used by Claude Code hooks."""
    _hook_main()


# --- serve ---


@app.command()
def serve(
    port: Annotated[
        int | None,
        typer.Option(help="Port to listen on (default: from config)"),
    ] = None,
    stop: Annotated[
        bool, typer.Option("--stop", help="Stop running daemon")
    ] = False,
    daemon: Annotated[
        bool, typer.Option("--daemon", help="Fork to background")
    ] = False,
) -> None:
    """Start a persistent daemon to serve rule retrieval requests."""
    from cuecard.serve import (
        daemon_status,
        run_server,
        stop_server,
    )

    home = _home_dir()

    if stop:
        stopped = stop_server(home)
        if stopped:
            console.print("[green]Daemon stopped.[/green]")
        else:
            console.print("[yellow]No daemon running.[/yellow]")
        return

    running, existing_pid = daemon_status(home)
    if running:
        err_console.print(
            f"[red]Daemon already running (PID {existing_pid}).[/red]"
            " Use 'cuecard serve --stop' to stop it.",
        )
        raise typer.Exit(1)

    if port is None:
        cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=home)
        port = cfg.serve_port

    if daemon:
        _fork_daemon(port, home)
        return

    console.print(f"Starting cuecard daemon on 127.0.0.1:{port} ...")
    run_server(port=port, home=home)


def _fork_daemon(port: int, home: Path) -> None:
    """Fork to background and run the server in the child process."""
    import os

    pid = os.fork()
    if pid > 0:
        # Parent — report and exit
        console.print(
            f"[green]Daemon started[/green] (PID {pid})"
            f" on 127.0.0.1:{port}"
        )
        return

    # Child — detach and run
    os.setsid()

    from cuecard.serve import _log_path, run_server

    log_file = _log_path(home)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)

    # Close stdin
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)

    import contextlib

    with contextlib.suppress(SystemExit):
        run_server(port=port, home=home)


# --- migrate ---


@app.command()
def migrate(
    input_file: Annotated[
        str, typer.Argument(help="Input .txt rule file")
    ],
    output: Annotated[
        str, typer.Option("--output", "-o", help="Output .toml file path")
    ] = "",
) -> None:
    """Convert a .txt rule file to .toml format.

    Comments are NOT preserved (tomllib has no comment API).
    The original .txt file is NOT deleted — switch when ready.
    """
    from cuecard.indexing.parser import _parse_txt

    input_path = Path(input_file)
    if not input_path.exists():
        err_console.print(f"[red]File not found:[/red] {input_file}")
        raise typer.Exit(1)

    if input_path.suffix.lower() != ".txt":
        err_console.print(
            f"[red]Expected a .txt file, got {input_path.suffix!r}[/red]",
        )
        raise typer.Exit(1)

    output_path = Path(output) if output else input_path.with_suffix(".toml")
    if output_path.exists():
        err_console.print(
            f"[red]Output file already exists:[/red] {output_path}\n"
            "Remove it first or choose a different --output path.",
        )
        raise typer.Exit(1)

    rules = _parse_txt(str(input_path))
    if not rules:
        err_console.print("[yellow]No rules found in input file.[/yellow]")
        raise typer.Exit(1)

    lines: list[str] = [
        "# Converted from: " + input_path.name,
        "# Events and tools are empty — use 'cuecard index' to infer them.",
        "",
    ]
    for rule in rules:
        lines.append("[[rules]]")
        # Escape backslashes and quotes for TOML string
        escaped = rule.text.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'text = "{escaped}"')
        lines.append("events = []")
        lines.append("tools = []")
        lines.append("")

    output_path.write_text("\n".join(lines))
    console.print(
        f"[green]Migrated {len(rules)} rules[/green] → {output_path}\n"
        f"[dim]Original .txt file preserved: {input_path}[/dim]",
    )


# --- Register commands from submodules ---
# These imports MUST be at the bottom so that app/rules_app are defined first.
# Each submodule registers its commands on app or rules_app at import time.

import cuecard.cli.eval_cmd as _cli_eval  # noqa: E402, F401
import cuecard.cli.hooks as _cli_hooks  # noqa: E402, F401
import cuecard.cli.rules as _cli_rules  # noqa: E402, F401
import cuecard.cli.setup as _cli_setup  # noqa: E402, F401
