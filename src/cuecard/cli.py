"""Typer CLI for cuecard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

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

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
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


# --- setup ---


@app.command()
def setup(
    model: Annotated[
        str, typer.Option(help="Embedding model name")
    ] = _DEFAULT_MODEL,
    check: Annotated[
        bool, typer.Option("--check", help="Verify setup only")
    ] = False,
    allow_custom_model: Annotated[
        bool, typer.Option("--allow-custom-model", help="Allow non-allowlisted model")
    ] = False,
) -> None:
    """Download model, build initial index, verify permissions."""
    from cuecard.config import _ALLOWED_MODELS
    from cuecard.security import ensure_directory

    home = _home_dir()
    cuecard_dir = home / ".cuecard"

    if check:
        _run_check(cuecard_dir)
        return

    # Validate model against allowlist
    if model not in _ALLOWED_MODELS:
        if not allow_custom_model:
            err_console.print(
                f"[red]Model {model!r} is not in the allowed model list.[/red]\n"
                "Use --allow-custom-model to override."
            )
            raise typer.Exit(1)
        err_console.print(
            f"[yellow]Warning: model {model!r} is not in the"
            " allowed model list.[/yellow]"
        )

    # Create directory structure
    ensure_directory(cuecard_dir)
    rules_dir = cuecard_dir / "rules"
    ensure_directory(rules_dir)
    index_dir = cuecard_dir / "index"
    ensure_directory(index_dir)

    # Create template config if not exists
    config_path = cuecard_dir / "config.toml"
    if not config_path.exists():
        _atomic_write_new_file(
            config_path,
            '# cuecard configuration\n'
            '# See: https://github.com/bhargavchippada/cuecard\n\n'
            '[sources]\n'
            'rules = ["rules/global.txt"]\n\n'
            '[embedding]\n'
            f'model = "{model}"\n\n'
            '[retrieval]\n'
            'top_k = 5\n'
            'threshold = 0.30\n',
        )
        console.print(f"Created config: {config_path}")

    # Create template rules file if not exists
    rules_path = rules_dir / "global.txt"
    if not rules_path.exists():
        _atomic_write_new_file(
            rules_path,
            "# Global rules — one per line\n"
            "# These apply to all projects\n\n"
            "# Never commit secrets (API keys, tokens, passwords) to git\n"
            "# Always validate user input at system boundaries\n",
        )
        console.print(f"Created rules: {rules_path}")
        console.print(
            "\n[bold]Add your rules to[/bold] "
            f"{rules_path}\n"
            "[bold]then run[/bold] [green]cuecard setup[/green] [bold]again.[/bold]"
        )
        return

    # Download model + build index
    console.print(f"Downloading model: {model} ...")
    from fastembed import TextEmbedding

    embedding_model = TextEmbedding(model_name=model)

    config = _load_config_or_exit(
        home_dir=home, allow_custom_model=allow_custom_model,
    )

    from cuecard.freshness import check_freshness
    from cuecard.indexer import build_index, save_index
    from cuecard.parser import parse_rules

    total_rules = 0
    for _label, cache_dir, source_paths in _iter_scoped_sources(config):
        rules = parse_rules(source_paths)
        if not rules:
            continue
        freshness = check_freshness(source_paths, {})
        idx = build_index(
            tuple(rules), freshness.updated_sources, config.model_name,
            model=embedding_model,  # type: ignore[arg-type]
        )
        save_index(idx, cache_dir)
        total_rules += idx.size

    if total_rules == 0:
        console.print("[yellow]No rules found in configured sources.[/yellow]")
        return

    console.print(
        f"[green]Setup complete![/green] "
        f"Indexed {total_rules} rules with {model}."
    )


def _run_check(cuecard_dir: Path) -> None:
    """Verify setup: config exists, model downloaded, index valid."""
    issues: list[str] = []

    config_path = cuecard_dir / "config.toml"
    if not config_path.exists():
        issues.append(f"Config not found: {config_path}")

    index_dir = cuecard_dir / "index"
    npz = index_dir / "embeddings.npz"
    meta = index_dir / "metadata.json"
    if not npz.exists() or not meta.exists():
        issues.append(f"Index not found in {index_dir}")

    if issues:
        for issue in issues:
            err_console.print(f"[red]\u2717[/red] {issue}")
        raise typer.Exit(1)

    console.print("[green]\u2713[/green] Config found")
    console.print("[green]\u2713[/green] Index found")
    console.print("[green]Setup OK.[/green]")


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
    table.add_row("dedup_threshold", str(cfg.dedup_threshold))
    table.add_row("query_max_length", str(cfg.query_max_length))
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

    from cuecard.parser import parse_rules

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
        from cuecard.indexer import load_index

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

    from cuecard.freshness import check_freshness
    from cuecard.indexer import build_index, save_index
    from cuecard.parser import parse_rules

    model = TextEmbedding(model_name=cfg.model_name)
    total_rules = 0

    for label, cache_dir, source_paths in _iter_scoped_sources(cfg):
        rules = parse_rules(source_paths)
        if not rules:
            console.print(f"[yellow]{label}: no rules found.[/yellow]")
            continue
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

    from cuecard.formatter import format_rules_verbose
    from cuecard.loader import load_or_build

    model = TextEmbedding(model_name=cfg.model_name)
    idx = load_or_build(cfg, model)  # type: ignore[arg-type]
    if idx is None:
        err_console.print(_SETUP_NOT_DONE)
        raise typer.Exit(1)

    # Determine effective mode: explicit flag > config > default
    effective_mode = mode if mode else cfg.pipeline.mode

    from cuecard.pipeline import VALID_MODES
    if effective_mode not in VALID_MODES:
        err_console.print(
            f"[red]Invalid mode {effective_mode!r}. "
            f"Valid: {sorted(VALID_MODES)}[/red]",
        )
        raise typer.Exit(1)

    if effective_mode != "embedding":
        from cuecard.pipeline import run_pipeline

        pipeline_result = run_pipeline(
            query, idx, cfg, embedding_model=model, mode=effective_mode,
        )
        results: Sequence[RankedResult] = pipeline_result.results
    else:
        from cuecard.retriever import retrieve as do_retrieve

        results = do_retrieve(
            idx,
            query,
            top_k=top_k if top_k > 0 else cfg.top_k,
            threshold=threshold if threshold > 0 else cfg.threshold,
            dedup_threshold=cfg.dedup_threshold,
            model=model,
            max_query_length=cfg.query_max_length,
        )

    console.print(format_rules_verbose(results))


# --- format ---


@app.command(name="format")
def format_cmd(
    query: Annotated[str, typer.Argument(help="Tool context query string")],
) -> None:
    """Show final injectable text for a query."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from fastembed import TextEmbedding

    from cuecard.formatter import format_rules
    from cuecard.loader import load_or_build
    from cuecard.retriever import retrieve as do_retrieve

    model = TextEmbedding(model_name=cfg.model_name)
    idx = load_or_build(cfg, model)  # type: ignore[arg-type]
    if idx is None:
        err_console.print(_SETUP_NOT_DONE)
        raise typer.Exit(1)

    results = do_retrieve(
        idx, query, top_k=cfg.top_k, threshold=cfg.threshold,
        dedup_threshold=cfg.dedup_threshold, model=model,
        max_query_length=cfg.query_max_length,
    )
    output = format_rules(results)
    if output:
        console.print(output)
    else:
        console.print("[yellow]No matching rules.[/yellow]")


# --- embed ---


@app.command()
def embed() -> None:
    """Show embedding stats (model, dim, count)."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from cuecard.indexer import load_index

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


# --- Register commands from submodules ---
# These imports MUST be at the bottom so that app/rules_app are defined first.
# Each submodule registers its commands on app or rules_app at import time.

import cuecard.cli_eval as _cli_eval  # noqa: E402, F401
import cuecard.cli_hooks as _cli_hooks  # noqa: E402, F401
import cuecard.cli_rules as _cli_rules  # noqa: E402, F401

# Re-export hook helpers so existing imports from cuecard.cli keep working.
_claude_settings_path = _cli_hooks._claude_settings_path
_load_claude_settings = _cli_hooks._load_claude_settings
_save_claude_settings = _cli_hooks._save_claude_settings
_has_cuecard_hook = _cli_hooks._has_cuecard_hook
