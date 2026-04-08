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
            'top_k = 7\n'
            'threshold = 0.30\n'
            '# fusion_k = 60                   '
            '# RRF parameter\n'
            '# sparse_enabled = true            '
            '# Enable BM25 hybrid retrieval\n\n'
            '# [pipeline]\n'
            '# mode = "embedding"              '
            '# embedding | llm-local | llm-haiku\n'
            '#\n'
            '# [pipeline.llm]\n'
            '# local_endpoint = "http://localhost:8081/v1"\n'
            '# thinking = false\n\n'
            '# [expansion]\n'
            '# max_per_rule = 10\n'
            '# max_expansion_length = 200\n',
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
    from cuecard.indexer import (
        build_index,
        load_rules_json,
        merge_rules_json,
        save_index,
        save_rules_json,
    )
    from cuecard.parser import parse_rules

    model = TextEmbedding(model_name=cfg.model_name)
    total_rules = 0

    for label, cache_dir, source_paths in _iter_scoped_sources(cfg):
        rules = parse_rules(source_paths)
        if not rules:
            console.print(f"[yellow]{label}: no rules found.[/yellow]")
            continue

        # Merge with cached rules.json to preserve expansions
        cached_rules = load_rules_json(cache_dir)
        if cached_rules is not None:
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

    from cuecard.formatter import format_rules_verbose
    from cuecard.loader import load_or_build

    model = TextEmbedding(model_name=cfg.model_name)
    loaded = load_or_build(cfg, model)  # type: ignore[arg-type]
    if loaded is None:
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

    from cuecard.pipeline import run_pipeline

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

    from cuecard.formatter import format_rules
    from cuecard.loader import load_or_build
    from cuecard.pipeline import run_pipeline

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
        int, typer.Option(help="Port to listen on")
    ] = 8452,
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


# --- configure ---

_VALID_PIPELINE_MODES = ("embedding", "llm-local", "llm-haiku")


def _load_existing_global_config(home: Path) -> dict[str, object]:
    """Load existing global config.toml as flat dict for defaults."""
    import tomllib

    config_path = home / ".cuecard" / "config.toml"
    if not config_path.exists():
        return {}
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)
    flat: dict[str, object] = {}
    pipeline = raw.get("pipeline", {})
    if "mode" in pipeline:
        flat["mode"] = pipeline["mode"]
    llm = pipeline.get("llm", {})
    if "local_endpoint" in llm:
        flat["local_endpoint"] = llm["local_endpoint"]
    if "thinking" in llm:
        flat["thinking"] = llm["thinking"]
    retrieval = raw.get("retrieval", {})
    if "top_k" in retrieval:
        flat["top_k"] = retrieval["top_k"]
    if "threshold" in retrieval:
        flat["threshold"] = retrieval["threshold"]
    if "sparse_enabled" in retrieval:
        flat["sparse_enabled"] = retrieval["sparse_enabled"]
    embedding = raw.get("embedding", {})
    if "model" in embedding:
        flat["model"] = embedding["model"]
    sources = raw.get("sources", {})
    if "rules" in sources:
        flat["rules"] = sources["rules"]
    return flat


def _format_rules_toml(rules: list[str]) -> str:
    """Format a list of rule paths as a TOML inline array."""
    items = ", ".join(f'"{r}"' for r in rules)
    return f"[{items}]"


def _build_config_toml(
    *,
    mode: str,
    model: str,
    top_k: int,
    threshold: float,
    sparse_enabled: bool,
    local_endpoint: str,
    rules: list[str],
) -> str:
    """Build a TOML config string from configure choices."""
    lines = [
        "# cuecard configuration",
        "# Generated by: cuecard configure",
        "",
        "[sources]",
        f"rules = {_format_rules_toml(rules)}",
        "",
        "[embedding]",
        f'model = "{model}"',
        "",
        "[retrieval]",
        f"top_k = {top_k}",
        f"threshold = {threshold:.2f}",
        f"sparse_enabled = {'true' if sparse_enabled else 'false'}",
        "",
        "[pipeline]",
        f'mode = "{mode}"',
    ]
    if mode in ("llm-local", "llm-haiku"):
        lines.append("")
        lines.append("[pipeline.llm]")
        if mode == "llm-local":
            lines.append(f'local_endpoint = "{local_endpoint}"')
        lines.append("thinking = false")
    lines.append("")
    return "\n".join(lines)


@app.command()
def configure() -> None:
    """Interactively configure cuecard settings."""
    from cuecard.security import ensure_directory

    home = _home_dir()
    cuecard_dir = home / ".cuecard"

    # Load existing config for defaults
    existing = _load_existing_global_config(home)

    # Pipeline mode
    default_mode = str(existing.get("mode", "embedding"))
    mode = typer.prompt(
        "Pipeline mode (embedding / llm-local / llm-haiku)",
        default=default_mode,
    )
    if mode not in _VALID_PIPELINE_MODES:
        err_console.print(
            f"[red]Invalid mode {mode!r}. "
            f"Valid: {list(_VALID_PIPELINE_MODES)}[/red]",
        )
        raise typer.Exit(1)

    # LLM-specific prompts
    default_endpoint = str(
        existing.get("local_endpoint", "http://localhost:8081/v1"),
    )
    local_endpoint = default_endpoint
    if mode == "llm-local":
        local_endpoint = typer.prompt(
            "Local LLM endpoint URL",
            default=default_endpoint,
        )
    elif mode == "llm-haiku":
        import os

        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if has_key:
            console.print("Anthropic API key found in environment.")
        else:
            confirmed = typer.confirm(
                "No ANTHROPIC_API_KEY found. Do you have one set up?",
                default=False,
            )
            if not confirmed:
                console.print(
                    "Set ANTHROPIC_API_KEY in your environment before"
                    " using llm-haiku mode.",
                )
                raise typer.Exit(1)

    # Retrieval settings
    raw_top_k = existing.get("top_k", 5)
    default_top_k = int(str(raw_top_k))
    top_k = typer.prompt("top_k (max results)", default=default_top_k, type=int)

    raw_threshold = existing.get("threshold", 0.30)
    default_threshold = float(str(raw_threshold))
    threshold = typer.prompt(
        "threshold (min similarity)",
        default=default_threshold,
        type=float,
    )

    default_sparse = bool(existing.get("sparse_enabled", True))
    sparse_enabled = typer.confirm(
        "Enable BM25 sparse retrieval?",
        default=default_sparse,
    )

    # Preserve existing values
    default_model = str(existing.get("model", _DEFAULT_MODEL))
    raw_rules = existing.get("rules", ["rules/global.txt"])
    default_rules: list[str] = (
        list(raw_rules) if isinstance(raw_rules, list) else ["rules/global.txt"]
    )

    # Build and write config
    config_content = _build_config_toml(
        mode=mode,
        model=default_model,
        top_k=top_k,
        threshold=threshold,
        sparse_enabled=sparse_enabled,
        local_endpoint=local_endpoint,
        rules=default_rules,
    )

    ensure_directory(cuecard_dir)
    config_path = cuecard_dir / "config.toml"
    config_path.write_text(config_content)
    config_path.chmod(0o600)

    console.print(f"\n[green]Config written to:[/green] {config_path}\n")
    console.print(config_content)
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. cuecard index     — rebuild index")
    console.print("  2. cuecard rules expand — generate expansions")
    console.print("  3. cuecard install   — install Claude Code hook")


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
    from cuecard.parser import _parse_txt

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

import cuecard.cli_eval as _cli_eval  # noqa: E402, F401
import cuecard.cli_hooks as _cli_hooks  # noqa: E402, F401
import cuecard.cli_rules as _cli_rules  # noqa: E402, F401

# Re-export hook helpers so existing imports from cuecard.cli keep working.
_claude_settings_path = _cli_hooks._claude_settings_path
_load_claude_settings = _cli_hooks._load_claude_settings
_save_claude_settings = _cli_hooks._save_claude_settings
_has_cuecard_hook = _cli_hooks._has_cuecard_hook
