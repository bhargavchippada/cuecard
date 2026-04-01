"""Typer CLI for cuecard."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from cuecard.models import ResolvedConfig

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

    rules = parse_rules(config.source_paths)
    if not rules:
        console.print("[yellow]No rules found in configured sources.[/yellow]")
        return

    freshness = check_freshness(config.source_paths, {})
    index = build_index(
        tuple(rules), freshness.updated_sources, config.model_name,
        model=embedding_model,  # type: ignore[arg-type]
    )
    save_index(index, config.global_cache_dir)

    console.print(
        f"[green]Setup complete![/green] "
        f"Indexed {index.size} rules with {model}."
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

        idx = load_index(cfg.global_cache_dir)
        if idx is None:
            console.print("[yellow]No valid index found.[/yellow]")
            return
        console.print(f"Model: {idx.model_name}")
        console.print(f"Rules: {idx.size}")
        console.print(f"Dim: {idx.dim}")
        console.print(f"Sources: {len(idx.sources)}")
        for path, meta in idx.sources.items():
            short_hash = meta.content_hash[:20]
            console.print(
                f"  {path}: {meta.rule_count} rules,"
                f" hash={short_hash}..."
            )
        return

    # Full rebuild
    from fastembed import TextEmbedding

    from cuecard.freshness import check_freshness
    from cuecard.indexer import build_index, save_index
    from cuecard.parser import parse_rules

    rules = parse_rules(cfg.source_paths)
    if not rules:
        console.print("[yellow]No rules found.[/yellow]")
        return

    model = TextEmbedding(model_name=cfg.model_name)
    freshness = check_freshness(cfg.source_paths, {})
    idx = build_index(
        tuple(rules), freshness.updated_sources, cfg.model_name,
        model=model,  # type: ignore[arg-type]
    )
    save_index(idx, cfg.global_cache_dir)
    console.print(f"[green]Index rebuilt:[/green] {idx.size} rules, dim={idx.dim}")


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

    from cuecard.indexer import load_index

    idx = load_index(cfg.global_cache_dir)
    if idx is None:
        err_console.print(_SETUP_NOT_DONE)
        raise typer.Exit(1)

    from fastembed import TextEmbedding

    from cuecard.formatter import format_rules_verbose

    model = TextEmbedding(model_name=cfg.model_name)

    # Determine effective mode: explicit flag > config > default
    effective_mode = mode if mode else cfg.pipeline.mode

    if effective_mode != "embedding":
        from cuecard.pipeline import run_pipeline

        pipeline_result = run_pipeline(
            query, idx, cfg, embedding_model=model, mode=effective_mode,
        )
        results = pipeline_result.results
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

    from cuecard.indexer import load_index

    idx = load_index(cfg.global_cache_dir)
    if idx is None:
        err_console.print(_SETUP_NOT_DONE)
        raise typer.Exit(1)

    from fastembed import TextEmbedding

    from cuecard.formatter import format_rules
    from cuecard.retriever import retrieve as do_retrieve

    model = TextEmbedding(model_name=cfg.model_name)
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


# --- rules subcommands ---


@rules_app.callback(invoke_without_command=True)
def rules_list(
    ctx: typer.Context,
    global_only: Annotated[
        bool, typer.Option("--global", help="Show only global rules")
    ] = False,
    project_only: Annotated[
        bool, typer.Option("--project", help="Show only project rules")
    ] = False,
) -> None:
    """List all rules with numbers and source files."""
    if ctx.invoked_subcommand is not None:
        return

    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from cuecard.parser import parse_rules

    home = _home_dir()
    global_cache = str(home / ".cuecard")

    all_rules = parse_rules(cfg.source_paths)
    if not all_rules:
        console.print("[yellow]No rules found.[/yellow]")
        return

    num = 1
    current_file = ""
    global_count = 0
    project_count = 0

    for rule in all_rules:
        is_global = rule.provenance.file.startswith(global_cache)

        if global_only and not is_global:
            num += 1
            continue
        if project_only and is_global:
            num += 1
            continue

        if rule.provenance.file != current_file:
            current_file = rule.provenance.file
            label = "Global" if is_global else "Project"
            console.print(f"\n[bold]{label}[/bold] ({current_file}):")

        console.print(f"  [cyan]{num:3d}[/cyan]  {rule.text}")

        if is_global:
            global_count += 1
        else:
            project_count += 1
        num += 1

    total = global_count + project_count
    console.print(f"\n{total} rules ({global_count} global, {project_count} project)")


@rules_app.command()
def add(
    text: Annotated[str, typer.Argument(help="Rule text to add")],
    global_scope: Annotated[
        bool, typer.Option("--global", help="Add to global rules")
    ] = False,
) -> None:
    """Add a rule to the rules file."""
    import os

    from cuecard.security import ensure_directory

    if global_scope:
        rules_path = _home_dir() / ".cuecard" / "rules" / "global.txt"
    else:
        rules_path = Path.cwd() / "rules.txt"

    ensure_directory(rules_path.parent)

    # Enforce max length
    from cuecard.models import MAX_RULE_LENGTH

    if len(text) > MAX_RULE_LENGTH:
        err_console.print(f"[red]Rule exceeds {MAX_RULE_LENGTH} character limit.[/red]")
        raise typer.Exit(1)

    if rules_path.exists():
        fd = os.open(str(rules_path), os.O_WRONLY | os.O_APPEND, 0o600)
    else:
        fd = os.open(str(rules_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, f"{text}\n".encode())
    finally:
        os.close(fd)

    console.print(f"Added to {rules_path}:\n  [green]{text}[/green]")


@rules_app.command()
def remove(
    number: Annotated[
        int,
        typer.Argument(help="Rule number (from 'cuecard rules' output)"),
    ],
) -> None:
    """Remove a rule by its number."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from cuecard.parser import parse_rules

    all_rules = parse_rules(cfg.source_paths)
    if number < 1 or number > len(all_rules):
        err_console.print(
            f"[red]Invalid rule number {number}."
            f" Valid range: 1-{len(all_rules)}[/red]"
        )
        raise typer.Exit(1)

    target = all_rules[number - 1]

    # Only allow removal from .txt files
    if not target.provenance.file.endswith(".txt"):
        err_console.print(
            f"[red]Rule #{number} is from {target.provenance.file} "
            f"(line {target.provenance.line_start}). "
            f"Edit the file directly to modify it.[/red]"
        )
        raise typer.Exit(1)

    # Read file, remove the line, write back
    file_path = Path(target.provenance.file)
    lines = file_path.read_text().splitlines(keepends=True)
    line_idx = target.provenance.line_start - 1

    if 0 <= line_idx < len(lines):
        removed_text = lines[line_idx].strip()
        del lines[line_idx]
        content = "".join(lines)
        import tempfile
        with tempfile.NamedTemporaryFile(
            dir=str(file_path.parent), mode="w",
            suffix=".txt", delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        import os as _os
        _os.chmod(tmp_path, 0o600)
        _os.replace(tmp_path, str(file_path))
        console.print(f"Removed from {file_path}:\n  [red]{removed_text}[/red]")
    else:
        err_console.print(
            f"[red]Line {target.provenance.line_start}"
            f" not found in {file_path}[/red]"
        )
        raise typer.Exit(1)


@rules_app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search term")],
) -> None:
    """Search through all rules."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from cuecard.parser import parse_rules

    home = _home_dir()
    global_cache = str(home / ".cuecard")
    all_rules = parse_rules(cfg.source_paths)
    query_lower = query.lower()

    matches = []
    for i, rule in enumerate(all_rules, 1):
        if query_lower in rule.text.lower():
            is_global = rule.provenance.file.startswith(
                global_cache,
            )
            scope = "global" if is_global else "project"
            matches.append((i, rule, scope))

    if not matches:
        console.print(f"[yellow]No rules matching '{query}'[/yellow]")
        return

    for num, rule, scope in matches:
        console.print(f"  [cyan]{num:3d}[/cyan]  {rule.text}  [dim]({scope})[/dim]")


@rules_app.command()
def sources() -> None:
    """Show configured rule file paths and status."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from cuecard.parser import parse_rules

    for path_str in cfg.source_paths:
        path = Path(path_str)
        if path.exists():
            rules = parse_rules((path_str,))
            console.print(f"[green]\u2713[/green] {path_str} ({len(rules)} rules)")
        else:
            console.print(f"[yellow]?[/yellow] {path_str} (not found)")

    if not cfg.source_paths:
        console.print("[yellow]No sources configured.[/yellow]")


# --- embed ---


@app.command()
def embed() -> None:
    """Show embedding stats (model, dim, count)."""
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from cuecard.indexer import load_index

    idx = load_index(cfg.global_cache_dir)
    if idx is None:
        err_console.print(_SETUP_NOT_DONE)
        raise typer.Exit(1)

    npz_path = Path(cfg.global_cache_dir) / "embeddings.npz"
    size_mb = npz_path.stat().st_size / (1024 * 1024) if npz_path.exists() else 0

    console.print(f"Model: {idx.model_name}")
    console.print(f"Dimensions: {idx.dim}")
    console.print(f"Rules embedded: {idx.size}")
    console.print(f"Index size: {size_mb:.2f} MB")


# --- install / uninstall ---

_HOOK_COMMAND = "uv run python -m cuecard.adapters.claude_code"
_HOOK_MARKER = "cuecard.adapters.claude_code"


def _claude_settings_path() -> Path:
    return _home_dir() / ".claude" / "settings.json"


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


def _has_cuecard_hook(settings: dict[str, object]) -> bool:
    """Check if cuecard hook is already registered."""
    hooks = settings.get("hooks", {})
    if not isinstance(hooks, dict):
        return False
    pre_tool = hooks.get("PreToolUse", [])
    if not isinstance(pre_tool, list):
        return False
    return any(
        isinstance(h, dict) and _HOOK_MARKER in str(h.get("command", ""))
        for h in pre_tool
    )


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

    pre_tool = hooks.get("PreToolUse")
    if not isinstance(pre_tool, list):
        pre_tool = []
        hooks["PreToolUse"] = pre_tool

    pre_tool.append({"type": "command", "command": _HOOK_COMMAND})
    _save_claude_settings(settings_path, settings)

    console.print(
        "[green]Installed[/green] cuecard PreToolUse hook"
        f" in {settings_path}"
    )


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
        pre_tool = hooks.get("PreToolUse", [])
        if isinstance(pre_tool, list):
            hooks["PreToolUse"] = [
                h for h in pre_tool
                if not (
                    isinstance(h, dict)
                    and _HOOK_MARKER in str(h.get("command", ""))
                )
            ]
            # Clean up empty lists
            if not hooks["PreToolUse"]:
                del hooks["PreToolUse"]
            if not hooks:
                del settings["hooks"]

    _save_claude_settings(settings_path, settings)
    console.print(
        "[green]Uninstalled[/green] cuecard hook"
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
    cfg = _load_config_or_exit(project_dir=Path.cwd(), home_dir=_home_dir())

    from cuecard.indexer import load_index

    idx = load_index(cfg.global_cache_dir)
    if idx is not None:
        console.print(f"[green]\u2713[/green] Index: {idx.size} rules, dim={idx.dim}")
        console.print(f"  Model: {idx.model_name}")
    else:
        console.print("[yellow]\u2717[/yellow] No valid index found")

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
    home = _home_dir()
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
        table.add_row("Coverage", f"{float(str(s['coverage'])) * 100:.1f}%")
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


# --- eval ---


@app.command(name="eval")
def eval_cmd(
    fixture_file: Annotated[
        Path, typer.Argument(help="Path to fixture JSON file")
    ],
    model: Annotated[
        str, typer.Option(help="Embedding model name")
    ] = _DEFAULT_MODEL,
    corpus_dir: Annotated[
        str | None,
        typer.Option("--corpus-dir", help="Directory containing rule corpora"),
    ] = None,
    top_k: Annotated[
        int, typer.Option(help="Max results per query")
    ] = 5,
    threshold: Annotated[
        float, typer.Option(help="Min similarity threshold")
    ] = 0.30,
    dedup_threshold: Annotated[
        float, typer.Option("--dedup-threshold", help="Dedup similarity threshold")
    ] = 0.95,
    mode: Annotated[str, typer.Option(help="Pipeline mode")] = "",
) -> None:
    """Run evaluation against a fixture file."""
    from cuecard.eval import format_eval_report, load_fixtures, run_eval

    try:
        fixtures = load_fixtures(str(fixture_file))
    except Exception as exc:
        err_console.print(f"[red]Failed to load fixtures:[/red] {exc}")
        raise typer.Exit(1) from None

    resolved_corpus_dir = (
        corpus_dir if corpus_dir is not None
        else str(fixture_file.parent)
    )

    try:
        from fastembed import TextEmbedding

        embedding_model = TextEmbedding(model_name=model)
    except Exception as exc:
        err_console.print(f"[red]Failed to load model:[/red] {exc}")
        raise typer.Exit(1) from None

    try:
        summary = run_eval(
            fixtures,
            resolved_corpus_dir,
            model,
            model=embedding_model,
            top_k=top_k,
            threshold=threshold,
            dedup_threshold=dedup_threshold,
            mode=mode if mode else None,
        )
    except Exception as exc:
        err_console.print(f"[red]Eval failed:[/red] {exc}")
        raise typer.Exit(1) from None

    report = format_eval_report(summary)
    console.print(report)
