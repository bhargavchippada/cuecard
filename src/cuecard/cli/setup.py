"""Setup and configure commands for cuecard CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from pathlib import Path

import typer

import cuecard.cli.main as _cli
from cuecard.cli.main import app, console, err_console

# --- setup ---


@app.command()
def setup(
    model: Annotated[
        str, typer.Option(help="Embedding model name")
    ] = _cli._DEFAULT_MODEL,
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

    home = _cli._home_dir()
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
        _cli._atomic_write_new_file(
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
        _cli._atomic_write_new_file(
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

    config = _cli._load_config_or_exit(
        home_dir=home, allow_custom_model=allow_custom_model,
    )

    from cuecard.indexing.freshness import check_freshness
    from cuecard.indexing.indexer import build_index, save_index
    from cuecard.indexing.parser import parse_rules

    total_rules = 0
    for _label, cache_dir, source_paths in _cli._iter_scoped_sources(config):
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

    home = _cli._home_dir()
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
    default_model = str(existing.get("model", _cli._DEFAULT_MODEL))
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
