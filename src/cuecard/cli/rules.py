"""Rules subcommands for cuecard CLI."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

import cuecard.cli.main as _cli
from cuecard.cli.main import console, err_console, rules_app

if TYPE_CHECKING:
    from cuecard.models import ExpandProgress, Rule


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

    cfg = _cli._load_config_or_exit(
        project_dir=Path.cwd(), home_dir=_cli._home_dir(),
    )

    from cuecard.indexing.parser import parse_rules

    home = _cli._home_dir()
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
    console.print(
        f"\n{total} rules ({global_count} global, {project_count} project)",
    )


@rules_app.command()
def add(
    text: Annotated[str, typer.Argument(help="Rule text to add")],
    global_scope: Annotated[
        bool, typer.Option("--global", help="Add to global rules")
    ] = False,
) -> None:
    """Add a rule to the rules file."""
    from cuecard.security import ensure_directory

    if global_scope:
        rules_path = _cli._home_dir() / ".cuecard" / "rules" / "global.txt"
    else:
        rules_path = Path.cwd() / "rules.txt"

    ensure_directory(rules_path.parent)

    # Enforce max length
    from cuecard.models import MAX_RULE_LENGTH

    if len(text) > MAX_RULE_LENGTH:
        err_console.print(
            f"[red]Rule exceeds {MAX_RULE_LENGTH} character limit.[/red]",
        )
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
    cfg = _cli._load_config_or_exit(
        project_dir=Path.cwd(), home_dir=_cli._home_dir(),
    )

    from cuecard.indexing.parser import parse_rules

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

    # Validate provenance file is within allowed source directories
    file_path = Path(target.provenance.file).resolve()
    allowed = set(cfg.global_source_paths) | set(cfg.project_source_paths)
    if str(file_path) not in allowed:
        err_console.print(
            f"[red]File {file_path} is not in configured source paths. "
            f"Edit the file directly.[/red]"
        )
        raise typer.Exit(1)

    lines = file_path.read_text().splitlines(keepends=True)
    line_idx = target.provenance.line_start - 1

    if 0 <= line_idx < len(lines):
        removed_text = lines[line_idx].strip()
        del lines[line_idx]
        content = "".join(lines)
        with tempfile.NamedTemporaryFile(
            dir=str(file_path.parent), mode="w",
            suffix=".txt", delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, str(file_path))
        console.print(
            f"Removed from {file_path}:\n  [red]{removed_text}[/red]",
        )
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
    cfg = _cli._load_config_or_exit(
        project_dir=Path.cwd(), home_dir=_cli._home_dir(),
    )

    from cuecard.indexing.parser import parse_rules

    home = _cli._home_dir()
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
        console.print(
            f"  [cyan]{num:3d}[/cyan]  {rule.text}  [dim]({scope})[/dim]",
        )


def _truncate_rule(text: str, max_len: int = 60) -> str:
    """Truncate rule text for progress display."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _expand_with_progress(
    rules: list[Rule],
    *,
    backend: str,
    endpoint: str = "http://localhost:8081/v1",
    missing_only: bool = False,
    dedup_threshold: float = 0.80,
) -> list[Rule]:
    """Run expand_rules with a Rich progress display on stderr."""
    from cuecard.indexing.expander import expand_rules

    skipped_count = 0

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]{task.fields[status]}"),
        TimeElapsedColumn(),
        console=err_console,
        transient=False,
    ) as progress:
        task_id = progress.add_task(
            "Expanding rules",
            total=len(rules),
            status="",
        )

        def _on_progress(p: ExpandProgress) -> None:
            nonlocal skipped_count
            if p.skipped:
                skipped_count += 1
            label = _truncate_rule(p.rule_text)
            status = (
                f'{p.expansions_generated} expansions | "{label}"'
            )
            progress.update(task_id, completed=p.rule_index + 1, status=status)

        result = expand_rules(
            rules,
            backend=backend,
            endpoint=endpoint,
            missing_only=missing_only,
            dedup_threshold=dedup_threshold,
            on_progress=_on_progress,
        )

    if skipped_count > 0:
        err_console.print(
            f"[dim]Skipped {skipped_count} rules with existing expansions[/dim]",
        )

    return result


@rules_app.command()
def expand(
    backend: Annotated[
        str, typer.Option(help="LLM backend: local or haiku")
    ] = "local",
    endpoint: Annotated[
        str, typer.Option(help="Local LLM endpoint URL")
    ] = "http://localhost:8081/v1",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be generated")
    ] = False,
    missing_only: Annotated[
        bool,
        typer.Option("--missing-only", help="Only expand rules without expansions"),
    ] = False,
) -> None:
    """Generate LLM expansions for rules."""
    from cuecard.indexing.indexer import (
        load_rules_json,
        merge_rules_json,
        save_rules_json,
    )
    from cuecard.indexing.parser import parse_rules

    cfg = _cli._load_config_or_exit(
        project_dir=Path.cwd(), home_dir=_cli._home_dir(),
    )

    for label, cache_dir, source_paths in _cli._iter_scoped_sources(cfg):
        # Always parse fresh source files
        parsed = parse_rules(source_paths)
        if not parsed:
            console.print(f"[yellow]{label}: no rules found.[/yellow]")
            continue
        # Merge with cached rules.json to preserve existing expansions
        cached_result = load_rules_json(cache_dir)
        if cached_result is not None:
            cached, _cached_aff = cached_result
            rules = merge_rules_json(parsed, cached)
        else:
            rules = parsed
        save_rules_json(rules, cache_dir)

        if dry_run:
            skipped = sum(1 for r in rules if missing_only and r.expansions)
            to_expand = len(rules) - skipped
            console.print(
                f"[bold]{label}:[/bold] would expand {to_expand} rules "
                f"(skipping {skipped} with existing expansions)",
            )
            continue

        console.print(
            f"[bold]{label}:[/bold] expanding {len(rules)} rules "
            f"via {backend}...",
        )

        try:
            expanded = _expand_with_progress(
                rules,
                backend=backend,
                endpoint=endpoint,
                missing_only=missing_only,
                dedup_threshold=cfg.expansion_dedup_threshold,
            )
        except Exception as exc:
            err_console.print(f"[red]Expansion failed: {exc}[/red]")
            raise typer.Exit(1) from None

        save_rules_json(expanded, cache_dir)

        new_count = sum(
            len(r.expansions) for r in expanded
        )
        old_count = sum(len(r.expansions) for r in rules)
        console.print(
            f"[green]{label}:[/green] {new_count} total expansions "
            f"({new_count - old_count} new).",
        )

        # Auto-rebuild index with new expansions
        from fastembed import TextEmbedding

        from cuecard.indexing.freshness import check_freshness
        from cuecard.indexing.indexer import build_index, save_index

        console.print(f"[bold]{label}:[/bold] rebuilding index...")
        model = TextEmbedding(model_name=cfg.model_name)
        freshness = check_freshness(source_paths, {})
        idx = build_index(
            tuple(expanded), freshness.updated_sources, cfg.model_name,
            model=model,  # type: ignore[arg-type]
        )
        save_index(idx, cache_dir)
        console.print(
            f"[green]{label}:[/green] index rebuilt"
            f" ({idx.size} rules, dim={idx.dim}).",
        )


@rules_app.command()
def sources() -> None:
    """Show configured rule file paths and status."""
    cfg = _cli._load_config_or_exit(
        project_dir=Path.cwd(), home_dir=_cli._home_dir(),
    )

    from cuecard.indexing.parser import parse_rules

    for path_str in cfg.source_paths:
        path = Path(path_str)
        if path.exists():
            rules = parse_rules((path_str,))
            console.print(
                f"[green]\u2713[/green] {path_str} ({len(rules)} rules)",
            )
        else:
            console.print(f"[yellow]?[/yellow] {path_str} (not found)")

    if not cfg.source_paths:
        console.print("[yellow]No sources configured.[/yellow]")
