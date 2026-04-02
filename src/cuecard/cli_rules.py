"""Rules subcommands for cuecard CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import cuecard.cli as _cli
from cuecard.cli import console, err_console, rules_app


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

    from cuecard.parser import parse_rules

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
    import os

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

    from cuecard.parser import parse_rules

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


@rules_app.command()
def sources() -> None:
    """Show configured rule file paths and status."""
    cfg = _cli._load_config_or_exit(
        project_dir=Path.cwd(), home_dir=_cli._home_dir(),
    )

    from cuecard.parser import parse_rules

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
