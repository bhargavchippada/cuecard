"""Eval command for cuecard CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

import cuecard.cli as _cli
from cuecard.cli import app, console, err_console


@app.command(name="eval")
def eval_cmd(
    fixture_file: Annotated[
        Path, typer.Argument(help="Path to fixture JSON file")
    ],
    model: Annotated[
        str, typer.Option(help="Embedding model name")
    ] = _cli._DEFAULT_MODEL,
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
    corpus_override: Annotated[
        str | None,
        typer.Option(
            "--corpus-override",
            help="Comma-separated corpus files for unified index",
        ),
    ] = None,
    sample_ratio: Annotated[
        float,
        typer.Option(
            "--sample-ratio",
            help="Fraction of fixtures to evaluate (0.0-1.0, stratified)",
        ),
    ] = 1.0,
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

    override_paths: tuple[str, ...] | None = None
    if corpus_override:
        override_paths = tuple(
            str(Path(resolved_corpus_dir) / p.strip())
            for p in corpus_override.split(",")
        )

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
            corpus_override=override_paths,
            sample_ratio=sample_ratio,
        )
    except Exception as exc:
        err_console.print(f"[red]Eval failed:[/red] {exc}")
        raise typer.Exit(1) from None

    report = format_eval_report(summary)
    console.print(report)
