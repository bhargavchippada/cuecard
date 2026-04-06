"""Fast entry point for cuecard CLI.

Routes ``cuecard hook`` directly to the adapter (skipping Typer)
so that hook calls avoid the CLI framework import overhead.
All other subcommands go through the full Typer app.
"""

from __future__ import annotations

import sys


def main() -> None:
    """Dispatch to hook fast-path or Typer CLI."""
    if len(sys.argv) >= 2 and sys.argv[1] == "hook":
        # Fast path: skip Typer, call adapter directly
        from cuecard.adapters.claude_code import main as hook_main

        hook_main()
    else:
        from cuecard.cli import app

        app()
