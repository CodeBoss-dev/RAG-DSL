"""CLI for PromptScript: compile / check / tokens."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click

from promptscript.parser import parse_file, ParseError
from promptscript.type_checker import check, TypeCheckFailed
from promptscript.runtime.context import RuntimeContext
from promptscript.compiler import compile_program, compile_segments
from promptscript.token_budget import total_tokens


@click.group()
@click.version_option(package_name="promptscript")
def cli():
    """PromptScript — typed DSL for RAG prompt construction."""


# ---------------------------------------------------------------------------
# compile
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--target", "-t", type=click.Choice(["markdown", "json"]), default="markdown",
              show_default=True, help="Output format.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output file path. Defaults to stdout.")
@click.option("--context", "-c", type=click.Path(exists=True, dir_okay=False), default=None,
              help="JSON file with variable bindings (injected into runtime context).")
@click.option("--budget", "-b", type=int, default=None,
              help="Token budget. Context chunks are dropped if exceeded.")
def compile(input_file: str, target: str, output: str | None, context: str | None, budget: int | None):
    """Parse, type-check, and compile a .ps file to markdown or JSON."""
    try:
        program = parse_file(input_file)
    except ParseError as exc:
        click.echo(f"Parse error: {exc}", err=True)
        sys.exit(1)

    errors = check(program)
    if errors:
        for e in errors:
            click.echo(f"Type error: {e.message}", err=True)
        sys.exit(1)

    variables: dict[str, Any] = {}
    if context:
        try:
            variables = json.loads(Path(context).read_text())
        except json.JSONDecodeError as exc:
            click.echo(f"Invalid context JSON: {exc}", err=True)
            sys.exit(1)

    ctx = RuntimeContext(variables=variables, token_budget=budget)
    result = compile_program(program, ctx, target=target)

    if output:
        Path(output).write_text(result, encoding="utf-8")
        click.echo(f"Written to {output}")
    else:
        click.echo(result)


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
def check_cmd(input_file: str):
    """Parse and type-check a .ps file without compiling."""
    try:
        program = parse_file(input_file)
    except ParseError as exc:
        click.echo(f"Parse error: {exc}", err=True)
        sys.exit(1)

    errors = check(program)
    if errors:
        for e in errors:
            click.echo(f"Type error: {e.message}", err=True)
        sys.exit(1)
    else:
        click.echo("OK — no errors found.")


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--context", "-c", type=click.Path(exists=True, dir_okay=False), default=None,
              help="JSON file with variable bindings.")
@click.option("--verbose", "-v", is_flag=True, help="Show per-segment token counts.")
def tokens(input_file: str, context: str | None, verbose: bool):
    """Show token counts per segment for a .ps file."""
    try:
        program = parse_file(input_file)
    except ParseError as exc:
        click.echo(f"Parse error: {exc}", err=True)
        sys.exit(1)

    variables: dict[str, Any] = {}
    if context:
        try:
            variables = json.loads(Path(context).read_text())
        except json.JSONDecodeError as exc:
            click.echo(f"Invalid context JSON: {exc}", err=True)
            sys.exit(1)

    ctx = RuntimeContext(variables=variables)
    segments = compile_segments(program, ctx)

    if verbose:
        for i, seg in enumerate(segments):
            var = seg.metadata.get("var", seg.metadata.get("compile_arg", ""))
            label = f"[{seg.role}] {var}" if var else f"[{seg.role}]"
            click.echo(f"  {i+1:3d}. {label:<30} {seg.token_count:>6} tokens")

    click.echo(f"\nTotal: {total_tokens(segments)} tokens across {len(segments)} segment(s).")


# Register 'check' command under its alias
cli.add_command(check_cmd, name="check")


if __name__ == "__main__":
    cli()
