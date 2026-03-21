"""Lark-based parser for PromptScript."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from lark import Lark
from lark.exceptions import UnexpectedInput

from promptscript.ast_nodes import Program
from promptscript.transformer import PromptScriptTransformer

_GRAMMAR_PATH = Path(__file__).parent / "grammar.lark"


@lru_cache(maxsize=1)
def _get_parser() -> Lark:
    grammar = _GRAMMAR_PATH.read_text()
    return Lark(grammar, parser="earley", ambiguity="resolve")


class ParseError(Exception):
    """Raised when the source cannot be parsed."""
    pass


def parse(source: str) -> Program:
    """Parse PromptScript source text and return a Program AST.

    Args:
        source: PromptScript source code as a string.

    Returns:
        A :class:`~promptscript.ast_nodes.Program` AST.

    Raises:
        ParseError: if the source contains syntax errors.
    """
    parser = _get_parser()
    transformer = PromptScriptTransformer()
    try:
        tree = parser.parse(source)
        return transformer.transform(tree)
    except UnexpectedInput as exc:
        raise ParseError(f"Syntax error: {exc}") from exc


def parse_file(path: str | os.PathLike) -> Program:
    """Parse a PromptScript file and return a Program AST."""
    source = Path(path).read_text(encoding="utf-8")
    return parse(source)
