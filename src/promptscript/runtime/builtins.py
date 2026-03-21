"""Built-in functions available in PromptScript programs."""

from __future__ import annotations

from typing import Any

from promptscript.runtime.context import RuntimeContext, RetrievedChunk


# ---------------------------------------------------------------------------
# Built-in function registry
# ---------------------------------------------------------------------------

def call_builtin(func_name: str, args: list[Any], ctx: RuntimeContext) -> Any:
    """Dispatch a dotted function call to the appropriate built-in.

    Args:
        func_name: Dotted name, e.g. "retriever.fetch" or "len".
        args: Evaluated argument values.
        ctx: The current RuntimeContext.

    Returns:
        The result of the built-in function.

    Raises:
        NameError: if the function is not known.
    """
    if func_name == "retriever.fetch":
        query = str(args[0]) if args else ""
        top_k = int(args[1]) if len(args) > 1 else 5
        return ctx.fetch(query, top_k)

    if func_name == "retriever.search":
        # Alias for retriever.fetch
        query = str(args[0]) if args else ""
        top_k = int(args[1]) if len(args) > 1 else 5
        return ctx.fetch(query, top_k)

    if func_name == "len":
        if not args:
            raise TypeError("len() requires one argument")
        return len(args[0])

    if func_name == "str":
        return str(args[0]) if args else ""

    if func_name == "int":
        return int(args[0]) if args else 0

    if func_name == "float":
        return float(args[0]) if args else 0.0

    raise NameError(f"Unknown built-in function: '{func_name}'")
