"""Compiler: AST -> flat list of PromptSegments -> rendered output.

Two-phase design:
1. Evaluation  – walk the AST, resolve identifiers, expand loops/conditionals,
                 produce a flat list of PromptSegment objects.
2. Rendering   – dispatch segments to the selected target renderer.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from promptscript.ast_nodes import (
    Program, Declaration, Assignment, SetParam, ForLoop, IfBlock, CompileCall,
    TypeStr, TypeInt, TypeFloat, TypeBool, TypePersona, TypeInstruct, TypeContextList,
    StringLiteral, LongStringLiteral, NumberLiteral, BoolLiteral,
    Identifier, FuncCall, ListExpr,
    BinaryCondition, NotCondition, ExprCondition,
)
from promptscript.runtime.context import RuntimeContext, RetrievedChunk
from promptscript.runtime.builtins import call_builtin
from promptscript.token_budget import PromptSegment, enforce_budget, count_tokens

logger = logging.getLogger(__name__)

Target = Literal["markdown", "json"]


# ---------------------------------------------------------------------------
# Role inference from type spec
# ---------------------------------------------------------------------------

def _role_for_type(type_spec: Any) -> str:
    mapping = {
        TypePersona: "persona",
        TypeInstruct: "instruct",
        TypeContextList: "context",
        TypeStr: "user",
        TypeInt: "user",
        TypeFloat: "user",
        TypeBool: "user",
    }
    return mapping.get(type(type_spec), "user")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """Walks the AST and produces PromptSegments."""

    def __init__(self, ctx: RuntimeContext):
        self.ctx = ctx
        self.segments: list[PromptSegment] = []
        self.api_params: dict[str, Any] = {}
        # Maps variable name -> declared TypeSpec (for role inference)
        self._type_map: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, program: Program) -> tuple[list[PromptSegment], dict[str, Any]]:
        for stmt in program.statements:
            self._exec(stmt)
        return self.segments, self.api_params

    # ------------------------------------------------------------------
    # Statement execution
    # ------------------------------------------------------------------

    def _exec(self, stmt: Any) -> None:
        if isinstance(stmt, Declaration):
            self._exec_declaration(stmt)
        elif isinstance(stmt, Assignment):
            self._exec_assignment(stmt)
        elif isinstance(stmt, SetParam):
            self._exec_set_param(stmt)
        elif isinstance(stmt, ForLoop):
            self._exec_for_loop(stmt)
        elif isinstance(stmt, IfBlock):
            self._exec_if_block(stmt)
        elif isinstance(stmt, CompileCall):
            self._exec_compile_call(stmt)

    def _exec_declaration(self, decl: Declaration) -> None:
        value = self._eval(decl.value)
        self.ctx.set(decl.name, value)
        self._type_map[decl.name] = decl.type_spec

        role = _role_for_type(decl.type_spec)

        if isinstance(decl.type_spec, TypeContextList):
            # Each retrieved chunk becomes its own context segment
            chunks = value if isinstance(value, list) else [value]
            for i, chunk in enumerate(chunks):
                text = chunk.text if isinstance(chunk, RetrievedChunk) else str(chunk)
                confidence = chunk.score if isinstance(chunk, RetrievedChunk) else 1.0
                doc_id = chunk.doc_id if isinstance(chunk, RetrievedChunk) else str(i)
                self.segments.append(PromptSegment(
                    role="context",
                    content=text,
                    confidence=confidence,
                    metadata={"var": decl.name, "doc_id": doc_id, "index": i},
                ))
        else:
            self.segments.append(PromptSegment(
                role=role,
                content=str(value),
                metadata={"var": decl.name},
            ))

    def _exec_assignment(self, asgn: Assignment) -> None:
        value = self._eval(asgn.value)
        self.ctx.set(asgn.name, value)

    def _exec_set_param(self, sp: SetParam) -> None:
        self.api_params[sp.name] = self._eval(sp.value)

    def _exec_for_loop(self, loop: ForLoop) -> None:
        iterable = self._eval(loop.iterable)
        if not hasattr(iterable, "__iter__"):
            logger.warning("for loop over non-iterable '%s'; skipping", loop.var)
            return
        for item in iterable:
            child_ctx = self.ctx.child()
            child_ctx.set(loop.var, item)
            child_eval = Evaluator(child_ctx)
            child_eval._type_map = dict(self._type_map)
            for stmt in loop.body:
                child_eval._exec(stmt)
            self.segments.extend(child_eval.segments)
            self.api_params.update(child_eval.api_params)

    def _exec_if_block(self, block: IfBlock) -> None:
        cond_result = self._eval_condition(block.condition)
        branch = block.then_body if cond_result else block.else_body
        for stmt in branch:
            self._exec(stmt)

    def _exec_compile_call(self, call: CompileCall) -> None:
        # prompt.compile() with explicit args: emit each arg as a segment
        for arg in call.args:
            value = self._eval(arg)
            name = arg.name if isinstance(arg, Identifier) else None
            ts = self._type_map.get(name) if name else None
            role = _role_for_type(ts) if ts else "user"

            if isinstance(value, list):
                for i, item in enumerate(value):
                    text = item.text if isinstance(item, RetrievedChunk) else str(item)
                    conf = item.score if isinstance(item, RetrievedChunk) else 1.0
                    self.segments.append(PromptSegment(
                        role="context", content=text, confidence=conf,
                        metadata={"compile_arg": name, "index": i},
                    ))
            else:
                self.segments.append(PromptSegment(
                    role=role,
                    content=str(value),
                    metadata={"compile_arg": name},
                ))

    # ------------------------------------------------------------------
    # Expression evaluation
    # ------------------------------------------------------------------

    def _eval(self, expr: Any) -> Any:
        if isinstance(expr, StringLiteral):
            return expr.value
        if isinstance(expr, LongStringLiteral):
            return expr.value
        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, BoolLiteral):
            return expr.value
        if isinstance(expr, Identifier):
            return self.ctx.get(expr.name)
        if isinstance(expr, FuncCall):
            args = [self._eval(a) for a in expr.args]
            return call_builtin(expr.func, args, self.ctx)
        if isinstance(expr, ListExpr):
            return [self._eval(e) for e in expr.elements]
        # Fallback: return as-is (shouldn't happen with well-formed AST)
        return expr

    def _eval_condition(self, cond: Any) -> bool:
        if isinstance(cond, BinaryCondition):
            left = self._eval(cond.left)
            right = self._eval(cond.right)
            return {
                "==": lambda a, b: a == b,
                "!=": lambda a, b: a != b,
                "<":  lambda a, b: a < b,
                ">":  lambda a, b: a > b,
                "<=": lambda a, b: a <= b,
                ">=": lambda a, b: a >= b,
            }[cond.op](left, right)
        if isinstance(cond, NotCondition):
            return not bool(self._eval(cond.operand))
        if isinstance(cond, ExprCondition):
            return bool(self._eval(cond.expr))
        return False


# ---------------------------------------------------------------------------
# Public compiler API
# ---------------------------------------------------------------------------

def compile_program(
    program: Program,
    ctx: RuntimeContext,
    target: Target = "markdown",
) -> str:
    """Compile a parsed Program to a rendered prompt string.

    Args:
        program: Parsed PromptScript AST.
        ctx: Runtime context with variable bindings and retriever.
        target: Output format — "markdown" or "json".

    Returns:
        Rendered prompt as a string.
    """
    evaluator = Evaluator(ctx)
    segments, api_params = evaluator.run(program)

    if ctx.token_budget is not None:
        segments = enforce_budget(segments, ctx.token_budget)

    if target == "markdown":
        from promptscript.targets.markdown import render
        return render(segments)
    elif target == "json":
        from promptscript.targets.json_api import render_json
        return render_json(segments, api_params)
    else:
        raise ValueError(f"Unknown target: '{target}'. Choose 'markdown' or 'json'.")


def compile_to_dict(
    program: Program,
    ctx: RuntimeContext,
) -> dict[str, Any]:
    """Compile to an OpenAI-compatible dict (not serialized to string)."""
    evaluator = Evaluator(ctx)
    segments, api_params = evaluator.run(program)
    if ctx.token_budget is not None:
        segments = enforce_budget(segments, ctx.token_budget)
    from promptscript.targets.json_api import render
    return render(segments, api_params)


def compile_segments(
    program: Program,
    ctx: RuntimeContext,
) -> list[PromptSegment]:
    """Return the raw PromptSegments (after budget enforcement) without rendering."""
    evaluator = Evaluator(ctx)
    segments, _ = evaluator.run(program)
    if ctx.token_budget is not None:
        segments = enforce_budget(segments, ctx.token_budget)
    return segments
