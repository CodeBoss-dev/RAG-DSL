"""Static type validation for PromptScript ASTs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from promptscript.ast_nodes import (
    Program, Declaration, Assignment, SetParam, ForLoop, IfBlock, CompileCall,
    TypeStr, TypeInt, TypeFloat, TypeBool, TypePersona, TypeInstruct, TypeContextList,
    StringLiteral, LongStringLiteral, NumberLiteral, BoolLiteral,
    Identifier, FuncCall, ListExpr,
    BinaryCondition, NotCondition, ExprCondition,
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

@dataclass
class TypeCheckError:
    message: str


class TypeCheckFailed(Exception):
    def __init__(self, errors: list[TypeCheckError]):
        self.errors = errors
        super().__init__("\n".join(e.message for e in errors))


# ---------------------------------------------------------------------------
# Type compatibility helpers
# ---------------------------------------------------------------------------

def _type_name(type_spec: Any) -> str:
    return {
        TypeStr: "str",
        TypeInt: "int",
        TypeFloat: "float",
        TypeBool: "bool",
        TypePersona: "persona",
        TypeInstruct: "instruct",
        TypeContextList: "context[]",
    }.get(type(type_spec), "unknown")


def _infer_expr_type(expr: Any, symbols: dict[str, Any]) -> str:
    """Infer the type name of an expression. Returns 'unknown' if indeterminate."""
    if isinstance(expr, (StringLiteral, LongStringLiteral)):
        return "str"
    if isinstance(expr, NumberLiteral):
        return "float" if isinstance(expr.value, float) else "int"
    if isinstance(expr, BoolLiteral):
        return "bool"
    if isinstance(expr, Identifier):
        if expr.name in symbols:
            return _type_name(symbols[expr.name])
        return "unknown"
    if isinstance(expr, FuncCall):
        # retriever.fetch always returns context[]
        if expr.func in ("retriever.fetch", "retriever.search"):
            return "context[]"
        return "unknown"
    if isinstance(expr, ListExpr):
        return "context[]"
    return "unknown"


def _expr_type_compatible(declared: Any, actual_type_name: str) -> bool:
    """Check whether `actual_type_name` is compatible with `declared` TypeSpec."""
    declared_name = _type_name(declared)
    if declared_name == actual_type_name:
        return True
    # int literal is compatible with float declaration
    if declared_name == "float" and actual_type_name == "int":
        return True
    # persona and instruct accept str-like values
    if declared_name in ("persona", "instruct") and actual_type_name in ("str", "unknown"):
        return True
    # context[] accepts unknown (e.g., variable that may be a list)
    if declared_name == "context[]" and actual_type_name in ("context[]", "unknown"):
        return True
    # allow unknown through (runtime will catch it)
    if actual_type_name == "unknown":
        return True
    return False


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

class TypeChecker:
    def __init__(self):
        self.symbols: dict[str, Any] = {}  # name -> TypeSpec
        self.errors: list[TypeCheckError] = []

    def check(self, program: Program) -> list[TypeCheckError]:
        for stmt in program.statements:
            self._check_statement(stmt)
        return self.errors

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _check_statement(self, stmt: Any) -> None:
        if isinstance(stmt, Declaration):
            self._check_declaration(stmt)
        elif isinstance(stmt, Assignment):
            self._check_assignment(stmt)
        elif isinstance(stmt, SetParam):
            self._check_set_param(stmt)
        elif isinstance(stmt, ForLoop):
            self._check_for_loop(stmt)
        elif isinstance(stmt, IfBlock):
            self._check_if_block(stmt)
        elif isinstance(stmt, CompileCall):
            self._check_compile_call(stmt)

    def _check_declaration(self, decl: Declaration) -> None:
        actual = _infer_expr_type(decl.value, self.symbols)
        if not _expr_type_compatible(decl.type_spec, actual):
            self.errors.append(TypeCheckError(
                f"Type mismatch: '{decl.name}' declared as {_type_name(decl.type_spec)}"
                f" but assigned {actual}"
            ))
        # context[] must come from a function call or list literal
        if isinstance(decl.type_spec, TypeContextList):
            if not isinstance(decl.value, (FuncCall, ListExpr, Identifier)):
                self.errors.append(TypeCheckError(
                    f"'{decl.name}' is context[] but value is not a function call or list"
                ))
        self.symbols[decl.name] = decl.type_spec

    def _check_assignment(self, asgn: Assignment) -> None:
        if asgn.name not in self.symbols:
            # Allow undeclared assignments (they become dynamically typed)
            self.symbols[asgn.name] = TypeStr()  # default assumption
            return
        declared = self.symbols[asgn.name]
        actual = _infer_expr_type(asgn.value, self.symbols)
        if not _expr_type_compatible(declared, actual):
            self.errors.append(TypeCheckError(
                f"Type mismatch: reassignment of '{asgn.name}' ({_type_name(declared)})"
                f" with incompatible type {actual}"
            ))

    def _check_set_param(self, sp: SetParam) -> None:
        if not isinstance(sp.value, (NumberLiteral, BoolLiteral)):
            self.errors.append(TypeCheckError(
                f"set_param '{sp.name}' must be a numeric or boolean literal"
            ))

    def _check_for_loop(self, loop: ForLoop) -> None:
        iterable_type = _infer_expr_type(loop.iterable, self.symbols)
        if iterable_type not in ("context[]", "unknown"):
            self.errors.append(TypeCheckError(
                f"for loop iterates over '{iterable_type}'; expected context[]"
            ))
        # Add loop variable as context type for body checking
        saved = self.symbols.get(loop.var)
        self.symbols[loop.var] = TypeStr()  # individual chunk is str-like
        for stmt in loop.body:
            self._check_statement(stmt)
        # Restore symbol table
        if saved is None:
            del self.symbols[loop.var]
        else:
            self.symbols[loop.var] = saved

    def _check_if_block(self, block: IfBlock) -> None:
        for stmt in block.then_body:
            self._check_statement(stmt)
        for stmt in block.else_body:
            self._check_statement(stmt)

    def _check_compile_call(self, call: CompileCall) -> None:
        for arg in call.args:
            if isinstance(arg, Identifier) and arg.name not in self.symbols:
                self.errors.append(TypeCheckError(
                    f"prompt.compile references undeclared variable '{arg.name}'"
                ))


def check(program: Program) -> list[TypeCheckError]:
    """Run type checking on a parsed Program. Returns list of errors (empty = OK)."""
    return TypeChecker().check(program)


def check_or_raise(program: Program) -> None:
    """Run type checking and raise TypeCheckFailed if there are errors."""
    errors = check(program)
    if errors:
        raise TypeCheckFailed(errors)
