"""Dataclass-based AST node definitions for PromptScript."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Type specs
# ---------------------------------------------------------------------------

@dataclass
class TypeStr:
    pass

@dataclass
class TypeInt:
    pass

@dataclass
class TypeFloat:
    pass

@dataclass
class TypeBool:
    pass

@dataclass
class TypePersona:
    pass

@dataclass
class TypeInstruct:
    pass

@dataclass
class TypeContextList:
    pass

# Union alias used in type annotations
TypeSpec = TypeStr | TypeInt | TypeFloat | TypeBool | TypePersona | TypeInstruct | TypeContextList


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------

@dataclass
class StringLiteral:
    value: str

@dataclass
class LongStringLiteral:
    value: str

@dataclass
class NumberLiteral:
    value: float

@dataclass
class BoolLiteral:
    value: bool

@dataclass
class Identifier:
    name: str

@dataclass
class FuncCall:
    func: str          # dotted name, e.g. "retriever.fetch"
    args: list[Any] = field(default_factory=list)

@dataclass
class ListExpr:
    elements: list[Any] = field(default_factory=list)

# Union alias
Expr = (
    StringLiteral
    | LongStringLiteral
    | NumberLiteral
    | BoolLiteral
    | Identifier
    | FuncCall
    | ListExpr
)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

@dataclass
class BinaryCondition:
    left: Any
    op: str
    right: Any

@dataclass
class NotCondition:
    operand: Any

@dataclass
class ExprCondition:
    expr: Any

Condition = BinaryCondition | NotCondition | ExprCondition


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass
class Declaration:
    type_spec: TypeSpec
    name: str
    value: Any

@dataclass
class Assignment:
    name: str
    value: Any

@dataclass
class SetParam:
    name: str
    value: Any  # NumberLiteral or BoolLiteral

@dataclass
class ForLoop:
    var: str
    iterable: Any
    body: list[Any] = field(default_factory=list)

@dataclass
class IfBlock:
    condition: Any
    then_body: list[Any] = field(default_factory=list)
    else_body: list[Any] = field(default_factory=list)

@dataclass
class CompileCall:
    args: list[Any] = field(default_factory=list)

Statement = Declaration | Assignment | SetParam | ForLoop | IfBlock | CompileCall


# ---------------------------------------------------------------------------
# Top-level Program
# ---------------------------------------------------------------------------

@dataclass
class Program:
    statements: list[Any] = field(default_factory=list)
