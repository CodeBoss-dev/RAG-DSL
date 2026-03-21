"""Tests for the PromptScript Lark Transformer."""

import pytest

from promptscript.parser import parse
from promptscript.ast_nodes import (
    Program, Declaration, FuncCall, ListExpr, Identifier,
    StringLiteral, NumberLiteral, BoolLiteral, BinaryCondition,
)


def test_func_call_with_args():
    program = parse('context[] docs = retriever.fetch("topic", 5)')
    decl = program.statements[0]
    assert isinstance(decl.value, FuncCall)
    assert decl.value.func == "retriever.fetch"
    assert len(decl.value.args) == 2
    assert isinstance(decl.value.args[0], StringLiteral)
    assert decl.value.args[0].value == "topic"
    assert isinstance(decl.value.args[1], NumberLiteral)
    assert decl.value.args[1].value == 5


def test_list_expr():
    program = parse('context[] docs = ["a", "b", "c"]')
    decl = program.statements[0]
    assert isinstance(decl.value, ListExpr)
    assert len(decl.value.elements) == 3
    assert all(isinstance(e, StringLiteral) for e in decl.value.elements)


def test_string_escape():
    program = parse(r'str x = "hello \"world\""')
    decl = program.statements[0]
    assert decl.value.value == 'hello "world"'


def test_negative_number():
    program = parse("float x = -3.14")
    decl = program.statements[0]
    assert decl.value.value == pytest.approx(-3.14)


def test_binary_condition_operators():
    for op in ("==", "!=", "<", ">", "<=", ">="):
        src = f"""
int a = 1
int b = 2
if a {op} b {{
    str msg = "yes"
}}
"""
        program = parse(src)
        from promptscript.ast_nodes import IfBlock
        if_block = program.statements[2]
        assert isinstance(if_block, IfBlock)
        assert isinstance(if_block.condition, BinaryCondition)
        assert if_block.condition.op == op


def test_program_returns_program_node():
    program = parse('str x = "hello"')
    assert isinstance(program, Program)
