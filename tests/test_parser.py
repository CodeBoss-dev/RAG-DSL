"""Tests for the PromptScript parser."""

import pytest

from promptscript.parser import parse, ParseError
from promptscript.ast_nodes import (
    Program, Declaration, SetParam, CompileCall,
    TypeStr, TypePersona, TypeInstruct, TypeContextList, TypeBool, TypeInt, TypeFloat,
    StringLiteral, LongStringLiteral, NumberLiteral, BoolLiteral,
    Identifier, FuncCall,
)


class TestBasicDeclarations:
    def test_str_declaration(self):
        program = parse('str x = "hello"')
        assert len(program.statements) == 1
        decl = program.statements[0]
        assert isinstance(decl, Declaration)
        assert isinstance(decl.type_spec, TypeStr)
        assert decl.name == "x"
        assert isinstance(decl.value, StringLiteral)
        assert decl.value.value == "hello"

    def test_int_declaration(self):
        program = parse("int n = 42")
        decl = program.statements[0]
        assert isinstance(decl.type_spec, TypeInt)
        assert isinstance(decl.value, NumberLiteral)
        assert decl.value.value == 42

    def test_float_declaration(self):
        program = parse("float temp = 0.7")
        decl = program.statements[0]
        assert isinstance(decl.type_spec, TypeFloat)
        assert decl.value.value == pytest.approx(0.7)

    def test_bool_declaration(self):
        program = parse("bool flag = true")
        decl = program.statements[0]
        assert isinstance(decl.type_spec, TypeBool)
        assert isinstance(decl.value, BoolLiteral)
        assert decl.value.value is True

    def test_persona_declaration(self):
        program = parse('persona role = "You are helpful."')
        decl = program.statements[0]
        assert isinstance(decl.type_spec, TypePersona)

    def test_instruct_declaration(self):
        program = parse('instruct task = "Answer concisely."')
        decl = program.statements[0]
        assert isinstance(decl.type_spec, TypeInstruct)

    def test_context_list_declaration(self):
        program = parse('context[] docs = retriever.fetch("query", 3)')
        decl = program.statements[0]
        assert isinstance(decl.type_spec, TypeContextList)
        assert isinstance(decl.value, FuncCall)
        assert decl.value.func == "retriever.fetch"

    def test_long_string_declaration(self):
        program = parse('instruct task = """Do this\nand that."""')
        decl = program.statements[0]
        assert isinstance(decl.value, LongStringLiteral)
        assert "Do this" in decl.value.value


class TestSetParam:
    def test_numeric_param(self):
        program = parse("set_param temperature = 0.5")
        stmt = program.statements[0]
        assert isinstance(stmt, SetParam)
        assert stmt.name == "temperature"
        assert isinstance(stmt.value, NumberLiteral)
        assert stmt.value.value == pytest.approx(0.5)

    def test_bool_param(self):
        program = parse("set_param stream = false")
        stmt = program.statements[0]
        assert isinstance(stmt, SetParam)
        assert stmt.value.value is False

    def test_integer_param(self):
        program = parse("set_param max_tokens = 1024")
        stmt = program.statements[0]
        assert stmt.value.value == 1024


class TestCompileCall:
    def test_empty_compile(self):
        program = parse("prompt.compile()")
        stmt = program.statements[0]
        assert isinstance(stmt, CompileCall)
        assert stmt.args == []

    def test_compile_with_args(self):
        program = parse("prompt.compile(role, query)")
        stmt = program.statements[0]
        assert isinstance(stmt, CompileCall)
        assert len(stmt.args) == 2
        assert isinstance(stmt.args[0], Identifier)
        assert stmt.args[0].name == "role"


class TestControlFlow:
    def test_for_loop(self):
        src = """
context[] docs = retriever.fetch("x", 2)
for doc in docs {
    str item = doc
}
"""
        program = parse(src)
        assert len(program.statements) == 2
        from promptscript.ast_nodes import ForLoop
        loop = program.statements[1]
        assert isinstance(loop, ForLoop)
        assert loop.var == "doc"
        assert len(loop.body) == 1

    def test_if_else(self):
        src = """
bool flag = true
if flag == true {
    str msg = "yes"
} else {
    str msg = "no"
}
"""
        program = parse(src)
        from promptscript.ast_nodes import IfBlock, BinaryCondition
        if_block = program.statements[1]
        assert isinstance(if_block, IfBlock)
        assert isinstance(if_block.condition, BinaryCondition)
        assert len(if_block.then_body) == 1
        assert len(if_block.else_body) == 1

    def test_if_no_else(self):
        src = """
bool flag = false
if flag == true {
    str msg = "yes"
}
"""
        program = parse(src)
        from promptscript.ast_nodes import IfBlock
        if_block = program.statements[1]
        assert isinstance(if_block, IfBlock)
        assert if_block.else_body == []


class TestComments:
    def test_comment_ignored(self):
        src = """
// This is a comment
str x = "value"
// Another comment
"""
        program = parse(src)
        assert len(program.statements) == 1
        assert isinstance(program.statements[0], Declaration)


class TestParseErrors:
    def test_invalid_syntax_raises(self):
        with pytest.raises(ParseError):
            parse("str = this is invalid @@@@")

    def test_unclosed_brace(self):
        with pytest.raises(ParseError):
            parse("for x in docs { str y = x")
