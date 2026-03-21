"""Tests for the PromptScript type checker."""

import pytest

from promptscript.parser import parse
from promptscript.type_checker import check, check_or_raise, TypeCheckFailed


class TestTypeCompatibility:
    def test_str_accepts_string_literal(self):
        program = parse('str x = "hello"')
        errors = check(program)
        assert errors == []

    def test_int_accepts_int_literal(self):
        program = parse("int n = 42")
        errors = check(program)
        assert errors == []

    def test_float_accepts_float_literal(self):
        program = parse("float f = 1.5")
        errors = check(program)
        assert errors == []

    def test_float_accepts_int_literal(self):
        # int is compatible with float
        program = parse("float f = 2")
        errors = check(program)
        assert errors == []

    def test_bool_accepts_bool_literal(self):
        program = parse("bool b = false")
        errors = check(program)
        assert errors == []

    def test_persona_accepts_string(self):
        program = parse('persona role = "You are helpful."')
        errors = check(program)
        assert errors == []

    def test_instruct_accepts_string(self):
        program = parse('instruct task = "Do this."')
        errors = check(program)
        assert errors == []

    def test_context_list_accepts_func_call(self):
        program = parse('context[] docs = retriever.fetch("q", 3)')
        errors = check(program)
        assert errors == []

    def test_context_list_accepts_list_expr(self):
        program = parse('context[] docs = ["a", "b"]')
        errors = check(program)
        assert errors == []


class TestTypeErrors:
    def test_str_rejects_int_literal(self):
        # str x = 42 should be a mismatch
        program = parse("str x = 42")
        errors = check(program)
        assert len(errors) == 1
        assert "x" in errors[0].message

    def test_int_rejects_string_literal(self):
        program = parse('int n = "hello"')
        errors = check(program)
        assert len(errors) == 1

    def test_bool_rejects_string_literal(self):
        program = parse('bool b = "true"')
        errors = check(program)
        assert len(errors) == 1

    def test_set_param_rejects_string(self):
        # Grammar itself rejects string literals for set_param (only NUMBER/BOOL allowed)
        from promptscript.parser import ParseError
        with pytest.raises(ParseError):
            parse('set_param temperature = "hot"')


class TestSetParam:
    def test_numeric_set_param_ok(self):
        program = parse("set_param max_tokens = 512")
        errors = check(program)
        assert errors == []

    def test_bool_set_param_ok(self):
        program = parse("set_param stream = true")
        errors = check(program)
        assert errors == []


class TestCompileCallCheck:
    def test_compile_with_declared_vars_ok(self):
        program = parse("""
persona role = "helper"
str query = "question"
prompt.compile(role, query)
""")
        errors = check(program)
        assert errors == []

    def test_compile_with_undeclared_var_errors(self):
        program = parse("prompt.compile(undefined_var)")
        errors = check(program)
        assert any("undefined_var" in e.message for e in errors)


class TestCheckOrRaise:
    def test_raises_on_errors(self):
        program = parse("str x = 42")
        with pytest.raises(TypeCheckFailed) as exc_info:
            check_or_raise(program)
        assert "x" in str(exc_info.value)

    def test_no_raise_on_clean(self):
        program = parse('str x = "hello"')
        check_or_raise(program)  # should not raise
