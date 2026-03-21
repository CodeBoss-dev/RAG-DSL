"""Tests for the Markdown compiler target."""

import pytest

from promptscript.parser import parse
from promptscript.compiler import compile_program, compile_segments
from promptscript.runtime.context import RuntimeContext, RetrievedChunk


def make_ctx(retriever=None, budget=None):
    return RuntimeContext(retriever=retriever, token_budget=budget)


def fixed_retriever(n=2):
    def _r(query, top_k):
        return [
            RetrievedChunk(doc_id=f"d{i}", text=f"Doc {i}: info about {query}.", score=1.0 - i * 0.1)
            for i in range(min(top_k, n))
        ]
    return _r


class TestMarkdownBasic:
    def test_persona_becomes_persona_section(self):
        program = parse('persona role = "You are helpful."')
        result = compile_program(program, make_ctx(), target="markdown")
        assert "## Persona" in result
        assert "You are helpful." in result

    def test_instruct_becomes_instructions_section(self):
        program = parse('instruct task = "Answer concisely."')
        result = compile_program(program, make_ctx(), target="markdown")
        assert "## Instructions" in result
        assert "Answer concisely." in result

    def test_str_becomes_query_section(self):
        program = parse('str q = "What is Python?"')
        result = compile_program(program, make_ctx(), target="markdown")
        assert "## Query" in result
        assert "What is Python?" in result

    def test_context_becomes_context_section(self, mock_retriever):
        program = parse('context[] docs = retriever.fetch("ai", 2)')
        result = compile_program(program, make_ctx(retriever=mock_retriever), target="markdown")
        assert "## Context" in result

    def test_section_order(self, mock_retriever):
        program = parse("""
persona role = "assistant"
context[] docs = retriever.fetch("topic", 1)
instruct task = "Do this."
str query = "My question"
""")
        result = compile_program(program, make_ctx(retriever=mock_retriever), target="markdown")
        persona_pos = result.index("## Persona")
        context_pos = result.index("## Context")
        instruct_pos = result.index("## Instructions")
        query_pos = result.index("## Query")
        assert persona_pos < context_pos < instruct_pos < query_pos


class TestMarkdownFixtures:
    def test_simple_fixture(self, fixtures_dir, mock_retriever):
        from promptscript.parser import parse_file
        program = parse_file(fixtures_dir / "simple.ps")
        result = compile_program(program, make_ctx(), target="markdown")
        assert "## Persona" in result
        assert "## Instructions" in result
        assert "## Query" in result

    def test_with_context_fixture(self, fixtures_dir, mock_retriever):
        from promptscript.parser import parse_file
        program = parse_file(fixtures_dir / "with_context.ps")
        result = compile_program(program, make_ctx(retriever=mock_retriever), target="markdown")
        assert "## Context" in result


class TestMarkdownTokenBudget:
    def test_budget_drops_context(self):
        # Create a program with context, then enforce a very tight budget
        program = parse('context[] docs = retriever.fetch("x", 5)')
        ctx = make_ctx(retriever=fixed_retriever(5), budget=5)
        segments = compile_segments(program, ctx)
        # With budget=5, most context segments should be dropped
        context_segs = [s for s in segments if s.role == "context"]
        total = sum(s.token_count for s in segments)
        assert total <= 5 or len(context_segs) < 5
