"""Tests for the JSON API compiler target."""

import json
import pytest

from promptscript.parser import parse
from promptscript.compiler import compile_program, compile_to_dict
from promptscript.runtime.context import RuntimeContext, RetrievedChunk


def make_ctx(retriever=None, budget=None):
    return RuntimeContext(retriever=retriever, token_budget=budget)


def fixed_retriever(n=2):
    def _r(query, top_k):
        return [
            RetrievedChunk(doc_id=f"d{i}", text=f"Chunk {i}.", score=1.0)
            for i in range(min(top_k, n))
        ]
    return _r


class TestJsonApiBasic:
    def test_output_is_valid_json(self):
        program = parse('str query = "Hello"')
        result = compile_program(program, make_ctx(), target="json")
        data = json.loads(result)
        assert "messages" in data

    def test_persona_maps_to_system(self):
        program = parse('persona role = "You are helpful."')
        data = compile_to_dict(program, make_ctx())
        system_msgs = [m for m in data["messages"] if m["role"] == "system"]
        assert len(system_msgs) >= 1
        assert "You are helpful." in system_msgs[0]["content"]

    def test_instruct_maps_to_system(self):
        program = parse('instruct task = "Answer concisely."')
        data = compile_to_dict(program, make_ctx())
        system_msgs = [m for m in data["messages"] if m["role"] == "system"]
        assert any("Answer concisely." in m["content"] for m in system_msgs)

    def test_str_maps_to_user(self):
        program = parse('str query = "My question."')
        data = compile_to_dict(program, make_ctx())
        user_msgs = [m for m in data["messages"] if m["role"] == "user"]
        assert any("My question." in m["content"] for m in user_msgs)

    def test_context_maps_to_user(self):
        program = parse('context[] docs = retriever.fetch("q", 1)')
        ctx = make_ctx(retriever=fixed_retriever(1))
        data = compile_to_dict(program, ctx)
        user_msgs = [m for m in data["messages"] if m["role"] == "user"]
        assert len(user_msgs) >= 1

    def test_set_param_in_api_body(self):
        program = parse("""
set_param temperature = 0.3
set_param max_tokens = 256
str query = "Hello"
""")
        data = compile_to_dict(program, make_ctx())
        assert data.get("temperature") == pytest.approx(0.3)
        assert data.get("max_tokens") == 256


class TestJsonMessages:
    def test_multiple_roles_produce_multiple_messages(self):
        program = parse("""
persona role = "assistant"
str query = "hello"
""")
        data = compile_to_dict(program, make_ctx())
        roles = [m["role"] for m in data["messages"]]
        assert "system" in roles
        assert "user" in roles

    def test_consecutive_system_segments_merged(self):
        program = parse("""
persona role = "assistant"
instruct task = "be helpful"
""")
        data = compile_to_dict(program, make_ctx())
        system_msgs = [m for m in data["messages"] if m["role"] == "system"]
        # Both persona and instruct are system — should be merged into one message
        assert len(system_msgs) == 1
        assert "assistant" in system_msgs[0]["content"]
        assert "be helpful" in system_msgs[0]["content"]
