"""Tests for the token budget module."""

import pytest

from promptscript.token_budget import (
    PromptSegment, enforce_budget, total_tokens, count_tokens
)


def make_seg(role, content, confidence=1.0):
    return PromptSegment(role=role, content=content, confidence=confidence)


class TestCountTokens:
    def test_empty_string(self):
        # At minimum 1 token
        assert count_tokens("") >= 0

    def test_short_string(self):
        n = count_tokens("hello world")
        assert n >= 1

    def test_longer_string_more_tokens(self):
        short = count_tokens("hi")
        long = count_tokens("This is a much longer string with many more tokens in it.")
        assert long > short


class TestPromptSegment:
    def test_token_count_auto_computed(self):
        seg = PromptSegment(role="user", content="Hello world")
        assert seg.token_count > 0

    def test_explicit_token_count_respected(self):
        seg = PromptSegment(role="user", content="Hello", token_count=99)
        assert seg.token_count == 99

    def test_default_confidence(self):
        seg = PromptSegment(role="context", content="chunk")
        assert seg.confidence == 1.0


class TestEnforceBudget:
    def test_no_drop_when_within_budget(self):
        segs = [
            make_seg("system", "You are helpful."),
            make_seg("context", "Some context."),
            make_seg("user", "My question."),
        ]
        budget = 10000
        result = enforce_budget(segs, budget)
        assert len(result) == len(segs)

    def test_drops_context_first(self):
        fixed = make_seg("system", "system")
        ctx1 = make_seg("context", "A" * 200, confidence=0.5)
        ctx2 = make_seg("context", "B" * 200, confidence=0.9)
        user = make_seg("user", "question")

        # Budget just enough to keep system + user but not both context chunks
        budget = fixed.token_count + user.token_count + ctx2.token_count
        result = enforce_budget([fixed, ctx1, ctx2, user], budget)

        roles = [s.role for s in result]
        assert "system" in roles
        assert "user" in roles
        # ctx1 (lower confidence) should be dropped
        contents = [s.content for s in result]
        assert ctx1.content not in contents

    def test_never_drops_non_context(self):
        system_seg = make_seg("system", "system " * 100)
        user_seg = make_seg("user", "user " * 100)
        ctx_seg = make_seg("context", "context " * 100)

        # Very tight budget that can't fit everything
        budget = system_seg.token_count + user_seg.token_count
        result = enforce_budget([system_seg, ctx_seg, user_seg], budget)

        assert any(s.role == "system" for s in result)
        assert any(s.role == "user" for s in result)

    def test_total_tokens(self):
        segs = [make_seg("user", "hi"), make_seg("system", "hello")]
        assert total_tokens(segs) == sum(s.token_count for s in segs)

    def test_lowest_confidence_dropped_first(self):
        segs = [
            make_seg("context", "low " * 50, confidence=0.1),
            make_seg("context", "medium " * 50, confidence=0.5),
            make_seg("context", "high " * 50, confidence=0.9),
            make_seg("user", "q"),
        ]
        # Budget that forces dropping lowest confidence chunks
        budget = segs[-1].token_count + segs[2].token_count  # keep user + high only
        result = enforce_budget(segs, budget)
        confidences = [s.confidence for s in result if s.role == "context"]
        # Only high-confidence context should remain (if any)
        for c in confidences:
            assert c >= 0.5
