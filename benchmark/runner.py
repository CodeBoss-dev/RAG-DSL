"""LLM API orchestration via Ollama's OpenAI-compatible endpoint.

Supports two prompt types:
  - plain_english: fills {{CONTEXT}} placeholder in .txt template
  - promptscript:  compiles .ps file using the PromptScript compiler
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from promptscript.parser import parse_file
from promptscript.type_checker import TypeChecker
from promptscript.compiler import compile_to_dict
from promptscript.runtime.context import RuntimeContext, RetrievedChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plain-English prompt builder
# ---------------------------------------------------------------------------

def _build_plain_english_prompt(
    template_path: Path,
    chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    """Fill the {{CONTEXT}} placeholder and return an OpenAI messages list."""
    template = template_path.read_text()
    context_text = "\n\n".join(
        f"[{c.doc_id}] {c.text}" for c in chunks
    )
    filled = template.replace("{{CONTEXT}}", context_text)
    return [{"role": "user", "content": filled}]


# ---------------------------------------------------------------------------
# PromptScript prompt builder
# ---------------------------------------------------------------------------

def _build_promptscript_prompt(
    ps_path: Path,
    chunks: list[RetrievedChunk],
    token_budget: int | None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Parse, type-check, and compile a .ps file.

    Returns (messages, extra_params) where extra_params may contain
    temperature / max_tokens extracted from set_param directives.
    """
    program = parse_file(str(ps_path))
    TypeChecker().check(program)

    # Build a retriever that returns the pre-fetched chunks
    def _preloaded_retriever(query: str, top_k: int) -> list[RetrievedChunk]:  # noqa: ARG001
        return chunks[:top_k]

    ctx = RuntimeContext(
        retriever=_preloaded_retriever,
        token_budget=token_budget,
    )
    result = compile_to_dict(program, ctx)
    messages: list[dict[str, str]] = result.get("messages", [])
    extra: dict[str, Any] = {
        k: v for k, v in result.items() if k != "messages"
    }
    return messages, extra


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def call_llm(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 512,
    timeout: float = 120,
) -> tuple[str, int, int]:
    """Call the LLM and return (response_text, prompt_tokens, completion_tokens)."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    text = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return text, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_task(
    *,
    task: dict,
    prompt_type: str,
    chunks: list[RetrievedChunk],
    prompt_dir: Path,
    client: OpenAI,
    model: str,
    default_temperature: float = 0.1,
    default_max_tokens: int = 512,
    timeout: float = 120,
    token_budget: int | None = None,
) -> dict[str, Any]:
    """Run a single task and return a result dict.

    The returned dict contains:
      task_id, prompt_type, response, prompt_tokens, completion_tokens,
      retrieved_doc_ids, messages (for debugging).
    """
    task_id: str = task["task_id"]

    if prompt_type == "plain_english":
        template_path = prompt_dir / f"{task_id}.txt"
        messages = _build_plain_english_prompt(template_path, chunks)
        extra_params: dict[str, Any] = {}
    elif prompt_type == "promptscript":
        ps_path = prompt_dir / f"{task_id}.ps"
        messages, extra_params = _build_promptscript_prompt(ps_path, chunks, token_budget)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type!r}")

    # set_param directives from .ps files override defaults; strip them so they
    # don't collide with the explicit keyword arguments in call_llm.
    temperature = float(extra_params.pop("temperature", default_temperature))
    max_tokens = int(extra_params.pop("max_tokens", default_max_tokens))

    response_text, prompt_tokens, completion_tokens = call_llm(
        client=client,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    return {
        "task_id": task_id,
        "prompt_type": prompt_type,
        "response": response_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "retrieved_doc_ids": [c.doc_id for c in chunks],
        "messages": messages,
    }
