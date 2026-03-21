"""JSON API body renderer for PromptScript (OpenAI / Ollama compatible)."""

from __future__ import annotations

import json
from typing import Any

from promptscript.token_budget import PromptSegment

# Roles that map to OpenAI message roles
_ROLE_MAP = {
    "persona": "system",
    "system": "system",
    "instruct": "system",
    "context": "user",
    "user": "user",
    "assistant": "assistant",
}

# Default API parameters
_DEFAULT_PARAMS: dict[str, Any] = {}


def render(
    segments: list[PromptSegment],
    api_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render PromptSegments to an OpenAI-compatible API request body dict.

    - `persona`, `system`, and `instruct` segments become `system` messages.
    - `context` and `user` segments become `user` messages.
      Multiple context chunks are joined with newlines under a single user message.
    - `assistant` segments become `assistant` messages.
    - `api_params` (from `set_param` directives) are merged at the top level.

    Args:
        segments: Output of the compiler / budget enforcer.
        api_params: Dict of API-level parameters (e.g. temperature, max_tokens).

    Returns:
        A dict suitable for ``json.dumps`` and POSTing to /v1/chat/completions.
    """
    params = dict(_DEFAULT_PARAMS)
    if api_params:
        params.update(api_params)

    messages: list[dict[str, str]] = []

    # Accumulate consecutive segments of the same OpenAI role
    def flush(role: str, parts: list[str]) -> None:
        if parts:
            messages.append({"role": role, "content": "\n\n".join(parts)})

    current_role: str | None = None
    current_parts: list[str] = []

    for seg in segments:
        api_role = _ROLE_MAP.get(seg.role.lower(), "user")
        if api_role != current_role:
            flush(current_role or "user", current_parts)
            current_role = api_role
            current_parts = [seg.content]
        else:
            current_parts.append(seg.content)

    flush(current_role or "user", current_parts)

    body: dict[str, Any] = {"messages": messages}
    body.update(params)
    return body


def render_json(
    segments: list[PromptSegment],
    api_params: dict[str, Any] | None = None,
    indent: int | None = 2,
) -> str:
    """Same as :func:`render` but returns a JSON string."""
    return json.dumps(render(segments, api_params), indent=indent, ensure_ascii=False)
