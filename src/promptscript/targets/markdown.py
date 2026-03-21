"""Markdown prompt renderer for PromptScript."""

from __future__ import annotations

from promptscript.token_budget import PromptSegment

# Section header order for consistent rendering
_ROLE_ORDER = ["persona", "system", "context", "instruct", "user", "assistant"]
_ROLE_HEADERS = {
    "persona": "## Persona",
    "system": "## System",
    "context": "## Context",
    "instruct": "## Instructions",
    "user": "## Query",
    "assistant": "## Assistant",
}


def render(segments: list[PromptSegment]) -> str:
    """Render a list of PromptSegments to a structured Markdown string.

    Segments with the same role are concatenated under one section header.
    Sections appear in canonical order (persona → system → context → instruct → user → assistant).

    Args:
        segments: Output of the compiler / budget enforcer.

    Returns:
        A Markdown string suitable for human review or saving to a file.
    """
    # Group by role, preserving within-role order
    groups: dict[str, list[str]] = {}
    for seg in segments:
        role = seg.role.lower()
        groups.setdefault(role, []).append(seg.content)

    parts: list[str] = []

    # Emit in canonical order, then any unexpected roles
    seen: set[str] = set()
    ordered_roles = _ROLE_ORDER + [r for r in groups if r not in _ROLE_ORDER]

    for role in ordered_roles:
        if role not in groups or role in seen:
            continue
        seen.add(role)
        header = _ROLE_HEADERS.get(role, f"## {role.capitalize()}")
        body = "\n\n".join(groups[role])
        parts.append(f"{header}\n\n{body}")

    return "\n\n---\n\n".join(parts)
