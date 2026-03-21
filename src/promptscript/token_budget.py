"""tiktoken-based token budget enforcement for PromptScript."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _enc = None
    logger.warning("tiktoken not installed; token counts will be approximate (4 chars/token).")


def count_tokens(text: str) -> int:
    """Return the token count for a string using cl100k_base encoding."""
    if _enc is not None:
        return len(_enc.encode(text))
    # Rough fallback: ~4 characters per token
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# PromptSegment
# ---------------------------------------------------------------------------

@dataclass
class PromptSegment:
    """A single logical segment of a compiled prompt."""
    role: str           # "system", "user", "assistant", "context", "instruct", "persona"
    content: str
    token_count: int = field(default=0)
    metadata: dict[str, Any] = field(default_factory=dict)
    # confidence in [0, 1]; lower = dropped first when over budget
    confidence: float = field(default=1.0)

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = count_tokens(self.content)


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

def enforce_budget(
    segments: list[PromptSegment],
    budget: int,
) -> list[PromptSegment]:
    """Drop context segments (lowest confidence first) until total tokens <= budget.

    Only segments with role=="context" are candidates for dropping.
    Non-context segments are always preserved.

    Args:
        segments: List of PromptSegments produced by the compiler.
        budget: Maximum total token count.

    Returns:
        Filtered list of segments within budget.
    """
    total = sum(s.token_count for s in segments)
    if total <= budget:
        return segments

    # Separate context candidates from fixed segments
    fixed = [s for s in segments if s.role != "context"]
    candidates = [s for s in segments if s.role == "context"]

    # Sort candidates ascending by confidence so we drop least confident first
    candidates.sort(key=lambda s: s.confidence)

    kept: list[PromptSegment] = list(candidates)
    for candidate in candidates:
        if total <= budget:
            break
        kept.remove(candidate)
        logger.info(
            "Token budget: dropped context chunk (confidence=%.2f, tokens=%d). "
            "Remaining: %d / %d",
            candidate.confidence,
            candidate.token_count,
            sum(s.token_count for s in fixed) + sum(s.token_count for s in kept),
            budget,
        )
        total -= candidate.token_count

    # Rebuild in original order, skipping dropped candidates
    kept_set = set(id(s) for s in kept)
    result = [s for s in segments if s.role != "context" or id(s) in kept_set]
    return result


def total_tokens(segments: list[PromptSegment]) -> int:
    return sum(s.token_count for s in segments)
