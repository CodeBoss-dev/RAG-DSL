"""Runtime context: variable bindings and retriever stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RetrievedChunk:
    """A single document chunk returned by the retriever."""
    doc_id: str
    text: str
    score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


# Type alias for a retriever callable
RetrieverFn = Callable[[str, int], list[RetrievedChunk]]


class RuntimeContext:
    """Holds variable bindings and the retriever function for a single compilation run."""

    def __init__(
        self,
        variables: dict[str, Any] | None = None,
        retriever: RetrieverFn | None = None,
        token_budget: int | None = None,
    ):
        self._vars: dict[str, Any] = dict(variables or {})
        self._retriever: RetrieverFn = retriever or _null_retriever
        self.token_budget = token_budget  # None = unlimited

    # ------------------------------------------------------------------
    # Variable access
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        if name not in self._vars:
            raise NameError(f"Undefined variable '{name}'")
        return self._vars[name]

    def set(self, name: str, value: Any) -> None:
        self._vars[name] = value

    def has(self, name: str) -> bool:
        return name in self._vars

    # ------------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------------

    def fetch(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self._retriever(query, top_k)

    # ------------------------------------------------------------------
    # Scope management (for loops / if blocks)
    # ------------------------------------------------------------------

    def child(self) -> "RuntimeContext":
        """Create a child context that inherits but isolates new bindings."""
        child = RuntimeContext(
            variables=dict(self._vars),
            retriever=self._retriever,
            token_budget=self.token_budget,
        )
        return child


def _null_retriever(query: str, top_k: int) -> list[RetrievedChunk]:
    """No-op retriever used when no retriever is supplied."""
    return []
