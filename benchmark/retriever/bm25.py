"""BM25 retriever backed by rank_bm25.

Loads the corpus once at construction time, then answers queries
deterministically — no GPU, no randomness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi

from promptscript.runtime.context import RetrievedChunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


class BM25Retriever:
    """Load all corpus docs and answer BM25 queries."""

    def __init__(self, corpus_dir: str | Path):
        self._docs: list[dict] = []
        self._index: BM25Okapi = self._build_index(Path(corpus_dir))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_index(self, corpus_dir: Path) -> BM25Okapi:
        doc_files = sorted(corpus_dir.glob("doc_*.json"))
        if not doc_files:
            raise FileNotFoundError(f"No doc_*.json files found in {corpus_dir}")

        tokenized: list[list[str]] = []
        for path in doc_files:
            with path.open() as f:
                doc = json.load(f)
            self._docs.append(doc)
            tokenized.append(_tokenize(doc["text"]))

        logger.info("BM25 index built over %d documents", len(self._docs))
        return BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return top-k chunks for *query*, sorted by descending BM25 score."""
        tokens = _tokenize(query)
        scores: list[float] = self._index.get_scores(tokens).tolist()

        # Pair each doc with its score and sort descending
        ranked = sorted(
            zip(scores, self._docs),
            key=lambda x: x[0],
            reverse=True,
        )

        results: list[RetrievedChunk] = []
        for score, doc in ranked[:top_k]:
            results.append(
                RetrievedChunk(
                    doc_id=doc["doc_id"],
                    text=doc["text"],
                    score=float(score),
                    metadata={"title": doc.get("title", ""), "topic": doc.get("topic", "")},
                )
            )
        return results

    def as_retriever_fn(self, top_k: int | None = None):
        """Return a callable compatible with RuntimeContext's RetrieverFn signature."""
        default_k = top_k

        def _fn(query: str, k: int = 5) -> list[RetrievedChunk]:
            return self.retrieve(query, top_k=default_k if default_k is not None else k)

        return _fn
