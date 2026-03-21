"""Shared pytest fixtures."""

from pathlib import Path
import pytest

from promptscript.runtime.context import RuntimeContext, RetrievedChunk


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def mock_retriever():
    """A deterministic retriever that returns fixed chunks."""
    def retriever(query: str, top_k: int) -> list[RetrievedChunk]:
        chunks = [
            RetrievedChunk(doc_id=f"doc_{i}", text=f"Chunk {i} about {query}.", score=1.0 - i * 0.1)
            for i in range(top_k)
        ]
        return chunks
    return retriever


@pytest.fixture
def runtime_ctx(mock_retriever) -> RuntimeContext:
    return RuntimeContext(retriever=mock_retriever)
