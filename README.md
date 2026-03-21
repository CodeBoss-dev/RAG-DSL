# RAG DSL — PromptScript

A lightweight typed DSL for constructing RAG prompts, benchmarked against equivalent plain-English prompts across 50 standardized tasks.

## Overview

PromptScript lets you write structured, type-checked prompts instead of freeform English. The benchmark empirically tests whether this improves **faithfulness**, **format compliance**, and **token efficiency** compared to plain-English prompts.

## Project Structure

```
├── src/promptscript/       # DSL implementation
│   ├── grammar.lark        # Lark EBNF grammar
│   ├── parser.py           # Lark-based parser
│   ├── ast_nodes.py        # Dataclass AST nodes
│   ├── transformer.py      # Parse tree -> AST
│   ├── type_checker.py     # Static type validation
│   ├── compiler.py         # AST -> PromptSegments -> output
│   ├── token_budget.py     # tiktoken-based budget enforcement
│   ├── targets/            # Markdown + JSON API renderers
│   ├── runtime/            # Variable bindings, built-ins
│   └── cli.py              # CLI entry point
│
├── benchmark/              # Evaluation pipeline
│   ├── dataset/            # 200-doc corpus + 50 tasks (JSONL)
│   ├── prompts/            # Paired plain-English + PromptScript files
│   ├── retriever/          # BM25 retriever
│   ├── runner.py           # LLM API orchestration (Ollama)
│   ├── scorer.py           # 4-metric scoring
│   └── pipeline.py         # End-to-end evaluation
│
├── report/                 # Analysis, figures, statistical tests
└── tests/                  # Unit + integration tests
```

## DSL Example

```
persona role = "You are a precise research assistant."
str query = "What caused the 2008 financial crisis?"
context[] docs = retriever.fetch(query, top_k=5)

set_param temperature = 0.2
set_param max_tokens = 512

instruct instructions = """
Answer using only the provided context.
If the answer is not in the context, say "I don't know."
Cite sources by document ID.
"""

prompt.compile(role, docs, query, instructions)
```

## CLI

```bash
# Compile to markdown or JSON
promptscript compile input.ps --target markdown --output out.md
promptscript compile input.ps --target json --output out.json

# Type-check only
promptscript check input.ps

# Show token counts per segment
promptscript tokens input.ps
```

## Benchmark

50 tasks across four types:

| Task Type | Count |
|---|---|
| Factual QA | 15 |
| Multi-hop Reasoning | 12 |
| Summarization with Citations | 13 |
| Out-of-Context Detection | 10 |

Both prompt types receive the **same retrieved documents** per task (BM25, deterministic). Prompt format is the sole independent variable.

### Metrics (each [0, 1])

1. **Answer Correctness** — 0.4 × exact match + 0.6 × ROUGE-L F1
2. **Faithfulness** — claim-level ROUGE overlap against retrieved chunks
3. **Format Compliance** — task-type-specific structural checks
4. **Token Efficiency** — prompt tokens normalized by correctness

Statistical significance tested with paired Wilcoxon signed-rank (N=50).

## Setup

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Run benchmark (requires Ollama)

```bash
python -m benchmark.pipeline
```

## Dependencies

**Core:** `lark`, `tiktoken`, `click`, `pyyaml`
**Benchmark:** `openai`, `rank-bm25`, `rouge-score`, `matplotlib`, `pandas`, `scipy`
**Dev:** `pytest`, `pytest-cov`, `ruff`
