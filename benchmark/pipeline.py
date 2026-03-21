"""End-to-end evaluation pipeline.

Flow per task:
  1. BM25 retrieve (once, shared between both prompt types)
  2. For each prompt_type: compile prompt → call LLM → score
  3. Write result row to results.jsonl

Usage:
  python -m benchmark.pipeline [--config benchmark/config.yaml] [--tasks task_001,task_002]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import yaml
from openai import OpenAI

from benchmark.retriever.bm25 import BM25Retriever
from benchmark.runner import run_task
from benchmark.scorer import score_result

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_tasks(tasks_file: str | Path) -> list[dict]:
    tasks = []
    with open(tasks_file) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    config_path: str | Path = "benchmark/config.yaml",
    task_ids: list[str] | None = None,
) -> list[dict]:
    """Run the full benchmark pipeline. Returns list of result dicts."""
    cfg = load_config(config_path)

    # Resolve all paths relative to the project root (where config sits)
    root = Path(config_path).parent.parent

    corpus_dir = root / cfg["retriever"]["corpus_dir"]
    tasks_file = root / cfg["dataset"]["tasks_file"]
    plain_dir = root / cfg["dataset"]["plain_english_dir"]
    ps_dir = root / cfg["dataset"]["promptscript_dir"]
    results_file = root / cfg["output"]["results_file"]
    results_file.parent.mkdir(parents=True, exist_ok=True)

    llm_cfg = cfg["llm"]
    client = OpenAI(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg.get("api_key", "ollama"),
    )

    retriever = BM25Retriever(corpus_dir)
    retriever_fn = retriever.as_retriever_fn(top_k=cfg["retriever"]["top_k"])

    all_tasks = load_tasks(tasks_file)
    # Filter by task_ids if specified (CLI or config)
    filter_ids = task_ids or cfg["run"].get("task_ids")
    if filter_ids:
        filter_set = set(filter_ids)
        all_tasks = [t for t in all_tasks if t["task_id"] in filter_set]

    prompt_types: list[str] = cfg["run"].get("prompt_types", ["plain_english", "promptscript"])
    token_budget: int | None = cfg.get("token_budget")
    scoring_cfg = cfg.get("scoring", {})

    results: list[dict] = []
    total = len(all_tasks) * len(prompt_types)
    done = 0

    with open(results_file, "w") as out_f:
        for task in all_tasks:
            task_id = task["task_id"]

            # --- Retrieve once, share between both prompt types ---
            chunks = retriever_fn(task["query"], cfg["retriever"]["top_k"])
            retrieved_texts = [c.text for c in chunks]

            for prompt_type in prompt_types:
                done += 1
                logger.info("[%d/%d] %s | %s", done, total, task_id, prompt_type)

                prompt_dir = plain_dir if prompt_type == "plain_english" else ps_dir

                try:
                    run_res = run_task(
                        task=task,
                        prompt_type=prompt_type,
                        chunks=chunks,
                        prompt_dir=prompt_dir,
                        client=client,
                        model=llm_cfg["model"],
                        default_temperature=llm_cfg.get("default_temperature", 0.1),
                        default_max_tokens=llm_cfg.get("default_max_tokens", 512),
                        timeout=llm_cfg.get("timeout", 120),
                        token_budget=token_budget,
                    )
                except Exception as exc:
                    logger.error("FAILED %s/%s: %s", task_id, prompt_type, exc)
                    run_res = {
                        "task_id": task_id,
                        "prompt_type": prompt_type,
                        "response": "",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "retrieved_doc_ids": [c.doc_id for c in chunks],
                        "error": str(exc),
                    }

                scores = score_result(
                    task=task,
                    run_result=run_res,
                    retrieved_texts=retrieved_texts,
                )

                row = {
                    **run_res,
                    **scores.as_dict(),
                    "task_type": task.get("task_type"),
                    "difficulty": task.get("difficulty"),
                }
                # Don't write messages to results file (too large)
                row.pop("messages", None)

                results.append(row)
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()

    logger.info("Pipeline complete. %d results written to %s", len(results), results_file)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--config",
    default="benchmark/config.yaml",
    show_default=True,
    help="Path to config.yaml",
)
@click.option(
    "--tasks",
    default=None,
    help="Comma-separated task IDs to run (default: all)",
)
def main(config: str, tasks: str | None) -> None:
    """Run the PromptScript benchmark evaluation pipeline."""
    task_ids = [t.strip() for t in tasks.split(",")] if tasks else None
    results = run_pipeline(config_path=config, task_ids=task_ids)
    click.echo(f"Done. {len(results)} results written.")


if __name__ == "__main__":
    main()
