"""Four-metric scoring for PromptScript benchmark.

Metrics (all in [0, 1]):
  1. answer_correctness  = 0.4 * relaxed_exact_match + 0.6 * rouge_l_f1
  2. faithfulness        = claim-level ROUGE-1 overlap against retrieved chunks
  3. format_compliance   = task-type-specific structural checks
  4. token_efficiency    = prompt_tokens / (correctness + 0.01), normalized
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rouge_score import rouge_scorer

# ---------------------------------------------------------------------------
# ROUGE helper
# ---------------------------------------------------------------------------

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def _rouge_l_f1(prediction: str, reference: str) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    scores = _ROUGE.score(reference, prediction)
    return float(scores["rougeL"].fmeasure)


def _rouge1_recall(prediction: str, reference: str) -> float:
    """Fraction of reference unigrams covered by prediction."""
    if not prediction.strip() or not reference.strip():
        return 0.0
    scores = _ROUGE.score(reference, prediction)
    return float(scores["rouge1"].recall)


# ---------------------------------------------------------------------------
# Metric 1: Answer Correctness
# ---------------------------------------------------------------------------

def _relaxed_exact_match(prediction: str, reference: str) -> float:
    """1.0 if the key answer tokens appear in the prediction, else 0.0."""
    pred_lower = prediction.lower()
    ref_lower = reference.lower()
    # Relaxed: check if every word in the reference (≥3 chars) appears in prediction
    ref_words = [w for w in re.findall(r"\b\w{3,}\b", ref_lower)]
    if not ref_words:
        return 1.0 if ref_lower in pred_lower else 0.0
    hits = sum(1 for w in ref_words if w in pred_lower)
    return hits / len(ref_words)


def score_correctness(prediction: str, ground_truth: dict) -> float:
    """0.4 * relaxed_exact_match + 0.6 * ROUGE-L F1."""
    reference = ground_truth.get("answer", "")
    if not reference:
        return 0.0
    rem = _relaxed_exact_match(prediction, reference)
    rl = _rouge_l_f1(prediction, reference)
    return 0.4 * rem + 0.6 * rl


# ---------------------------------------------------------------------------
# Metric 2: Faithfulness
# ---------------------------------------------------------------------------

def score_faithfulness(prediction: str, retrieved_texts: list[str]) -> float:
    """Claim-level ROUGE-1 recall of prediction against the retrieved context.

    Splits prediction into sentences and measures how well each is supported
    by any retrieved chunk, then averages.
    """
    if not prediction.strip() or not retrieved_texts:
        return 0.0

    combined_context = " ".join(retrieved_texts)
    sentences = [s.strip() for s in re.split(r"[.!?]+", prediction) if s.strip()]
    if not sentences:
        return 0.0

    recalls = [_rouge1_recall(sentence, combined_context) for sentence in sentences]
    return sum(recalls) / len(recalls)


# ---------------------------------------------------------------------------
# Metric 3: Format Compliance
# ---------------------------------------------------------------------------

_REFUSAL_PHRASES = [
    "i don't know", "i do not know", "cannot answer", "can't answer",
    "not mentioned", "not provided", "no information", "the context does not",
    "the provided context", "based on the context",
]


def _has_refusal(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _REFUSAL_PHRASES)


def score_format_compliance(
    prediction: str,
    task_type: str,
    ground_truth: dict,
) -> float:
    """Task-type-specific format checks.

    Returns a score in [0, 1] based on how many checks pass.
    """
    checks: list[bool] = []

    answerable: bool = ground_truth.get("answerable", True)

    if task_type == "factual_qa":
        # Should be concise (≤ 100 words)
        word_count = len(prediction.split())
        checks.append(word_count <= 100)
        # Should not be empty
        checks.append(bool(prediction.strip()))
        # If unanswerable, should signal that; if answerable, should not refuse
        if not answerable:
            checks.append(_has_refusal(prediction))
        else:
            checks.append(not _has_refusal(prediction) or word_count > 5)

    elif task_type == "multi_hop":
        # Should provide some reasoning (≥ 30 words)
        checks.append(len(prediction.split()) >= 30)
        checks.append(bool(prediction.strip()))
        if not answerable:
            checks.append(_has_refusal(prediction))

    elif task_type == "summarization":
        # Should cite doc IDs or mention sources
        citations = ground_truth.get("citations", [])
        if citations:
            lower = prediction.lower()
            cited_count = sum(1 for c in citations if c.lower() in lower)
            checks.append(cited_count >= max(1, len(citations) // 2))
        else:
            checks.append(True)
        # Should be substantive (≥ 50 words)
        checks.append(len(prediction.split()) >= 50)

    elif task_type == "out_of_context":
        # Must explicitly signal inability to answer from context
        checks.append(_has_refusal(prediction))
        # Should be brief
        checks.append(len(prediction.split()) <= 80)

    else:
        # Unknown type: just check non-empty
        checks.append(bool(prediction.strip()))

    if not checks:
        return 1.0
    return sum(checks) / len(checks)


# ---------------------------------------------------------------------------
# Metric 4: Token Efficiency
# ---------------------------------------------------------------------------

def score_token_efficiency(
    prompt_tokens: int,
    correctness: float,
    *,
    max_tokens_seen: int = 4096,
) -> float:
    """Lower prompt_tokens relative to correctness → higher efficiency.

    raw = prompt_tokens / (correctness + 0.01)
    Normalized by capping at max_tokens_seen / 0.01 (worst case ratio).
    Score = 1 - (raw / worst_case), clipped to [0, 1].
    """
    if prompt_tokens <= 0:
        return 1.0
    raw = prompt_tokens / (correctness + 0.01)
    worst_case = max_tokens_seen / 0.01
    score = 1.0 - (raw / worst_case)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Aggregated scorer
# ---------------------------------------------------------------------------

@dataclass
class ScoreResult:
    task_id: str
    prompt_type: str
    answer_correctness: float
    faithfulness: float
    format_compliance: float
    token_efficiency: float

    def as_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt_type": self.prompt_type,
            "answer_correctness": round(self.answer_correctness, 4),
            "faithfulness": round(self.faithfulness, 4),
            "format_compliance": round(self.format_compliance, 4),
            "token_efficiency": round(self.token_efficiency, 4),
        }


def score_result(
    *,
    task: dict,
    run_result: dict,
    retrieved_texts: list[str],
    max_tokens_seen: int = 4096,
) -> ScoreResult:
    """Compute all four metrics for a single task run."""
    prediction: str = run_result.get("response", "")
    ground_truth: dict = task.get("ground_truth", {})
    task_type: str = task.get("task_type", "factual_qa")
    prompt_tokens: int = run_result.get("prompt_tokens", 0)

    correctness = score_correctness(prediction, ground_truth)
    faithfulness = score_faithfulness(prediction, retrieved_texts)
    fmt = score_format_compliance(prediction, task_type, ground_truth)
    efficiency = score_token_efficiency(
        prompt_tokens, correctness, max_tokens_seen=max_tokens_seen
    )

    return ScoreResult(
        task_id=task["task_id"],
        prompt_type=run_result["prompt_type"],
        answer_correctness=correctness,
        faithfulness=faithfulness,
        format_compliance=fmt,
        token_efficiency=efficiency,
    )
