"""
reranker.py
===========================================================
Unified Retrieval–Classification–Uncertainty Reranking Layer

This is the missing "system coupling" component that makes
your framework publishable as a single coherent method.

It fuses:
- FAISS retrieval score
- Bloom classifier confidence
- Uncertainty penalty (SPU / entropy)

Result: uncertainty-aware cognitive retrieval ranking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------
# Input result from retriever
# ---------------------------------------------------------
@dataclass
class RetrievalResult:
    rank: int
    doc_id: int
    text: str
    score: float

    bloom_level: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None

    final_score: float = 0.0  # computed later


# ---------------------------------------------------------
# Reranking function
# ---------------------------------------------------------
def rerank_with_uncertainty(
    results: List[RetrievalResult],
    classifier_confidence: float,
    ordinal_risk: float,
    uncertainty_score: float,
    alpha: float = 0.6,
    beta: float = 0.3,
    gamma: float = 0.1,
) -> List[RetrievalResult]:
    """
    Unified scoring function:

    final_score =
        α * retrieval_score +
        β * classifier_confidence +
        γ * (1 - uncertainty)

    Optionally penalized by ordinal drift.
    """

    for r in results:

        # uncertainty term (higher uncertainty → lower score)
        uncertainty_term = 1.0 - uncertainty_score

        # ordinal penalty (higher drift → lower score)
        ordinal_penalty = 1.0 - ordinal_risk

        r.final_score = (
            alpha * r.score +
            beta * classifier_confidence +
            gamma * uncertainty_term
        ) * ordinal_penalty

    return sorted(results, key=lambda x: x.final_score, reverse=True)


# ---------------------------------------------------------
# Optional helper (for debugging / paper reporting)
# ---------------------------------------------------------
def explain_rerank(result: RetrievalResult) -> str:
    return (
        f"FinalScore={result.final_score:.4f} | "
        f"BaseScore={result.score:.4f} | "
        f"Adjusted by classifier + uncertainty + ordinal penalty"
    )