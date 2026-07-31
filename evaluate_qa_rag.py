"""Small offline academic QA benchmark for RAG retrieval + grounding checks."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence

from retriever import PrivacyRetriever
from runtime_utils import exact_match, token_f1


DOCUMENTS: List[Dict[str, str]] = [
    {
        "doc_id": "circuits_ohm",
        "text": "Ohm's law states that voltage equals current times resistance in a linear resistor.",
    },
    {
        "doc_id": "circuits_power",
        "text": "Electrical power is measured in watts and equals voltage multiplied by current.",
    },
    {
        "doc_id": "networks_tcp",
        "text": "TCP provides reliable ordered delivery. UDP has lower transport overhead but does not guarantee ordering or delivery.",
    },
    {
        "doc_id": "algorithms_merge_sort",
        "text": "Merge sort recursively divides a list and runs in O(n log n) time.",
    },
    {
        "doc_id": "algorithms_binary_search",
        "text": "Binary search halves a sorted search interval at each step.",
    },
    {
        "doc_id": "statistics_p_value",
        "text": "A p-value is computed assuming the null hypothesis is true.",
    },
    {
        "doc_id": "statistics_alpha",
        "text": "At alpha 0.05, a p-value below alpha leads to rejecting the null hypothesis.",
    },
    {
        "doc_id": "bloom_compare",
        "text": "Compare and contrast questions usually signal the Analyze level in Bloom's taxonomy.",
    },
    {
        "doc_id": "rag_grounding",
        "text": "A grounded RAG system should decline when retrieved context lacks support for the answer.",
    },
    {
        "doc_id": "privacy_role",
        "text": "Protected exam corpora must not be retrieved for student-facing queries.",
    },
]

QA_ITEMS: List[Dict[str, object]] = [
    {
        "question": "What relationship does Ohm's law define?",
        "support_doc": "circuits_ohm",
        "answers": ["voltage equals current times resistance", "voltage current resistance"],
    },
    {
        "question": "What unit is electrical power measured in?",
        "support_doc": "circuits_power",
        "answers": ["watts", "watt"],
    },
    {
        "question": "Which protocol provides reliable ordered delivery?",
        "support_doc": "networks_tcp",
        "answers": ["tcp", "transmission control protocol"],
    },
    {
        "question": "What tradeoff does UDP make compared with TCP?",
        "support_doc": "networks_tcp",
        "answers": [
            "udp has lower transport overhead but does not guarantee ordering or delivery",
            "lower overhead without guaranteed delivery",
        ],
    },
    {
        "question": "What is the time complexity of merge sort?",
        "support_doc": "algorithms_merge_sort",
        "answers": ["o(n log n)", "n log n", "o n log n"],
    },
    {
        "question": "How does binary search reduce the search interval?",
        "support_doc": "algorithms_binary_search",
        "answers": ["halves a sorted search interval", "halves the interval"],
    },
    {
        "question": "What does a p-value assume about the null hypothesis?",
        "support_doc": "statistics_p_value",
        "answers": ["the null hypothesis is true", "null hypothesis is true"],
    },
    {
        "question": "At alpha 0.05, what decision follows from a p-value of 0.03?",
        "support_doc": "statistics_alpha",
        "answers": ["reject the null hypothesis", "reject null"],
    },
    {
        "question": "Which Bloom level is usually signaled by compare?",
        "support_doc": "bloom_compare",
        "answers": ["analyze", "analyzing"],
    },
    {
        "question": "What should a grounded RAG system do when context lacks support?",
        "support_doc": "rag_grounding",
        "answers": ["decline", "say it does not know", "refuse to answer"],
    },
]


def _exact_match(prediction: str, references: Sequence[str]) -> float:
    return float(any(exact_match(prediction, ref) for ref in references))


def _token_f1(prediction: str, reference: str) -> float:
    return token_f1(prediction, reference)


def _best_gold_f1(prediction: str, references: Sequence[str]) -> float:
    return max((_token_f1(prediction, ref) for ref in references), default=0.0)


@lru_cache(maxsize=1)
def _retriever() -> PrivacyRetriever:
    retr = PrivacyRetriever()
    retr.build_index([doc["text"] for doc in DOCUMENTS])
    return retr


def _rank(query: str, system: str) -> List[Dict[str, object]]:
    del system
    hits = _retriever().retrieve(query, top_k=len(DOCUMENTS))
    ranked: List[Dict[str, object]] = []
    for hit in hits:
        doc = DOCUMENTS[int(hit.doc_id)]
        ranked.append(
            {
                "doc_id": doc["doc_id"],
                "text": hit.text,
                "score": float(hit.cosine),
            }
        )
    return ranked
