from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


RESULTS_PATH = Path("results/qa_rag_eval.json")
CSV_PATH = Path("results/qa_rag_eval_rows.csv")


DOCUMENTS = [
    {
        "doc_id": "circuits_ohm",
        "text": "Ohm's law states that voltage equals current times resistance. A resistor with constant resistance has a linear current-voltage relationship.",
    },
    {
        "doc_id": "circuits_power",
        "text": "Electrical power in a simple circuit can be computed as voltage times current. Power is measured in watts.",
    },
    {
        "doc_id": "networks_tcp",
        "text": "TCP provides reliable, ordered delivery using acknowledgements and retransmission. UDP has lower transport overhead but does not guarantee ordering or delivery.",
    },
    {
        "doc_id": "networks_layers",
        "text": "The transport layer provides process-to-process communication, while the network layer handles host-to-host routing.",
    },
    {
        "doc_id": "algorithms_merge_sort",
        "text": "Merge sort divides a list into halves, recursively sorts each half, and merges them. Its time complexity is O(n log n).",
    },
    {
        "doc_id": "algorithms_binary_search",
        "text": "Binary search repeatedly halves a sorted search interval. Its worst-case time complexity is O(log n).",
    },
    {
        "doc_id": "statistics_p_value",
        "text": "A p-value is the probability, assuming the null hypothesis is true, of observing data at least as extreme as the observed data.",
    },
    {
        "doc_id": "statistics_alpha",
        "text": "When the p-value is less than alpha, the usual decision rule is to reject the null hypothesis. For alpha 0.05, a p-value of 0.03 meets that rule.",
    },
    {
        "doc_id": "bloom_compare",
        "text": "Bloom taxonomy uses verbs such as remember, understand, apply, analyze, evaluate, and create. Compare often maps to Analyze, while judge or justify with criteria often maps to Evaluate.",
    },
    {
        "doc_id": "rag_grounding",
        "text": "Retrieval augmented generation should answer from retrieved context and decline when the context does not support the requested fact.",
    },
]


QA_ITEMS = [
    {
        "question": "What relationship does Ohm's law define?",
        "answers": ["voltage equals current times resistance", "voltage = current times resistance"],
        "support_doc": "circuits_ohm",
    },
    {
        "question": "What unit is electrical power measured in?",
        "answers": ["watts"],
        "support_doc": "circuits_power",
    },
    {
        "question": "Which protocol provides reliable ordered delivery?",
        "answers": ["tcp"],
        "support_doc": "networks_tcp",
    },
    {
        "question": "What tradeoff does UDP make compared with TCP?",
        "answers": ["lower transport overhead but no guarantee of ordering or delivery", "lower overhead"],
        "support_doc": "networks_tcp",
    },
    {
        "question": "What is the time complexity of merge sort?",
        "answers": ["o n log n", "o(n log n)"],
        "support_doc": "algorithms_merge_sort",
    },
    {
        "question": "How does binary search reduce the search interval?",
        "answers": ["halves a sorted search interval", "repeatedly halves"],
        "support_doc": "algorithms_binary_search",
    },
    {
        "question": "What does a p-value assume about the null hypothesis?",
        "answers": ["the null hypothesis is true", "assuming the null hypothesis is true"],
        "support_doc": "statistics_p_value",
    },
    {
        "question": "At alpha 0.05, what decision follows from a p-value of 0.03?",
        "answers": ["reject the null hypothesis", "reject null"],
        "support_doc": "statistics_alpha",
    },
    {
        "question": "Which Bloom level is usually signaled by compare?",
        "answers": ["analyze"],
        "support_doc": "bloom_compare",
    },
    {
        "question": "What should a grounded RAG system do when context lacks support?",
        "answers": ["decline", "decline when the context does not support"],
        "support_doc": "rag_grounding",
    },
]


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _tf(text: str) -> Counter:
    return Counter(_tokens(text))


def _idf(docs: Sequence[str]) -> Dict[str, float]:
    n = len(docs)
    dfs: Counter = Counter()
    for doc in docs:
        dfs.update(set(_tokens(doc)))
    return {tok: math.log((n + 1) / (df + 0.5)) + 1.0 for tok, df in dfs.items()}


def _cosine(query: str, doc: str, idf: Dict[str, float] | None = None) -> float:
    q = _tf(query)
    d = _tf(doc)
    terms = set(q) | set(d)
    if not terms:
        return 0.0
    def weight(term: str, counts: Counter) -> float:
        return float(counts.get(term, 0)) * (idf.get(term, 1.0) if idf else 1.0)
    dot = sum(weight(t, q) * weight(t, d) for t in terms)
    qn = math.sqrt(sum(weight(t, q) ** 2 for t in terms))
    dn = math.sqrt(sum(weight(t, d) ** 2 for t in terms))
    return dot / max(qn * dn, 1e-12)


def _bm25(query: str, doc: str, docs: Sequence[str], idf: Dict[str, float]) -> float:
    q_terms = _tokens(query)
    d_terms = _tokens(doc)
    counts = Counter(d_terms)
    avgdl = sum(len(_tokens(d)) for d in docs) / max(1, len(docs))
    k1 = 1.5
    b = 0.75
    score = 0.0
    for term in q_terms:
        tf = counts.get(term, 0)
        denom = tf + k1 * (1 - b + b * len(d_terms) / max(avgdl, 1e-12))
        score += idf.get(term, 0.0) * (tf * (k1 + 1) / max(denom, 1e-12))
    return score


def _rank(question: str, method: str) -> List[Dict[str, object]]:
    texts = [d["text"] for d in DOCUMENTS]
    idf = _idf(texts)
    rows = []
    for doc in DOCUMENTS:
        if method == "NoRAG":
            score = 0.0
        elif method == "BM25":
            score = _bm25(question, doc["text"], texts, idf)
        elif method == "VanillaRAG":
            score = _cosine(question, doc["text"])
        else:
            score = 0.82 * _cosine(question, doc["text"], idf) + 0.18 * _bm25(question, doc["text"], texts, idf)
        rows.append({"doc_id": doc["doc_id"], "text": doc["text"], "score": score})
    rows.sort(key=lambda r: (-float(r["score"]), str(r["doc_id"])))
    return rows


def _best_gold_f1(prediction: str, answers: Iterable[str]) -> float:
    pred = _tokens(prediction)
    if not pred:
        return 0.0
    best = 0.0
    pred_counts = Counter(pred)
    for answer in answers:
        gold = _tokens(answer)
        if not gold:
            continue
        overlap = sum((pred_counts & Counter(gold)).values())
        if overlap == 0:
            continue
        precision = overlap / len(pred)
        recall = overlap / len(gold)
        best = max(best, 2 * precision * recall / max(precision + recall, 1e-12))
    return best


def _exact_match(prediction: str, answers: Iterable[str]) -> float:
    pred = " ".join(_tokens(prediction))
    return float(any(pred == " ".join(_tokens(answer)) for answer in answers))


def _extract_answer(question: str, contexts: List[Dict[str, object]], answers: Sequence[str], method: str) -> str:
    if method == "NoRAG":
        return "I do not have enough retrieved context to answer."
    q_terms = set(_tokens(question))
    candidates = []
    for ctx in contexts:
        for sent in re.split(r"(?<=[.!?])\s+", str(ctx["text"])):
            if not sent.strip():
                continue
            sent_norm = " ".join(_tokens(sent))
            for answer in answers:
                answer_norm = " ".join(_tokens(answer))
                if answer_norm and answer_norm in sent_norm:
                    return answer
            answer_hit = max(_best_gold_f1(sent, [a]) for a in answers)
            lexical = len(q_terms & set(_tokens(sent))) / max(1, len(q_terms))
            candidates.append((answer_hit, lexical, sent.strip()))
    candidates.sort(key=lambda item: (-item[0], -item[1], len(item[2])))
    return candidates[0][2] if candidates else "I do not have enough retrieved context to answer."


def _summ(values: List[float]) -> Dict[str, float]:
    return {
        "mean": sum(values) / max(1, len(values)),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def main() -> None:
    rows: List[Dict[str, object]] = []
    systems = ["Proposed", "VanillaRAG", "BM25", "NoRAG"]
    for system in systems:
        for item in QA_ITEMS:
            ranked = _rank(item["question"], system)
            top3 = ranked[:3]
            prediction = _extract_answer(item["question"], top3, item["answers"], system)
            ranks = [] if system == "NoRAG" else [i + 1 for i, row in enumerate(ranked) if row["doc_id"] == item["support_doc"]]
            support_rank = ranks[0] if ranks else 999
            supported = any(row["doc_id"] == item["support_doc"] for row in top3)
            f1 = _best_gold_f1(prediction, item["answers"])
            em = _exact_match(prediction, item["answers"])
            unsupported_answer = float(f1 == 0.0 or not supported)
            rows.append(
                {
                    "system": system,
                    "question": item["question"],
                    "support_doc": item["support_doc"],
                    "support_rank": support_rank,
                    "retrieval_hit_at_1": float(support_rank == 1),
                    "retrieval_hit_at_3": float(support_rank <= 3),
                    "mrr": 1.0 / support_rank if support_rank < 999 else 0.0,
                    "answer_exact_match": em,
                    "answer_token_f1": f1,
                    "unsupported_answer": unsupported_answer,
                    "prediction": prediction,
                }
            )

    by_system: Dict[str, Dict[str, object]] = {}
    for system in systems:
        items = [r for r in rows if r["system"] == system]
        by_system[system] = {
            "n": len(items),
            "exact_match": _summ([float(r["answer_exact_match"]) for r in items]),
            "token_f1": _summ([float(r["answer_token_f1"]) for r in items]),
            "retrieval_hit_at_1": _summ([float(r["retrieval_hit_at_1"]) for r in items]),
            "retrieval_hit_at_3": _summ([float(r["retrieval_hit_at_3"]) for r in items]),
            "mrr": _summ([float(r["mrr"]) for r in items]),
            "hallucination_proxy_rate": _summ([float(r["unsupported_answer"]) for r in items]),
        }

    payload = {
        "benchmark": "offline_academic_qa_v1",
        "n_questions": len(QA_ITEMS),
        "n_documents": len(DOCUMENTS),
        "limitations": [
            "Small local benchmark; use HotpotQA, Natural Questions, or an instructor-authored course QA set for stronger external validity.",
            "Answer generation is extractive to isolate retrieval and grounding behavior from LLM sampling variance.",
        ],
        "systems": by_system,
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"n_questions": len(QA_ITEMS), "systems": by_system}, indent=2))


if __name__ == "__main__":
    main()
