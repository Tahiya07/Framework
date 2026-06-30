"""Shared helpers for student/teacher RAG quality control."""

from __future__ import annotations

import re
from typing import List, Sequence

from retriever import RetrievalResult

# Chunks below this cosine similarity are dropped before generation.
DEFAULT_MIN_COSINE = 0.22

_REFUSAL = "I don't know based on the provided context."


def filter_relevant_chunks(
    chunks: Sequence[RetrievalResult],
    *,
    min_cosine: float = DEFAULT_MIN_COSINE,
) -> List[RetrievalResult]:
    """Keep only chunks that are semantically close to the query."""
    if not chunks:
        return []
    kept = [c for c in chunks if float(c.cosine) >= float(min_cosine)]
    return list(kept) if kept else [max(chunks, key=lambda c: float(c.cosine))]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _normalize(text)))


def chunk_copy_ratio(answer: str, chunk_texts: Sequence[str]) -> float:
    """Fraction of answer tokens that appear in the best-matching source chunk."""
    ans = _token_set(answer)
    if not ans:
        return 0.0
    best = 0.0
    for chunk in chunk_texts:
        cset = _token_set(chunk)
        if not cset:
            continue
        overlap = len(ans & cset) / len(ans)
        best = max(best, overlap)
    return best


def sanitize_rag_answer(
    answer: str,
    chunk_texts: Sequence[str],
    *,
    max_copy_ratio: float = 0.88,
) -> str:
    """Strip verbatim chunk dumps; refuse when the model only pasted context."""
    text = (answer or "").strip()
    if not text:
        return _REFUSAL

    # Remove numbered context echoes like "[1] ..." or "Context:" blocks.
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\[\d+\]\s", stripped):
            continue
        if stripped.lower().startswith(("context:", "bounded context", "[bounded context]")):
            continue
        lines.append(stripped)
    if lines:
        text = " ".join(lines).strip()
    else:
        text = re.sub(r"^\[\d+\]\s*", "", text).strip()
    text = re.sub(r"\[\d+\]\s*", "", text).strip()

    ratio = chunk_copy_ratio(text, chunk_texts)
    # Refuse only when the model dumped a long passage with almost no rewriting.
    if ratio >= max_copy_ratio and len(_token_set(text)) >= 20:
        return _REFUSAL
    return text


def trim_chunks_for_context(
    chunks: Sequence[RetrievalResult],
    *,
    max_chars_per_chunk: int = 700,
    max_total_chars: int = 2800,
) -> List[RetrievalResult]:
    """Trim long chunks so the LLM prompt fits and stays focused."""
    out: List[RetrievalResult] = []
    total = 0
    for chunk in chunks:
        body = (chunk.text or "").strip()
        if len(body) > max_chars_per_chunk:
            cut = body[:max_chars_per_chunk]
            for sep in (". ", ".\n", "\n"):
                j = cut.rfind(sep)
                if j > max_chars_per_chunk // 3:
                    body = cut[: j + len(sep)].strip()
                    break
            else:
                body = cut.strip()
        if total + len(body) > max_total_chars and out:
            break
        out.append(
            RetrievalResult(
                rank=chunk.rank,
                doc_id=chunk.doc_id,
                text=body,
                score=chunk.score,
                cosine=chunk.cosine,
                infonce_risk=chunk.infonce_risk,
                privacy_score=chunk.privacy_score,
                l2_distance=chunk.l2_distance,
                bloom_level=chunk.bloom_level,
                topic=chunk.topic,
                difficulty=chunk.difficulty,
                final_score=chunk.final_score,
            )
        )
        total += len(body)
    return out
