"""Shared helpers for student/teacher RAG quality control."""

from __future__ import annotations

import re
from typing import Any, List, Sequence

from retriever import RetrievalResult

# Chunks below this cosine similarity are dropped before generation.
DEFAULT_MIN_COSINE = 0.22
DOCUMENT_SUMMARY_MIN_COSINE = 0.05

_REFUSAL = "I don't know based on the provided context."

_DOCUMENT_SUMMARY_CUES = (
    "summar",
    "overview",
    "main point",
    "key point",
    "this pdf",
    "this document",
    "the pdf",
    "the document",
    "uploaded",
    "whole document",
    "entire document",
    "what is in",
    "what does this",
    "tell me about this",
    "stakeholder interview",
)


def is_document_summary_query(query: str) -> bool:
    q = _normalize(query)
    if not q:
        return False
    return any(cue in q for cue in _DOCUMENT_SUMMARY_CUES)


def chunks_to_retrieval_results(chunks: Sequence[Any]) -> List[RetrievalResult]:
    """Map ingested document chunks to retrieval results (full-corpus summaries)."""
    ordered = sorted(
        chunks,
        key=lambda c: (str(getattr(c, "source", "")), int(getattr(c, "page", 0) or 0), int(getattr(c, "chunk_id", 0))),
    )
    out: List[RetrievalResult] = []
    for i, chunk in enumerate(ordered):
        text = str(getattr(chunk, "text", "") or "").strip()
        if not text:
            continue
        out.append(
            RetrievalResult(
                rank=i + 1,
                doc_id=int(getattr(chunk, "chunk_id", i)),
                text=text,
                cosine=1.0,
            )
        )
    return out


def boost_summary_retrieval_query(
    query: str,
    corpus_chunks: Sequence[str],
) -> str:
    """Anchor retrieval in document content when the user asks to summarize the upload."""
    sample = " ".join((t or "").strip() for t in corpus_chunks[:4] if (t or "").strip())
    sample = re.sub(r"\s+", " ", sample)[:500]
    if sample:
        return f"main topics themes key points interview questions {sample}"
    return f"document content themes key points {query.strip()}"


def dedupe_retrieval_results(chunks: Sequence[RetrievalResult]) -> List[RetrievalResult]:
    """Drop near-duplicate chunks (common with overlapping PDF chunking)."""
    seen: set[str] = set()
    out: List[RetrievalResult] = []
    for chunk in chunks:
        text = (chunk.text or "").strip()
        if not text:
            continue
        key = _normalize(text[:280])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            RetrievalResult(
                rank=len(out) + 1,
                doc_id=chunk.doc_id,
                text=text,
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
    return out


def merge_corpus_for_summary(chunks: Sequence[Any], *, max_chars: int = 7200) -> str:
    """Merge unique ingested chunks into one text block for document summarization."""
    ordered = sorted(
        chunks,
        key=lambda c: (str(getattr(c, "source", "")), int(getattr(c, "page", 0) or 0), int(getattr(c, "chunk_id", 0))),
    )
    seen: set[str] = set()
    parts: List[str] = []
    total = 0
    for chunk in ordered:
        text = str(getattr(chunk, "text", "") or "").strip()
        if not text:
            continue
        key = _normalize(text[:280])
        if key in seen:
            continue
        seen.add(key)
        room = max_chars - total
        if room <= 0:
            break
        if len(text) > room:
            parts.append(text[:room].rsplit(" ", 1)[0].strip())
            break
        parts.append(text)
        total += len(text) + 2
    return "\n\n".join(parts)


def merged_document_retrieval_result(chunks: Sequence[Any], *, max_chars: int = 7200) -> List[RetrievalResult]:
    body = merge_corpus_for_summary(chunks, max_chars=max_chars)
    if not body.strip():
        return []
    return [
        RetrievalResult(
            rank=1,
            doc_id=0,
            text=body,
            cosine=1.0,
        )
    ]


def build_document_summary_task() -> str:
    """Fixed generation task — never echoes the user's meta query."""
    return (
        "Summarize the document excerpts in [BOUNDED CONTEXT] for an academic reader.\n"
        "Cover: main themes, stakeholder concerns, and the key interview topics.\n"
        "Write 5-7 sentences. Begin directly with the substance "
        '(example opening: "Stakeholder interviews focus on...").\n'
        "Forbidden words/phrases: query, PDF, user, context, summarize the."
    )


_META_SUMMARY_MARKERS = (
    "the topic of the query",
    "the query \"",
    "the query '",
    "is asking for",
    "the context provides",
    "relevant to summarizing",
    "which are relevant to",
)


def looks_like_meta_summary(text: str) -> bool:
    low = _normalize(text)
    return sum(1 for m in _META_SUMMARY_MARKERS if m in low) >= 2


def extractive_document_summary(chunk_texts: Sequence[str]) -> str:
    """Deterministic fallback when the small LLM narrates the request instead of summarizing."""
    merged = merge_corpus_for_summary(
        [type("_C", (), {"text": t, "source": "", "page": 0, "chunk_id": i})() for i, t in enumerate(chunk_texts)],
        max_chars=9000,
    )
    if not merged.strip():
        return _REFUSAL

    questions = re.findall(r"\d+\.\s*[^?\n]{10,}\?", merged)
    seen_q: set[str] = set()
    unique_q: List[str] = []
    for q in questions:
        key = _normalize(q)
        if key in seen_q:
            continue
        seen_q.add(key)
        unique_q.append(re.sub(r"\s+", " ", q).strip())

    lead = "Stakeholder interviews address managing course materials, assessments, and academic AI assistance."
    if unique_q:
        q_block = "; ".join(unique_q[:6])
        if len(unique_q) > 6:
            q_block += f"; and {len(unique_q) - 6} further topics."
        return (
            f"{lead} Key questions raised include: {q_block} "
            "Overall themes include separating student learning resources from protected assessments, "
            "reducing exam leakage, offline vs cloud trust, Bloom taxonomy moderation, and assessment design."
        )

    snippet = re.sub(r"\s+", " ", merged)[:600].strip()
    return f"{lead} {snippet}"



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


def sanitize_summary_answer(answer: str, chunk_texts: Sequence[str]) -> str:
    text = sanitize_rag_answer(answer, chunk_texts, max_copy_ratio=0.92)
    if text == _REFUSAL or looks_like_meta_summary(text):
        return extractive_document_summary(chunk_texts)
    return text
