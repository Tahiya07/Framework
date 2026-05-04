from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from ingestion import DocumentChunk, DocumentIngestor
from privacy.privacy_guard import STUDENT_REFUSAL, screen_generation_output


SUPPORTED_RAG_TYPES = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".txt",
    ".md",
}


@dataclass
class RAGCitation:
    source: str
    page: int | None
    modality: str
    chunk_id: int
    score: float
    text: str


@dataclass
class RAGAnswer:
    answer: str
    citations: List[RAGCitation] = field(default_factory=list)
    refused: bool = False
    refusal_reason: str = ""


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _idf(chunks: Sequence[DocumentChunk]) -> Dict[str, float]:
    dfs: Counter = Counter()
    for chunk in chunks:
        dfs.update(set(_tokens(chunk.text)))
    n = max(1, len(chunks))
    return {tok: math.log((n + 1) / (df + 0.5)) + 1.0 for tok, df in dfs.items()}


def _score(query: str, chunk: DocumentChunk, idf: Dict[str, float]) -> float:
    q = Counter(_tokens(query))
    d = Counter(_tokens(chunk.text))
    if not q or not d:
        return 0.0
    terms = set(q) | set(d)
    dot = sum(q[t] * d[t] * idf.get(t, 1.0) for t in terms)
    qn = math.sqrt(sum((q[t] * idf.get(t, 1.0)) ** 2 for t in terms))
    dn = math.sqrt(sum((d[t] * idf.get(t, 1.0)) ** 2 for t in terms))
    lexical_overlap = len(set(q) & set(d)) / max(1, len(set(q)))
    return float((dot / max(qn * dn, 1e-12)) + 0.15 * lexical_overlap)


class MultiModalAcademicRAG:
    """Small local RAG wrapper for PDF, image, and text ingestion.

    This class is intentionally extractive and dependency-light. It gives the
    paper and demo a concrete PDF/image RAG surface even when the heavier FAISS
    retriever or LLM generator is not available. Image files still require an
    OCR backend through ``DocumentIngestor``; when OCR is unavailable, ingestion
    raises a clear runtime error instead of silently pretending to be multimodal.
    """

    def __init__(self, *, chunk_size: int = 220, chunk_overlap: int = 32) -> None:
        self.ingestor = DocumentIngestor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.public_chunks: List[DocumentChunk] = []
        self.protected_chunks: List[DocumentChunk] = []

    @property
    def chunks(self) -> List[DocumentChunk]:
        return self.public_chunks + self.protected_chunks

    def ingest_path(
        self,
        path: str | Path,
        *,
        access_level: str = "public",
        content_type: str = "study_material",
    ) -> List[DocumentChunk]:
        path = Path(path)
        if path.suffix.lower() not in SUPPORTED_RAG_TYPES:
            raise ValueError(f"Unsupported RAG file type: {path.suffix or '<none>'}")
        chunks = self.ingestor.process(path, access_level=access_level, content_type=content_type)
        target = self.protected_chunks if access_level == "protected" else self.public_chunks
        target.extend(chunks)
        return chunks

    def ingest_text(
        self,
        text: str,
        *,
        source: str = "<text>",
        access_level: str = "public",
        content_type: str = "study_material",
    ) -> List[DocumentChunk]:
        chunks = self.ingestor.chunk_text(
            text,
            source=source,
            modality="text",
            access_level=access_level,
            content_type=content_type,
        )
        target = self.protected_chunks if access_level == "protected" else self.public_chunks
        target.extend(chunks)
        return chunks

    def retrieve(
        self,
        query: str,
        *,
        role: str = "student",
        top_k: int = 4,
        include_protected_for_teacher: bool = True,
    ) -> List[RAGCitation]:
        role_lc = role.lower().strip()
        searchable = list(self.public_chunks)
        if role_lc in {"teacher", "moderator", "admin"} and include_protected_for_teacher:
            searchable.extend(self.protected_chunks)
        if not searchable:
            return []
        idf = _idf(searchable)
        ranked = [
            RAGCitation(
                source=chunk.source,
                page=chunk.page,
                modality=chunk.modality,
                chunk_id=chunk.chunk_id,
                score=_score(query, chunk, idf),
                text=chunk.text,
            )
            for chunk in searchable
            if chunk.text.strip()
        ]
        ranked.sort(key=lambda row: (-row.score, row.source, row.chunk_id))
        return ranked[: max(1, int(top_k))]

    def answer(self, query: str, *, role: str = "student", top_k: int = 4) -> RAGAnswer:
        citations = self.retrieve(query, role=role, top_k=top_k)
        if not citations:
            return RAGAnswer("I do not have enough indexed context to answer.", [])
        answer = self._extractive_answer(query, citations)
        decision = screen_generation_output(role, query, answer, self.protected_chunks)
        if not decision.allowed:
            return RAGAnswer(STUDENT_REFUSAL, citations, refused=True, refusal_reason=decision.reason)
        return RAGAnswer(answer, citations)

    @staticmethod
    def _extractive_answer(query: str, citations: Sequence[RAGCitation]) -> str:
        q_terms = set(_tokens(query))
        candidates: List[tuple[float, int, str]] = []
        for citation in citations:
            for sentence in re.split(r"(?<=[.!?])\s+", citation.text):
                sentence = sentence.strip()
                if not sentence:
                    continue
                s_terms = set(_tokens(sentence))
                overlap = len(q_terms & s_terms) / max(1, len(q_terms))
                candidates.append((overlap + citation.score, -len(sentence), sentence))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        selected = [item[2] for item in candidates[:2]]
        return " ".join(selected) if selected else citations[0].text


def _self_test() -> None:
    rag = MultiModalAcademicRAG(chunk_size=32, chunk_overlap=4)
    rag.ingest_text("Ohm's law states that voltage equals current times resistance.", source="note.txt")
    out = rag.answer("What does Ohm's law relate?")
    assert "voltage" in out.answer.lower()
    assert out.citations and out.citations[0].source == "note.txt"
    print("[OK] multimodal RAG self-test passed")


if __name__ == "__main__":
    _self_test()
