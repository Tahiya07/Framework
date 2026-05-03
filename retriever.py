"""
retriever.py
==============================================================================

Local retrieval layer for the privacy-constrained academic assistant.

The retriever supports two corpus representations:

* raw public text chunks, used for ordinary student-facing RAG over learning
  materials;
* safe metadata chunks, used when protected exam material should be indexed as
  an abstract semantic proxy rather than as reconstructable text.

This module does not claim formal privacy. It provides auditable, local
controls that can be evaluated with reconstruction and leakage probes.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import faiss  # type: ignore
import numpy as np

from encoder_backends import StableTextEncoder


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

logger = logging.getLogger("retriever")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class SafeDocumentChunk:
    chunk_id: int
    source: str
    modality: str
    bloom_level: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class RetrievalResult:
    rank: int
    doc_id: int
    text: str
    score: float = 0.0
    cosine: float = 0.0
    infonce_risk: float = 0.0
    privacy_score: float = 0.0
    l2_distance: float = 0.0
    bloom_level: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    final_score: float = 0.0


class PrivacyRetriever:
    """CPU-only dense retriever with a privacy-aware candidate re-ranker."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        normalize: bool = True,
        temperature: float = 0.07,
        lambda_privacy: float = 0.5,
        model: Optional[StableTextEncoder] = None,
        query_perturb_sigma: float = 0.0,
        query_perturb_seed_base: int = 42,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if lambda_privacy < 0:
            raise ValueError("lambda_privacy must be >= 0")
        if query_perturb_sigma < 0:
            raise ValueError("query_perturb_sigma must be >= 0")

        self.model_name = model_name
        self.normalize = bool(normalize)
        self.temperature = float(temperature)
        self.lambda_privacy = float(lambda_privacy)
        self.query_perturb_sigma = float(query_perturb_sigma)
        self.query_perturb_seed_base = int(query_perturb_seed_base)
        self._model = model

        self._index: Optional[faiss.Index] = None
        self._docs: List[Union[str, SafeDocumentChunk, object]] = []
        self._index_texts: List[str] = []
        self._return_texts: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._dim = 0

    @property
    def model(self) -> StableTextEncoder:
        if self._model is None:
            logger.info("Loading encoder: %s", self.model_name)
            self._model = StableTextEncoder(
                self.model_name,
                device="cpu",
                local_files_only=True,
                n_features=EMBED_DIM,
            )
        return self._model

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        ).astype(np.float32, copy=False)
        if emb.ndim == 1:
            emb = emb[None, :]
        if self.normalize and emb.size:
            norm = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norm, 1e-12, None)
        return np.ascontiguousarray(emb, dtype=np.float32)

    @staticmethod
    def _safe_chunk_text(chunk: SafeDocumentChunk) -> str:
        parts = [
            chunk.bloom_level or "",
            chunk.topic or "",
            chunk.difficulty or "",
            chunk.summary or "",
        ]
        text = " | ".join([p for p in parts if str(p).strip()])
        return text or f"{chunk.modality} chunk from {chunk.source}"

    def _document_texts(self, doc: Union[str, SafeDocumentChunk, object]) -> tuple[str, str]:
        if isinstance(doc, str):
            return doc, doc
        if isinstance(doc, SafeDocumentChunk):
            safe = self._safe_chunk_text(doc)
            return safe, safe

        text = str(getattr(doc, "text", "") or "").strip()
        access = str(getattr(doc, "access_level", "public") or "public").lower()
        content_type = str(getattr(doc, "content_type", "") or "").lower()

        if access == "protected" or content_type == "exam_paper":
            bloom = str(getattr(doc, "bloom_level", "") or "")
            topic = str(getattr(doc, "topic", "") or "")
            difficulty = str(getattr(doc, "difficulty", "") or "")
            summary = str(getattr(doc, "safe_summary", "") or "")
            safe = " | ".join([p for p in (bloom, topic, difficulty, summary) if p.strip()])
            return safe or "protected academic assessment item", text

        return text, text

    def build_index(self, documents: Sequence[Union[str, SafeDocumentChunk, object]]) -> None:
        if not documents:
            raise ValueError("documents must be a non-empty sequence")

        self._docs = list(documents)
        pairs = [self._document_texts(doc) for doc in self._docs]
        self._index_texts = [p[0] for p in pairs]
        self._return_texts = [p[1] for p in pairs]

        emb = self._encode(self._index_texts)
        self._embeddings = emb
        self._dim = int(emb.shape[1])

        index = faiss.IndexFlatL2(self._dim)
        index.add(emb)
        self._index = index
        logger.info("FAISS index built: N=%d, D=%d", len(self._docs), self._dim)

    def _perturb_query(self, query: str, q_emb: np.ndarray) -> np.ndarray:
        if self.query_perturb_sigma <= 0:
            return q_emb
        digest = hashlib.sha256(
            f"{self.query_perturb_seed_base}:{query}".encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False) % (2**32)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.query_perturb_sigma, size=q_emb.shape).astype(np.float32)
        out = q_emb + noise
        if self.normalize:
            out = out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-12, None)
        return np.ascontiguousarray(out, dtype=np.float32)

    def retrieve(
        self,
        query: Union[str, Sequence[str]],
        top_k: int = 5,
        candidate_pool: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if self._index is None or self._embeddings is None:
            raise RuntimeError("index not built")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        if isinstance(query, str):
            query_text = query
        else:
            if len(query) != 1:
                raise ValueError("Only single-query retrieval is supported")
            query_text = str(query[0])

        pool_n = min(len(self._docs), max(int(candidate_pool or top_k), int(top_k)))
        q_emb = self._perturb_query(query_text, self._encode([query_text]))
        distances, indices = self._index.search(q_emb, pool_n)

        cand_idx = [int(i) for i in indices[0] if int(i) >= 0]
        if not cand_idx:
            return []
        if len(self._return_texts) != len(self._docs):
            pairs = [self._document_texts(doc) for doc in self._docs]
            self._index_texts = [p[0] for p in pairs]
            self._return_texts = [p[1] for p in pairs]

        cand_emb = self._embeddings[cand_idx]
        cosine = (cand_emb @ q_emb[0]).astype(np.float64)
        logits = cosine / max(self.temperature, 1e-12)
        logits = logits - float(np.max(logits))
        exp_logits = np.exp(logits)
        infonce = exp_logits / max(float(np.sum(exp_logits)), 1e-12)

        rows = []
        for pos, doc_idx in enumerate(cand_idx):
            privacy_score = float(cosine[pos] - self.lambda_privacy * infonce[pos])
            rows.append(
                (
                    privacy_score,
                    float(cosine[pos]),
                    int(doc_idx),
                    float(infonce[pos]),
                    float(distances[0][pos]),
                )
            )
        rows.sort(key=lambda item: (-item[0], -item[1], item[2]))

        out: List[RetrievalResult] = []
        for rank, (privacy_score, cos, doc_idx, risk, dist) in enumerate(rows[:top_k], start=1):
            doc = self._docs[doc_idx]
            out.append(
                RetrievalResult(
                    rank=rank,
                    doc_id=doc_idx,
                    text=self._return_texts[doc_idx],
                    score=privacy_score,
                    cosine=cos,
                    infonce_risk=risk,
                    privacy_score=privacy_score,
                    l2_distance=dist,
                    bloom_level=getattr(doc, "bloom_level", None),
                    topic=getattr(doc, "topic", None),
                    difficulty=getattr(doc, "difficulty", None),
                )
            )
        return out


def _self_test() -> None:
    docs = [
        "Photosynthesis converts light into chemical energy.",
        "Bloom taxonomy includes remember, understand, apply, analyze, evaluate, create.",
        SafeDocumentChunk(2, "exam.pdf", "pdf", "Analyze", "circuits", "medium"),
    ]
    retr = PrivacyRetriever(temperature=0.07, lambda_privacy=0.1)
    retr.build_index(docs)
    out = retr.retrieve("What is photosynthesis?", top_k=2, candidate_pool=3)
    assert len(out) == 2
    assert all(r.text for r in out)
    print("[OK] retriever self-test passed")


if __name__ == "__main__":
    _self_test()
