"""
retriever.py
==============================================================================

Privacy-preserving FAISS retriever aligned with:

- ingestion.py (SafeDocumentChunk only)
- classifier.py (Bloom labels)
- uncertainty.py (optional scoring hooks)

Key upgrade:
------------------------------------------------------------
1. Indexes SAFE structured chunks (not raw text)
2. Uses metadata-aware retrieval representation
3. Compatible with Bloom classifier outputs
4. Supports future uncertainty-aware reranking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import faiss  # type: ignore

from encoder_backends import StableTextEncoder

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384

logger = logging.getLogger("retriever")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Input structure (MATCHES ingestion.py)
# -----------------------------------------------------------------------------
@dataclass
class SafeDocumentChunk:
    chunk_id: int
    source: str
    modality: str

    bloom_level: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None

    summary: Optional[str] = None


# -----------------------------------------------------------------------------
# Retrieval output
# -----------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    rank: int
    doc_id: int
    text: str
    score: float

    # publishable metadata (important for paper)
    bloom_level: Optional[str] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None


# -----------------------------------------------------------------------------
# Retriever
# -----------------------------------------------------------------------------
class PrivacyRetriever:

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        normalize: bool = True,
        model: Optional[StableTextEncoder] = None,
    ) -> None:

        self.model_name = model_name
        self.normalize = normalize
        self._model = model

        self._index = None
        self._docs: List[SafeDocumentChunk] = []
        self._dim = 0

    # -------------------------------------------------------------------------
    @property
    def model(self) -> StableTextEncoder:
        if self._model is None:
            logger.info(f"Loading encoder: {self.model_name}")
            self._model = StableTextEncoder(
                self.model_name,
                device="cpu",
                local_files_only=True,
                n_features=EMBED_DIM,
            )
        return self._model

    # -------------------------------------------------------------------------
    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        ).astype(np.float32)

        if emb.ndim == 1:
            emb = emb[None, :]

        return np.ascontiguousarray(emb)

    # -------------------------------------------------------------------------
    def _chunk_to_text(self, c: SafeDocumentChunk) -> str:
        """
        Metadata-aware embedding representation (IMPORTANT upgrade)

        This is where your publishable contribution lives:
        - NOT raw text
        - structured semantic proxy
        """

        parts = [
            c.bloom_level or "",
            c.topic or "",
            c.difficulty or "",
            c.summary or "",
        ]

        return " | ".join([p for p in parts if p])

    # -------------------------------------------------------------------------
    def build_index(self, documents: Sequence[SafeDocumentChunk]) -> None:
        if not documents:
            raise ValueError("Empty document list")

        self._docs = list(documents)

        texts = [self._chunk_to_text(d) for d in self._docs]
        emb = self._encode(texts)

        self._dim = emb.shape[1]

        index = faiss.IndexFlatIP(self._dim)
        index.add(emb)

        self._index = index

        logger.info(
            f"FAISS index built: N={len(self._docs)}, D={self._dim}"
        )

    # -------------------------------------------------------------------------
    def retrieve(
        self,
        query: Union[str, Sequence[str]],
        top_k: int = 5,
    ) -> List[RetrievalResult]:

        if self._index is None:
            raise RuntimeError("Index not built")

        if isinstance(query, str):
            query = [query]

        if len(query) != 1:
            raise ValueError("Only single-query retrieval supported")

        q_emb = self._encode(query)

        if self.normalize:
            q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)

        scores, idx = self._index.search(q_emb, top_k)

        idx = idx[0]
        scores = scores[0]

        results: List[RetrievalResult] = []

        for rank, (i, s) in enumerate(zip(idx, scores), start=1):
            doc = self._docs[int(i)]

            results.append(
                RetrievalResult(
                    rank=rank,
                    doc_id=doc.chunk_id,
                    text=self._chunk_to_text(doc),
                    score=float(s),

                    bloom_level=doc.bloom_level,
                    topic=doc.topic,
                    difficulty=doc.difficulty,
                )
            )

        return results


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def _self_test():
    docs = [
        SafeDocumentChunk(0, "a", "pdf", "Apply", "Algebra", "Medium"),
        SafeDocumentChunk(1, "a", "pdf", "Analyze", "Physics", "Hard"),
        SafeDocumentChunk(2, "a", "pdf", "Understand", "ML", "Easy"),
    ]

    r = PrivacyRetriever()
    r.build_index(docs)

    out = r.retrieve("neural networks", top_k=2)

    assert len(out) == 2
    assert out[0].score >= out[1].score

    print("✔ retriever aligned with safe ingestion pipeline")


if __name__ == "__main__":
    _self_test()