"""
ingestion.py (PRIVACY-ENHANCED + MODEL-INTEGRATED VERSION)
==============================================================================

Key properties:
------------------------------------------------------------
1. Strict raw vs safe separation (no leakage into retrieval index)
2. Bloom classifier-driven semantic abstraction
3. Retrieval-safe document representation only
4. CPU-only deterministic pipeline
5. Compatible with classifier.py + uncertainty.py
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

try:
    import torch
    torch.manual_seed(42)
except Exception:
    torch = None


# ---------------------------------------------------------------------
# Optional PDF support
# ---------------------------------------------------------------------
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    fitz = None
    _HAS_PYMUPDF = False


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logger = logging.getLogger("ingestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------
# SAFE representation (ONLY goes to retrieval system)
# ---------------------------------------------------------------------
@dataclass
class SafeDocumentChunk:
    chunk_id: int
    source: str
    modality: str

    # semantic metadata (derived, NOT raw text)
    bloom_level: str
    bloom_confidence: float
    ordinal_risk: float

    topic: Optional[str] = None
    difficulty: Optional[str] = None


# ---------------------------------------------------------------------
# RAW internal representation (NEVER leaves ingestion module)
# ---------------------------------------------------------------------
@dataclass
class RawDocumentChunk:
    chunk_id: int
    source: str
    text: str
    page: Optional[int]
    modality: str


# ---------------------------------------------------------------------
# Ingestion system
# ---------------------------------------------------------------------
class DocumentIngestor:
    """
    Privacy-preserving ingestion pipeline:
    - Raw text is processed internally
    - Only semantic abstractions are exported
    """

    def __init__(
        self,
        classifier: Callable[[str], object],
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        normalize_whitespace: bool = True,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap >= chunk_size:
            raise ValueError("invalid overlap")

        self.classifier = classifier  # <<< KEY FIX
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.normalize_whitespace = normalize_whitespace

    # ---------------------------------------------------------
    def _normalize(self, text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # ---------------------------------------------------------
    def _chunk(self, text: str) -> List[str]:
        tokens = text.split()
        step = self.chunk_size - self.chunk_overlap

        return [
            " ".join(tokens[i:i + self.chunk_size])
            for i in range(0, len(tokens), step)
        ]

    # ---------------------------------------------------------
    # RAW PDF ingestion (internal only)
    # ---------------------------------------------------------
    def _load_pdf_raw(self, path: Path) -> List[RawDocumentChunk]:
        if not _HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF required")

        doc = fitz.open(str(path))
        chunks: List[RawDocumentChunk] = []
        cid = 0

        try:
            for i in range(len(doc)):
                text = doc[i].get_text("text") or ""

                if self.normalize_whitespace:
                    text = self._normalize(text)

                for piece in self._chunk(text):
                    chunks.append(
                        RawDocumentChunk(
                            chunk_id=cid,
                            source=str(path),
                            text=piece,
                            page=i + 1,
                            modality="pdf",
                        )
                    )
                    cid += 1
        finally:
            doc.close()

        return chunks

    # ---------------------------------------------------------
    # Semantic abstraction (THIS IS THE KEY CONTRIBUTION NOW)
    # ---------------------------------------------------------
    def _to_safe(self, raw: RawDocumentChunk) -> SafeDocumentChunk:
        """
        Converts raw text → Bloom-aware semantic representation.
        """

        pred = self.classifier.predict(raw.text)

        return SafeDocumentChunk(
            chunk_id=raw.chunk_id,
            source=raw.source,
            modality=raw.modality,

            bloom_level=pred.label,
            bloom_confidence=float(pred.confidence),
            ordinal_risk=float(pred.ordinal_risk),

            topic=None,
            difficulty=None,
        )

    # ---------------------------------------------------------
    def to_safe_chunks(
        self,
        raw_chunks: List[RawDocumentChunk],
    ) -> List[SafeDocumentChunk]:

        return [self._to_safe(c) for c in raw_chunks]

    # ---------------------------------------------------------
    # PUBLIC API (SAFE ONLY)
    # ---------------------------------------------------------
    def process_pdf_safe(self, path: Union[str, Path]) -> List[SafeDocumentChunk]:
        path = Path(path)

        raw = self._load_pdf_raw(path)
        safe = self.to_safe_chunks(raw)

        return safe


# ---------------------------------------------------------------------
# SELF TEST
# ---------------------------------------------------------------------
def _self_test():

    # dummy classifier (replace with your BloomClassifier)
    class Dummy:
        def predict(self, x):
            class P:
                label = "Apply"
                confidence = 0.8
                ordinal_risk = 0.2
            return P()

    ing = DocumentIngestor(classifier=Dummy(), chunk_size=20, chunk_overlap=5)

    raw = [
        RawDocumentChunk(0, "x", "this is a sample exam question", 1, "pdf")
    ]

    safe = ing.to_safe_chunks(raw)

    assert safe[0].bloom_level == "Apply"
    assert safe[0].bloom_confidence > 0

    print("✔ ingestion pipeline OK (privacy + classifier integrated)")


if __name__ == "__main__":
    _self_test()