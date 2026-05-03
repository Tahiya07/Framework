"""
ingestion.py
==============================================================================

Document ingestion for the local academic assistant.

Public learning material can be chunked as raw text for RAG. Protected exam
material is tagged at ingestion time so retrieval and generation layers can
route it through teacher-only workflows and output screening.
"""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union


random.seed(42)
try:
    import numpy as np

    np.random.seed(42)
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch

    torch.manual_seed(42)
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import fitz  # PyMuPDF

    _HAS_PYMUPDF = True
except Exception:  # pragma: no cover
    fitz = None
    _HAS_PYMUPDF = False

logger = logging.getLogger("ingestion")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class DocumentChunk:
    chunk_id: int
    source: str
    text: str
    page: Optional[int] = None
    modality: str = "text"
    access_level: str = "public"
    content_type: str = "study_material"
    bloom_level: Optional[str] = None
    bloom_confidence: float = 0.0
    ordinal_risk: float = 0.0
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    safe_summary: Optional[str] = None


@dataclass
class SafeDocumentChunk:
    chunk_id: int
    source: str
    modality: str
    bloom_level: str
    bloom_confidence: float
    ordinal_risk: float
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    summary: Optional[str] = None


@dataclass
class RawDocumentChunk:
    chunk_id: int
    source: str
    text: str
    page: Optional[int]
    modality: str


class DocumentIngestor:
    """Extract and chunk PDF, image, and text inputs for local RAG."""

    def __init__(
        self,
        classifier: Optional[Callable[[str], object]] = None,
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        normalize_whitespace: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        self.classifier = classifier
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.normalize_whitespace = bool(normalize_whitespace)

    def _normalize(self, text: str) -> str:
        text = (text or "").replace("\x00", " ")
        return re.sub(r"\s+", " ", text).strip() if self.normalize_whitespace else text

    def _split_text(self, text: str) -> List[str]:
        tokens = self._normalize(text).split()
        if not tokens:
            return []
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [
            " ".join(tokens[i : i + self.chunk_size])
            for i in range(0, len(tokens), step)
            if tokens[i : i + self.chunk_size]
        ]

    def chunk_text(
        self,
        text: str,
        *,
        source: str = "<text>",
        modality: str = "text",
        access_level: str = "public",
        content_type: str = "study_material",
        page: Optional[int] = None,
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        for idx, piece in enumerate(self._split_text(text)):
            chunks.append(
                DocumentChunk(
                    chunk_id=idx,
                    source=source,
                    text=piece,
                    page=page,
                    modality=modality,
                    access_level=access_level,
                    content_type=content_type,
                )
            )
        return chunks

    def _load_pdf_raw(self, path: Path) -> List[RawDocumentChunk]:
        if not _HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF is required for PDF ingestion")
        doc = fitz.open(str(path))
        chunks: List[RawDocumentChunk] = []
        cid = 0
        try:
            for page_i in range(len(doc)):
                text = self._normalize(doc[page_i].get_text("text") or "")
                for piece in self._split_text(text):
                    chunks.append(RawDocumentChunk(cid, str(path), piece, page_i + 1, "pdf"))
                    cid += 1
        finally:
            doc.close()
        return chunks

    def _load_image_text(self, path: Path) -> str:
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
        except Exception as exc:  # pragma: no cover
            try:
                import easyocr  # type: ignore
            except Exception as easy_exc:  # pragma: no cover
                raise RuntimeError(
                    "Image OCR requires pillow plus pytesseract or easyocr. "
                    "Install an OCR backend or upload extracted text."
                ) from easy_exc
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            return " ".join(str(piece) for piece in reader.readtext(str(path), detail=0))
        tesseract_cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
        if not tesseract_cmd:
            for candidate in [
                Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
                Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
            ]:
                if candidate.is_file():
                    tesseract_cmd = str(candidate)
                    break
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return str(pytesseract.image_to_string(Image.open(path)) or "")

    def process(
        self,
        path: Union[str, Path],
        *,
        access_level: str = "public",
        content_type: str = "study_material",
    ) -> List[DocumentChunk]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            raw_chunks = self._load_pdf_raw(path)
            return [
                DocumentChunk(
                    chunk_id=raw.chunk_id,
                    source=raw.source,
                    text=raw.text,
                    page=raw.page,
                    modality=raw.modality,
                    access_level=access_level,
                    content_type=content_type,
                )
                for raw in raw_chunks
            ]
        if suffix in {".txt", ".md", ".csv"}:
            return self.chunk_text(
                path.read_text(encoding="utf-8", errors="ignore"),
                source=str(path),
                modality="text",
                access_level=access_level,
                content_type=content_type,
            )
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            return self.chunk_text(
                self._load_image_text(path),
                source=str(path),
                modality="image",
                access_level=access_level,
                content_type=content_type,
            )
        raise ValueError(f"Unsupported file type: {path.suffix or '<none>'}")

    def _predict_bloom(self, text: str) -> tuple[Optional[str], float, float]:
        if self.classifier is None:
            return None, 0.0, 0.0
        predictor = getattr(self.classifier, "predict", self.classifier)
        pred = predictor(text)
        return (
            getattr(pred, "label", None),
            float(getattr(pred, "confidence", 0.0) or 0.0),
            float(getattr(pred, "ordinal_risk", 0.0) or 0.0),
        )

    def _to_safe(self, raw: RawDocumentChunk) -> SafeDocumentChunk:
        label, confidence, risk = self._predict_bloom(raw.text)
        return SafeDocumentChunk(
            chunk_id=raw.chunk_id,
            source=raw.source,
            modality=raw.modality,
            bloom_level=label or "Unknown",
            bloom_confidence=confidence,
            ordinal_risk=risk,
            summary=None,
        )

    def to_safe_chunks(self, raw_chunks: List[RawDocumentChunk]) -> List[SafeDocumentChunk]:
        return [self._to_safe(chunk) for chunk in raw_chunks]

    def process_pdf_safe(self, path: Union[str, Path]) -> List[SafeDocumentChunk]:
        return self.to_safe_chunks(self._load_pdf_raw(Path(path)))


def _self_test() -> None:
    ing = DocumentIngestor(chunk_size=5, chunk_overlap=1)
    chunks = ing.chunk_text("one two three four five six", source="demo")
    assert len(chunks) == 2
    assert chunks[0].text == "one two three four five"
    print("[OK] ingestion self-test passed")


if __name__ == "__main__":
    _self_test()
