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
from typing import List, Optional, Union


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

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_TEXT_SUFFIXES = {".txt", ".md", ".csv"}


def resolve_tesseract_cmd() -> Optional[str]:
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if cmd:
        return cmd
    for candidate in (
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def ocr_backend_status() -> dict[str, object]:
    """Report whether image OCR can run (Tesseract or easyocr)."""
    try:
        import PIL  # noqa: F401
    except Exception as exc:
        return {"available": False, "engine": "none", "reason": f"pillow_unavailable: {exc}"}

    tesseract_reason = "not_configured"
    cmd = resolve_tesseract_cmd()
    if cmd:
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = cmd
            pytesseract.get_tesseract_version()
            return {"available": True, "engine": "pytesseract", "reason": "ok", "command": cmd}
        except Exception as exc:
            tesseract_reason = str(exc)

    try:
        import easyocr  # noqa: F401

        return {"available": True, "engine": "easyocr", "reason": "ok", "command": None}
    except Exception as exc:
        return {
            "available": False,
            "engine": "none",
            "reason": f"tesseract={tesseract_reason}; easyocr={exc}",
            "install_hint": (
                "Install Tesseract OCR and add it to PATH, or set TESSERACT_CMD to tesseract.exe. "
                "Windows: winget install UB-Mannheim.TesseractOCR. "
                "Alternatively pip install easyocr, or upload PDF/TXT and paste text instead of images."
            ),
        }


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
        chunk_size: int = 256,
        chunk_overlap: int = 32,
        normalize_whitespace: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
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
        if not chunks:
            raise RuntimeError(
                f"No extractable text in {path.name}. Scanned/image-only PDFs need OCR — "
                "paste the text, or re-export the PDF with a text layer."
            )
        return chunks

    def _load_image_text_easyocr(self, path: Path) -> str:
        import easyocr  # type: ignore

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        pieces = reader.readtext(str(path), detail=0)
        return " ".join(str(piece) for piece in pieces)

    def _load_image_text_tesseract(self, path: Path) -> str:
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore

        cmd = resolve_tesseract_cmd()
        if not cmd:
            raise RuntimeError("tesseract executable not found")
        pytesseract.pytesseract.tesseract_cmd = cmd
        pytesseract.get_tesseract_version()
        return str(pytesseract.image_to_string(Image.open(path)) or "")

    def _load_image_text(self, path: Path) -> str:
        status = ocr_backend_status()
        if not status.get("available"):
            hint = status.get("install_hint") or status.get("reason")
            raise RuntimeError(f"Image OCR is unavailable. {hint}")

        engine = str(status.get("engine") or "")
        if engine == "pytesseract":
            try:
                return self._load_image_text_tesseract(path)
            except Exception as exc:
                logger.warning("Tesseract OCR failed for %s (%s); trying easyocr", path.name, exc)

        try:
            return self._load_image_text_easyocr(path)
        except Exception as exc:
            raise RuntimeError(
                f"Image OCR failed for {path.name}. Install Tesseract and add it to PATH "
                f"(or set TESSERACT_CMD), or pip install easyocr. Details: {exc}"
            ) from exc

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
            text = self._load_image_text(path)
            if not self._normalize(text):
                raise RuntimeError(
                    f"No readable text was extracted from image '{path.name}'. "
                    "Try a clearer scan, paste the text directly, or upload a PDF/TXT file."
                )
            return self.chunk_text(
                text,
                source=str(path),
                modality="image",
                access_level=access_level,
                content_type=content_type,
            )
        raise ValueError(f"Unsupported file type: {path.suffix or '<none>'}")


def _self_test() -> None:
    ing = DocumentIngestor(chunk_size=5, chunk_overlap=1)
    chunks = ing.chunk_text("one two three four five six", source="demo")
    assert len(chunks) == 2
    assert chunks[0].text == "one two three four five"
    print("[OK] ingestion self-test passed")


if __name__ == "__main__":
    _self_test()
