from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List
import os
import shutil


RESULTS_PATH = Path("results/ocr_image_pipeline_eval.json")
CSV_PATH = Path("results/ocr_image_pipeline_eval_rows.csv")


SAMPLES = [
    {
        "sample_id": "typed_exam_item",
        "text": "Q1 Explain Ohm's law and voltage current resistance.",
        "access_level": "protected",
    },
    {
        "sample_id": "typed_study_note",
        "text": "Merge sort recursively divides a list and runs in O n log n time.",
        "access_level": "public",
    },
    {
        "sample_id": "typed_statistics_note",
        "text": "A p value below alpha supports rejecting the null hypothesis.",
        "access_level": "public",
    },
]


def _tokens(text: str) -> List[str]:
    import re

    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _token_f1(prediction: str, gold: str) -> float:
    from runtime_utils import token_f1

    return token_f1(prediction, gold)


def _word_error_rate(prediction: str, gold: str) -> float:
    pred = _tokens(prediction)
    ref = _tokens(gold)
    if not ref:
        return 0.0
    prev = list(range(len(pred) + 1))
    for i, ref_tok in enumerate(ref, start=1):
        cur = [i] + [0] * len(pred)
        for j, pred_tok in enumerate(pred, start=1):
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + (0 if ref_tok == pred_tok else 1),
            )
        prev = cur
    return prev[-1] / len(ref)


def _make_image(path: Path, text: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1100, 180), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    draw.text((30, 55), text, fill="black", font=font)
    img.save(path)


def _ocr_available() -> Dict[str, object]:
    try:
        import PIL  # noqa: F401
    except Exception as exc:
        return {"available": False, "engine": "none", "reason": f"pillow_unavailable: {exc}"}
    try:
        import pytesseract

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
        pytesseract.get_tesseract_version()

        return {"available": True, "engine": "pytesseract", "reason": "ok", "command": tesseract_cmd}
    except Exception as exc:
        pytesseract_reason = str(exc)
    try:
        import easyocr  # noqa: F401

        return {"available": True, "engine": "easyocr", "reason": "ok"}
    except Exception as exc:
        return {"available": False, "engine": "none", "reason": f"ocr_backend_unavailable: pytesseract={pytesseract_reason}; easyocr={exc}"}


def main() -> None:
    rows: List[Dict[str, object]] = []
    status = _ocr_available()
    if status["available"]:
        from ingestion import DocumentIngestor

        ingestor = DocumentIngestor(chunk_size=128, chunk_overlap=0)
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            for sample in SAMPLES:
                image_path = tmpdir / f"{sample['sample_id']}.png"
                _make_image(image_path, sample["text"])
                chunks = ingestor.process(
                    image_path,
                    access_level=str(sample["access_level"]),
                    content_type="exam_paper" if sample["access_level"] == "protected" else "study_material",
                )
                extracted = " ".join(chunk.text for chunk in chunks)
                rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "engine": status["engine"],
                        "access_level_preserved": float(all(chunk.access_level == sample["access_level"] for chunk in chunks)),
                        "modality_preserved": float(all(chunk.modality == "image" for chunk in chunks)),
                        "token_f1": _token_f1(extracted, sample["text"]),
                        "word_error_rate": _word_error_rate(extracted, sample["text"]),
                        "n_chunks": len(chunks),
                        "gold_text": sample["text"],
                        "extracted_text": extracted,
                    }
                )

    def mean(key: str) -> float:
        vals = [float(row[key]) for row in rows]
        return sum(vals) / max(1, len(vals))

    payload = {
        "benchmark": "synthetic_image_ocr_pipeline_v1",
        "backend_status": status,
        "n_samples": len(rows),
        "metrics": {
            "mean_token_f1": mean("token_f1") if rows else None,
            "mean_word_error_rate": mean("word_error_rate") if rows else None,
            "access_level_preservation": mean("access_level_preserved") if rows else None,
            "modality_preservation": mean("modality_preserved") if rows else None,
        },
        "limitations": [
            "Synthetic typed images only; scanned handwriting, low-resolution photos, and DocVQA-style layouts remain future work.",
            "If backend_status.available is false, this run validates evaluator readiness but not OCR quality.",
        ],
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"backend_status": status, "n_samples": len(rows), "metrics": payload["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
