from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Dict, List

from multimodal_rag import MultiModalAcademicRAG


RESULTS_PATH = Path("results/multimodal_rag_eval.json")
CSV_PATH = Path("results/multimodal_rag_eval_rows.csv")


def _contains(answer: str, expected: str) -> float:
    return float(expected.lower() in answer.lower())


def _make_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawString(72, 720, "Ohm's law states that voltage equals current times resistance.")
    c.drawString(72, 700, "Merge sort divides a list and runs in O(n log n) time.")
    c.save()


def _make_image(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1100, 180), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    draw.text((28, 60), "TCP provides reliable ordered delivery.", fill="black", font=font)
    img.save(path)


def _run_pdf_case(tmpdir: Path) -> Dict[str, object]:
    pdf_path = tmpdir / "study_packet.pdf"
    _make_pdf(pdf_path)
    rag = MultiModalAcademicRAG(chunk_size=64, chunk_overlap=8)
    chunks = rag.ingest_path(pdf_path, access_level="public", content_type="study_material")
    out = rag.answer("What is the time complexity of merge sort?", role="student", top_k=2)
    return {
        "case": "pdf_rag",
        "status": "ok",
        "n_chunks": len(chunks),
        "answer_contains_expected": _contains(out.answer, "O(n log n)"),
        "top_modality": out.citations[0].modality if out.citations else "",
        "top_source_suffix": Path(out.citations[0].source).suffix.lower() if out.citations else "",
        "refused": float(out.refused),
        "answer": out.answer,
    }


def _run_image_case(tmpdir: Path) -> Dict[str, object]:
    image_path = tmpdir / "network_note.png"
    _make_image(image_path)
    rag = MultiModalAcademicRAG(chunk_size=64, chunk_overlap=8)
    try:
        chunks = rag.ingest_path(image_path, access_level="public", content_type="study_material")
        out = rag.answer("Which protocol provides reliable ordered delivery?", role="student", top_k=2)
        return {
            "case": "image_rag",
            "status": "ok",
            "n_chunks": len(chunks),
            "answer_contains_expected": _contains(out.answer, "TCP"),
            "top_modality": out.citations[0].modality if out.citations else "",
            "top_source_suffix": Path(out.citations[0].source).suffix.lower() if out.citations else "",
            "refused": float(out.refused),
            "answer": out.answer,
        }
    except Exception as exc:
        return {
            "case": "image_rag",
            "status": "backend_unavailable",
            "n_chunks": 0,
            "answer_contains_expected": 0.0,
            "top_modality": "",
            "top_source_suffix": ".png",
            "refused": 0.0,
            "answer": str(exc),
        }


def main() -> None:
    rows: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for runner in [_run_pdf_case, _run_image_case]:
            try:
                rows.append(runner(tmpdir))
            except Exception as exc:
                rows.append(
                    {
                        "case": runner.__name__.replace("_run_", "").replace("_case", ""),
                        "status": "error",
                        "n_chunks": 0,
                        "answer_contains_expected": 0.0,
                        "top_modality": "",
                        "top_source_suffix": "",
                        "refused": 0.0,
                        "answer": str(exc),
                    }
                )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    payload = {
        "benchmark": "multimodal_rag_smoke_v1",
        "n_cases": len(rows),
        "n_ok": len(ok_rows),
        "pdf_rag_ok": any(r["case"] == "pdf_rag" and r["status"] == "ok" for r in rows),
        "image_rag_ok": any(r["case"] == "image_rag" and r["status"] == "ok" for r in rows),
        "answer_accuracy_on_ok_cases": (
            sum(float(r["answer_contains_expected"]) for r in ok_rows) / max(1, len(ok_rows))
        ),
        "limitations": [
            "PDF RAG uses native text extraction from a generated text PDF.",
            "Image RAG requires a working OCR backend; backend-unavailable is reported explicitly.",
        ],
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: payload[k] for k in ["pdf_rag_ok", "image_rag_ok", "answer_accuracy_on_ok_cases"]}, indent=2))


if __name__ == "__main__":
    main()
