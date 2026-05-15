from __future__ import annotations

import csv
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from evaluate_qa_rag import DOCUMENTS, QA_ITEMS, _best_gold_f1, _exact_match, _rank, _tokens
from evaluate_multimodal_rag import _make_image, _make_pdf
from multimodal_rag import MultiModalAcademicRAG
from multi_slm import task_registry_report
from qwen_gguf_cli import QwenGgufCliGenerator


RESULTS_PATH = Path("results/qwen_rag_eval.json")
CSV_PATH = Path("results/qwen_rag_eval_rows.csv")


def _contains(answer: str, expected: str) -> float:
    return float(expected.lower() in answer.lower())


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def _summ(values: List[float]) -> Dict[str, float]:
    return {
        "mean": _mean(values),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def _run_academic_qa(qwen: QwenGgufCliGenerator) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    docs_by_id = {doc["doc_id"]: doc["text"] for doc in DOCUMENTS}
    for item in QA_ITEMS:
        ranked = _rank(item["question"], "Proposed")
        top3 = ranked[:3]
        contexts = [str(row["text"]) for row in top3]
        gen = qwen.generate(item["question"], contexts)
        ranks = [i + 1 for i, row in enumerate(ranked) if row["doc_id"] == item["support_doc"]]
        support_rank = ranks[0] if ranks else 999
        supported = support_rank <= 3
        f1 = _best_gold_f1(gen.answer, item["answers"])
        em = _exact_match(gen.answer, item["answers"])
        rows.append(
            {
                "task": "academic_qa",
                "question": item["question"],
                "support_doc": item["support_doc"],
                "support_rank": support_rank,
                "retrieval_hit_at_1": float(support_rank == 1),
                "retrieval_hit_at_3": float(supported),
                "answer_exact_match": em,
                "answer_token_f1": f1,
                "unsupported_answer": float(f1 == 0.0 or not supported),
                "latency_s": gen.elapsed_s,
                "host_total_mib": gen.memory.get("host_total_mib", ""),
                "model_file_mb": gen.model_file_bytes / 1_000_000,
                "slm_task": gen.task_id,
                "slm_model_path": gen.model_path,
                "prediction": gen.answer,
                "gold_answers": json.dumps(item["answers"]),
                "support_text": docs_by_id[item["support_doc"]],
            }
        )
    return rows


def _run_multimodal(
    pdf_slm: QwenGgufCliGenerator,
    image_slm: QwenGgufCliGenerator,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        pdf_path = tmpdir / "study_packet.pdf"
        image_path = tmpdir / "network_note.png"
        _make_pdf(pdf_path)
        _make_image(image_path)

        rag = MultiModalAcademicRAG(chunk_size=64, chunk_overlap=8)
        pdf_chunks = rag.ingest_path(pdf_path, access_level="public", content_type="study_material")
        image_chunks = rag.ingest_path(image_path, access_level="public", content_type="study_material")

        cases = [
            {
                "task": "pdf_rag",
                "question": "What is the time complexity of merge sort?",
                "expected": "O(n log n)",
                "modality": "pdf",
                "n_chunks": len(pdf_chunks),
            },
            {
                "task": "image_rag",
                "question": "Which protocol provides reliable ordered delivery?",
                "expected": "TCP",
                "modality": "image",
                "n_chunks": len(image_chunks),
            },
        ]
        for case in cases:
            citations = rag.retrieve(case["question"], role="student", top_k=2)
            contexts = [citation.text for citation in citations]
            slm = pdf_slm if case["task"] == "pdf_rag" else image_slm
            gen = slm.generate(case["question"], contexts)
            rows.append(
                {
                    "task": case["task"],
                    "question": case["question"],
                    "expected": case["expected"],
                    "answer_contains_expected": _contains(gen.answer, case["expected"]),
                    "top_modality": citations[0].modality if citations else "",
                    "n_chunks": case["n_chunks"],
                    "latency_s": gen.elapsed_s,
                    "host_total_mib": gen.memory.get("host_total_mib", ""),
                    "model_file_mb": gen.model_file_bytes / 1_000_000,
                    "slm_task": gen.task_id,
                    "slm_model_path": gen.model_path,
                    "prediction": gen.answer,
                }
            )
    return rows


def main() -> None:
    qa_slm = QwenGgufCliGenerator.for_task("academic_qa", max_tokens=32, ctx_size=512, threads=2)
    pdf_slm = QwenGgufCliGenerator.for_task("pdf_rag", max_tokens=32, ctx_size=512, threads=2)
    image_slm = QwenGgufCliGenerator.for_task("image_rag", max_tokens=32, ctx_size=512, threads=2)
    qa_rows = _run_academic_qa(qa_slm)
    mm_rows = _run_multimodal(pdf_slm, image_slm)
    rows = qa_rows + mm_rows

    payload = {
        "benchmark": "qwen_rag_eval_v1",
        "architecture": "multi_slm_task_specialists",
        "slm_registry": task_registry_report(),
        "backend": "llama.cpp-cli-cpu-gguf",
        "runtime_config": {
            "qa_ctx_size": qa_slm.ctx_size,
            "qa_max_tokens": qa_slm.max_tokens,
            "threads": qa_slm.threads,
            "device": "none",
            "gpu_layers": 0,
            "repack": False,
        },
        "academic_qa": {
            "n": len(qa_rows),
            "exact_match": _summ([float(r["answer_exact_match"]) for r in qa_rows]),
            "token_f1": _summ([float(r["answer_token_f1"]) for r in qa_rows]),
            "retrieval_hit_at_1": _summ([float(r["retrieval_hit_at_1"]) for r in qa_rows]),
            "retrieval_hit_at_3": _summ([float(r["retrieval_hit_at_3"]) for r in qa_rows]),
            "hallucination_proxy_rate": _summ([float(r["unsupported_answer"]) for r in qa_rows]),
            "latency_s": _summ([float(r["latency_s"]) for r in qa_rows]),
        },
        "multimodal_rag": {
            "n": len(mm_rows),
            "pdf_rag_ok": any(r["task"] == "pdf_rag" and float(r["answer_contains_expected"]) == 1.0 for r in mm_rows),
            "image_rag_ok": any(r["task"] == "image_rag" and float(r["answer_contains_expected"]) == 1.0 for r in mm_rows),
            "answer_accuracy": _summ([float(r["answer_contains_expected"]) for r in mm_rows]),
            "latency_s": _summ([float(r["latency_s"]) for r in mm_rows]),
        },
        "limitations": [
            "Qwen is run from Qwen2.5-1.5B-Instruct-Q4_K_M.gguf using standalone llama.cpp CLI because llama-cpp-python could not be built without a native compiler toolchain.",
            "The GGUF file is below 1 GB; runtime host memory may exceed 1 GB because llama.cpp still allocates context and compute buffers.",
            "The benchmark remains small; use a larger course QA set or HotpotQA/NQ subset for stronger publication evidence.",
        ],
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "architecture": payload["architecture"],
                "academic_qa": payload["academic_qa"],
                "multimodal_rag": payload["multimodal_rag"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
