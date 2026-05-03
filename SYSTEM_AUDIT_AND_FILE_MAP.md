# System Audit and File Map

## Sanity Check Summary

The current system implements a privacy-constrained educational RAG pipeline with separate evidence for QA/RAG quality, privacy screening, PDF RAG, image OCR, image RAG, and persistent FAISS vector storage.

Latest verified results:

- Privacy evaluation: 125 total rows, 104 student attack prompts, student attack block rate 1.000, benign student allow rate 0.867, teacher moderation allow rate 1.000.
- QA/RAG evaluation: 10 offline academic QA items; Proposed token-F1 0.973, exact match 0.900, retrieval hit@3 1.000, unsupported-answer proxy 0.000.
- OCR evaluation: Tesseract via pytesseract is available; synthetic typed-image OCR token-F1 0.963, word error rate 0.048, modality preservation 1.000, access-level preservation 1.000.
- Multimodal RAG smoke test: PDF RAG passed, image RAG passed, answer accuracy on successful cases 1.000.
- Qwen GGUF evaluation: `Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` is used through standalone llama.cpp CLI. The model file is 986.049 MB. Qwen academic QA token-F1 is 0.914, exact match is 0.800, retrieval hit@3 is 1.000, and unsupported-answer proxy is 0.000. Qwen PDF RAG and image RAG both pass.
- Vector database: FAISS index save/load self-test passed with persisted metadata and retrievable chunk text.

## Core Mechanism

The system separates public learning material from protected exam material. Public chunks are available to student-facing RAG. Protected chunks are available to teacher/moderator workflows and are screened before any student-facing output. The paper should frame this as a deployment pattern and evaluation protocol for privacy-constrained educational RAG, not as a new privacy algorithm or a formal privacy guarantee.

## File Responsibilities

`ingestion.py`

Implements document ingestion for text, PDF, and image files. Text and Markdown are chunked directly. PDFs are parsed with PyMuPDF. Images are OCRed with pytesseract/Tesseract when available, with EasyOCR as a fallback path. Each chunk keeps metadata: source, page, modality, access level, content type, Bloom metadata fields, and safe-summary fields.

`retriever.py`

Implements the FAISS-backed vector retriever. It supports ordinary text chunks and metadata-preserving document chunks. Protected exam chunks are indexed through safe metadata when appropriate. It now includes `save_vector_store()` and `load_vector_store()` so the vector database can be persisted locally with both `index.faiss` and `metadata.json`.

`privacy_guard.py`

Implements query and output screening. It detects reconstruction intent, protected artifact references, copied spans, n-gram containment, and semantic concept overlap. Teacher/moderator roles are allowed to access protected moderation workflows; students are blocked from reconstructing protected exam material.

`multimodal_rag.py`

Provides a lightweight local RAG wrapper for PDF, image, and text sources. It uses `DocumentIngestor` for file ingestion, keeps public/protected corpora separate, retrieves relevant chunks with a dependency-light lexical scorer, and applies `privacy_guard.py` before returning student-facing answers. This gives a concrete PDF/image RAG implementation even when the heavier LLM path is unavailable.

`qwen_gguf_cli.py`

Runs the required local Qwen model, `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`, through standalone llama.cpp CLI. This avoids the unavailable `llama-cpp-python` wheel on Python 3.13 while still evaluating the exact GGUF model. The configuration is CPU-only, `gpu_layers=0`, `ctx_size=512`, no weight repacking, and deterministic decoding with temperature 0.

`evaluate_qwen_rag.py`

Runs Qwen-generated evaluation over the offline academic QA set and the PDF/image RAG smoke cases. It writes `results/qwen_rag_eval.json` and `results/qwen_rag_eval_rows.csv`. These are the main generator-backed results; the extractive QA/RAG metrics remain diagnostic controls.

`models.py`

Contains the original Qwen GGUF generator intended for `llama-cpp-python`. The import is now optional so the app does not fail when the Python binding cannot be installed. On this machine, Qwen evaluation uses `qwen_gguf_cli.py` because the standalone llama.cpp runtime works and the Python binding requires a native compiler toolchain.

`streamlit_app.py`

Implements the interactive demo. Users can upload PDF, image, text, or Markdown files into either the public learning corpus or protected exam corpus. It now indexes full `DocumentChunk` objects instead of plain strings, preserving source/page/modality/access metadata. It saves public and protected vector databases under `data/vector_store/public` and `data/vector_store/protected`.

`evaluate_privacy_guard.py`

Runs the expanded role-aware privacy evaluation. It evaluates direct reconstruction, indirect leakage, paraphrase probes, partial-span extraction, jailbreak-style prompts, semantic reconstruction, benign student prompts, and teacher moderation prompts. Outputs are written to `results/privacy_guard_eval.json` and `results/privacy_guard_eval_rows.csv`.

`evaluate_qa_rag.py`

Runs a small offline academic QA/RAG benchmark with deterministic extractive controls. It reports answer exact match, token-F1, retrieval hit@1, retrieval hit@3, MRR, and unsupported-answer proxy. This fills the missing QA/RAG evidence gap, while remaining clearly scoped as a small local benchmark.

`evaluate_ocr_pipeline.py`

Runs OCR-specific evaluation on synthetic typed images. It verifies that image ingestion preserves modality and access-level metadata, and reports OCR token-F1 and word error rate. It discovers the Windows Tesseract path explicitly when PATH has not refreshed.

`evaluate_multimodal_rag.py`

Runs end-to-end smoke tests for PDF RAG and image RAG. It generates a small PDF and image, ingests them, retrieves from them, and checks whether the answer contains the expected evidence.

`consolidate_paper_results.py`

Collects Bloom, domain shift, privacy, QA/RAG, OCR, and multimodal RAG results into unified tables. It writes JSON, CSV, Markdown, and figure-table CSV outputs under `results/` and `figures/`.

`build_high_venue_paper.py`

Builds the evidence-aligned manuscript PDF and summary Markdown. It keeps the paper title unchanged and updates the claims, contributions, tables, QA/RAG section, privacy section, and OCR/image RAG section based on the latest result files.

`paper_draft.pdf`

Regenerated manuscript PDF built from the current evidence files.

`paper_high_venue.md`

Short evidence summary and claim-scope checklist generated with the paper.

## Evidence Interpretation

The strongest defensible contribution is not model novelty. The contribution is a deployment pattern plus an evaluation protocol for privacy-constrained educational RAG systems:

- role-separated public and protected corpora;
- student-facing leakage screening;
- teacher-only protected moderation;
- Bloom classifier evaluation framed as domain-shift insight;
- QA/RAG correctness and retrieval evaluation;
- Qwen2.5-1.5B-Instruct-Q4_K_M.gguf generator evaluation;
- OCR/image ingestion evaluation;
- persistent local vector database support.

The claims should remain conservative. The system demonstrates bounded empirical leakage resistance under the evaluated attack taxonomy, not formal privacy. The QA/RAG and OCR results are useful prototype evidence, not a full large-scale user or benchmark study.
