# Lightweight Multi-Modal Tiny LLM Framework

Privacy-preserving academic assistance for university environments: student RAG over study materials and teacher Bloom question moderation, with role-aware privacy controls and federated privacy-risk training.

## Quick start

See **[RUNBOOK.md](RUNBOOK.md)** for a full list of runnable scripts and recommended command sequences.

```bash
pip install -r requirements.txt

# Train Bloom LoRA (teacher labels)
python train_qwen_bloom.py

# Option A — centralized LoRA
python train_qwen_bloom.py
python merge_model.py

# Option B — federated teacher Bloom LoRA (matches architecture diagram)
python federated/run_simulation.py
python merge_model.py   # auto-picks models/qwen_bloom_federated

# Full stack: federated LoRA + merge + privacy guard + evaluation
python federated/run_full_stack.py

# Run publication evaluation pipeline
python run_evaluation_pipeline.py

# Enable federated LoRA inside the eval pipeline
set RUN_FEDERATED_LORA=1
python run_evaluation_pipeline.py

# Interactive demo
streamlit run streamlit_app.py
```

## Evaluation pipeline (publication)

| Step | Script | Output |
|------|--------|--------|
| Bloom LoRA + optional SVM baseline | `evaluate_bloom.py` | `results/bloom_lora_eval.json` |
| Student RAG (FAISS + GGUF) | `evaluate_qwen_rag.py` | `results/qwen_rag_eval.json` |
| Multimodal smoke (PDF/image) | `evaluate_multimodal_rag.py` | `results/multimodal_rag_eval.json` |
| OCR readiness | `evaluate_ocr_pipeline.py` | `results/ocr_image_pipeline_eval.json` |
| Privacy guard | `privacy/evaluate_privacy_guard.py` | `results/privacy_guard_eval.json` |
| Privacy baselines | `privacy/evaluate_privacy_benchmarks.py` | `results/privacy_benchmark_baselines.json` |
| Federated privacy model | `privacy/train_federated_privacy_guard.py` | `results/federated_privacy_guard.json` |
| Unified table | `consolidate_paper_results.py` | `results/unified_results_table.{json,csv,md}` |
| Figures | `generate_paper_figures.py` | `figures/` |

Bloom **labels** use the trained LoRA (`predict_bloom.py`). Teacher **reason / rewrite** text uses local GGUF generation (`bloom_prompt.py`).

## Core modules

- `ingestion.py` — PyMuPDF, Tesseract OCR, text chunking
- `retriever.py` — all-MiniLM + FAISS, public vs protected corpora
- `models.py` — Qwen2.5-1.5B-Instruct GGUF RAG generation
- `predict_bloom.py` — Qwen LoRA Bloom classification
- `bloom_prompt.py` — teacher moderation narrative (GGUF)
- `privacy/` — PrivacyGuard + federated risk model

## Environment

- Offline: `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`
- Cap Bloom test size for smoke runs: `EVAL_BLOOM_MAX_TEST=50 python run_evaluation_pipeline.py`

git clone https://github.com/ggml-org/llama.cpp.git