# Runbook — scripts to execute

All commands assume the project root and an activated venv:

```powershell
cd C:\Users\tahiy\PycharmProjects\Framework
.\.venv\Scripts\Activate.ps1
```

Install once: `pip install -r requirements.txt`

Required on disk: `models/qwen.gguf`, trained LoRA under `models/qwen_bloom_3000/` (or federated output), and datasets referenced by training/eval scripts.

---

## Quick paths

| Goal | Command |
|------|---------|
| Interactive demo | `streamlit run streamlit_app.py` |
| Architecture compliance | `python architecture_compliance.py` |
| Train Bloom LoRA (centralized) | `python train_qwen_bloom.py` |
| Merge LoRA for inference | `python merge_model.py` |
| Full publication eval | `python run_evaluation_pipeline.py` |
| Federated LoRA + eval | `python federated/run_full_stack.py` |
| Bloom baselines (SVM / zero-shot / LoRA) | `python bloom_evaluation.py` then `python build_bloom_comparison.py` |

Optional env vars for eval:

- `RUN_FEDERATED_LORA=1` — run FL before merge in `run_evaluation_pipeline.py`
- `EVAL_BLOOM_MAX_TEST=50` — cap Bloom test size for smoke runs

---

## 1. Demo

| Script | Command |
|--------|---------|
| `streamlit_app.py` | `streamlit run streamlit_app.py` |

---

## 2. Architecture check

| Script | Command | Output |
|--------|---------|--------|
| `architecture_compliance.py` | `python architecture_compliance.py` | `results/architecture_compliance.json` |

---

## 3. Training and model prep

| Script | Command |
|--------|---------|
| `train_qwen_bloom.py` | `python train_qwen_bloom.py` |
| `merge_model.py` | `python merge_model.py` |
| `privacy/train_federated_privacy_guard.py` | `python privacy/train_federated_privacy_guard.py` |

---

## 4. Federated learning

| Script | Command |
|--------|---------|
| `federated/run_simulation.py` | `python federated/run_simulation.py` |
| `federated/run_full_stack.py` | `python federated/run_full_stack.py` |

Called internally by `run_simulation.py` (not usually run alone): `federated/partition.py`, `federated/client_train.py`, `federated/server_aggregate.py`.

---

## 5. Evaluation pipeline (orchestrator)

`python run_evaluation_pipeline.py` runs, in order:

1. `merge_model.py`
2. `evaluate_bloom.py` (`--svm-baseline`)
3. `build_bloom_comparison.py`
4. `evaluate_qwen_rag.py` (optional if GGUF unavailable)
5. `evaluate_multimodal_rag.py`
6. `evaluate_ocr_pipeline.py`
7. `privacy/train_federated_privacy_guard.py`
8. `privacy/evaluate_privacy_guard.py`
9. `privacy/evaluate_privacy_benchmarks.py`
10. `consolidate_paper_results.py`
11. `generate_paper_figures.py`

---

## 6. Standalone evaluation

| Script | Typical output |
|--------|----------------|
| `bloom_evaluation.py` | `evaluation_outputs/`, `results/bloom_baseline_comparison.json` |
| `evaluate_bloom.py` | `results/bloom_lora_eval.json` |
| `evaluate_qwen_rag.py` | `results/qwen_rag_eval.json` |
| `evaluate_multimodal_rag.py` | `results/multimodal_rag_eval.json` |
| `evaluate_ocr_pipeline.py` | `results/ocr_image_pipeline_eval.json` |
| `privacy/evaluate_privacy_guard.py` | `results/privacy_guard_eval.json` |
| `privacy/evaluate_privacy_benchmarks.py` | `results/privacy_benchmark_baselines.json` |
| `build_bloom_comparison.py` | `results/bloom_baseline_comparison.json` |
| `consolidate_paper_results.py` | `results/unified_results_table.*` |
| `generate_paper_figures.py` | `figures/` |

---

## 7. Optional CLI / smoke tests

| Script | Command |
|--------|---------|
| `predict_bloom.py` | `python predict_bloom.py` |
| `bloom_prompt.py` | `python bloom_prompt.py` |
| `ingestion.py` | `python ingestion.py` |
| `retriever.py` | `python retriever.py` |
| `models.py` | `python models.py` |
| `summarizer.py` | `python summarizer.py` |
| `multimodal_rag.py` | `python multimodal_rag.py` |

---

## 8. CSE 400 report (LaTeX)

Compile `CSE_400_Report/main.tex` in Overleaf or with `pdflatex`. See `CSE_400_Report/OVERLEAF_COMPILE.md` if present.

---

## Library modules (import only)

`role_access.py`, `runtime_utils.py`, `encoder_backends.py`, `multi_slm.py`, `uncertainty.py`, `privacy/privacy_guard.py`, `privacy/federated_privacy.py`, `federated/config.py`, `federated/secure_bundle.py`, `federated/lora_state.py`, `evaluate_qa_rag.py`, `qwen_gguf_cli.py`.
