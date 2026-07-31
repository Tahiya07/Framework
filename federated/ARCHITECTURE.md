# Federated architecture (publication map)

## Primary claim — federatedly trained Bloom LoRA (Layer 2)

Federate **only** the Qwen2.5-0.5B Bloom taxonomy LoRA adapter + `score` head.

| Step | Script | Output |
|------|--------|--------|
| Partition (IID / Dirichlet non-IID) | `federated/partition.py` | Per-client CSV + label mix log |
| Local train (FedAvg or FedProx) | `federated/client_train.py` | Encrypted LoRA+score bundle |
| Server FedAvg | `federated/server_aggregate.py` | `models/qwen_bloom_federated0.5B_{tag}/` |
| Simulation | `federated/run_simulation.py` | `results/federated_lora_{tag}.json` |
| Merge + eval | `merge_model.py`, `evaluate_bloom.py` | Separate federated merged/eval paths |
| Comparison table | `build_federated_comparison.py` | `results/federated_bloom_comparison.*` |

**Architecture (must match centralized baseline):**

- Base: `Qwen/Qwen2.5-0.5B-Instruct` (frozen)
- LoRA: `r=32`, `alpha=64`, targets `q/k/v/o/gate/up/down_proj`
- `modules_to_save=["score"]`
- Prompt: shared `predict_bloom.build_prompt` ending in `Bloom Level:`

**Paper experiment matrix:**

1. Centralized LoRA (locked: ~84.86% / 84.21%) — do not overwrite
2. FedAvg + IID
3. FedProx (μ=0.01) + IID
4. FedAvg + non-IID (Dirichlet α=0.5)
5. FedProx (μ=0.01) + non-IID (Dirichlet α=0.5)

Primary runs are **from-scratch** (not warm-started from the centralized adapter).

**Transport:** bundles use XOR + SHA-256 integrity (`FEDERATED_UPDATE_KEY`). Prototype only — replace with TLS + secure aggregation for production.

**Privacy wording:** Federated training keeps raw client data local during collaborative optimization. It does **not** by itself provide formal protection against inference attacks on updates, secure aggregation, or differential privacy.

## Optional case study — privacy-risk FL (Layer 1)

- **Module:** `privacy/federated_privacy.py`
- **Train:** `python privacy/train_federated_privacy_guard.py`
- **Enable in stack:** `python federated/run_full_stack.py --with-privacy-guard`
- Hashed-feature logistic prototype — **not** the main federated Bloom contribution.

## Student side (not LLM-federated)

Student learning uses **local RAG** on **public** corpora only (`ingestion`, `retriever`, `models` / GGUF). Privacy is corpus isolation + `PrivacyGuard`, not federating student LoRA.

## Recommended commands

```bash
# Smoke (4 clients × 2 rounds, IID FedAvg)
python federated/run_simulation.py --clients 4 --rounds 2 --partition iid --algorithm fedavg --from-scratch

# Paper-scale cell example
python federated/run_simulation.py --clients 8 --rounds 5 --partition non_iid_label --alpha 0.5 --algorithm fedprox --prox-mu 0.01

# Merge + evaluate into SEPARATE paths (never overwrite centralized)
python merge_model.py --model-size 0.5b --lora-dir models/qwen_bloom_federated0.5B_fedavg_iid --output-dir models/qwen_bloom_federated0.5B_fedavg_iid_merged --force
python evaluate_bloom.py --model-size 0.5b --model-dir models/qwen_bloom_federated0.5B_fedavg_iid_merged --results-json results/federated_bloom_eval_fedavg_iid.json

# Comparison table
python build_federated_comparison.py
```
