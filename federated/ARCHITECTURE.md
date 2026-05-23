# Federated architecture (implementation map)

## Layer 1 — Federated privacy guard (existing)

- **Module:** `privacy/federated_privacy.py`
- **Train:** `python privacy/train_federated_privacy_guard.py`
- **Role:** Hashed-feature logistic; server sees clipped/noised weight deltas only.
- **Runtime:** `privacy/privacy_guard.py` blocks reconstruction-style student queries.

## Layer 2 — Federated teacher Bloom LoRA (new)

| Step | Script | Output |
|------|--------|--------|
| Partition data by synthetic site | `federated/partition.py` | Per-client CSV under `federated/updates/` |
| Local train | `federated/client_train.py` | Encrypted bundle JSON |
| Server FedAvg | `federated/server_aggregate.py` | `models/qwen_bloom_federated/` |
| Full simulation | `federated/run_simulation.py` | `results/federated_lora_simulation.json` |
| Merge for inference | `merge_model.py` | `models/qwen_bloom_merged/` |

**Transport:** bundles use XOR + SHA-256 integrity (`FEDERATED_UPDATE_KEY` env). Replace with TLS + secure aggregation in production.

## Layer 3 — Student side (not LLM-federated)

Student learning uses **local RAG** (`ingestion`, `retriever`, `models`) on **public** corpora only. Privacy is enforced by corpus isolation + `PrivacyGuard`, not by federating student LoRA in this prototype.

## Recommended command sequence

```bash
python federated/run_simulation.py --clients 4 --rounds 3 --local-epochs 1
python merge_model.py
python privacy/train_federated_privacy_guard.py
python run_evaluation_pipeline.py
```
