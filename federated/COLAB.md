# Federated Bloom LoRA on GPU (Colab / Kaggle)

Train the privacy-preserving federated adapter on a free GPU, then deploy merged/quantized models on your local CPU machine.

## Why GPU here?

- Federated local training on a 1.5B model needs many optimizer steps per client.
- CPU (~30 s/step) → many hours per round; GPU (T4) → minutes per client.
- **Training on GPU + inference on CPU** matches the architecture: server-side FL aggregation, lightweight offline deployment.

## Colab (recommended)

1. **Runtime → Change runtime type → T4 GPU**
2. Upload this repo as a zip, or clone from GitHub:

```python
# Cell 1 — get the code
!git clone <YOUR_REPO_URL> Framework
%cd Framework
```

Or upload `Framework.zip` and unzip:

```python
!unzip -q Framework.zip
%cd Framework
```

3. Install dependencies (Colab already has CUDA PyTorch):

```python
# Cell 2 — install
!pip install -q transformers==4.41.0 peft==0.11.1 accelerate==0.33.0 \
    pandas scikit-learn matplotlib safetensors huggingface-hub
```

4. Verify GPU + run the full pipeline:

```python
# Cell 3 — train federated + merge + test eval + zip adapter
!python federated/run_gpu_pipeline.py --clients 4 --rounds 3 --eval-each-round
```

5. Download artifacts:

```python
# Cell 4 — download
from google.colab import files
files.download("results/qwen_bloom_federated_adapter.zip")
files.download("results/federated_lora_simulation.json")
```

## Kaggle

1. Create a notebook with **GPU** accelerator (P100/T4).
2. Add this repo as a Kaggle dataset or upload files.
3. Same install + run commands as Colab.
4. Download `results/qwen_bloom_federated_adapter.zip` from the output panel.

## Local deploy (after download)

On your Windows machine with `.venv` activated:

```powershell
# Unzip adapter into models/
Expand-Archive -Force results\qwen_bloom_federated_adapter.zip -DestinationPath models\

# Merge for inference
python merge_model.py --lora-dir models/qwen_bloom_federated --output-dir models/qwen_bloom_federated_merged --force

# Evaluate on held-out test (compare vs centralized 84%)
python evaluate_bloom.py --model_dir models/qwen_bloom_federated_merged --eval_csv data/figshare_bloom_v1_test.csv
```

`predict_bloom.py` and `streamlit_app.py` will auto-prefer the federated adapter when present.

## Smoke test (cheap)

Before a full 4×3 run:

```bash
python federated/run_simulation.py --clients 2 --rounds 1 --eval-each-round
```

Watch client `train_loss` — it should fall toward **< 1.0** (not stall at ~6).

## Paper table (three rows)

| Setting | Command | Test CSV |
|---------|---------|----------|
| SVM baseline | `python evaluate_bloom.py --svm-baseline` | official test |
| Centralized LoRA | `evaluate_bloom.py --model_dir models/qwen_bloom_merged` | `figshare_bloom_v1_test.csv` |
| Federated LoRA | `evaluate_bloom.py --model_dir models/qwen_bloom_federated_merged` | same |

Report accuracy, macro-F1, within-one-level accuracy, and model size.
