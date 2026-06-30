from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TEACHER_ROLE = "teacher"
STUDENT_ROLE = "student"

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_GLOBAL_LORA = ROOT / "models" / "qwen_bloom_federated"
DEFAULT_LORA_FALLBACK = ROOT / "models" / "qwen_bloom_3000"
UPDATES_DIR = ROOT / "federated" / "updates"
BUNDLES_DIR = ROOT / "federated" / "bundles"

BLOOM_LABELS = {
    "Remember": 0,
    "Understand": 1,
    "Apply": 2,
    "Analyze": 3,
    "Evaluate": 4,
    "Create": 5,
}

# Shared LoRA target modules. MUST be identical on every client and on the
# server aggregator, otherwise adapter shapes diverge and FedAvg breaks.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


@dataclass
class FederatedLoraConfig:
    base_model: str = DEFAULT_BASE_MODEL
    global_adapter_dir: str = str(DEFAULT_GLOBAL_LORA)
    num_clients: int = 4
    rounds: int = 3
    # Local training recipe aligned with the centralized trainer
    # (train_qwen_bloom.py) so federated vs centralized is a fair comparison.
    # NOTE on stability: the SEQ_CLS head is randomly initialized, so each
    # client's *first* round trains a fresh head on a small shard. A 1e-4 peak
    # over a short 2-epoch cosine cycle overshoots (grad explosion -> loss spike
    # -> poisoned FedAvg). A 5e-5 peak + longer warmup + larger effective batch
    # converges stably; rounds 2+ warm-start from the aggregate and refine.
    local_epochs: float = 3.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    label_smoothing: float = 0.05
    use_class_weights: bool = True
    max_grad_norm: float = 1.0
    max_length: int = 256
    batch_size: int = 2
    # Effective batch = batch_size * grad_accum = 16 (matches centralized);
    # larger batch denoises the gradient direction on size-2 micro-batches.
    grad_accum: int = 8
    max_samples_per_client: int = 400
    client_fraction: float = 1.0
    clip_norm: float = 1.0
    dp_noise: float = 0.0
    seed: int = 42
    train_csv: str = str(ROOT / "data" / "figshare_bloom_v1_train.csv")
    text_col: str = "question"
    label_col: str = "bloom_level"
