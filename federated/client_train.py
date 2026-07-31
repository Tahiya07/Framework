#!/usr/bin/env python
"""Local teacher-client Bloom LoRA training; exports encrypted parameter bundle only.

Matches centralized train_qwen_bloom.py architecture:
  LoRA r=32, alpha=64, modules_to_save=["score"], shared Bloom Level: prompt.

Supports FedProx: L = L_task + (μ/2) ||w - w_global||^2 over trainable params only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.config import (  # noqa: E402
    BLOOM_LABELS,
    FederatedLoraConfig,
    TEACHER_ROLE,
    make_peft_lora_config,
)
from federated.class_weights import resolve_class_weights  # noqa: E402
from federated.lora_state import (  # noqa: E402
    extract_trainable_state,
    trainable_nbytes,
    trainable_param_count,
)
from federated.secure_bundle import pack_update, save_bundle  # noqa: E402
from predict_bloom import build_prompt  # noqa: E402


class _BloomDS(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class _WeightedTrainer(Trainer):
    """Centralized BloomTrainer loss + optional FedProx proximal term."""

    def __init__(
        self,
        class_weights=None,
        label_smoothing: float = 0.0,
        prox_mu: float = 0.0,
        global_params: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.prox_mu = float(prox_mu)
        # CPU clones of global trainable tensors; moved to device on demand.
        self._global_params = {k: v.detach().cpu().clone() for k, v in (global_params or {}).items()}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None,
            label_smoothing=self.label_smoothing,
        )
        loss = loss_fn(logits, labels)

        if self.prox_mu > 0.0 and self._global_params:
            prox = loss.new_zeros(())
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                g_cpu = self._global_params.get(name)
                if g_cpu is None:
                    continue
                g = g_cpu.to(device=param.device, dtype=param.dtype)
                prox = prox + torch.sum((param - g) ** 2)
            loss = loss + 0.5 * self.prox_mu * prox

        return (loss, outputs) if return_outputs else loss


def _load_model_stack(config: FederatedLoraConfig, global_dir: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(BLOOM_LABELS),
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    if global_dir and (global_dir / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(base, str(global_dir), is_trainable=True)
        print(f"[client] warm-started from global adapter {global_dir}")
    else:
        model = get_peft_model(base, make_peft_lora_config(config))
        print(
            f"[client] fresh LoRA r={config.lora_r} alpha={config.lora_alpha} "
            f"modules_to_save={config.modules_to_save}"
        )
    return tokenizer, model


def train_local_adapter(
    df: pd.DataFrame,
    config: FederatedLoraConfig,
    global_dir: Path | None,
) -> tuple[dict, int, dict]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    tokenizer, model = _load_model_stack(config, global_dir)
    global_state = extract_trainable_state(model)
    # FedProx anchors must use named_parameters keys (exact match during compute_loss).
    global_named = {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    texts = [build_prompt(str(q)) for q in df[config.text_col]]
    labels = [BLOOM_LABELS[str(l)] for l in df[config.label_col]]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=config.max_length)
    ds = _BloomDS(enc, labels)

    warm_started = bool(global_dir and (global_dir / "adapter_config.json").is_file())
    lr = config.finetune_learning_rate if warm_started else config.learning_rate
    if warm_started:
        print(f"[client] warm-start finetune lr={lr}")

    class_weights = resolve_class_weights(config, labels)
    if class_weights is not None:
        print(
            f"[client] class weights ({config.class_weight_source}):",
            [round(float(w), 3) for w in class_weights.tolist()],
        )

    prox_mu = float(config.prox_mu) if config.algorithm == "fedprox" else 0.0
    if prox_mu > 0:
        print(f"[client] FedProx μ={prox_mu}")

    args = TrainingArguments(
        output_dir=str(ROOT / "federated" / "_client_cache"),
        learning_rate=lr,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.local_epochs,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        seed=config.seed,
    )

    trainer = _WeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        class_weights=class_weights,
        label_smoothing=config.label_smoothing,
        prox_mu=prox_mu,
        global_params=global_named if prox_mu > 0 else None,
    )
    trainer.train()

    local_state = extract_trainable_state(model)
    stats = {
        "trainable_parameters": trainable_param_count(local_state),
        "update_bytes": trainable_nbytes(local_state),
        "prox_mu": prox_mu,
        "algorithm": config.algorithm,
    }
    return local_state, len(df), stats


def _configure_cpu_threads() -> None:
    """Avoid OpenMP/BLAS oversubscription on CPU hosts (can be 10× slower)."""
    import os

    n = int(os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or "8")
    n = max(1, min(n, os.cpu_count() or 8))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(max(1, min(4, n)))
    except RuntimeError:
        pass


def main() -> int:
    _configure_cpu_threads()
    parser = argparse.ArgumentParser(description="Federated teacher client local LoRA training.")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--csv", required=True, help="Client-local CSV (question, bloom_level).")
    parser.add_argument("--global-adapter", default=None)
    parser.add_argument("--out-bundle", required=True)
    parser.add_argument("--no-encrypt", action="store_true")
    parser.add_argument("--local-epochs", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--prox-mu", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config-json", default=None, help="Optional JSON overrides from the simulator.")
    args = parser.parse_args()

    cfg = FederatedLoraConfig()
    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        for key, value in payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    if args.local_epochs is not None:
        cfg.local_epochs = args.local_epochs
    if args.base_model:
        cfg.base_model = args.base_model
    if args.seed is not None:
        cfg.seed = args.seed
    cfg.algorithm = args.algorithm
    if args.prox_mu is not None:
        cfg.prox_mu = args.prox_mu

    df = pd.read_csv(args.csv).dropna()
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=cfg.seed)

    global_dir = Path(args.global_adapter) if args.global_adapter else None
    local_state, n, stats = train_local_adapter(df, cfg, global_dir)

    bundle = pack_update(
        client_id=args.client_id,
        round_idx=args.round,
        role=TEACHER_ROLE,
        n_samples=n,
        state=local_state,
        encrypt=not args.no_encrypt,
    )
    bundle["trainable_parameters"] = stats["trainable_parameters"]
    bundle["update_bytes"] = stats["update_bytes"]
    bundle["prox_mu"] = stats["prox_mu"]
    bundle["algorithm"] = stats["algorithm"]
    save_bundle(Path(args.out_bundle), bundle)
    print(
        f"[client] saved bundle -> {args.out_bundle} "
        f"(n={n}, params={stats['trainable_parameters']}, bytes={stats['update_bytes']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
