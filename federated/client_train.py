#!/usr/bin/env python
"""Local teacher-client Bloom LoRA training; exports encrypted parameter bundle only."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.config import BLOOM_LABELS, FederatedLoraConfig, LORA_TARGET_MODULES, TEACHER_ROLE  # noqa: E402
from federated.lora_state import extract_trainable_state  # noqa: E402
from federated.secure_bundle import pack_update, save_bundle  # noqa: E402


def build_prompt(question: str) -> str:
    return (
        "Classify Bloom's Taxonomy level.\n"
        "Focus on reasoning depth, not verbs.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


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
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=LORA_TARGET_MODULES,
        )
        model = get_peft_model(base, lora_cfg)
    return tokenizer, model


def train_local_adapter(
    df: pd.DataFrame,
    config: FederatedLoraConfig,
    global_dir: Path | None,
) -> tuple[dict, int]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    tokenizer, model = _load_model_stack(config, global_dir)
    global_state = extract_trainable_state(model)

    texts = [build_prompt(str(q)) for q in df[config.text_col]]
    labels = [BLOOM_LABELS[str(l)] for l in df[config.label_col]]
    enc = tokenizer(texts, truncation=True, padding=True, max_length=config.max_length)
    ds = _BloomDS(enc, labels)

    args = TrainingArguments(
        output_dir=str(ROOT / "federated" / "_client_cache"),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.local_epochs,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    local_state = extract_trainable_state(model)
    return local_state, len(df)


def main() -> int:
    parser = argparse.ArgumentParser(description="Federated teacher client local LoRA training.")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--csv", required=True, help="Client-local CSV (question, bloom_level).")
    parser.add_argument("--global-adapter", default=None)
    parser.add_argument("--out-bundle", required=True)
    parser.add_argument("--no-encrypt", action="store_true")
    parser.add_argument("--local-epochs", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    cfg = FederatedLoraConfig(local_epochs=args.local_epochs)
    df = pd.read_csv(args.csv).dropna()
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=cfg.seed)

    global_dir = Path(args.global_adapter) if args.global_adapter else None
    local_state, n = train_local_adapter(df, cfg, global_dir)

    bundle = pack_update(
        client_id=args.client_id,
        round_idx=args.round,
        role=TEACHER_ROLE,
        n_samples=n,
        state=local_state,
        encrypt=not args.no_encrypt,
    )
    save_bundle(Path(args.out_bundle), bundle)
    print(f"[client] saved encrypted bundle -> {args.out_bundle} (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
