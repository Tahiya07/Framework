#!/usr/bin/env python

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Dict, List

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)

from peft import LoraConfig, TaskType, get_peft_model


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# LABELS
# ============================================================

LABELS = {
    "Remember": 0,
    "Understand": 1,
    "Apply": 2,
    "Analyze": 3,
    "Evaluate": 4,
    "Create": 5,
}


# ============================================================
# DATASET
# ============================================================

@dataclass
class BloomDataset(torch.utils.data.Dataset):
    encodings: Dict[str, List[int]]
    labels: List[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ============================================================
# AUGMENTATION (SAFE)
# ============================================================

def augment(text):
    if random.random() < 0.2:
        text = text.replace("explain", "describe")
    if random.random() < 0.2:
        text = text.replace("analyze", "examine")
    return text


# ============================================================
# PROMPT
# ============================================================

def build_prompt(q):
    return (
        "Classify Bloom's Taxonomy level.\n"
        f"Question: {q}\n"
        "Answer:"
    )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


# ============================================================
# LOSS-FREE TRAINER (stable)
# ============================================================

class BloomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
            if self.class_weights is not None else None,
            label_smoothing=0.05
        )

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, default="data/figshare_bloom_v1.csv")
    parser.add_argument("--text_col", type=str, default="question")
    parser.add_argument("--label_col", type=str, default="bloom_level")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="models/qwen_bloom_3000")

    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=10)

    # ✅ CHANGED: 3000 SAMPLE LIMIT
    parser.add_argument("--sample_size", type=int, default=3000)

    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = pd.read_csv(args.csv).dropna()

    # ✔ enforce 3000 cap safely
    if len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=42)

    df["label"] = df[args.label_col].map(LABELS)

    df = df.dropna(subset=["label"])  # safety fix

    texts = [build_prompt(augment(t)) for t in df[args.text_col]]
    labels = df["label"].astype(int).tolist()

    # ========================================================
    # SAFE SPLIT (avoids stratify crash)
    # ========================================================

    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
    except:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42
        )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ========================================================
    # MODEL
    # ========================================================

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    # ========================================================
    # LoRA (slightly stronger for 3k data)
    # ========================================================

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,                 # ↑ capacity for 3000 samples
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )

    cw = torch.tensor(cw, dtype=torch.float)

    # ========================================================
    # TOKENIZATION
    # ========================================================

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=args.max_length)
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=args.max_length)

    train_ds = BloomDataset(train_enc, train_labels)
    val_ds = BloomDataset(val_enc, val_labels)

    # ========================================================
    # TRAINING CONFIG
    # ========================================================

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,

        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,

        num_train_epochs=args.epochs,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,

        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        report_to="none",
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = BloomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        class_weights=cw,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete →", args.output_dir)


if __name__ == "__main__":
    \
    main()