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
    trainer,
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
    # Must stay identical to predict_bloom.build_prompt / federated client_train
    # to avoid train/inference skew.
    return (
        "Classify Bloom's Taxonomy level.\n"
        "Focus on reasoning depth, not verbs.\n\n"
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

    # Dedicated, pre-made split (held-out test stays untouched for evaluate_bloom.py).
    parser.add_argument("--train_csv", type=str, default="data/figshare_bloom_v1_train.csv")
    parser.add_argument("--val_csv", type=str, default="data/figshare_bloom_v1_val.csv")
    # Deprecated single-file mode: if set, it overrides train_csv and is split 80/20.
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--text_col", type=str, default="question")
    parser.add_argument("--label_col", type=str, default="bloom_level")

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="models/qwen_bloom_trained")

    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Bloom level is highly verb-sensitive; augmentation is OFF unless requested.
    parser.add_argument("--augment", action="store_true")

    # 0 = use all available training rows (no cap).
    parser.add_argument("--sample_size", type=int, default=0)

    args = parser.parse_args()

    set_seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # ========================================================
    # LOAD DATA (dedicated train / val files; the held-out
    # test split is never touched here so evaluate_bloom.py
    # reports leakage-free numbers)
    # ========================================================

    def _read(csv_path):
        d = pd.read_csv(csv_path).dropna(subset=[args.text_col, args.label_col])
        d["label"] = d[args.label_col].map(LABELS)
        d = d.dropna(subset=["label"])
        d["label"] = d["label"].astype(int)
        return d

    def _split(d):
        return train_test_split(
            d,
            test_size=0.2,
            random_state=42,
            stratify=d["label"],
        )

    if args.csv:
        train_df, val_df = _split(_read(args.csv))
    else:
        train_df = _read(args.train_csv)
        if args.val_csv and os.path.isfile(args.val_csv):
            val_df = _read(args.val_csv)
        else:
            train_df, val_df = _split(train_df)

    if args.sample_size and len(train_df) > args.sample_size:
        train_df = train_df.sample(args.sample_size, random_state=42)

    def _prep(text):
        return build_prompt(augment(text) if args.augment else text)

    train_texts = [_prep(t) for t in train_df[args.text_col]]
    train_labels = train_df["label"].tolist()

    val_texts = [build_prompt(t) for t in val_df[args.text_col]]
    val_labels = val_df["label"].tolist()

    print(f"[train] train={len(train_texts)}  val={len(val_texts)}")

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
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        # Adapt all attention + MLP projections (not just q/v): biggest
        # accuracy lever for a 1.5B classifier on a few-thousand-row set.
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
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
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",

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
        seed=42,
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
    )

    trainer.train()
    metrics = trainer.evaluate()
 
    print("\n" + "=" * 40)
    print("Final Validation Results")
    print("=" * 40)
    print(f"Accuracy : {metrics['eval_accuracy']:.2%}")
    print(f"Macro F1 : {metrics['eval_f1_macro']:.2%}")
    print(f"Loss      : {metrics['eval_loss']:.4f}")
    print("=" * 40)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete →", args.output_dir)


if __name__ == "__main__":

    main()