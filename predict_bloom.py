#!/usr/bin/env python

# ============================================================
# BLOOM TAXONOMY EVALUATION + INFERENCE
# FOR TRAINED QWEN LoRA CLASSIFIER
# ============================================================

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from peft import PeftModel

from bloom_model_profiles import (
    DEFAULT_MODEL_SIZE,
    get_profile,
    resolve_checkpoint_dir as resolve_profile_checkpoint_dir,
)


def _active_model_size(model_size: str | None = None) -> str:
    return model_size or os.environ.get("BLOOM_MODEL_SIZE") or DEFAULT_MODEL_SIZE


_DEFAULT_PROFILE = get_profile(DEFAULT_MODEL_SIZE)
DEFAULT_LORA_DIR = _DEFAULT_PROFILE.lora_dir
DEFAULT_FEDERATED_LORA_DIR = _DEFAULT_PROFILE.federated_lora_dir
DEFAULT_MERGED_DIR = _DEFAULT_PROFILE.merged_dir
DEFAULT_QUANTIZED_DIR = _DEFAULT_PROFILE.quantized_dir
DEFAULT_BASE_MODEL = _DEFAULT_PROFILE.base_model

LABELS = {
    0: "Remember",
    1: "Understand",
    2: "Apply",
    3: "Analyze",
    4: "Evaluate",
    5: "Create",
}

# Canonical order for probability vectors and UI tables.
BLOOM_LABELS: list[str] = [LABELS[i] for i in range(len(LABELS))]

# Lowercase keys used by RAGGenerator / CognitiveSummarizer.
BLOOM_LEVELS: list[str] = [
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
]

LABEL_TO_RAG_KEY: dict[str, str] = dict(zip(BLOOM_LABELS, BLOOM_LEVELS))
RAG_KEY_TO_LABEL: dict[str, str] = {v: k for k, v in LABEL_TO_RAG_KEY.items()}

# Back-compat alias
DEFAULT_MODEL_DIR = DEFAULT_MERGED_DIR


def _load_tokenizer(model_path: str):
    kwargs = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)


# ============================================================
# PROMPT TEMPLATE
# ============================================================

def build_prompt(question):
    return (
        "Classify Bloom's Taxonomy level.\n"
        "Focus on reasoning depth, not verbs.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


# ============================================================
# LOAD MODEL (merged checkpoint or LoRA adapter)
# ============================================================

def is_lora_adapter(model_dir: str | os.PathLike) -> bool:
    return (Path(model_dir) / "adapter_config.json").is_file()


def is_quantized_checkpoint(model_dir: str | os.PathLike) -> bool:
    return (Path(model_dir) / "model_int8.pt").is_file()


def resolve_model_dir(
    model_dir: str | None = None,
    *,
    prefer_merged: bool = True,
    prefer_quantized: bool = False,
    model_size: str | None = None,
) -> str:
    profile = get_profile(_active_model_size(model_size))
    return resolve_profile_checkpoint_dir(
        profile,
        model_dir=model_dir,
        prefer_quantized=prefer_quantized,
    )


def load_quantized_model(model_dir: str):
    path = Path(model_dir)
    if not is_quantized_checkpoint(path):
        raise FileNotFoundError(
            f"No INT8 model at {path}. Run: python quantize_bloom.py"
        )
    print(f"\nLoading quantized Bloom classifier from {path}...")
    tokenizer = _load_tokenizer(str(path))
    model = torch.load(path / "model_int8.pt", map_location="cpu", weights_only=False)
    model.eval()
    return tokenizer, model


def load_model(model_dir: str, base_model: str | None = None, *, quantized: bool = False):
    if quantized or is_quantized_checkpoint(model_dir):
        return load_quantized_model(model_dir)

    path = Path(model_dir)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if is_lora_adapter(path):
        if not base_model:
            base_model = DEFAULT_BASE_MODEL
        print(f"\nLoading LoRA adapter from {path} (base={base_model})...")
        tokenizer = _load_tokenizer(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=6,
            dtype=dtype,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(path))
    else:
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"No merged model at {path}. Run: python merge_model.py"
            )
        print(f"\nLoading merged Bloom classifier from {path}...")
        tokenizer = _load_tokenizer(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path),
            dtype=dtype,
            trust_remote_code=True,
        )

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return tokenizer, model


# ============================================================
# SINGLE PREDICTION
# ============================================================

def predict(question, tokenizer, model, max_length=256):

    prompt = build_prompt(question)

    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():

        outputs = model(**inputs)

        logits = outputs.logits

        probs = torch.softmax(logits, dim=-1)[0]

        pred_id = torch.argmax(probs).item()

    confidence = probs[pred_id].item()

    return {
        "prediction": LABELS[pred_id],
        "confidence": round(confidence, 4),
        "probabilities": {
            LABELS[i]: round(probs[i].item(), 4)
            for i in range(len(LABELS))
        },
    }


def probabilities_to_distribution(probabilities: dict[str, float]) -> np.ndarray:
    """Map label probabilities to a (6,) vector in ``BLOOM_LEVELS`` order."""
    return np.asarray(
        [float(probabilities.get(RAG_KEY_TO_LABEL[level], 0.0)) for level in BLOOM_LEVELS],
        dtype=np.float32,
    )


def label_to_rag_key(label: str) -> str:
    key = (label or "").strip()
    if key.lower() in BLOOM_LEVELS:
        return key.lower()
    return LABEL_TO_RAG_KEY.get(key, "understand")


class QwenBloomPredictor:
    """Lazy-loaded Qwen2.5 Bloom classifier (merged, LoRA, or INT8 quantized)."""

    def __init__(
        self,
        model_dir: str | None = None,
        base_model: str | None = None,
        *,
        model_size: str | None = None,
        prefer_merged: bool = True,
        prefer_quantized: bool = False,
        quantized: bool = False,
    ) -> None:
        self.profile = get_profile(_active_model_size(model_size))
        self.quantized = quantized or prefer_quantized
        self.model_dir = resolve_model_dir(
            model_dir,
            prefer_merged=prefer_merged,
            prefer_quantized=self.quantized,
            model_size=self.profile.key,
        )
        self.base_model = base_model or self.profile.base_model
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.quantized or is_quantized_checkpoint(self.model_dir):
            self._tokenizer, self._model = load_quantized_model(self.model_dir)
            return
        base = self.base_model if is_lora_adapter(self.model_dir) else None
        self._tokenizer, self._model = load_model(self.model_dir, base)

    def predict(self, question: str, max_length: int = 256) -> dict:
        self._ensure_loaded()
        raw = predict(question, self._tokenizer, self._model, max_length=max_length)
        dist = probabilities_to_distribution(raw["probabilities"])
        return {
            **raw,
            "rag_key": label_to_rag_key(raw["prediction"]),
            "distribution": dist,
        }


# ============================================================
# ORDINAL ERROR
# ============================================================

def ordinal_metrics(y_true, y_pred) -> dict:

    distances = [
        abs(t - p)
        for t, p in zip(y_true, y_pred)
    ]

    mean_distance = np.mean(distances)

    within_one = np.mean([
        d <= 1 for d in distances
    ])

    severe_error = np.mean([
        d >= 3 for d in distances
    ])

    return {
        "mean_ordinal_distance": round(float(mean_distance), 4),
        "within_one_level_accuracy": round(float(within_one), 4),
        "severe_error_rate": round(float(severe_error), 4),
    }


# ============================================================
# EVALUATE CSV
# ============================================================

def evaluate_csv(
    csv_path,
    text_col,
    label_col,
    tokenizer,
    model,
    output_dir,
):

    print("\nLoading evaluation dataset...")

    df = pd.read_csv(csv_path).dropna()

    label2id = {
        "Remember": 0,
        "Understand": 1,
        "Apply": 2,
        "Analyze": 3,
        "Evaluate": 4,
        "Create": 5,
    }

    df = df[df[label_col].isin(label2id)]

    texts = df[text_col].tolist()

    y_true = [
        label2id[x]
        for x in df[label_col]
    ]

    y_pred = []

    print("\nRunning inference...\n")

    for idx, text in enumerate(texts):

        result = predict(text, tokenizer, model)

        pred_label = result["prediction"]

        pred_id = label2id[pred_label]

        y_pred.append(pred_id)

        if idx % 50 == 0:
            print(f"{idx}/{len(texts)} complete")

    # ========================================================
    # METRICS
    # ========================================================

    acc = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(LABELS.values()),
        digits=4,
    )

    ord_metrics = ordinal_metrics(y_true, y_pred)

    # ========================================================
    # PRINT
    # ========================================================

    print("\n==============================")
    print("EVALUATION RESULTS")
    print("==============================\n")

    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")

    print("\nOrdinal Metrics")
    print(ord_metrics)

    print("\nClassification Report\n")
    print(report)

    # ========================================================
    # SAVE REPORT
    # ========================================================

    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        **ord_metrics,
    }

    with open(
        os.path.join(output_dir, "metrics.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)

    with open(
        os.path.join(output_dir, "classification_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 8))

    plt.imshow(cm)

    plt.title("Bloom Taxonomy Confusion Matrix")

    plt.colorbar()

    tick_marks = np.arange(len(LABELS))

    plt.xticks(
        tick_marks,
        LABELS.values(),
        rotation=45,
    )

    plt.yticks(
        tick_marks,
        LABELS.values(),
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    cm_path = os.path.join(
        output_dir,
        "confusion_matrix.png",
    )

    plt.savefig(cm_path)

    print(f"\nSaved confusion matrix -> {cm_path}")

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Merged dir or LoRA adapter dir (default: merged if present).",
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )

    parser.add_argument(
        "--eval_csv",
        type=str,
        default="data/figshare_bloom_v1.csv",
    )

    parser.add_argument(
        "--text_col",
        type=str,
        default="question",
    )

    parser.add_argument(
        "--label_col",
        type=str,
        default="bloom_level",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
    )

    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    base = args.base_model if is_lora_adapter(model_dir) else None
    tokenizer, model = load_model(model_dir, base)

    # ========================================================
    # FULL EVALUATION
    # ========================================================

    evaluate_csv(
        csv_path=args.eval_csv,
        text_col=args.text_col,
        label_col=args.label_col,
        tokenizer=tokenizer,
        model=model,
        output_dir=args.output_dir,
    )

    # ========================================================
    # INTERACTIVE MODE
    # ========================================================

    print("\n==============================")
    print("INTERACTIVE BLOOM CLASSIFIER")
    print("==============================")

    while True:

        q = input("\nEnter question (or 'exit'): ")

        if q.lower() == "exit":
            break

        result = predict(q, tokenizer, model)

        print("\nPrediction:")
        print(result["prediction"])

        print("\nConfidence:")
        print(result["confidence"])

        print("\nClass Probabilities:")

        for k, v in result["probabilities"].items():
            print(f"{k:12s} -> {v}")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()