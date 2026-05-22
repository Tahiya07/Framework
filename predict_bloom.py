#!/usr/bin/env python

# ============================================================
# BLOOM TAXONOMY EVALUATION + INFERENCE
# FOR TRAINED QWEN LoRA CLASSIFIER
# ============================================================

import os
import json
import argparse
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


# ============================================================
# LABELS
# ============================================================

LABELS = {
    0: "Remember",
    1: "Understand",
    2: "Apply",
    3: "Analyze",
    4: "Evaluate",
    5: "Create",
}


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
# LOAD MODEL
# ============================================================

def load_model(model_dir, base_model):

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print("Loading base model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=6,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, model_dir)

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
        }
    }


# ============================================================
# ORDINAL ERROR
# ============================================================

def ordinal_metrics(y_true, y_pred):

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
        default="models/qwen_bloom_3000",
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

    tokenizer, model = load_model(
        args.model_dir,
        args.base_model,
    )

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