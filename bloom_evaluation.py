#!/usr/bin/env python
# ============================================================
# Bloom taxonomy baseline comparison
# TF-IDF + LinearSVC | Qwen2.5 zero-shot (GGUF) | trained LoRA
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bloom_prompt import zero_shot_bloom_label
from predict_bloom import BLOOM_LABELS, QwenBloomPredictor, ordinal_metrics

OUTPUT_DIR = Path("evaluation_outputs")
RESULTS_JSON = Path("results/bloom_baseline_comparison.json")

LABEL_NORMALIZATION = {
    "Knowledge": "Remember",
    "Remembering": "Remember",
    "Remember": "Remember",
    "Recall": "Remember",
    "Comprehension": "Understand",
    "Understanding": "Understand",
    "Understand": "Understand",
    "Application": "Apply",
    "Applying": "Apply",
    "Apply": "Apply",
    "Analysis": "Analyze",
    "Analyzing": "Analyze",
    "Analyze": "Analyze",
    "Evaluation": "Evaluate",
    "Evaluating": "Evaluate",
    "Evaluate": "Evaluate",
    "Synthesis": "Create",
    "Creating": "Create",
    "Create": "Create",
}

LABEL_TO_INT = {label: i for i, label in enumerate(BLOOM_LABELS)}


def normalize_label(value: str) -> str:
    token = str(value).strip()
    return LABEL_NORMALIZATION.get(token, token)


def _detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    question_candidates = ["QUESTION", "question", "text", "Text", "Questions"]
    label_candidates = ["BT LEVEL", "label", "Label", "bloom", "Bloom"]
    question_col = next((c for c in df.columns if c in question_candidates), None)
    label_col = next((c for c in df.columns if c in label_candidates), None)
    if question_col is None or label_col is None:
        raise ValueError(f"Could not detect question/label columns in {df.columns.tolist()}")
    return question_col, label_col


def _load_split(
    csv_path: Path,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    question_col, label_col = _detect_columns(df)
    df = df[[question_col, label_col]].dropna()
    df[question_col] = df[question_col].astype(str)
    df[label_col] = df[label_col].astype(str).map(normalize_label)
    df = df[df[label_col].isin(BLOOM_LABELS)]
    return train_test_split(
        df[question_col],
        df[label_col],
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )


def _labels_to_int(labels: list[str]) -> list[int]:
    return [LABEL_TO_INT[label] for label in labels]


def _evaluate_strings(y_true: list[str], y_pred: list[str]) -> dict:
    y_true_i = _labels_to_int(y_true)
    y_pred_i = _labels_to_int(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        **ordinal_metrics(y_true_i, y_pred_i),
        "classification_report": classification_report(
            y_true, y_pred, labels=BLOOM_LABELS, zero_division=0, output_dict=True
        ),
    }


def _plot_cm(y_true: list[str], y_pred: list[str], title: str, filename: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=BLOOM_LABELS)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=BLOOM_LABELS, yticklabels=BLOOM_LABELS)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def _agreement(a: list[str], b: list[str]) -> float:
    return float(np.mean(np.array(a) == np.array(b)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Bloom SVM vs zero-shot vs LoRA comparison")
    parser.add_argument("--dataset", default="data/figshare_combined_dataset.csv")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0, help="Cap test set (0 = all)")
    parser.add_argument("--skip-zero-shot", action="store_true")
    parser.add_argument("--skip-lora", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    RESULTS_JSON.parent.mkdir(exist_ok=True)

    print("Loading dataset...")
    X_train, X_test, y_train, y_test = _load_split(
        Path(args.dataset),
        test_size=args.test_size,
        random_state=args.random_state,
    )
    if args.max_samples > 0:
        X_test = X_test.iloc[: args.max_samples]
        y_test = y_test.iloc[: args.max_samples]

    y_true = y_test.tolist()
    n_test = len(y_true)
    print(f"Train={len(X_train)}  Test={n_test}")

    print("Training TF-IDF + LinearSVC baseline...")
    svm_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test).tolist()
    svm_results = _evaluate_strings(y_true, svm_preds)

    zero_shot_preds: list[str] | None = None
    zero_shot_results: dict | None = None
    if not args.skip_zero_shot:
        print("Running Qwen zero-shot (GGUF) predictions...")
        zero_shot_preds = []
        for i, question in enumerate(X_test):
            zero_shot_preds.append(zero_shot_bloom_label(question))
            if i % 10 == 0:
                print(f"  zero-shot [{i}/{n_test}] {zero_shot_preds[-1]}", flush=True)
        zero_shot_results = _evaluate_strings(y_true, zero_shot_preds)

    lora_preds: list[str] | None = None
    lora_results: dict | None = None
    if not args.skip_lora:
        print("Running trained LoRA predictions...")
        predictor = QwenBloomPredictor()
        lora_preds = [predictor.predict(q)["prediction"] for q in X_test]
        lora_results = _evaluate_strings(y_true, lora_preds)

    report: dict = {
        "dataset": args.dataset,
        "n_train": int(len(X_train)),
        "n_test": n_test,
        "random_state": args.random_state,
        "test_size": args.test_size,
        "SVM": svm_results,
    }
    if zero_shot_results is not None:
        report["zero_shot_gguf"] = zero_shot_results
        report["QWEN"] = zero_shot_results
    if lora_results is not None:
        report["lora_trained"] = lora_results
        report["trained_lora"] = lora_results

    if svm_preds and lora_preds:
        report["svm_lora_agreement"] = _agreement(svm_preds, lora_preds)
    if svm_preds and zero_shot_preds:
        report["svm_zero_shot_agreement"] = _agreement(svm_preds, zero_shot_preds)
        report["agreement_rate"] = report["svm_zero_shot_agreement"]
    if zero_shot_preds and lora_preds:
        report["zero_shot_lora_agreement"] = _agreement(zero_shot_preds, lora_preds)

    trained_metrics_path = Path("evaluation_results/metrics.json")
    if trained_metrics_path.is_file():
        report["trained_model_reference"] = json.loads(
            trained_metrics_path.read_text(encoding="utf-8")
        )

    RESULTS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(OUTPUT_DIR / "evaluation_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    _plot_cm(y_true, svm_preds, "SVM Bloom CM", "svm_cm.png")
    if zero_shot_preds:
        _plot_cm(y_true, zero_shot_preds, "Qwen Zero-Shot Bloom CM", "qwen_cm.png")
    if lora_preds:
        _plot_cm(y_true, lora_preds, "Qwen LoRA Bloom CM", "lora_cm.png")
    if svm_preds and zero_shot_preds:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix(svm_preds, zero_shot_preds, labels=BLOOM_LABELS),
            annot=True,
            fmt="d",
            xticklabels=BLOOM_LABELS,
            yticklabels=BLOOM_LABELS,
        )
        plt.title("SVM vs Qwen Zero-Shot Agreement")
        plt.xlabel("Zero-shot")
        plt.ylabel("SVM")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "agreement.png", dpi=300)
        plt.close()

    print("\n================ FINAL RESULTS ================\n")
    print("SVM:", json.dumps({k: v for k, v in svm_results.items() if k != "classification_report"}, indent=2))
    if zero_shot_results:
        print(
            "\nZero-shot GGUF:",
            json.dumps(
                {k: v for k, v in zero_shot_results.items() if k != "classification_report"},
                indent=2,
            ),
        )
    if lora_results:
        print(
            "\nTrained LoRA:",
            json.dumps(
                {k: v for k, v in lora_results.items() if k != "classification_report"},
                indent=2,
            ),
        )
    print(f"\nWrote {RESULTS_JSON} and figures under {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
