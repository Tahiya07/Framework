# ============================================================
# EDUGUARD-RAG
# BLOOM TAXONOMY COMPARISON + EVALUATION (FIXED VERSION)
#
# Publication-ready, aligned with bloom_prompt.py
# ============================================================

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from bloom_prompt import predict_bloom_label

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# OUTPUT DIR
# ============================================================

OUTPUT_DIR = Path("evaluation_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# BLOOM LABEL SPACE (STRICT)
# ============================================================

LABELS = [
    "Remembering",
    "Understanding",
    "Applying",
    "Analyzing",
    "Evaluating",
    "Creating"
]

LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}


# ============================================================
# LABEL NORMALIZATION (CRITICAL FIX)
# ============================================================

LABEL_NORMALIZATION = {
    # Old Bloom / dataset variants
    "Knowledge": "Remembering",
    "Remember": "Remembering",
    "Recall": "Remembering",

    "Comprehension": "Understanding",
    "Understand": "Understanding",

    "Application": "Applying",
    "Apply": "Applying",

    "Analysis": "Analyzing",
    "Analyze": "Analyzing",

    "Evaluation": "Evaluating",
    "Evaluate": "Evaluating",

    "Synthesis": "Creating",
    "Create": "Creating",
}


def normalize_label(x: str) -> str:
    x = str(x).strip()
    return LABEL_NORMALIZATION.get(x, x)


# ============================================================
# LOAD DATASET (ROBUST)
# ============================================================

print("\n===================================================")
print(" Loading Dataset ")
print("===================================================\n")

DATASET_PATH = "data/figshare_combined_dataset.csv"
df = pd.read_csv(DATASET_PATH)

print("Detected Columns:", df.columns.tolist())


# ============================================================
# AUTO COLUMN DETECTION (FIXED)
# ============================================================

QUESTION_COL = None
LABEL_COL = None

question_candidates = ["QUESTION", "question", "text", "Text", "Questions"]
label_candidates = ["BT LEVEL", "label", "Label", "bloom", "Bloom"]

for c in df.columns:
    if c in question_candidates:
        QUESTION_COL = c
    if c in label_candidates:
        LABEL_COL = c

if QUESTION_COL is None:
    raise ValueError("Question column not found in dataset")

if LABEL_COL is None:
    raise ValueError("Label column not found in dataset")


print(f"\nQuestion Column: {QUESTION_COL}")
print(f"Label Column: {LABEL_COL}")


# ============================================================
# CLEAN + NORMALIZE
# ============================================================

df = df[[QUESTION_COL, LABEL_COL]].dropna()

df[QUESTION_COL] = df[QUESTION_COL].astype(str)
df[LABEL_COL] = df[LABEL_COL].astype(str).apply(normalize_label)

# keep only valid labels
df = df[df[LABEL_COL].isin(LABELS)]

print("\nLabel distribution:")
print(df[LABEL_COL].value_counts())


# ============================================================
# SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df[QUESTION_COL],
    df[LABEL_COL],
    test_size=0.15,
    random_state=42,
    stratify=df[LABEL_COL]
)

print(f"\nTrain size: {len(X_train)}")
print(f"Test size : {len(X_test)}")


# ============================================================
# CLASSICAL BASELINE (SVM)
# ============================================================

print("\n===================================================")
print(" Training Classical ML Baseline ")
print("===================================================\n")

svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("clf", LinearSVC(class_weight="balanced"))
])

svm_model.fit(X_train, y_train)


# ============================================================
# METRICS
# ============================================================

def ordinal_error(y_true, y_pred):
    return float(np.mean([
        abs(LABEL_TO_INT[a] - LABEL_TO_INT[b])
        for a, b in zip(y_true, y_pred)
    ]))


def within_one(y_true, y_pred):
    return float(np.mean([
        abs(LABEL_TO_INT[a] - LABEL_TO_INT[b]) <= 1
        for a, b in zip(y_true, y_pred)
    ]))


def severe_error(y_true, y_pred):
    return float(np.mean([
        abs(LABEL_TO_INT[a] - LABEL_TO_INT[b]) > 1
        for a, b in zip(y_true, y_pred)
    ]))


def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "ordinal_error": ordinal_error(y_true, y_pred),
        "within_one_level": within_one(y_true, y_pred),
        "severe_error": severe_error(y_true, y_pred),
        "kappa": float(cohen_kappa_score(y_true, y_pred))
    }


# ============================================================
# PREDICTIONS
# ============================================================

print("\n===================================================")
print(" Running SVM Predictions ")
print("===================================================\n")

svm_preds = svm_model.predict(X_test)


print("\n===================================================")
print(" Running Qwen Predictions ")
print("===================================================\n")

llm_preds = []

for i, q in enumerate(X_test):
    pred = predict_bloom_label(q)
    llm_preds.append(pred)

    if i % 10 == 0:
        print(f"[{i}/{len(X_test)}] {pred}")


# ============================================================
# EVALUATION
# ============================================================

print("\n===================================================")
print(" Evaluating Models ")
print("===================================================\n")

svm_results = evaluate(y_test, svm_preds)
qwen_results = evaluate(y_test, llm_preds)

agreement = float(np.mean(np.array(svm_preds) == np.array(llm_preds)))


# ============================================================
# REPORT
# ============================================================

report = {
    "SVM": svm_results,
    "QWEN": qwen_results,
    "agreement_rate": agreement,
    "num_disagreements": int(np.sum(np.array(svm_preds) != np.array(llm_preds)))
}


# ============================================================
# SAVE OUTPUT
# ============================================================

with open(OUTPUT_DIR / "evaluation_report.json", "w") as f:
    json.dump(report, f, indent=4)


# ============================================================
# CONFUSION MATRICES
# ============================================================

def plot_cm(y_true, y_pred, title, file):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=LABELS,
                yticklabels=LABELS)

    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / file, dpi=300)
    plt.close()


plot_cm(y_test, svm_preds, "SVM Bloom CM", "svm_cm.png")
plot_cm(y_test, llm_preds, "Qwen Bloom CM", "qwen_cm.png")


# ============================================================
# AGREEMENT HEATMAP
# ============================================================

plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(svm_preds, llm_preds, labels=LABELS),
    annot=True,
    fmt="d",
    xticklabels=LABELS,
    yticklabels=LABELS
)
plt.title("SVM vs Qwen Agreement")
plt.xlabel("Qwen")
plt.ylabel("SVM")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "agreement.png", dpi=300)
plt.close()


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n================ FINAL RESULTS ================\n")

print("SVM:\n", json.dumps(svm_results, indent=4))
print("\nQWEN:\n", json.dumps(qwen_results, indent=4))

print("\nAgreement:", agreement)

print("\nSaved to evaluation_outputs/")