from __future__ import annotations

import json
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from scipy.stats import ttest_rel

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from bloom_prompt import predict_bloom_label

# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = Path("pbbs_univ_v2_outputs")
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LABEL SPACE
# ============================================================

LABELS = [
    "Remembering",
    "Understanding",
    "Applying",
    "Analyzing",
    "Evaluating",
    "Creating"
]

IDX = {l: i for i, l in enumerate(LABELS)}

# ============================================================
# LOAD DATASET (ROBUST FIX)
# ============================================================

with open("pbbs_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# FIX COLUMN STANDARDIZATION
if "gold_label" in df.columns:
    df["label"] = df["gold_label"]

required = ["question", "label", "domain"]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=required)

print("\nLoaded Dataset:")
print(df[required].head())

# ============================================================
# SPLIT
# ============================================================

X_train, X_test, y_train, y_test, d_train, d_test = train_test_split(
    df["question"],
    df["label"],
    df["domain"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

# ============================================================
# MODEL
# ============================================================

def train_svm(X, y):
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LinearSVC(class_weight="balanced"))
    ])
    model.fit(X, y)
    return model

svm = train_svm(X_train, y_train)
svm_preds = svm.predict(X_test)

qwen_preds = [predict_bloom_label(x) for x in X_test]

# ============================================================
# CORE METRICS
# ============================================================

def ordinal_error(y_true, y_pred):
    return np.mean([
        abs(IDX[a] - IDX[b]) for a, b in zip(y_true, y_pred)
    ])

def within_one(y_true, y_pred):
    return np.mean([
        abs(IDX[a] - IDX[b]) <= 1 for a, b in zip(y_true, y_pred)
    ])

def evaluate(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "ordinal_error": float(ordinal_error(y_true, y_pred)),
        "within_one": float(within_one(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred))
    }

# ============================================================
# PARADOX COMPLEXITY INDEX (FIXED VERSION)
# true disagreement rate (NOT label distance hack)
# ============================================================

def PCI(pred1, pred2):
    return float(np.mean([a != b for a, b in zip(pred1, pred2)]))

# ============================================================
# ENTROPY (STABLE)
# ============================================================

def entropy(preds):
    c = Counter(preds)
    p = np.array([v / len(preds) for v in c.values()])
    return float(-np.sum(p * np.log(p + 1e-9)))

# ============================================================
# CALIBRATION GAP (FIXED ECE-LITE)
# ============================================================

def calibration_gap(y_true, y_pred):
    correct = np.array([a == b for a, b in zip(y_true, y_pred)])
    confidence = np.ones_like(correct) * 0.8  # fixed proxy confidence

    return float(np.mean(np.abs(confidence - correct)))

# ============================================================
# DOMAIN STABILITY (FIXED)
# ============================================================

def domain_stability(preds, domains):
    out = {}
    for d in sorted(set(domains)):
        subset = [preds[i] for i in range(len(preds)) if domains.iloc[i] == d]

        out[d] = {
            "entropy": entropy(subset),
            "diversity": len(set(subset))
        }
    return out

# ============================================================
# SEMANTIC ROBUSTNESS
# ============================================================

def paraphrase(q):
    return q.replace("Analyze", "Critically analyze") \
            .replace("Evaluate", "Critically evaluate") \
            .replace("Design", "Construct")

def semantic_score(fn, X):
    return float(np.mean([
        fn(x) == fn(paraphrase(x)) for x in X
    ]))

# ============================================================
# EVALUATION
# ============================================================

print("\nPBBS-UNIV v2.0 (CLEAN EVAL)\n")

svm_metrics = evaluate(y_test, svm_preds)
qwen_metrics = evaluate(y_test, qwen_preds)

svm_metrics["semantic_score"] = semantic_score(lambda x: svm.predict([x])[0], X_test)
qwen_metrics["semantic_score"] = semantic_score(lambda x: predict_bloom_label(x), X_test)

svm_metrics["domain_stability"] = domain_stability(svm_preds, d_test)
qwen_metrics["domain_stability"] = domain_stability(qwen_preds, d_test)

qwen_metrics["prediction_entropy"] = entropy(qwen_preds)

# FIXED PCI
qwen_metrics["PCI"] = PCI(svm_preds, qwen_preds)

svm_metrics["calibration_gap"] = calibration_gap(y_test, svm_preds)
qwen_metrics["calibration_gap"] = calibration_gap(y_test, qwen_preds)

# ============================================================
# STAT TEST (SAFE)
# ============================================================

svm_acc = np.array([int(a == b) for a, b in zip(y_test, svm_preds)])
qwen_acc = np.array([int(a == b) for a, b in zip(y_test, qwen_preds)])

t_stat, p_value = ttest_rel(svm_acc.astype(float), qwen_acc.astype(float))

# ============================================================
# REPORT
# ============================================================

report = {
    "SVM": svm_metrics,
    "QWEN": qwen_metrics,
    "contributions": {
        "PCI": "Prediction disagreement rate across models",
        "Entropy": "Uncertainty structure of predictions",
        "CalibrationGap": "Confidence vs correctness mismatch proxy"
    },
    "statistical_test": {
        "t_stat": float(t_stat),
        "p_value": float(p_value)
    }
}

with open(OUT_DIR / "pbbs_univ_v2_report.json", "w") as f:
    json.dump(report, f, indent=4)

print(json.dumps(report, indent=4))
print("\nSaved → pbbs_univ_v2_outputs/\n")