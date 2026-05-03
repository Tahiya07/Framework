from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from classifier import BLOOM_LEVELS, _find_figshare_exam_dataset, _normalise_bloom
from bloom.bloom_models import (
    HierarchicalBloomClassifier,
    OrdinalThresholdClassifier,
    make_linear_svm_pipeline,
    make_logreg_pipeline,
    make_minilm_logreg_pipeline,
)

# -----------------------------
# CONFIG
# -----------------------------
SEED = 42
VERSION = "figshare_bloom_v1"

DATA_DIR = Path("../data")
MODELS_DIR = Path("../models")
RESULTS_DIR = Path("../results")


# -----------------------------
# DATA LOADING
# -----------------------------
LABEL_MAPPING = {
    "Knowledge": "Remember",
    "Comprehension": "Understand",
    "Application": "Apply",
    "Analysis": "Analyze",
    "Evaluation": "Evaluate",
    "Synthesis": "Create",
}


def _clean_text(x: object) -> str:
    return " ".join(str(x).strip().split())


def _load_dataset() -> Tuple[pd.DataFrame, Dict[str, object]]:
    path = _find_figshare_exam_dataset()
    if path is None:
        raise FileNotFoundError("Figshare dataset not found.")

    raw = pd.read_csv(path, low_memory=False)

    df = pd.DataFrame({
        "question": raw["QUESTION"].map(_clean_text),
        "label_raw": raw["BT LEVEL"].astype(str).str.strip(),
    })

    audit = {
        "dataset": "figshare",
        "path": str(path),
        "version": VERSION,
        "label_mapping": LABEL_MAPPING,
        "raw_rows": len(df),
    }

    df = df.dropna()
    df = df[df["question"].str.len() > 0]

    df["bloom_level"] = df["label_raw"].map(_normalise_bloom)
    df = df[df["bloom_level"].notna()].copy()
    df["bloom_level"] = df["bloom_level"].astype(str)

    audit["final_rows"] = len(df)
    audit["distribution"] = df["bloom_level"].value_counts().to_dict()

    # clean duplicates + conflicts
    df = df.drop_duplicates(subset=["question", "bloom_level"])

    conflict = df.groupby("question")["bloom_level"].nunique()
    bad = set(conflict[conflict > 1].index)

    df = df[~df["question"].isin(bad)]
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)

    audit["conflicts_removed"] = len(bad)

    return df, audit


# -----------------------------
# SPLIT
# -----------------------------
def _split(df: pd.DataFrame):
    train, temp = train_test_split(
        df,
        test_size=0.3,
        random_state=SEED,
        stratify=df["bloom_level"],
    )

    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=SEED,
        stratify=temp["bloom_level"],
    )

    return train, val, test


# -----------------------------
# MODELS
# -----------------------------
def _build_models():
    return {
        "logreg": make_logreg_pipeline(class_weight="balanced"),
        "svm": make_linear_svm_pipeline(class_weight="balanced"),
        "hierarchical": HierarchicalBloomClassifier(class_weight="balanced"),
        "ordinal": OrdinalThresholdClassifier(class_weight="balanced"),
        "minilm": make_minilm_logreg_pipeline(class_weight="balanced"),
    }


# -----------------------------
# METRICS (single source of truth)
# -----------------------------
def _metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=BLOOM_LEVELS)

    idx = {l: i for i, l in enumerate(BLOOM_LEVELS)}
    yt = np.array([idx[x] for x in y_true])
    yp = np.array([idx[x] for x in y_pred])

    dist = np.abs(yt - yp)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "mean_ordinal_error": float(dist.mean()),
        "within_one_level_accuracy": float((dist <= 1).mean()),
        "severe_error_rate": float((dist >= 2).mean()),
        "classification_report": classification_report(
            y_true, y_pred, labels=BLOOM_LEVELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": cm.tolist(),
    }


# -----------------------------
# CROSS VALIDATION (clean)
# -----------------------------
def _cross_validate(models, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results = {}

    for name, model in models.items():
        f1s, ord_err = [], []

        for tr, va in skf.split(X, y):
            m = clone(model)

            xtr = [X[i] for i in tr]
            ytr = [y[i] for i in tr]
            xva = [X[i] for i in va]
            yva = [y[i] for i in va]

            m.fit(xtr, ytr)
            pred = m.predict(xva)

            f1s.append(f1_score(yva, pred, average="macro"))

            idx = {l: i for i, l in enumerate(BLOOM_LEVELS)}
            dist = np.abs(np.array([idx[a] for a in yva]) - np.array([idx[p] for p in pred]))
            ord_err.append(dist.mean())

        results[name] = {
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
            "mean_ordinal_error": float(np.mean(ord_err)),
        }

    return results


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)

    df, audit = _load_dataset()
    train, val, test = _split(df)

    X_train, y_train = train["question"].tolist(), train["bloom_level"].tolist()
    X_val, y_val = val["question"].tolist(), val["bloom_level"].tolist()
    X_test, y_test = test["question"].tolist(), test["bloom_level"].tolist()

    models = _build_models()

    # -----------------------------
    # STEP 1: CV selection
    # -----------------------------
    cv_results = _cross_validate(models, X_train, y_train)

    best_model_name = max(
        cv_results.items(),
        key=lambda x: x[1]["macro_f1_mean"]
    )[0]

    # -----------------------------
    # STEP 2: validation comparison
    # -----------------------------
    val_results = {}

    for name, model in models.items():
        m = clone(model)
        m.fit(X_train, y_train)
        pred = m.predict(X_val)
        val_results[name] = _metrics(y_val, pred)

    # -----------------------------
    # STEP 3: final training
    # -----------------------------
    best_model = clone(models[best_model_name])

    X_dev = X_train + X_val
    y_dev = y_train + y_val

    t0 = time.time()
    best_model.fit(X_dev, y_dev)
    train_time = time.time() - t0

    # -----------------------------
    # STEP 4: test evaluation
    # -----------------------------
    test_pred = best_model.predict(X_test)

    results = {
        "dataset": "figshare",
        "version": VERSION,
        "seed": SEED,
        "audit": audit,

        "cv_results": cv_results,
        "selected_model": best_model_name,
        "validation_results": val_results,

        "validation_metrics": _metrics(y_val, best_model.predict(X_val)),
        "test_metrics": _metrics(y_test, test_pred),

        "training_time_sec": train_time,
    }

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    (RESULTS_DIR / f"{VERSION}.json").write_text(json.dumps(results, indent=2))
    joblib.dump(best_model, MODELS_DIR / "figshare_model.joblib")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()