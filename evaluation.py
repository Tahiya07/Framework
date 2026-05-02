"""
evaluation.py
===========================================================
Publishable evaluation suite for Bloom classifier system
"""

import numpy as np
from typing import Dict, Sequence


# ---------------------------------------------------------
def bloom_distance(y_true, y_pred):
    return abs(int(y_true) - int(y_pred))


def mean_bloom_distance(y_true, y_pred):
    return float(np.mean([bloom_distance(t, p) for t, p in zip(y_true, y_pred)]))


# ---------------------------------------------------------
def macro_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1s = []

    for c in np.unique(y_true):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)

        f1s.append(2 * prec * rec / (prec + rec + 1e-12))

    return float(np.mean(f1s))


# ---------------------------------------------------------
def evaluate_system(probs, y_true, uncertainties):
    y_pred = np.argmax(probs, axis=1)

    acc = float(np.mean(y_pred == y_true))
    f1 = macro_f1(y_true, y_pred)
    md = mean_bloom_distance(y_true, y_pred)

    errors = (y_pred != np.array(y_true)).astype(int)

    if np.std(uncertainties) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(uncertainties, errors)[0, 1])

    return {
        "accuracy": acc,
        "macro_f1": f1,
        "mean_bloom_distance": md,
        "uncertainty_error_corr": corr,
    }