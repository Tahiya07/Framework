"""Publication-grade metrics for Bloom taxonomy evaluation."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)

from predict_bloom import BLOOM_LABELS, ordinal_metrics
from uncertainty import UncertaintyEngine


def _ids_to_labels(y: Sequence[int]) -> list[str]:
    return [BLOOM_LABELS[int(i)] for i in y]


def bootstrap_ci(
    y_true: list[int],
    y_pred: list[int],
    metric_fn: Callable[[list[int], list[int]], float],
    *,
    n_samples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    n = len(y_true)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_bootstrap": n_samples}
    rng = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, n)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        scores.append(float(metric_fn(yt, yp)))
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_low": round(float(np.percentile(arr, 100 * alpha / 2)), 4),
        "ci_high": round(float(np.percentile(arr, 100 * (1 - alpha / 2))), 4),
        "n_bootstrap": n_samples,
    }


def mcnemar_test(y_true: list[int], y_pred_a: list[int], y_pred_b: list[int]) -> dict[str, float | int | str]:
    """Paired significance test for two classifiers on the same examples."""
    from scipy.stats import binomtest

    b = 0  # A wrong, B right
    c = 0  # A right, B wrong
    for t, a, bb in zip(y_true, y_pred_a, y_pred_b):
        a_ok = a == t
        b_ok = bb == t
        if not a_ok and b_ok:
            b += 1
        elif a_ok and not b_ok:
            c += 1
    n = b + c
    if n == 0:
        return {
            "statistic": "mcnemar_exact",
            "n_discordant": 0,
            "b_wrong_a_right": 0,
            "c_right_a_wrong": 0,
            "p_value": 1.0,
        }
    p_value = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue)
    return {
        "statistic": "mcnemar_exact",
        "n_discordant": n,
        "b_wrong_a_right": b,
        "c_right_a_wrong": c,
        "p_value": round(p_value, 6),
    }


def selective_prediction_metrics(
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    thresholds: Sequence[float] = (0.7, 0.8, 0.9),
) -> list[dict[str, float]]:
    conf = np.asarray(confidences, dtype=np.float64)
    correct = np.asarray([t == p for t, p in zip(y_true, y_pred)], dtype=np.float64)
    rows: list[dict[str, float]] = []
    n = len(y_true)
    for threshold in thresholds:
        mask = conf >= threshold
        covered = int(mask.sum())
        if covered == 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "coverage": 0.0,
                    "accuracy": 0.0,
                    "n_accepted": 0.0,
                    "n_total": float(n),
                }
            )
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": round(float(covered / n), 4),
                "accuracy": round(float(correct[mask].mean()), 4),
                "n_accepted": float(covered),
                "n_total": float(n),
            }
        )
    return rows


def calibration_metrics(
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    *,
    n_bins: int = 10,
) -> dict:
    engine = UncertaintyEngine(n_bins=n_bins)
    correct = [1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)]
    ece = engine.compute_ece(confidences, correct, n_bins=n_bins)
    reliability = engine.reliability_data(confidences, correct, n_bins=n_bins)
    nonempty = reliability["bin_counts"] > 0
    return {
        "ece": round(float(ece), 4),
        "reliability_bins": {
            "bin_centers": [
                round(float(x), 4) for x in reliability["bin_centers"][nonempty]
            ],
            "bin_accuracy": [
                round(float(x), 4) for x in reliability["bin_accuracy"][nonempty]
            ],
            "bin_confidence": [
                round(float(x), 4) for x in reliability["bin_confidence"][nonempty]
            ],
            "bin_counts": [int(x) for x in reliability["bin_counts"][nonempty]],
        },
    }


def per_class_f1(y_true: list[int], y_pred: list[int]) -> dict[str, dict[str, float]]:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(BLOOM_LABELS))),
        target_names=BLOOM_LABELS,
        output_dict=True,
        zero_division=0,
    )
    out: dict[str, dict[str, float]] = {}
    for label in BLOOM_LABELS:
        stats = report.get(label, {})
        if not isinstance(stats, dict):
            continue
        out[label] = {
            "precision": round(float(stats.get("precision", 0.0)), 4),
            "recall": round(float(stats.get("recall", 0.0)), 4),
            "f1": round(float(stats.get("f1-score", 0.0)), 4),
            "support": int(stats.get("support", 0)),
        }
    return out


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
    *,
    confidences: list[float] | None = None,
    bootstrap_samples: int = 2000,
) -> dict:
    y_true_labels = _ids_to_labels(y_true)
    y_pred_labels = _ids_to_labels(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true_labels, y_pred_labels)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(y_true, y_pred, weights="quadratic")
        ),
        **ordinal_metrics(y_true, y_pred),
    }
    for key, value in metrics.items():
        if isinstance(value, float):
            metrics[key] = round(value, 4)

    metrics["per_class"] = per_class_f1(y_true, y_pred)
    metrics["bootstrap_ci"] = {
        "accuracy": bootstrap_ci(
            y_true,
            y_pred,
            lambda yt, yp: accuracy_score(yt, yp),
            n_samples=bootstrap_samples,
        ),
        "macro_f1": bootstrap_ci(
            y_true,
            y_pred,
            lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
            n_samples=bootstrap_samples,
        ),
        "quadratic_weighted_kappa": bootstrap_ci(
            y_true,
            y_pred,
            lambda yt, yp: cohen_kappa_score(yt, yp, weights="quadratic"),
            n_samples=bootstrap_samples,
        ),
    }

    if confidences is not None and len(confidences) == len(y_true):
        cal = calibration_metrics(y_true, y_pred, confidences)
        metrics["ece"] = cal["ece"]
        metrics["calibration"] = cal
        metrics["selective_prediction"] = selective_prediction_metrics(
            y_true, y_pred, confidences
        )

    return metrics


def predictions_from_rows(rows: list[dict], label2id: dict[str, int]) -> tuple[list[int], list[int], list[float]]:
    y_true = [label2id[str(r["gold"])] for r in rows]
    y_pred = [label2id[str(r["prediction"])] for r in rows]
    confidences = [float(r.get("confidence", 0.0)) for r in rows]
    return y_true, y_pred, confidences


def compare_model_predictions(
    rows_a: list[dict],
    rows_b: list[dict],
    *,
    key: str = "question",
) -> dict | None:
    """McNemar on aligned examples between two saved prediction tables."""
    index_b = {str(r[key]): r for r in rows_b}
    y_true: list[int] = []
    y_pred_a: list[int] = []
    y_pred_b: list[int] = []
    label2id = {label: i for i, label in enumerate(BLOOM_LABELS)}
    for row in rows_a:
        other = index_b.get(str(row[key]))
        if other is None:
            continue
        y_true.append(label2id[str(row["gold"])])
        y_pred_a.append(label2id[str(row["prediction"])])
        y_pred_b.append(label2id[str(other["prediction"])])
    if len(y_true) < 2:
        return None
    result = mcnemar_test(y_true, y_pred_a, y_pred_b)
    result["n_aligned"] = len(y_true)
    return result
