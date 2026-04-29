from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone

from bloom_models import (
    make_bloom_cue_logreg_pipeline,
    make_logreg_pipeline,
    make_domain_robust_logreg_pipeline,
)
from evaluate_cross_domain_bloom import (
    RESULTS_DIR,
    SEED,
    _fit_predict,
    _make_reduced,
    _metric_bundle,
    _read_figshare,
    _read_moocradar,
    _stratified_cap,
)


def _score_model(
    model: object,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: Sequence[str],
) -> Dict[str, object]:
    fitted = clone(model)
    fitted.fit(train_df["question"].tolist(), train_df["label"].tolist())
    pred = fitted.predict(test_df["question"].tolist())
    return _metric_bundle(test_df["label"].tolist(), list(pred), labels)


def _directional_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: Sequence[str],
) -> Dict[str, object]:
    models = {
        "cue_only_logreg": make_bloom_cue_logreg_pipeline(class_weight="balanced"),
        "content_tfidf_logreg": make_logreg_pipeline(class_weight="balanced"),
        "combined_cue_content_logreg": make_domain_robust_logreg_pipeline(class_weight="balanced"),
    }
    scored: Dict[str, object] = {}
    for name, model in models.items():
        scored[name] = _score_model(model, train_df, test_df, labels)
    best_name = max(
        scored.items(),
        key=lambda kv: (
            kv[1]["within_one_level_accuracy"],
            -kv[1]["severe_error_rate"],
            kv[1]["macro_f1"],
        ),
    )[0]
    return {
        "candidate_metrics": scored,
        "ordinal_stability_model": best_name,
        "interpretation": (
            "Cue-only performance estimates how much Bloom signal is carried "
            "by action verbs and question structure; content TF-IDF estimates "
            "topic-vocabulary transfer."
        ),
    }


def _summarise_direction(block: Dict[str, object]) -> Dict[str, object]:
    metrics = block["candidate_metrics"]
    cue = metrics["cue_only_logreg"]
    content = metrics["content_tfidf_logreg"]
    combined = metrics["combined_cue_content_logreg"]
    return {
        "cue_minus_content_macro_f1": float(cue["macro_f1"] - content["macro_f1"]),
        "cue_minus_content_within_one": float(cue["within_one_level_accuracy"] - content["within_one_level_accuracy"]),
        "cue_minus_content_severe": float(cue["severe_error_rate"] - content["severe_error_rate"]),
        "combined_minus_content_macro_f1": float(combined["macro_f1"] - content["macro_f1"]),
        "combined_minus_content_within_one": float(combined["within_one_level_accuracy"] - content["within_one_level_accuracy"]),
        "combined_minus_content_severe": float(combined["severe_error_rate"] - content["severe_error_rate"]),
    }


def main() -> None:
    max_per_label = int(os.environ.get("BLOOM_TRANSFER_MAX_PER_LABEL", "500"))
    fig = _stratified_cap(_make_reduced(_read_figshare(), "ternary"), max_per_label=max_per_label)
    mooc_raw, mooc_audit = _read_moocradar()
    mooc = _stratified_cap(_make_reduced(mooc_raw, "ternary"), max_per_label=max_per_label)
    labels = ["Low", "Mid", "High"]

    fig_to_mooc = _directional_analysis(fig, mooc, labels)
    mooc_to_fig = _directional_analysis(mooc, fig, labels)
    report = {
        "seed": SEED,
        "scheme": "ternary",
        "labels": labels,
        "max_per_label": max_per_label,
        "datasets": {
            "figshare_rows_used": int(len(fig)),
            "moocradar_rows_used": int(len(mooc)),
            "figshare_distribution": fig["label"].value_counts().to_dict(),
            "moocradar_distribution": mooc["label"].value_counts().to_dict(),
            "moocradar_audit": mooc_audit,
        },
        "figshare_to_moocradar": fig_to_mooc,
        "moocradar_to_figshare": mooc_to_fig,
        "summary": {
            "figshare_to_moocradar": _summarise_direction(fig_to_mooc),
            "moocradar_to_figshare": _summarise_direction(mooc_to_fig),
        },
        "hypothesis": (
            "Bloom levels should retain some cross-domain signal in action "
            "verbs and task structure, while content vocabulary shifts by "
            "dataset and subject matter."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "bloom_domain_shift_cue_analysis.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
