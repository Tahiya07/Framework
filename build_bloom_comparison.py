#!/usr/bin/env python
"""Merge Bloom metrics from evaluation_outputs/ and evaluation_results/ into one table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HOLDOUT_REPORT = Path("evaluation_outputs/evaluation_report.json")
LORA_METRICS = Path("evaluation_results/metrics.json")
OUT_JSON = Path("results/bloom_baseline_comparison.json")
OUT_CSV = Path("evaluation_outputs/results_table.csv")
OUT_MD = Path("results/bloom_comparison_table.md")


def _normalize_holdout(metrics: dict) -> dict:
    return {
        "accuracy": round(float(metrics["accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "weighted_f1": round(float(metrics.get("weighted_f1", 0)), 4),
        "mean_ordinal_distance": round(
            float(metrics.get("ordinal_error", metrics.get("mean_ordinal_distance", 0))), 4
        ),
        "within_one_level_accuracy": round(
            float(metrics.get("within_one_level", metrics.get("within_one_level_accuracy", 0))), 4
        ),
        "severe_error_rate": round(
            float(metrics.get("severe_error", metrics.get("severe_error_rate", 0))), 4
        ),
        "kappa": round(float(metrics.get("kappa", 0)), 4) if metrics.get("kappa") is not None else None,
    }


def _normalize_lora(metrics: dict) -> dict:
    return {
        "accuracy": round(float(metrics["accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "weighted_f1": round(float(metrics.get("weighted_f1", 0)), 4),
        "mean_ordinal_distance": round(float(metrics["mean_ordinal_distance"]), 4),
        "within_one_level_accuracy": round(float(metrics["within_one_level_accuracy"]), 4),
        "severe_error_rate": round(float(metrics["severe_error_rate"]), 4),
    }


def build_comparison() -> dict:
    if not HOLDOUT_REPORT.is_file():
        raise FileNotFoundError(f"Missing {HOLDOUT_REPORT}. Restore from git: git checkout HEAD -- evaluation_outputs/")
    if not LORA_METRICS.is_file():
        raise FileNotFoundError(f"Missing {LORA_METRICS}. Run train_qwen_bloom.py evaluation first.")

    holdout = json.loads(HOLDOUT_REPORT.read_text(encoding="utf-8"))
    lora_raw = json.loads(LORA_METRICS.read_text(encoding="utf-8"))

    svm = _normalize_holdout(holdout["SVM"])
    zero_shot = _normalize_holdout(holdout["QWEN"])
    lora = _normalize_lora(lora_raw)

    comparison = {
        "sources": {
            "holdout_baselines": str(HOLDOUT_REPORT),
            "trained_lora": str(LORA_METRICS),
            "holdout_split": "15% stratified hold-out (figshare_combined_dataset.csv, random_state=42)",
            "lora_split": "Official Figshare test split (evaluation_results from predict_bloom / training eval)",
        },
        "n_test_holdout": holdout.get("n_test") or 379,
        "n_test_lora": 2330,
        "SVM": svm,
        "zero_shot_gguf": zero_shot,
        "QWEN": zero_shot,
        "lora_trained": lora,
        "trained_model_reference": lora_raw,
        "agreement_rate": holdout.get("agreement_rate"),
        "svm_zero_shot_agreement": holdout.get("agreement_rate"),
        "num_disagreements": holdout.get("num_disagreements"),
    }
    return comparison


def _write_csv(comparison: dict) -> None:
    rows = [
        {
            "Model": "TF-IDF + LinearSVC (SVM)",
            "Evaluation split": "15% hold-out",
            **comparison["SVM"],
        },
        {
            "Model": "Qwen2.5 zero-shot (GGUF)",
            "Evaluation split": "15% hold-out",
            **comparison["zero_shot_gguf"],
        },
        {
            "Model": "Qwen2.5 LoRA (trained)",
            "Evaluation split": "Official test",
            **comparison["lora_trained"],
        },
    ]
    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)


def _write_md(comparison: dict) -> None:
    lines = [
        "# Bloom taxonomy comparison",
        "",
        "| Model | Split | Accuracy | Macro-F1 | Within-one | Severe error | Ordinal dist. |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, split, key in (
        ("TF-IDF + LinearSVC", "15% hold-out", "SVM"),
        ("Qwen2.5 zero-shot (GGUF)", "15% hold-out", "zero_shot_gguf"),
        ("Qwen2.5 LoRA (trained)", "Official test", "lora_trained"),
    ):
        m = comparison[key]
        lines.append(
            f"| {model} | {split} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
            f"{m['within_one_level_accuracy']:.3f} | {m['severe_error_rate']:.3f} | "
            f"{m['mean_ordinal_distance']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"SVM vs zero-shot agreement: {comparison.get('agreement_rate', 'n/a')}",
            "",
            "Hold-out baselines: `evaluation_outputs/evaluation_report.json`",
            "Trained LoRA: `evaluation_results/metrics.json`",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    comparison = build_comparison()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    _write_csv(comparison)
    _write_md(comparison)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
