#!/usr/bin/env python
"""Merge Bloom metrics into one publication comparison table (no federated required)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HOLDOUT_REPORT = Path("evaluation_outputs/evaluation_report.json")
LORA_EVAL = Path("results/bloom_lora_eval.json")
LORA_METRICS_LEGACY = Path("evaluation_results/metrics.json")
QUANT_EVAL = Path("results/bloom_quantized_eval.json")
OUT_JSON = Path("results/bloom_baseline_comparison.json")
OUT_CSV = Path("results/bloom_comparison_table.csv")
OUT_MD = Path("results/bloom_comparison_table.md")


def _normalize(metrics: dict) -> dict:
    return {
        "accuracy": round(float(metrics["accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "weighted_f1": round(float(metrics.get("weighted_f1", 0)), 4),
        "mean_ordinal_distance": round(
            float(metrics.get("mean_ordinal_distance", metrics.get("ordinal_error", 0))), 4
        ),
        "within_one_level_accuracy": round(
            float(metrics.get("within_one_level_accuracy", metrics.get("within_one_level", 0))), 4
        ),
        "severe_error_rate": round(
            float(metrics.get("severe_error_rate", metrics.get("severe_error", 0))), 4
        ),
    }


def _load_lora_metrics() -> tuple[dict, dict]:
    if LORA_EVAL.is_file():
        payload = json.loads(LORA_EVAL.read_text(encoding="utf-8"))
        return _normalize(payload["qwen_lora"]), {
            "source": str(LORA_EVAL),
            "split": payload.get("test_csv", "figshare_bloom_v1_test.csv"),
            "n_test": payload.get("n_test"),
            "checkpoint_type": payload.get("checkpoint_type"),
        }
    if LORA_METRICS_LEGACY.is_file():
        raw = json.loads(LORA_METRICS_LEGACY.read_text(encoding="utf-8"))
        return _normalize(raw), {
            "source": str(LORA_METRICS_LEGACY),
            "split": "legacy evaluation_results",
            "n_test": None,
            "checkpoint_type": "merged",
        }
    raise FileNotFoundError(f"Run: python evaluate_bloom.py --svm-baseline (writes {LORA_EVAL})")


def build_comparison() -> dict:
    lora, lora_meta = _load_lora_metrics()

    svm = None
    zero_shot = None
    n_holdout = None
    holdout_path = None

    if LORA_EVAL.is_file():
        eval_payload = json.loads(LORA_EVAL.read_text(encoding="utf-8"))
        if "tfidf_svm_baseline" in eval_payload:
            svm = _normalize(eval_payload["tfidf_svm_baseline"])

    if HOLDOUT_REPORT.is_file():
        holdout = json.loads(HOLDOUT_REPORT.read_text(encoding="utf-8"))
        holdout_path = str(HOLDOUT_REPORT)
        n_holdout = holdout.get("n_test") or 379
        if svm is None and "SVM" in holdout:
            svm = _normalize(holdout["SVM"])
        if "QWEN" in holdout:
            zero_shot = _normalize(holdout["QWEN"])

    quantized = None
    quant_meta = None
    if QUANT_EVAL.is_file():
        qpayload = json.loads(QUANT_EVAL.read_text(encoding="utf-8"))
        quantized = _normalize(qpayload["qwen_lora"])
        quant_meta = {
            "source": str(QUANT_EVAL),
            "quantization": qpayload.get("quantization"),
        }

    comparison = {
        "sources": {
            "trained_lora": lora_meta["source"],
            "lora_split": f"Official test split ({lora_meta.get('split')}, n={lora_meta.get('n_test')})",
            "holdout_baselines": holdout_path,
            "holdout_split": "15% stratified hold-out (figshare_combined_dataset.csv, random_state=42)",
            "quantized_eval": str(QUANT_EVAL) if quantized else None,
        },
        "n_test_lora": lora_meta.get("n_test"),
        "n_test_holdout": n_holdout,
        "lora_trained": lora,
        "trained_model_reference": lora,
        "checkpoint_type": lora_meta.get("checkpoint_type"),
    }
    if svm is not None:
        comparison["SVM"] = svm
        comparison["tfidf_svm_baseline"] = svm
    if zero_shot is not None:
        comparison["zero_shot_gguf"] = zero_shot
        comparison["QWEN"] = zero_shot
    if quantized is not None:
        comparison["lora_quantized_int8"] = quantized
        comparison["quantization"] = quant_meta
    return comparison


def _write_csv(comparison: dict) -> None:
    rows = []
    if "SVM" in comparison:
        rows.append({"Model": "TF-IDF + LinearSVC", "Evaluation split": "train→test (Figshare)", **comparison["SVM"]})
    if "zero_shot_gguf" in comparison:
        rows.append(
            {
                "Model": "Qwen2.5 zero-shot (GGUF)",
                "Evaluation split": "15% hold-out",
                **comparison["zero_shot_gguf"],
            }
        )
    rows.append(
        {
            "Model": "Qwen2.5 LoRA (merged, fp32)",
            "Evaluation split": "Official test",
            **comparison["lora_trained"],
        }
    )
    if "lora_quantized_int8" in comparison:
        rows.append(
            {
                "Model": "Qwen2.5 LoRA (INT8, CPU)",
                "Evaluation split": "Official test",
                **comparison["lora_quantized_int8"],
            }
        )
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)


def _write_md(comparison: dict) -> None:
    lines = [
        "# Bloom taxonomy comparison",
        "",
        "| Model | Split | Accuracy | Macro-F1 | Within-one | Severe error | Ordinal dist. |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    specs = [
        ("TF-IDF + LinearSVC", "train→test", "SVM"),
        ("Qwen2.5 zero-shot (GGUF)", "15% hold-out", "zero_shot_gguf"),
        ("Qwen2.5 LoRA (merged)", "Official test", "lora_trained"),
        ("Qwen2.5 LoRA (INT8)", "Official test", "lora_quantized_int8"),
    ]
    for model, split, key in specs:
        if key not in comparison:
            continue
        m = comparison[key]
        lines.append(
            f"| {model} | {split} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
            f"{m['within_one_level_accuracy']:.3f} | {m['severe_error_rate']:.3f} | "
            f"{m['mean_ordinal_distance']:.3f} |"
        )
    lines.extend(["", f"Primary LoRA eval: `{comparison['sources']['trained_lora']}`"])
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
