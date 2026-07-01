#!/usr/bin/env python
"""Merge Bloom metrics into one publication comparison table (no federated required)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bloom_model_profiles import BLOOM_MODEL_PROFILES

HOLDOUT_REPORT = Path("evaluation_outputs/evaluation_report.json")
LORA_METRICS_LEGACY = Path("evaluation_results/metrics.json")
OUT_JSON = Path("results/bloom_baseline_comparison.json")
OUT_CSV = Path("results/bloom_comparison_table.csv")
OUT_MD = Path("results/bloom_comparison_table.md")


def _normalize(metrics: dict) -> dict:
    return {
        "accuracy": round(float(metrics["accuracy"]), 4),
        "macro_f1": round(float(metrics["macro_f1"]), 4),
        "weighted_f1": round(float(metrics.get("weighted_f1", 0)), 4),
        "cohen_kappa": round(float(metrics.get("cohen_kappa", 0)), 4),
        "quadratic_weighted_kappa": round(
            float(metrics.get("quadratic_weighted_kappa", metrics.get("kappa", 0))), 4
        ),
        "ece": round(float(metrics.get("ece", 0)), 4) if metrics.get("ece") is not None else None,
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


def _load_eval_payload(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_lora_metrics(path: Path) -> tuple[dict, dict]:
    payload = _load_eval_payload(path)
    if payload is None:
        raise FileNotFoundError(f"Missing eval JSON: {path}")
    return _normalize(payload["qwen_lora"]), {
        "source": str(path),
        "model_size": payload.get("model_size"),
        "split": payload.get("test_csv", "figshare_bloom_v1_test.csv"),
        "n_test": payload.get("n_test"),
        "checkpoint_type": payload.get("checkpoint_type"),
        "base_model": payload.get("base_model"),
    }


def _primary_lora_eval() -> Path:
    primary = Path(BLOOM_MODEL_PROFILES["1.5b"].results_json)
    if primary.is_file():
        return primary
    for profile in BLOOM_MODEL_PROFILES.values():
        candidate = Path(profile.results_json)
        if candidate.is_file():
            return candidate
    if LORA_METRICS_LEGACY.is_file():
        raise FileNotFoundError(
            f"Legacy metrics at {LORA_METRICS_LEGACY} are not supported for multi-model comparison. "
            "Run: python evaluate_bloom.py --model-size 1.5b --svm-baseline"
        )
    raise FileNotFoundError(
        "Run evaluate_bloom.py for at least one model size, e.g. "
        "python evaluate_bloom.py --model-size 1.5b --svm-baseline"
    )


def build_comparison() -> dict:
    lora_eval_paths = {key: Path(profile.results_json) for key, profile in BLOOM_MODEL_PROFILES.items()}
    quant_eval_paths = {key: Path(profile.quant_results_json) for key, profile in BLOOM_MODEL_PROFILES.items()}

    primary_path = _primary_lora_eval()
    lora, lora_meta = _load_lora_metrics(primary_path)

    svm = None
    zero_shot = None
    n_holdout = None
    holdout_path = None

    for path in lora_eval_paths.values():
        eval_payload = _load_eval_payload(path)
        if eval_payload and "tfidf_svm_baseline" in eval_payload:
            svm = _normalize(eval_payload["tfidf_svm_baseline"])
            break

    if HOLDOUT_REPORT.is_file():
        holdout = json.loads(HOLDOUT_REPORT.read_text(encoding="utf-8"))
        holdout_path = str(HOLDOUT_REPORT)
        n_holdout = holdout.get("n_test") or 379
        if svm is None and "SVM" in holdout:
            svm = _normalize(holdout["SVM"])
        if "QWEN" in holdout:
            zero_shot = _normalize(holdout["QWEN"])

    lora_by_size: dict[str, dict] = {}
    lora_meta_by_size: dict[str, dict] = {}
    for key, path in lora_eval_paths.items():
        payload = _load_eval_payload(path)
        if payload is None:
            continue
        lora_by_size[key], lora_meta_by_size[key] = _load_lora_metrics(path)

    quantized_by_size: dict[str, dict] = {}
    quant_meta_by_size: dict[str, dict] = {}
    for key, path in quant_eval_paths.items():
        payload = _load_eval_payload(path)
        if payload is None:
            continue
        quantized_by_size[key] = _normalize(payload["qwen_lora"])
        quant_meta_by_size[key] = {
            "source": str(path),
            "quantization": payload.get("quantization"),
        }

    comparison = {
        "sources": {
            "trained_lora": lora_meta["source"],
            "lora_split": f"Official test split ({lora_meta.get('split')}, n={lora_meta.get('n_test')})",
            "holdout_baselines": holdout_path,
            "holdout_split": "15% stratified hold-out (figshare_combined_dataset.csv, random_state=42)",
            "lora_eval_by_size": {k: str(p) for k, p in lora_eval_paths.items() if p.is_file()},
            "quantized_eval_by_size": {k: str(p) for k, p in quant_eval_paths.items() if p.is_file()},
        },
        "n_test_lora": lora_meta.get("n_test"),
        "n_test_holdout": n_holdout,
        "lora_trained": lora,
        "trained_model_reference": lora,
        "checkpoint_type": lora_meta.get("checkpoint_type"),
        "lora_by_model_size": lora_by_size,
        "lora_meta_by_model_size": lora_meta_by_size,
    }
    if svm is not None:
        comparison["SVM"] = svm
        comparison["tfidf_svm_baseline"] = svm
    if zero_shot is not None:
        comparison["zero_shot_gguf"] = zero_shot
        comparison["QWEN"] = zero_shot
    if quantized_by_size:
        comparison["lora_quantized_by_model_size"] = quantized_by_size
        comparison["quantization_by_model_size"] = quant_meta_by_size
        if "1.5b" in quantized_by_size:
            comparison["lora_quantized_int8"] = quantized_by_size["1.5b"]
            comparison["quantization"] = quant_meta_by_size.get("1.5b")
        if "0.5b" in quantized_by_size and "lora_quantized_int8" not in comparison:
            comparison["lora_quantized_int8"] = quantized_by_size["0.5b"]
            comparison["quantization"] = quant_meta_by_size.get("0.5b")
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
    for key in ("1.5b", "0.5b"):
        metrics = comparison.get("lora_by_model_size", {}).get(key)
        if metrics is None:
            continue
        label = BLOOM_MODEL_PROFILES[key].display_name
        rows.append(
            {
                "Model": f"{label} LoRA (merged, fp32)",
                "Evaluation split": "Official test",
                **metrics,
            }
        )
    if "lora_trained" in comparison and not comparison.get("lora_by_model_size"):
        rows.append(
            {
                "Model": "Qwen2.5 LoRA (merged, fp32)",
                "Evaluation split": "Official test",
                **comparison["lora_trained"],
            }
        )
    for key in ("1.5b", "0.5b"):
        metrics = comparison.get("lora_quantized_by_model_size", {}).get(key)
        if metrics is None:
            continue
        label = BLOOM_MODEL_PROFILES[key].display_name
        rows.append(
            {
                "Model": f"{label} LoRA (INT8, CPU)",
                "Evaluation split": "Official test",
                **metrics,
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
    ]
    for key in ("1.5b", "0.5b"):
        metrics = comparison.get("lora_by_model_size", {}).get(key)
        if metrics is None:
            continue
        label = BLOOM_MODEL_PROFILES[key].display_name
        specs.append((f"{label} LoRA (merged)", "Official test", f"lora_{key}"))
        comparison[f"lora_{key}"] = metrics
    for key in ("1.5b", "0.5b"):
        metrics = comparison.get("lora_quantized_by_model_size", {}).get(key)
        if metrics is None:
            continue
        label = BLOOM_MODEL_PROFILES[key].display_name
        specs.append((f"{label} LoRA (INT8)", "Official test", f"lora_quant_{key}"))
        comparison[f"lora_quant_{key}"] = metrics
    if not comparison.get("lora_by_model_size") and "lora_trained" in comparison:
        specs.append(("Qwen2.5 LoRA (merged)", "Official test", "lora_trained"))
    if "lora_quantized_int8" in comparison and not comparison.get("lora_quantized_by_model_size"):
        specs.append(("Qwen2.5 LoRA (INT8)", "Official test", "lora_quantized_int8"))
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
