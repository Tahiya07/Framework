#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from bloom_eval_metrics import (
    compare_model_predictions,
    evaluate_predictions,
    mcnemar_test,
)
from bloom_model_profiles import (
    BLOOM_MODEL_PROFILES,
    COMBINED_COMPARISON_TABLE_FIG,
    get_profile,
    resolve_checkpoint_dir,
)
from predict_bloom import (
    BLOOM_LABELS,
    QwenBloomPredictor,
    is_lora_adapter,
    is_quantized_checkpoint,
)

FIG_DIR = Path("figures")

METRIC_COLUMNS: list[tuple[str, str]] = [
    ("Accuracy", "accuracy"),
    ("Macro-F1", "macro_f1"),
    ("QWK", "quadratic_weighted_kappa"),
    ("Within-one", "within_one_level_accuracy"),
    ("Severe error", "severe_error_rate"),
    ("ECE", "ece"),
]


def _load_split(csv_path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    df = df[df[label_col].isin(BLOOM_LABELS)].copy()
    return df


def _run_svm_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    *,
    bootstrap_samples: int,
) -> dict:
    label2id = {label: i for i, label in enumerate(BLOOM_LABELS)}
    clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
            ("clf", LinearSVC(class_weight="balanced")),
        ]
    )
    clf.fit(train_df[text_col], train_df[label_col])
    preds = clf.predict(test_df[text_col])
    y_true = [label2id[x] for x in test_df[label_col]]
    y_pred = [label2id[x] for x in preds]
    return {
        "metrics": evaluate_predictions(y_true, y_pred, bootstrap_samples=bootstrap_samples),
        "predictions": list(preds),
        "y_pred": y_pred,
    }


def _run_lora(
    test_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    predictor: QwenBloomPredictor,
    *,
    bootstrap_samples: int,
) -> dict:
    label2id = {label: i for i, label in enumerate(BLOOM_LABELS)}
    rows = []
    y_true, y_pred, confidences = [], [], []
    for _, row in test_df.iterrows():
        text = str(row[text_col])
        gold = str(row[label_col])
        out = predictor.predict(text)
        pred = out["prediction"]
        rows.append(
            {
                "question": text,
                "gold": gold,
                "prediction": pred,
                "confidence": out["confidence"],
                "rag_key": out["rag_key"],
            }
        )
        y_true.append(label2id[gold])
        y_pred.append(label2id[pred])
        confidences.append(float(out["confidence"]))
        if len(rows) % 25 == 0:
            print(f"[lora] {len(rows)}/{len(test_df)}")
    return {
        "metrics": evaluate_predictions(
            y_true, y_pred, confidences=confidences, bootstrap_samples=bootstrap_samples
        ),
        "rows": rows,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confidences,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=BLOOM_LABELS,
            digits=4,
            zero_division=0,
        ),
    }


def _save_confusion(cm: list[list[int]], title: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(BLOOM_LABELS)), BLOOM_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(BLOOM_LABELS)), BLOOM_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_per_class_table(per_class: dict[str, dict[str, float]], title: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = []
    for label in BLOOM_LABELS:
        stats = per_class.get(label, {})
        rows.append(
            [
                label,
                f"{stats.get('precision', 0.0):.3f}",
                f"{stats.get('recall', 0.0):.3f}",
                f"{stats.get('f1', 0.0):.3f}",
                str(int(stats.get("support", 0))),
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Bloom level", "Precision", "Recall", "F1", "Support"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
    ax.set_title(title, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_reliability_plot(calibration: dict, title: str, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    bins = calibration.get("reliability_bins", {})
    conf = bins.get("bin_confidence", [])
    acc = bins.get("bin_accuracy", [])
    if not conf or not acc:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#888888", label="Perfect calibration")
    ax.plot(conf, acc, "o-", color="#1f77b4", label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={calibration.get('ece', 0.0):.3f}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _format_metric(metrics: dict, key: str) -> str:
    value = metrics.get(key)
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def _metrics_row(label: str, metrics: dict) -> list[str]:
    return [label] + [_format_metric(metrics, key) for _, key in METRIC_COLUMNS]


def _load_eval_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_svm_baseline(current_payload: dict) -> dict | None:
    svm = current_payload.get("tfidf_svm_baseline")
    if svm is not None:
        return svm
    for profile in BLOOM_MODEL_PROFILES.values():
        payload = _load_eval_json(Path(profile.results_json))
        if payload and "tfidf_svm_baseline" in payload:
            return payload["tfidf_svm_baseline"]
    return None


def _build_comparison_rows(
    current_payload: dict,
    current_profile,
    *,
    quantized: bool,
) -> list[list[str]]:
    rows: list[list[str]] = []
    svm = _find_svm_baseline(current_payload)
    if svm is not None:
        rows.append(_metrics_row("TF-IDF + LinearSVC", svm))

    for key in ("1.5b", "0.5b"):
        profile = BLOOM_MODEL_PROFILES[key]
        if quantized and key == current_profile.key:
            payload = current_payload
            label = f"{profile.display_name} LoRA (INT8)"
        else:
            payload = _load_eval_json(Path(profile.results_json))
            if payload is None:
                if key == current_profile.key and not quantized:
                    payload = current_payload
                else:
                    continue
            label = f"{profile.display_name} LoRA"
        rows.append(_metrics_row(label, payload["qwen_lora"]))
    return rows


def _comparison_subtitle(payload: dict) -> str | None:
    parts: list[str] = []
    if payload.get("lora_svm_agreement") is not None:
        parts.append(f"LoRA–SVM agreement: {payload['lora_svm_agreement']:.3f}")
    macro_ci = payload.get("qwen_lora", {}).get("bootstrap_ci", {}).get("macro_f1")
    if macro_ci:
        macro = payload["qwen_lora"]["macro_f1"]
        parts.append(f"Macro-F1 {macro:.3f} [{macro_ci['ci_low']:.3f}, {macro_ci['ci_high']:.3f}]")
    return " | ".join(parts) if parts else None


def _save_comparison_table(
    rows: list[list[str]],
    *,
    title: str,
    path: Path,
    subtitle: str | None = None,
) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Model"] + [name for name, _ in METRIC_COLUMNS]
    fig_w = max(12.0, 1.2 * len(header))
    fig_h = max(2.8, 0.45 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=header,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif col == 0:
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f8fc")
    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=11, pad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _build_statistical_tests(
    lora: dict,
    *,
    svm: dict | None,
    profile_key: str,
) -> dict:
    tests: dict = {
        "bootstrap": lora["metrics"].get("bootstrap_ci", {}),
    }
    if svm is not None:
        tests["mcnemar_vs_svm"] = mcnemar_test(lora["y_true"], lora["y_pred"], svm["y_pred"])

    other_key = "0.5b" if profile_key == "1.5b" else "1.5b"
    other_profile = BLOOM_MODEL_PROFILES[other_key]
    other_rows_path = Path(other_profile.results_rows)
    if other_rows_path.is_file():
        other_rows = pd.read_csv(other_rows_path).to_dict("records")
        paired = compare_model_predictions(lora["rows"], other_rows)
        if paired is not None:
            tests[f"mcnemar_vs_{other_key}"] = {
                **paired,
                "reference_model": other_profile.display_name,
            }
    return tests


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained Qwen Bloom LoRA.")
    parser.add_argument(
        "--model-size",
        choices=sorted(BLOOM_MODEL_PROFILES),
        default="1.5b",
        help="Model variant: 0.5b or 1.5b (sets default paths and output files).",
    )
    parser.add_argument("--test-csv", type=Path, default=Path("data/figshare_bloom_v1_test.csv"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/figshare_bloom_v1_train.csv"))
    parser.add_argument("--text-col", default="question")
    parser.add_argument("--label-col", default="bloom_level")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Merged checkpoint or LoRA dir (default: profile merged dir if present).",
    )
    parser.add_argument("--lora-dir", default=None, help="Override profile LoRA training dir.")
    parser.add_argument("--merged-dir", default=None, help="Override profile merged checkpoint dir.")
    parser.add_argument("--base-model", default=None, help="Override Hugging Face base model id.")
    parser.add_argument("--svm-baseline", action="store_true")
    parser.add_argument("--quantized", action="store_true", help="Evaluate INT8 quantized merged model.")
    parser.add_argument("--quantized-dir", default=None, help="Override profile quantized dir.")
    parser.add_argument("--results-json", type=Path, default=None, help="Override output JSON path.")
    parser.add_argument("--max-test", type=int, default=0, help="Cap test rows (0 = all).")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap resamples for 95%% confidence intervals.",
    )
    args = parser.parse_args()

    profile = get_profile(args.model_size)
    lora_dir = args.lora_dir or profile.lora_dir
    merged_dir = args.merged_dir or profile.merged_dir
    base_model = args.base_model or profile.base_model

    test_df = _load_split(args.test_csv, args.text_col, args.label_col)
    if args.max_test > 0:
        test_df = test_df.head(args.max_test)
    print(f"[eval] model_size={profile.key} ({profile.display_name})")
    print(f"Test rows: {len(test_df)}")

    if args.quantized:
        model_dir = resolve_checkpoint_dir(profile, model_dir=args.quantized_dir, quantized=True)
        if not is_quantized_checkpoint(model_dir):
            raise FileNotFoundError(
                f"Quantized model missing at {model_dir}. "
                f"Run: python quantize_bloom.py --model-size {profile.key}"
            )
        predictor = QwenBloomPredictor(model_dir=model_dir, quantized=True)
        checkpoint_type = "quantized_int8"
    else:
        model_dir = resolve_checkpoint_dir(profile, model_dir=args.model_dir, quantized=False)
        predictor = QwenBloomPredictor(model_dir=model_dir, base_model=base_model)
        checkpoint_type = "lora_adapter" if is_lora_adapter(model_dir) else "merged"
    lora = _run_lora(
        test_df,
        args.text_col,
        args.label_col,
        predictor,
        bootstrap_samples=args.bootstrap_samples,
    )

    svm_result = None
    if args.svm_baseline:
        train_df = _load_split(args.train_csv, args.text_col, args.label_col)
        svm_result = _run_svm_baseline(
            train_df,
            test_df,
            args.text_col,
            args.label_col,
            bootstrap_samples=args.bootstrap_samples,
        )

    payload: dict = {
        "benchmark": "bloom_lora_figshare_v1",
        "model_size": profile.key,
        "model_dir": model_dir,
        "lora_dir": lora_dir,
        "merged_dir": merged_dir,
        "base_model": base_model,
        "checkpoint_type": checkpoint_type,
        "test_csv": str(args.test_csv),
        "n_test": len(test_df),
        "qwen_lora": lora["metrics"],
        "confusion_matrix": lora["confusion_matrix"],
        "classification_report": lora["classification_report"],
        "statistical_tests": _build_statistical_tests(lora, svm=svm_result, profile_key=profile.key),
        "limitations": [
            f"Bloom labels come from the trained {profile.display_name} LoRA classifier (train_qwen_bloom.py).",
            "Teacher moderation text (reason/rewrite) is evaluated separately via bloom_prompt.py.",
            "Federated LoRA evaluation is optional (skipped when GPU/time unavailable).",
            "McNemar vs the other model size requires both evaluate_bloom.py runs on the same test CSV.",
        ],
    }

    quant_meta = Path(model_dir) / "quantization.json"
    if quant_meta.is_file():
        payload["quantization"] = json.loads(quant_meta.read_text(encoding="utf-8"))

    if svm_result is not None:
        payload["tfidf_svm_baseline"] = svm_result["metrics"]
        agree = float(
            np.mean([a == b for a, b in zip(svm_result["predictions"], [r["prediction"] for r in lora["rows"]])])
        )
        payload["lora_svm_agreement"] = agree

    out_json = args.results_json or (
        Path(profile.quant_results_json) if args.quantized else Path(profile.results_json)
    )
    results_rows = Path(profile.results_rows)
    confusion_fig = Path(profile.quant_confusion_fig if args.quantized else profile.confusion_fig)
    per_class_fig = FIG_DIR / f"bloom_per_class_f1_{profile.key.replace('.', '')}.png"
    reliability_fig = FIG_DIR / f"bloom_reliability_{profile.key.replace('.', '')}.png"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.quantized:
        results_rows.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(lora["rows"]).to_csv(results_rows, index=False)

    title = f"Qwen LoRA Bloom Confusion Matrix ({profile.display_name})"
    if args.quantized:
        title += " (INT8)"
    _save_confusion(lora["confusion_matrix"], title, confusion_fig)
    _save_per_class_table(
        lora["metrics"].get("per_class", {}),
        f"Per-class F1 ({profile.display_name})",
        per_class_fig,
    )
    calibration = lora["metrics"].get("calibration")
    if calibration:
        _save_reliability_plot(
            calibration,
            f"Reliability diagram ({profile.display_name})",
            reliability_fig,
        )

    comparison_rows = _build_comparison_rows(payload, profile, quantized=args.quantized)
    comparison_subtitle = _comparison_subtitle(payload)
    comparison_title = f"Bloom Evaluation Comparison ({profile.display_name}, n={len(test_df)})"
    comparison_fig = Path(profile.comparison_table_fig)
    combined_fig = Path(COMBINED_COMPARISON_TABLE_FIG)
    _save_comparison_table(
        comparison_rows,
        title=comparison_title,
        subtitle=comparison_subtitle,
        path=comparison_fig,
    )
    _save_comparison_table(
        comparison_rows,
        title="Bloom Evaluation Comparison (0.5B vs 1.5B)",
        subtitle=comparison_subtitle,
        path=combined_fig,
    )

    summary = {
        "model_size": profile.key,
        "accuracy": payload["qwen_lora"]["accuracy"],
        "macro_f1": payload["qwen_lora"]["macro_f1"],
        "quadratic_weighted_kappa": payload["qwen_lora"]["quadratic_weighted_kappa"],
        "ece": payload["qwen_lora"].get("ece"),
        "bootstrap_macro_f1": payload["qwen_lora"].get("bootstrap_ci", {}).get("macro_f1"),
        "statistical_tests": payload["statistical_tests"],
    }
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_json}")
    if not args.quantized:
        print(f"Wrote {results_rows}")
    print(f"Wrote {confusion_fig}")
    print(f"Wrote {per_class_fig}")
    if calibration:
        print(f"Wrote {reliability_fig}")
    print(f"Wrote {comparison_fig}")
    print(f"Wrote {combined_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
