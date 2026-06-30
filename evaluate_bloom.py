#!/usr/bin/env python


from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from predict_bloom import (
    BLOOM_LABELS,
    DEFAULT_BASE_MODEL,
    DEFAULT_LORA_DIR,
    DEFAULT_MERGED_DIR,
    DEFAULT_QUANTIZED_DIR,
    QwenBloomPredictor,
    ordinal_metrics,
    is_lora_adapter,
    is_quantized_checkpoint,
    resolve_model_dir,
)


RESULTS_JSON = Path("results/bloom_lora_eval.json")
RESULTS_ROWS = Path("results/bloom_lora_eval_rows.csv")
FIG_DIR = Path("figures")


def _load_split(csv_path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    df = df[df[label_col].isin(BLOOM_LABELS)].copy()
    return df


def _evaluate_labels(y_true: list[int], y_pred: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        **ordinal_metrics(y_true, y_pred),
    }


def _run_svm_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame, text_col: str, label_col: str) -> dict:
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
        "metrics": _evaluate_labels(y_true, y_pred),
        "predictions": list(preds),
    }


def _run_lora(test_df: pd.DataFrame, text_col: str, label_col: str, predictor: QwenBloomPredictor) -> dict:
    label2id = {label: i for i, label in enumerate(BLOOM_LABELS)}
    rows = []
    y_true, y_pred = [], []
    for idx, row in test_df.iterrows():
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
        if len(rows) % 25 == 0:
            print(f"[lora] {len(rows)}/{len(test_df)}")
    return {
        "metrics": _evaluate_labels(y_true, y_pred),
        "rows": rows,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained Qwen Bloom LoRA.")
    parser.add_argument("--test-csv", type=Path, default=Path("data/figshare_bloom_v1_test.csv"))
    parser.add_argument("--train-csv", type=Path, default=Path("data/figshare_bloom_v1_train.csv"))
    parser.add_argument("--text-col", default="question")
    parser.add_argument("--label-col", default="bloom_level")
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Merged checkpoint or LoRA dir (default: models/qwen_bloom_merged if present).",
    )
    parser.add_argument("--lora-dir", default=DEFAULT_LORA_DIR)
    parser.add_argument("--merged-dir", default=DEFAULT_MERGED_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--svm-baseline", action="store_true")
    parser.add_argument("--quantized", action="store_true", help="Evaluate INT8 quantized merged model.")
    parser.add_argument("--quantized-dir", default=DEFAULT_QUANTIZED_DIR)
    parser.add_argument("--results-json", type=Path, default=None, help="Override output JSON path.")
    parser.add_argument("--max-test", type=int, default=0, help="Cap test rows (0 = all).")
    args = parser.parse_args()

    test_df = _load_split(args.test_csv, args.text_col, args.label_col)
    if args.max_test > 0:
        test_df = test_df.head(args.max_test)
    print(f"Test rows: {len(test_df)}")

    if args.quantized:
        model_dir = str(args.quantized_dir)
        if not is_quantized_checkpoint(model_dir):
            raise FileNotFoundError(f"Quantized model missing at {model_dir}. Run: python quantize_bloom.py")
        predictor = QwenBloomPredictor(model_dir=model_dir, quantized=True)
        checkpoint_type = "quantized_int8"
    else:
        model_dir = resolve_model_dir(args.model_dir)
        predictor = QwenBloomPredictor(model_dir=model_dir, base_model=args.base_model)
        checkpoint_type = "lora_adapter" if is_lora_adapter(model_dir) else "merged"
    lora = _run_lora(test_df, args.text_col, args.label_col, predictor)

    payload: dict = {
        "benchmark": "bloom_lora_figshare_v1",
        "model_dir": model_dir,
        "lora_dir": str(args.lora_dir),
        "merged_dir": str(args.merged_dir),
        "base_model": args.base_model,
        "checkpoint_type": checkpoint_type,
        "test_csv": str(args.test_csv),
        "n_test": len(test_df),
        "qwen_lora": lora["metrics"],
        "confusion_matrix": lora["confusion_matrix"],
        "limitations": [
            "Bloom labels come from the trained Qwen2.5-1.5B LoRA classifier (train_qwen_bloom.py).",
            "Teacher moderation text (reason/rewrite) is evaluated separately via bloom_prompt.py.",
            "Federated LoRA evaluation is optional (skipped when GPU/time unavailable).",
        ],
    }

    quant_meta = Path(model_dir) / "quantization.json"
    if quant_meta.is_file():
        payload["quantization"] = json.loads(quant_meta.read_text(encoding="utf-8"))

    if args.svm_baseline:
        train_df = _load_split(args.train_csv, args.text_col, args.label_col)
        svm = _run_svm_baseline(train_df, test_df, args.text_col, args.label_col)
        payload["tfidf_svm_baseline"] = svm["metrics"]
        agree = float(
            np.mean([a == b for a, b in zip(svm["predictions"], [r["prediction"] for r in lora["rows"]])])
        )
        payload["lora_svm_agreement"] = agree

    out_json = args.results_json or (Path("results/bloom_quantized_eval.json") if args.quantized else RESULTS_JSON)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not args.quantized:
        pd.DataFrame(lora["rows"]).to_csv(RESULTS_ROWS, index=False)
    _save_confusion(
        lora["confusion_matrix"],
        "Qwen LoRA Bloom Confusion Matrix" + (" (INT8)" if args.quantized else ""),
        FIG_DIR / ("bloom_quantized_confusion_matrix.png" if args.quantized else "bloom_lora_confusion_matrix.png"),
    )
    print(json.dumps({"qwen_lora": payload["qwen_lora"]}, indent=2))
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
