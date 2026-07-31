#!/usr/bin/env python
"""Bloom taxonomy inference for trained Qwen LoRA / merged / deploy checkpoints.

Batch evaluation lives in ``evaluate_bloom.py`` (writes ``results/``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bloom_model_profiles import (
    DEFAULT_MODEL_SIZE,
    get_profile,
    resolve_checkpoint_dir as resolve_profile_checkpoint_dir,
)


def _active_model_size(model_size: str | None = None) -> str:
    return model_size or os.environ.get("BLOOM_MODEL_SIZE") or DEFAULT_MODEL_SIZE


_DEFAULT_PROFILE = get_profile(DEFAULT_MODEL_SIZE)
DEFAULT_LORA_DIR = _DEFAULT_PROFILE.lora_dir
DEFAULT_FEDERATED_LORA_DIR = _DEFAULT_PROFILE.federated_lora_dir
DEFAULT_MERGED_DIR = _DEFAULT_PROFILE.merged_dir
DEFAULT_QUANTIZED_DIR = _DEFAULT_PROFILE.quantized_dir
DEFAULT_BASE_MODEL = _DEFAULT_PROFILE.base_model

LABELS = {
    0: "Remember",
    1: "Understand",
    2: "Apply",
    3: "Analyze",
    4: "Evaluate",
    5: "Create",
}

# Canonical order for probability vectors and UI tables.
BLOOM_LABELS: list[str] = [LABELS[i] for i in range(len(LABELS))]

# Lowercase keys used by RAGGenerator / CognitiveSummarizer.
BLOOM_LEVELS: list[str] = [
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
]

LABEL_TO_RAG_KEY: dict[str, str] = dict(zip(BLOOM_LABELS, BLOOM_LEVELS))
RAG_KEY_TO_LABEL: dict[str, str] = {v: k for k, v in LABEL_TO_RAG_KEY.items()}

# Back-compat alias
DEFAULT_MODEL_DIR = DEFAULT_MERGED_DIR


def _load_tokenizer(model_path: str, *, fallback_path: str | None = None):
    kwargs = {"trust_remote_code": True}
    candidates = [model_path]
    if fallback_path and fallback_path != model_path:
        candidates.append(fallback_path)

    last_err: Exception | None = None
    for path in candidates:
        try:
            try:
                return AutoTokenizer.from_pretrained(path, fix_mistral_regex=True, **kwargs)
            except TypeError:
                return AutoTokenizer.from_pretrained(path, **kwargs)
        except Exception as exc:  # noqa: BLE001 — fall back to base tokenizer if merge copy is corrupt
            last_err = exc
            print(f"[warn] tokenizer load failed for {path}: {exc}")
    raise RuntimeError(f"Could not load tokenizer from {candidates}") from last_err


def build_prompt(question: str) -> str:
    """Canonical Bloom classifier prompt — MUST match train_qwen_bloom exactly.

    Train/val metrics (~84%) were produced with this template. Using a shorter
    ``Answer:`` variant at inference collapses held-out accuracy (~49%).
    """
    return (
        "You are an expert in educational assessment and Bloom's Taxonomy.\n\n"
        "Your task is to classify the following question into exactly one of the six Bloom's Taxonomy cognitive levels.\n\n"
        "Bloom's Taxonomy Levels:\n"
        "- Remember: Recall facts, definitions, or basic concepts.\n"
        "- Understand: Explain ideas or interpret concepts.\n"
        "- Apply: Use knowledge to solve problems or complete tasks.\n"
        "- Analyze: Break information into parts, identify relationships, or compare concepts.\n"
        "- Evaluate: Make judgments, justify decisions, or critique based on evidence.\n"
        "- Create: Generate, design, develop, or produce something original.\n\n"
        "Focus on the reasoning required to answer the question rather than relying only on action verbs.\n\n"
        f"Question:\n{question}\n\n"
        "Bloom Level:"
    )


def is_lora_adapter(model_dir: str | os.PathLike) -> bool:
    return (Path(model_dir) / "adapter_config.json").is_file()


def is_deploy_checkpoint(model_dir: str | os.PathLike) -> bool:
    path = Path(model_dir)
    meta = path / "quantization.json"
    if meta.is_file():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        if payload.get("format") == "fp16_merged" and (path / "config.json").is_file():
            return True
        if payload.get("format") == "torch_dynamic_int8" and (path / "model_int8.pt").is_file():
            return True
    return (path / "model_int8.pt").is_file()


def load_deploy_model(model_dir: str):
    path = Path(model_dir)
    if not is_deploy_checkpoint(path):
        raise FileNotFoundError(
            f"No lightweight deploy model at {path}. Run: python quantize_bloom.py --model-size 0.5b --force"
        )

    meta_path = path / "quantization.json"
    fmt = ""
    if meta_path.is_file():
        fmt = json.loads(meta_path.read_text(encoding="utf-8")).get("format", "")

    print(f"\nLoading lightweight Bloom classifier from {path} ({fmt or 'legacy'})...")
    tokenizer = _load_tokenizer(str(path), fallback_path=DEFAULT_BASE_MODEL)

    if fmt == "fp16_merged":
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path),
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model.eval().cpu()
        return tokenizer, model

    if (path / "model_int8.pt").is_file():
        model = torch.load(path / "model_int8.pt", map_location="cpu", weights_only=False)
        model.eval().cpu()
        return tokenizer, model

    raise FileNotFoundError(f"Unsupported deploy format at {path}")


def resolve_model_dir(
    model_dir: str | None = None,
    *,
    prefer_merged: bool = True,
    prefer_quantized: bool = False,
    model_size: str | None = None,
) -> str:
    profile = get_profile(_active_model_size(model_size))
    return resolve_profile_checkpoint_dir(
        profile,
        model_dir=model_dir,
        prefer_quantized=prefer_quantized,
    )


def load_model(model_dir: str, base_model: str | None = None, *, quantized: bool = False):
    if quantized or is_deploy_checkpoint(model_dir):
        return load_deploy_model(model_dir)

    path = Path(model_dir)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if is_lora_adapter(path):
        if not base_model:
            base_model = DEFAULT_BASE_MODEL
        print(f"\nLoading LoRA adapter from {path} (base={base_model})...")
        tokenizer = _load_tokenizer(base_model)
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=6,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, str(path))
    else:
        if not (path / "config.json").is_file():
            raise FileNotFoundError(
                f"No merged model at {path}. Run: python merge_model.py"
            )
        print(f"\nLoading merged Bloom classifier from {path}...")
        tokenizer = _load_tokenizer(str(path), fallback_path=base_model or DEFAULT_BASE_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path),
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return tokenizer, model


def predict(question, tokenizer, model, max_length=256):
    prompt = build_prompt(question)

    inputs = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()

    confidence = probs[pred_id].item()

    return {
        "prediction": LABELS[pred_id],
        "confidence": round(confidence, 4),
        "probabilities": {
            LABELS[i]: round(probs[i].item(), 4)
            for i in range(len(LABELS))
        },
    }


def probabilities_to_distribution(probabilities: dict[str, float]) -> np.ndarray:
    """Map label probabilities to a (6,) vector in ``BLOOM_LEVELS`` order."""
    return np.asarray(
        [float(probabilities.get(RAG_KEY_TO_LABEL[level], 0.0)) for level in BLOOM_LEVELS],
        dtype=np.float32,
    )


def label_to_rag_key(label: str) -> str:
    key = (label or "").strip()
    if key.lower() in BLOOM_LEVELS:
        return key.lower()
    return LABEL_TO_RAG_KEY.get(key, "understand")


class QwenBloomPredictor:
    """Lazy-loaded Qwen2.5 Bloom classifier (merged, LoRA, or INT8 quantized)."""

    def __init__(
        self,
        model_dir: str | None = None,
        base_model: str | None = None,
        *,
        model_size: str | None = None,
        prefer_merged: bool = True,
        prefer_quantized: bool = False,
        quantized: bool = False,
    ) -> None:
        self.profile = get_profile(_active_model_size(model_size))
        self.quantized = quantized or prefer_quantized
        self.model_dir = resolve_model_dir(
            model_dir,
            prefer_merged=prefer_merged,
            prefer_quantized=self.quantized,
            model_size=self.profile.key,
        )
        self.base_model = base_model or self.profile.base_model
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self.quantized or is_deploy_checkpoint(self.model_dir):
            self._tokenizer, self._model = load_deploy_model(self.model_dir)
            return
        base = self.base_model if is_lora_adapter(self.model_dir) else None
        self._tokenizer, self._model = load_model(self.model_dir, base)

    def predict(self, question: str, max_length: int = 256) -> dict:
        self._ensure_loaded()
        raw = predict(question, self._tokenizer, self._model, max_length=max_length)
        dist = probabilities_to_distribution(raw["probabilities"])
        return {
            **raw,
            "rag_key": label_to_rag_key(raw["prediction"]),
            "distribution": dist,
        }


def ordinal_metrics(y_true, y_pred) -> dict:
    distances = [abs(t - p) for t, p in zip(y_true, y_pred)]
    mean_distance = np.mean(distances)
    within_one = np.mean([d <= 1 for d in distances])
    severe_error = np.mean([d >= 3 for d in distances])
    return {
        "mean_ordinal_distance": round(float(mean_distance), 4),
        "within_one_level_accuracy": round(float(within_one), 4),
        "severe_error_rate": round(float(severe_error), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Interactive Bloom classifier (use evaluate_bloom.py for batch eval).")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Merged dir or LoRA adapter dir (default: merged if present).",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    base = args.base_model if is_lora_adapter(model_dir) else None
    tokenizer, model = load_model(model_dir, base)

    print("\n==============================")
    print("INTERACTIVE BLOOM CLASSIFIER")
    print("==============================")
    print("(Batch evaluation: python evaluate_bloom.py)")

    while True:
        q = input("\nEnter question (or 'exit'): ")
        if q.lower() == "exit":
            break

        result = predict(q, tokenizer, model)
        print("\nPrediction:")
        print(result["prediction"])
        print("\nConfidence:")
        print(result["confidence"])
        print("\nClass Probabilities:")
        for k, v in result["probabilities"].items():
            print(f"{k:12s} -> {v}")


if __name__ == "__main__":
    main()
