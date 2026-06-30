#!/usr/bin/env python
"""Merge Qwen Bloom LoRA adapter into a single deployable classifier checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_LORA_DIR = "models/qwen_bloom_trained"
DEFAULT_FEDERATED_LORA_DIR = "models/qwen_bloom_federated"
DEFAULT_MERGED_DIR = "models/qwen_bloom_merged"


def resolve_lora_dir(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    fed = Path(DEFAULT_FEDERATED_LORA_DIR)
    if (fed / "adapter_config.json").is_file():
        return fed
    return Path(DEFAULT_LORA_DIR)

LABELS = {
    "Remember": 0,
    "Understand": 1,
    "Apply": 2,
    "Analyze": 3,
    "Evaluate": 4,
    "Create": 5,
}
ID2LABEL = {v: k for k, v in LABELS.items()}


def merge_lora(
    *,
    lora_dir: str | Path = DEFAULT_LORA_DIR,
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: str | Path = DEFAULT_MERGED_DIR,
    force: bool = False,
) -> Path:
    lora_path = Path(lora_dir)
    out_path = Path(output_dir)

    if not (lora_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter not found at {lora_path}. Train with train_qwen_bloom.py first."
        )

    if out_path.is_dir() and (out_path / "config.json").is_file() and not force:
        print(f"[merge] Using existing merged model at {out_path}")
        return out_path

    print(f"[merge] Base: {base_model}")
    print(f"[merge] LoRA: {lora_path}")
    print(f"[merge] Out:  {out_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABELS,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(lora_path))
    merged = model.merge_and_unload()

    out_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"[merge] Saved merged classifier -> {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge Bloom LoRA into a full model folder.")
    parser.add_argument(
        "--lora-dir",
        default=None,
        help="LoRA adapter (default: federated if present, else qwen_bloom_3000).",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_MERGED_DIR)
    parser.add_argument("--force", action="store_true", help="Re-merge even if output exists.")
    args = parser.parse_args()
    lora_dir = resolve_lora_dir(args.lora_dir)
    merge_lora(
        lora_dir=lora_dir,
        base_model=args.base_model,
        output_dir=args.output_dir,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
