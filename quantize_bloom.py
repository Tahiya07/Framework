#!/usr/bin/env python
"""Quantize merged Bloom classifier for lightweight CPU deployment (dynamic INT8)."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bloom_model_profiles import BLOOM_MODEL_PROFILES, get_profile
from predict_bloom import build_prompt, predict
BENCHMARK_QUESTION = (
    "Compare and contrast the advantages of array-based and linked-list implementations "
    "of a stack. Justify which you would choose for a memory-constrained embedded system."
)


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def quantize_merged(
    merged_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    merged_path = Path(merged_dir)
    out_path = Path(output_dir)

    if not (merged_path / "config.json").is_file():
        raise FileNotFoundError(f"Merged model not found at {merged_path}. Run: python merge_model.py")

    artifact = out_path / "model_int8.pt"
    if artifact.is_file() and not force:
        print(f"[quantize] Using existing quantized model at {out_path}")
        return out_path

    print(f"[quantize] Loading merged model from {merged_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(merged_path), trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(merged_path),
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    print("[quantize] Applying dynamic INT8 quantization (Linear layers)...")
    quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_path)
    shutil.copy2(merged_path / "config.json", out_path / "config.json")
    torch.save(quantized, out_path / "model_int8.pt")

    merged_size = _dir_size_bytes(merged_path)
    quant_size = _dir_size_bytes(out_path)
    meta = {
        "format": "torch_dynamic_int8",
        "source_merged_dir": str(merged_path),
        "merged_size_mb": round(merged_size / (1024**2), 2),
        "quantized_size_mb": round(quant_size / (1024**2), 2),
        "compression_ratio": round(merged_size / max(quant_size, 1), 2),
    }
    (out_path / "quantization.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[quantize] Saved -> {out_path} ({meta['quantized_size_mb']} MiB, was {meta['merged_size_mb']} MiB)")
    return out_path


def benchmark_latency(model_dir: Path, *, runs: int = 5) -> dict:
    from predict_bloom import load_model, load_quantized_model, is_quantized_checkpoint

    if is_quantized_checkpoint(model_dir):
        tokenizer, model = load_quantized_model(str(model_dir))
        label = "int8"
    else:
        tokenizer, model = load_model(str(model_dir))
        label = "fp32"

    # Warm-up
    predict(BENCHMARK_QUESTION, tokenizer, model)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        predict(BENCHMARK_QUESTION, tokenizer, model)
        times.append(time.perf_counter() - t0)

    return {
        "backend": label,
        "runs": runs,
        "mean_latency_s": round(float(sum(times) / len(times)), 4),
        "min_latency_s": round(float(min(times)), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize merged Bloom classifier for CPU deploy.")
    parser.add_argument(
        "--model-size",
        choices=sorted(BLOOM_MODEL_PROFILES),
        default="1.5b",
        help="Model variant: 0.5b or 1.5b (sets default merged / quantized paths).",
    )
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Compare fp32 vs INT8 latency.")
    args = parser.parse_args()
    profile = get_profile(args.model_size)
    merged_dir = args.merged_dir or profile.merged_dir
    output_dir = args.output_dir or profile.quantized_dir

    out = quantize_merged(merged_dir, output_dir, force=args.force)

    if args.benchmark:
        merged = Path(merged_dir)
        fp32 = benchmark_latency(merged)
        int8 = benchmark_latency(out)
        bench = {"fp32_merged": fp32, "int8_quantized": int8}
        bench_path = Path("results/bloom_quantization_benchmark.json")
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        bench_path.write_text(json.dumps(bench, indent=2), encoding="utf-8")
        print(json.dumps(bench, indent=2))
        print(f"[quantize] benchmark -> {bench_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
