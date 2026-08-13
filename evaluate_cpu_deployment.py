#!/usr/bin/env python
"""Compare Bloom classifier deployment artifacts on an offline CPU host.

The script measures the things that matter for a Railway-like CPU deployment:
on-disk size, cold load time, warm single-request latency, batch throughput,
and resident-memory growth.  It always evaluates Hugging Face merged and
lightweight checkpoints when supplied.  An ONNX artifact is optional because
exporting a custom Qwen classifier must be validated separately.

Examples
--------
python evaluate_cpu_deployment.py --model-size 0.5b
python evaluate_cpu_deployment.py \
  --merged-dir models/qwen_bloom_merged0.5B \
  --quantized-dir models/qwen_bloom_quantized0.5B \
  --onnx-dir models/qwen_bloom_onnx_int8
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

try:
    import psutil
except ImportError:  # keep size/latency evaluation usable in a minimal environment
    psutil = None

from bloom_model_profiles import DEFAULT_MODEL_SIZE, get_profile
from predict_bloom import build_prompt, load_deploy_model, load_model, predict


SAMPLE_QUESTION = (
    "Compare array-based and linked-list stack implementations and justify "
    "which you would choose for a memory-constrained embedded system."
)


def directory_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * fraction
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


@dataclass
class Benchmark:
    name: str
    path: str
    status: str
    format: str
    size_mib: float
    load_seconds: float | None = None
    rss_delta_mib: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    mean_latency_ms: float | None = None
    throughput_requests_per_second: float | None = None
    error: str | None = None


def benchmark_huggingface(name: str, path: Path, runs: int) -> Benchmark:
    result = Benchmark(name, str(path), "failed", "huggingface", round(directory_size_bytes(path) / 2**20, 2))
    process = psutil.Process() if psutil else None
    before = process.memory_info().rss if process else 0
    try:
        start = time.perf_counter()
        if (path / "quantization.json").is_file() or (path / "model_int8.pt").is_file():
            tokenizer, model = load_deploy_model(str(path))
            result.format = json.loads((path / "quantization.json").read_text()).get("format", "deploy") if (path / "quantization.json").is_file() else "torch_dynamic_int8"
        else:
            tokenizer, model = load_model(str(path))
            result.format = "merged_fp32"
        result.load_seconds = round(time.perf_counter() - start, 3)
        if process:
            result.rss_delta_mib = round((process.memory_info().rss - before) / 2**20, 2)

        # Exclude tokenizer/model initialization from the measurements.
        predict(SAMPLE_QUESTION, tokenizer, model)
        timings: list[float] = []
        start = time.perf_counter()
        for _ in range(runs):
            tick = time.perf_counter()
            predict(SAMPLE_QUESTION, tokenizer, model)
            timings.append((time.perf_counter() - tick) * 1000)
        elapsed = time.perf_counter() - start
        result.latency_p50_ms = round(percentile(timings, 0.50), 2)
        result.latency_p95_ms = round(percentile(timings, 0.95), 2)
        result.mean_latency_ms = round(statistics.mean(timings), 2)
        result.throughput_requests_per_second = round(runs / elapsed, 3)
        result.status = "ok"
    except Exception as exc:  # report an unusable variant instead of aborting comparison
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            del tokenizer, model
        except UnboundLocalError:
            pass
        gc.collect()
    return result


def benchmark_onnx(name: str, path: Path, runs: int) -> Benchmark:
    """Benchmark a standard ONNX classifier export when onnxruntime is installed."""
    result = Benchmark(name, str(path), "failed", "onnx", round(directory_size_bytes(path) / 2**20, 2))
    files = list(path.glob("*.onnx"))
    if not files:
        result.error = "No .onnx model file found."
        return result
    try:
        import numpy as np
        import onnxruntime as ort
        from predict_bloom import _load_tokenizer

        process = psutil.Process() if psutil else None
        before = process.memory_info().rss if process else 0
        start = time.perf_counter()
        session = ort.InferenceSession(str(files[0]), providers=["CPUExecutionProvider"])
        tokenizer = _load_tokenizer(str(path), fallback_path=None)
        result.load_seconds = round(time.perf_counter() - start, 3)
        if process:
            result.rss_delta_mib = round((process.memory_info().rss - before) / 2**20, 2)
        encoded = tokenizer(build_prompt(SAMPLE_QUESTION), truncation=True, max_length=256, return_tensors="np")
        feeds = {item.name: encoded[item.name] for item in session.get_inputs() if item.name in encoded}
        session.run(None, feeds)
        timings = []
        start = time.perf_counter()
        for _ in range(runs):
            tick = time.perf_counter()
            session.run(None, feeds)
            timings.append((time.perf_counter() - tick) * 1000)
        elapsed = time.perf_counter() - start
        result.latency_p50_ms = round(percentile(timings, 0.50), 2)
        result.latency_p95_ms = round(percentile(timings, 0.95), 2)
        result.mean_latency_ms = round(statistics.mean(timings), 2)
        result.throughput_requests_per_second = round(runs / elapsed, 3)
        result.status = "ok"
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def markdown_report(results: list[Benchmark], cpu: str) -> str:
    rows = [
        "# CPU deployment benchmark",
        "",
        f"CPU: `{cpu}`  ",
        f"Torch threads: `{torch.get_num_threads()}`",
        "",
        "| Variant | Status | Size (MiB) | Load (s) | RSS Δ (MiB) | p50 (ms) | p95 (ms) | Req/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        rows.append(
            f"| {item.name} ({item.format}) | {item.status} | {item.size_mib:.2f} | "
            f"{item.load_seconds or 0:.3f} | {item.rss_delta_mib or 0:.2f} | "
            f"{item.latency_p50_ms or 0:.2f} | {item.latency_p95_ms or 0:.2f} | "
            f"{item.throughput_requests_per_second or 0:.3f} |"
        )
    rows += [
        "",
        "## Deployment decision",
        "",
        "Use the smallest **accuracy-validated** variant with acceptable p95 latency and RSS. "
        "For this repository, dynamic PyTorch INT8 is explicitly rejected by `quantize_bloom.py` "
        "because it caused major accuracy loss. ONNX Runtime INT8 is a promising offline-CPU "
        "candidate, but only deploy it after matching classifier predictions/metrics against the merged FP32 checkpoint.",
    ]
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure CPU deployment cost for Bloom classifier artifacts.")
    parser.add_argument("--model-size", default=DEFAULT_MODEL_SIZE, choices=("0.5b", "1.5b"))
    parser.add_argument("--merged-dir")
    parser.add_argument("--quantized-dir")
    parser.add_argument("--onnx-dir")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=max(1, min(os.cpu_count() or 1, 4)))
    parser.add_argument("--output-json", type=Path, default=Path("results/cpu_deployment_benchmark.json"))
    parser.add_argument("--output-markdown", type=Path, default=Path("results/cpu_deployment_benchmark.md"))
    args = parser.parse_args()
    if args.runs < 3:
        parser.error("--runs must be at least 3")
    torch.set_num_threads(args.threads)
    profile = get_profile(args.model_size)
    candidates = [
        ("merged", Path(args.merged_dir or profile.merged_dir), benchmark_huggingface),
        ("lightweight", Path(args.quantized_dir or profile.quantized_dir), benchmark_huggingface),
    ]
    if args.onnx_dir:
        candidates.append(("onnx", Path(args.onnx_dir), benchmark_onnx))
    results = [
        runner(name, path, args.runs) if path.is_dir() else Benchmark(name, str(path), "missing", "unknown", 0, error="Directory not found")
        for name, path, runner in candidates
    ]
    payload: dict[str, Any] = {
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "torch_threads": torch.get_num_threads(),
        "runs": args.runs,
        "results": [asdict(item) for item in results],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_markdown.write_text(markdown_report(results, payload["cpu"]), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output_json} and {args.output_markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
