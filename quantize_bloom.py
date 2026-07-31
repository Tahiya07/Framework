#!/usr/bin/env python
"""Export lightweight Bloom classifier for CPU deploy (FP16 default; INT8 disabled)."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from bloom_model_profiles import BLOOM_MODEL_PROFILES, DEFAULT_MODEL_SIZE, get_profile
from predict_bloom import is_deploy_checkpoint, load_deploy_model, load_model, predict

BENCHMARK_QUESTION = (
    "Compare and contrast the advantages of array-based and linked-list implementations "
    "of a stack. Justify which you would choose for a memory-constrained embedded system."
)

DEFAULT_CALIBRATION_CSV = Path("data/figshare_bloom_v1_test.csv")
MIN_PREDICTION_AGREEMENT = 0.95


def _load_tokenizer(model_path: str):
    from predict_bloom import _load_tokenizer as load_tok

    return load_tok(model_path)


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _calibration_questions(csv_path: Path, n: int = 32) -> list[str]:
    if csv_path.is_file():
        df = pd.read_csv(csv_path).dropna()
        col = "question" if "question" in df.columns else df.columns[0]
        return [str(x) for x in df[col].head(n).tolist()]
    return [BENCHMARK_QUESTION]


def _prediction_agreement(
    tokenizer_a,
    model_a,
    tokenizer_b,
    model_b,
    questions: list[str],
) -> float:
    if not questions:
        return 1.0
    matches = 0
    for question in questions:
        pred_a = predict(question, tokenizer_a, model_a)["prediction"]
        pred_b = predict(question, tokenizer_b, model_b)["prediction"]
        matches += int(pred_a == pred_b)
    return matches / len(questions)


def export_lightweight(
    merged_dir: str | Path,
    output_dir: str | Path,
    *,
    profile_key: str,
    base_model: str,
    fmt: str = "fp16",
    calibration_csv: Path = DEFAULT_CALIBRATION_CSV,
    force: bool = False,
) -> Path:
    merged_path = Path(merged_dir)
    out_path = Path(output_dir)

    if not (merged_path / "config.json").is_file():
        raise FileNotFoundError(f"Merged model not found at {merged_path}. Run: python merge_model.py")

    meta_path = out_path / "quantization.json"
    if is_deploy_checkpoint(out_path) and meta_path.is_file() and not force:
        print(f"[quantize] Using existing lightweight model at {out_path}")
        return out_path

    if fmt != "fp16":
        raise ValueError(
            "Dynamic INT8 quantization breaks Qwen2 Bloom accuracy (~17% vs ~83%). "
            "Use --format fp16 (default) for lightweight deploy."
        )

    print(f"[quantize] Loading merged FP32 model from {merged_path}...")
    tokenizer, model = load_model(str(merged_path))
    model.cpu().eval()

    questions = _calibration_questions(calibration_csv)
    print(f"[quantize] Validating FP16 export on {len(questions)} calibration questions...")
    model_fp16 = AutoModelForSequenceClassification.from_pretrained(
        str(merged_path),
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model_fp16.eval().cpu().half()

    agreement = _prediction_agreement(tokenizer, model, tokenizer, model_fp16, questions)
    print(f"[quantize] FP32 vs FP16 prediction agreement: {agreement:.3f}")
    if agreement < MIN_PREDICTION_AGREEMENT:
        raise RuntimeError(
            f"FP16 export failed validation (agreement={agreement:.3f} < {MIN_PREDICTION_AGREEMENT}). "
            "Use merged FP32 model for deploy."
        )

    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    model_fp16.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    merged_size = _dir_size_bytes(merged_path)
    quant_size = _dir_size_bytes(out_path)
    meta = {
        "format": "fp16_merged",
        "model_size": profile_key,
        "base_model": base_model,
        "source_merged_dir": str(merged_path),
        "merged_size_mb": round(merged_size / (1024**2), 2),
        "quantized_size_mb": round(quant_size / (1024**2), 2),
        "compression_ratio": round(merged_size / max(quant_size, 1), 2),
        "calibration_questions": len(questions),
        "fp32_fp16_prediction_agreement": round(agreement, 4),
        "notes": (
            "FP16 merged weights (HF save_pretrained). "
            "Torch dynamic INT8 is not used for Qwen2 classifiers due to accuracy collapse."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[quantize] Saved FP16 deploy model -> {out_path} "
        f"({meta['quantized_size_mb']} MiB, was {meta['merged_size_mb']} MiB)"
    )
    return out_path


def benchmark_latency(model_dir: Path, *, runs: int = 5) -> dict:
    if is_deploy_checkpoint(model_dir):
        tokenizer, model = load_deploy_model(str(model_dir))
        meta = json.loads((model_dir / "quantization.json").read_text(encoding="utf-8"))
        label = meta.get("format", "deploy")
    else:
        tokenizer, model = load_model(str(model_dir))
        label = "fp32"

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
    parser = argparse.ArgumentParser(description="Export lightweight Bloom classifier for CPU deploy.")
    parser.add_argument(
        "--model-size",
        choices=sorted(BLOOM_MODEL_PROFILES),
        default=DEFAULT_MODEL_SIZE,
        help="Model variant: 0.5b or 1.5b (sets default merged / deploy paths).",
    )
    parser.add_argument(
        "--format",
        choices=("fp16",),
        default="fp16",
        help="Lightweight format (fp16). INT8 is not supported for Qwen2 Bloom.",
    )
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--calibration-csv", type=Path, default=DEFAULT_CALIBRATION_CSV)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Compare fp32 vs lightweight latency.")
    args = parser.parse_args()
    profile = get_profile(args.model_size)
    merged_dir = args.merged_dir or profile.merged_dir
    output_dir = args.output_dir or profile.quantized_dir

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        out = export_lightweight(
            merged_dir,
            output_dir,
            profile_key=profile.key,
            base_model=profile.base_model,
            fmt=args.format,
            calibration_csv=args.calibration_csv,
            force=args.force,
        )

    if args.benchmark:
        merged = Path(merged_dir)
        fp32 = benchmark_latency(merged)
        lite = benchmark_latency(out)
        bench = {
            "model_size": profile.key,
            "base_model": profile.base_model,
            "fp32_merged": fp32,
            "lightweight_deploy": lite,
            "speedup": round(fp32["mean_latency_s"] / max(lite["mean_latency_s"], 1e-6), 3),
        }
        bench_path = Path(profile.quant_benchmark_json)
        bench_path.parent.mkdir(parents=True, exist_ok=True)
        bench_path.write_text(json.dumps(bench, indent=2), encoding="utf-8")
        print(json.dumps(bench, indent=2))
        print(f"[quantize] benchmark -> {bench_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
