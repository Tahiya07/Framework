#!/usr/bin/env python
"""Deploy + evaluate pipeline (centralized model only; no federated training)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from bloom_model_profiles import BLOOM_MODEL_PROFILES, DEFAULT_MODEL_SIZE, get_profile

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], *, optional: bool = False) -> None:
    print("[deploy]", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        if optional:
            print(f"[warn] optional step failed: {exc}")
        else:
            raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge, quantize, and evaluate Bloom classifier for deploy.")
    parser.add_argument(
        "--model-size",
        choices=sorted(BLOOM_MODEL_PROFILES),
        default=DEFAULT_MODEL_SIZE,
        help="Model variant: 0.5b (lightweight default) or 1.5b.",
    )
    parser.add_argument("--skip-merge", action="store_true", help="Use existing merged model.")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-paper", action="store_true", help="Skip consolidate + figures.")
    parser.add_argument("--lora-dir", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--quantized-dir", default=None)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    profile = get_profile(args.model_size)
    lora_dir = args.lora_dir or profile.lora_dir
    merged_dir = args.merged_dir or profile.merged_dir
    quantized_dir = args.quantized_dir or profile.quantized_dir

    py = sys.executable
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("BLOOM_MODEL_SIZE", profile.key)

    max_test_flag = ["--max-test", str(args.max_test)] if args.max_test > 0 else []

    if not args.skip_merge:
        _run(
            [
                py,
                "merge_model.py",
                "--model-size",
                profile.key,
                "--lora-dir",
                lora_dir,
                "--output-dir",
                merged_dir,
                "--force",
            ]
        )

    if not args.skip_eval:
        _run(
            [
                py,
                "evaluate_bloom.py",
                "--model-size",
                profile.key,
                "--merged-dir",
                merged_dir,
                "--svm-baseline",
                *max_test_flag,
            ]
        )

    if not args.skip_quantize:
        _run(
            [
                py,
                "quantize_bloom.py",
                "--model-size",
                profile.key,
                "--merged-dir",
                merged_dir,
                "--output-dir",
                quantized_dir,
                "--benchmark",
            ]
        )
        if not args.skip_eval:
            _run(
                [
                    py,
                    "evaluate_bloom.py",
                    "--model-size",
                    profile.key,
                    "--quantized",
                    "--quantized-dir",
                    quantized_dir,
                    *max_test_flag,
                ]
            )

    _run([py, "build_bloom_comparison.py"])

    if not args.skip_paper:
        _run([py, "consolidate_paper_results.py"])
        _run([py, "generate_paper_figures.py"], optional=True)
        _run([py, "architecture_compliance.py"], optional=True)

    print(f"[deploy] centralized deploy pipeline complete ({profile.display_name}, federated skipped)")
    print("[deploy] outputs:")
    print(f"  - {profile.results_json}")
    print(f"  - {profile.quant_results_json}")
    print("  - results/bloom_baseline_comparison.json")
    print(f"  - {quantized_dir}/  (lightweight CPU inference)")
    print(f"  - {profile.quant_benchmark_json}  (fp32 vs INT8 latency)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
