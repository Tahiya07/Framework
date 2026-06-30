#!/usr/bin/env python
"""Deploy + evaluate pipeline (centralized model only; no federated training)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

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
    parser.add_argument("--skip-merge", action="store_true", help="Use existing merged model.")
    parser.add_argument("--skip-quantize", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-paper", action="store_true", help="Skip consolidate + figures.")
    parser.add_argument("--lora-dir", default="models/qwen_bloom_3000")
    parser.add_argument("--merged-dir", default="models/qwen_bloom_merged")
    parser.add_argument("--quantized-dir", default="models/qwen_bloom_quantized")
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()

    py = sys.executable
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    max_test_flag = ["--max-test", str(args.max_test)] if args.max_test > 0 else []

    if not args.skip_merge:
        _run(
            [
                py,
                "merge_model.py",
                "--lora-dir",
                args.lora_dir,
                "--output-dir",
                args.merged_dir,
                "--force",
            ]
        )

    if not args.skip_eval:
        _run(
            [
                py,
                "evaluate_bloom.py",
                "--model-dir",
                args.merged_dir,
                "--svm-baseline",
                *max_test_flag,
            ]
        )

    if not args.skip_quantize:
        _run(
            [
                py,
                "quantize_bloom.py",
                "--merged-dir",
                args.merged_dir,
                "--output-dir",
                args.quantized_dir,
                "--benchmark",
            ]
        )
        if not args.skip_eval:
            _run(
                [
                    py,
                    "evaluate_bloom.py",
                    "--quantized",
                    "--quantized-dir",
                    args.quantized_dir,
                    *max_test_flag,
                ]
            )

    _run([py, "build_bloom_comparison.py"])

    if not args.skip_paper:
        _run([py, "consolidate_paper_results.py"])
        _run([py, "generate_paper_figures.py"], optional=True)
        _run([py, "architecture_compliance.py"], optional=True)

    print("[deploy] centralized deploy pipeline complete (federated skipped)")
    print("[deploy] outputs:")
    print("  - results/bloom_lora_eval.json")
    print("  - results/bloom_quantized_eval.json")
    print("  - results/bloom_baseline_comparison.json")
    print("  - models/qwen_bloom_quantized/  (lightweight CPU inference)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
