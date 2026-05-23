#!/usr/bin/env python
"""End-to-end evaluation for publication tables and figures."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], *, optional: bool = False) -> None:
    print(f"[run] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        if optional:
            print(f"[warn] optional step failed: {exc}")
        else:
            raise


def main() -> int:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    py = sys.executable
    max_test = os.environ.get("EVAL_BLOOM_MAX_TEST", "0")

    run_fl = os.environ.get("RUN_FEDERATED_LORA", "0") == "1"
    steps = []
    if run_fl:
        steps.append(
            [
                py,
                "federated/run_simulation.py",
                "--clients",
                os.environ.get("FL_CLIENTS", "4"),
                "--rounds",
                os.environ.get("FL_ROUNDS", "2"),
                "--local-epochs",
                os.environ.get("FL_LOCAL_EPOCHS", "1"),
                "--max-samples-per-client",
                os.environ.get("FL_MAX_SAMPLES", "200"),
            ]
        )
    steps.extend(
        [
        [py, "merge_model.py"],
        [py, "evaluate_bloom.py", "--svm-baseline", "--max-test", max_test],
        [py, "build_bloom_comparison.py"],
        [py, "evaluate_qwen_rag.py"],
        [py, "evaluate_multimodal_rag.py"],
        [py, "evaluate_ocr_pipeline.py"],
        [py, "privacy/train_federated_privacy_guard.py", "--rounds", "5"],
        [py, "privacy/evaluate_privacy_guard.py"],
        [py, "privacy/evaluate_privacy_benchmarks.py"],
        [py, "consolidate_paper_results.py"],
        [py, "generate_paper_figures.py"],
        ]
    )
    for cmd in steps:
        optional = cmd[1] == "evaluate_qwen_rag.py"
        _run(cmd, optional=optional)
    print("[done] evaluation pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
