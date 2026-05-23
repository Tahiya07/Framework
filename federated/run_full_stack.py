#!/usr/bin/env python
"""Federated teacher LoRA + merge + privacy guard + evaluation pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, optional: bool = False) -> None:
    print("[stack]", " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(ROOT), check=True)
    except subprocess.CalledProcessError as exc:
        if optional:
            print(f"[warn] optional step failed: {exc}")
        else:
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--local-epochs", type=float, default=1.0)
    parser.add_argument("--max-samples-per-client", type=int, default=250)
    parser.add_argument("--skip-fl-simulation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if not args.skip_fl_simulation:
        _run(
            [
                py,
                "federated/run_simulation.py",
                "--clients",
                str(args.clients),
                "--rounds",
                str(args.rounds),
                "--local-epochs",
                str(args.local_epochs),
                "--max-samples-per-client",
                str(args.max_samples_per_client),
            ]
        )

    lora_dir = ROOT / "models" / "qwen_bloom_federated"
    if not (lora_dir / "adapter_config.json").is_file():
        print("[stack] federated adapter missing; falling back to centralized LoRA for merge.")
        lora_dir = ROOT / "models" / "qwen_bloom_3000"

    _run(
        [
            py,
            "merge_model.py",
            "--lora-dir",
            str(lora_dir),
            "--output-dir",
            "models/qwen_bloom_merged",
        ]
    )

    _run([py, "privacy/train_federated_privacy_guard.py", "--rounds", "10", "--threshold", "0.62"])

    if not args.skip_eval:
        _run([py, "run_evaluation_pipeline.py"])

    print("[stack] federated full stack complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
