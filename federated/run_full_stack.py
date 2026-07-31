#!/usr/bin/env python
"""Federated teacher LoRA + optional merge/eval/privacy case study.

Primary Bloom FL never overwrites locked centralized 0.5B models/results.
Privacy-risk FL is opt-in (--with-privacy-guard) and is not the main FL claim.
"""

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
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=float, default=3.0)
    parser.add_argument("--max-samples-per-client", type=int, default=0)
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--partition", choices=("iid", "non_iid_label", "hash"), default="iid")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument("--skip-fl-simulation", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--with-privacy-guard",
        action="store_true",
        help="Optional case study only — not the primary federated Bloom claim.",
    )
    args = parser.parse_args()

    py = sys.executable
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    from federated.config import setting_tag

    tag = setting_tag(algorithm=args.algorithm, partition=args.partition, alpha=args.alpha)
    adapter = ROOT / "models" / f"qwen_bloom_federated0.5B_{tag}"
    merged = ROOT / "models" / f"qwen_bloom_federated0.5B_{tag}_merged"

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
                "--algorithm",
                args.algorithm,
                "--partition",
                args.partition,
                "--alpha",
                str(args.alpha),
                "--prox-mu",
                str(args.prox_mu),
                "--from-scratch",
                "--global-adapter",
                str(adapter),
            ]
        )

    if not (adapter / "adapter_config.json").is_file():
        raise SystemExit(f"Federated adapter missing at {adapter}. Run simulation first.")

    _run(
        [
            py,
            "merge_model.py",
            "--model-size",
            "0.5b",
            "--lora-dir",
            str(adapter),
            "--output-dir",
            str(merged),
            "--force",
        ]
    )

    if args.with_privacy_guard:
        _run(
            [py, "privacy/train_federated_privacy_guard.py", "--rounds", "10", "--threshold", "0.62"],
            optional=True,
        )

    if not args.skip_eval:
        _run(
            [
                py,
                "evaluate_bloom.py",
                "--model-size",
                "0.5b",
                "--model-dir",
                str(merged),
                "--results-json",
                f"results/federated_bloom_eval_{tag}.json",
            ]
        )
        _run([py, "build_federated_comparison.py"], optional=True)

    print("[stack] federated Bloom LoRA stack complete")
    print(
        "Privacy disclaimer: data locality during FL is not formal SecAgg/DP. "
        "Use --with-privacy-guard only as a separate deployment case study."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
