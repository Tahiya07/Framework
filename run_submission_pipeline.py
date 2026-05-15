from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    # Full-sized knobs are user-overridable; defaults are publication-oriented.
    n_total = os.environ.get("SUBMIT_N_TOTAL", "600")
    n_test_qa = os.environ.get("SUBMIT_N_TEST_QA", "300")
    n_unc = os.environ.get("SUBMIT_N_UNCERTAINTY_POOL", "600")
    max_per_label = os.environ.get("BLOOM_TRANSFER_MAX_PER_LABEL", "2000")
    os.environ["BLOOM_TRANSFER_MAX_PER_LABEL"] = max_per_label
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    _run(
        [
            sys.executable,
            "evaluate.py",
            "--full",
            "--n-total",
            n_total,
            "--n-test-qa",
            n_test_qa,
            "--n-uncertainty-pool",
            n_unc,
        ]
    )

    _run([sys.executable, "evaluate_qwen_rag.py"])
    _run([sys.executable, "evaluate_multimodal_rag.py"])
    _run([sys.executable, "-m", "bloom.evaluate_figshare_bloom"])
    _run([sys.executable, "-m", "bloom.evaluate_cross_domain_bloom"])
    _run([sys.executable, "-m", "bloom.analyze_bloom_domain_shift"])
    _run([sys.executable, "package_cross_domain_results.py"])
    _run([sys.executable, "privacy/train_federated_privacy_guard.py", "--rounds", "5", "--threshold", "0.99", "--dp-noise", "1.0"])
    _run([sys.executable, "privacy/evaluate_privacy_guard.py"])
    _run([sys.executable, "privacy/evaluate_privacy_benchmarks.py"])
    _run([sys.executable, "consolidate_paper_results.py"])
    _run([sys.executable, "build_high_venue_paper.py"])
    print("[done] submission pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
