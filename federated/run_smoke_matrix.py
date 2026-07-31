#!/usr/bin/env python
"""Evidence-phase FL smoke runner (4x2 only). Does not run 8x5.

Smoke protocol (CPU-feasible validity check; NOT paper-scale):
  --clients 4 --rounds 2 --local-epochs 1 --max-samples-per-client 100
  --eval-csv data/figshare_bloom_v1_val.csv
  --test-csv data/figshare_bloom_v1_test.csv
  --from-scratch

Paper-scale later: 8 clients, 5 rounds, local-epochs 3, max-samples 0 (full shards).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Cap BLAS/OpenMP threads for CPU hosts (default HF/torch oversubscription is very slow).
for _k, _v in {
    "OMP_NUM_THREADS": "8",
    "MKL_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "8",
    "NUMEXPR_NUM_THREADS": "8",
    "TORCH_NUM_THREADS": "8",
}.items():
    os.environ.setdefault(_k, _v)

SMOKES = [
    {
        "name": "A_fedavg_iid",
        "algorithm": "fedavg",
        "partition": "iid",
        "alpha": "0.5",
        "prox_mu": "0.01",
    },
    {
        "name": "B_fedprox_iid",
        "algorithm": "fedprox",
        "partition": "iid",
        "alpha": "0.5",
        "prox_mu": "0.01",
    },
    {
        "name": "C_fedavg_noniid",
        "algorithm": "fedavg",
        "partition": "non_iid_label",
        "alpha": "0.5",
        "prox_mu": "0.01",
    },
    {
        "name": "D_fedprox_noniid",
        "algorithm": "fedprox",
        "partition": "non_iid_label",
        "alpha": "0.5",
        "prox_mu": "0.01",
    },
]


def main() -> int:
    py = sys.executable
    only = sys.argv[1] if len(sys.argv) > 1 else None
    summary = []
    for spec in SMOKES:
        if only and only not in (spec["name"], spec["algorithm"] + "_" + spec["partition"]):
            continue
        cmd = [
            py,
            str(ROOT / "federated" / "run_simulation.py"),
            "--clients",
            "4",
            "--rounds",
            "2",
            "--local-epochs",
            "1",
            "--max-samples-per-client",
            "100",
            "--algorithm",
            spec["algorithm"],
            "--partition",
            spec["partition"],
            "--alpha",
            spec["alpha"],
            "--prox-mu",
            spec["prox_mu"],
            "--from-scratch",
            "--seed",
            "42",
            "--eval-csv",
            str(ROOT / "data" / "figshare_bloom_v1_val.csv"),
            "--test-csv",
            str(ROOT / "data" / "figshare_bloom_v1_test.csv"),
            "--eval-each-round",
        ]
        print("[smoke]", spec["name"], " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)
        summary.append(spec["name"])
    out = ROOT / "results" / "federated_smoke_summary.json"
    out.write_text(
        json.dumps(
            {
                "completed": summary,
                "protocol": {
                    "clients": 4,
                    "rounds": 2,
                    "local_epochs": 1,
                    "max_samples_per_client": 100,
                    "eval_csv": "data/figshare_bloom_v1_val.csv",
                    "test_csv": "data/figshare_bloom_v1_test.csv",
                    "note": "Validity smoke only; paper-scale is 8x5 full shards.",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[smoke] done ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
