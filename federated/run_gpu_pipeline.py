#!/usr/bin/env python
"""One-shot GPU federated Bloom LoRA pipeline (Colab / Kaggle / local CUDA).

Primary paper path: from-scratch Qwen2.5-0.5B LoRA FedAvg/FedProx.

Example:
    python federated/run_gpu_pipeline.py --clients 8 --rounds 5 --partition iid --algorithm fedavg
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("[gpu-pipeline]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _check_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        print(
            "[warn] CUDA not available — training will fall back to CPU and be very slow.\n"
            "       In Colab: Runtime → Change runtime type → T4 GPU."
        )
    else:
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[gpu-pipeline] CUDA OK: {name} ({mem:.1f} GiB)")


def _zip_adapter(adapter_dir: Path, out_zip: Path) -> None:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in adapter_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(adapter_dir.parent))
    print(f"[gpu-pipeline] adapter zip -> {out_zip} ({out_zip.stat().st_size / 1024**2:.1f} MiB)")


def main() -> int:
    parser = argparse.ArgumentParser(description="GPU federated Bloom LoRA end-to-end pipeline.")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=float, default=3.0)
    parser.add_argument("--max-samples-per-client", type=int, default=0)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise", type=float, default=0.0)
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--partition", choices=("iid", "non_iid_label", "hash"), default="iid")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--prox-mu", type=float, default=0.01)
    parser.add_argument("--eval-each-round", action="store_true", default=True)
    parser.add_argument("--no-eval-each-round", action="store_true")
    parser.add_argument("--global-adapter", default=None)
    parser.add_argument("--merged-dir", default=None)
    parser.add_argument("--test-csv", default="data/figshare_bloom_v1_test.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-scratch", action="store_true", default=True)
    parser.add_argument("--skip-zip", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from federated.config import setting_tag

    tag = setting_tag(algorithm=args.algorithm, partition=args.partition, alpha=args.alpha)
    adapter = Path(args.global_adapter) if args.global_adapter else Path(f"models/qwen_bloom_federated0.5B_{tag}")
    merged = Path(args.merged_dir) if args.merged_dir else Path(f"models/qwen_bloom_federated0.5B_{tag}_merged")

    eval_each = args.eval_each_round and not args.no_eval_each_round
    py = sys.executable
    _check_cuda()

    sim_cmd = [
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
        "--clip-norm",
        str(args.clip_norm),
        "--dp-noise",
        str(args.dp_noise),
        "--algorithm",
        args.algorithm,
        "--partition",
        args.partition,
        "--alpha",
        str(args.alpha),
        "--prox-mu",
        str(args.prox_mu),
        "--global-adapter",
        str(adapter),
        "--test-csv",
        args.test_csv,
        "--from-scratch",
    ]
    if eval_each:
        sim_cmd.append("--eval-each-round")
    else:
        sim_cmd.append("--no-eval-each-round")
    if args.resume:
        sim_cmd.append("--resume")
    _run(sim_cmd)

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
    results_json = ROOT / "results" / f"federated_bloom_eval_{tag}.json"
    _run(
        [
            py,
            "evaluate_bloom.py",
            "--model-size",
            "0.5b",
            "--model-dir",
            str(merged),
            "--test-csv",
            args.test_csv,
            "--results-json",
            str(results_json),
        ]
    )
    _run([py, "build_federated_comparison.py"])

    if not args.skip_zip:
        _zip_adapter(adapter, ROOT / "results" / f"federated_adapter_{tag}.zip")

    summary = {
        "setting_tag": tag,
        "adapter": str(adapter),
        "merged": str(merged),
        "eval_json": str(results_json),
    }
    (ROOT / "results" / f"federated_gpu_pipeline_{tag}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
