#!/usr/bin/env python
"""One-shot GPU federated Bloom LoRA pipeline (Colab / Kaggle / local CUDA).

Runs: partition → federated rounds → merge → test evaluation → zip adapter.

Example (Colab after uploading/cloning this repo):
    python federated/run_gpu_pipeline.py --clients 4 --rounds 3 --eval-each-round
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=float, default=3.0)
    parser.add_argument("--max-samples-per-client", type=int, default=400)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise", type=float, default=0.0)
    parser.add_argument("--eval-each-round", action="store_true", default=True)
    parser.add_argument("--no-eval-each-round", action="store_true")
    parser.add_argument("--global-adapter", default="models/qwen_bloom_federated")
    parser.add_argument("--merged-dir", default="models/qwen_bloom_federated_merged")
    parser.add_argument("--test-csv", default="data/figshare_bloom_v1_test.csv")
    parser.add_argument("--resume", action="store_true", help="Keep existing global adapter.")
    parser.add_argument("--skip-zip", action="store_true")
    args = parser.parse_args()

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
        "--global-adapter",
        args.global_adapter,
    ]
    if eval_each:
        sim_cmd.append("--eval-each-round")
    if args.resume:
        sim_cmd.append("--resume")
    _run(sim_cmd)

    _run(
        [
            py,
            "merge_model.py",
            "--lora-dir",
            args.global_adapter,
            "--output-dir",
            args.merged_dir,
            "--force",
        ]
    )

    eval_out = ROOT / "results" / "federated_bloom_test_eval.json"
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            py,
            "evaluate_bloom.py",
            "--model_dir",
            args.merged_dir,
            "--eval_csv",
            args.test_csv,
            "--output_dir",
            str(eval_out.parent / "federated_bloom_test"),
        ]
    )

    summary = {
        "pipeline": "federated_gpu_bloom_lora",
        "global_adapter": args.global_adapter,
        "merged_dir": args.merged_dir,
        "test_csv": args.test_csv,
        "simulation_report": "results/federated_lora_simulation.json",
        "test_eval_dir": str(eval_out.parent / "federated_bloom_test"),
    }
    summary_path = ROOT / "results" / "federated_gpu_pipeline.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[gpu-pipeline] summary -> {summary_path}")

    if not args.skip_zip:
        adapter = ROOT / args.global_adapter
        if (adapter / "adapter_config.json").is_file():
            _zip_adapter(adapter, ROOT / "results" / "qwen_bloom_federated_adapter.zip")

    print("[gpu-pipeline] done. Download results/qwen_bloom_federated_adapter.zip for local deploy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
