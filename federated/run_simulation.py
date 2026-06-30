#!/usr/bin/env python
"""Single-machine simulation of federated teacher Bloom LoRA (architecture prototype)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.config import BUNDLES_DIR, FederatedLoraConfig, UPDATES_DIR  # noqa: E402
from federated.partition import partition_csv  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("[sim]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate federated teacher Bloom LoRA rounds.")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=float, default=2.0)
    parser.add_argument("--max-samples-per-client", type=int, default=300)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise", type=float, default=0.0)
    parser.add_argument("--train-csv", default=str(ROOT / "data" / "figshare_bloom_v1_train.csv"))
    parser.add_argument("--global-adapter", default=str(ROOT / "models" / "qwen_bloom_federated"))
    parser.add_argument("--skip-train", action="store_true", help="Only aggregate existing bundles.")
    args = parser.parse_args()

    cfg = FederatedLoraConfig(
        num_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        max_samples_per_client=args.max_samples_per_client,
        clip_norm=args.clip_norm,
        dp_noise=args.dp_noise,
        train_csv=args.train_csv,
        global_adapter_dir=args.global_adapter,
    )

    global_dir = Path(cfg.global_adapter_dir)
    global_dir.mkdir(parents=True, exist_ok=True)

    if UPDATES_DIR.exists():
        shutil.rmtree(UPDATES_DIR)
    if BUNDLES_DIR.exists():
        shutil.rmtree(BUNDLES_DIR)
    UPDATES_DIR.mkdir(parents=True, exist_ok=True)
    BUNDLES_DIR.mkdir(parents=True, exist_ok=True)

    parts = partition_csv(
        cfg.train_csv,
        num_clients=cfg.num_clients,
        text_col=cfg.text_col,
        label_col=cfg.label_col,
        max_per_client=cfg.max_samples_per_client,
    )

    client_csvs: dict[str, Path] = {}
    for client_id, frame in parts.items():
        path = UPDATES_DIR / f"{client_id}.csv"
        frame[[cfg.text_col, cfg.label_col]].to_csv(path, index=False)
        client_csvs[client_id] = path
        print(f"[sim] client {client_id}: {len(frame)} rows")

    py = sys.executable
    history = []

    for round_idx in range(1, cfg.rounds + 1):
        bundle_paths: list[Path] = []
        for client_id, csv_path in client_csvs.items():
            bundle_path = BUNDLES_DIR / f"round{round_idx:02d}_{client_id}.json"
            bundle_paths.append(bundle_path)
            if not args.skip_train:
                _run(
                    [
                        py,
                        str(ROOT / "federated" / "client_train.py"),
                        "--client-id",
                        client_id,
                        "--round",
                        str(round_idx),
                        "--csv",
                        str(csv_path),
                        "--global-adapter",
                        str(global_dir),
                        "--out-bundle",
                        str(bundle_path),
                        "--local-epochs",
                        str(cfg.local_epochs),
                    ]
                )

        if not bundle_paths or not all(p.is_file() for p in bundle_paths):
            raise FileNotFoundError("missing client bundles; run without --skip-train")

        _run(
            [
                py,
                str(ROOT / "federated" / "server_aggregate.py"),
                "--bundles",
                *[str(p) for p in bundle_paths],
                "--global-adapter",
                str(global_dir),
                "--clip-norm",
                str(cfg.clip_norm),
                "--dp-noise",
                str(cfg.dp_noise),
            ]
        )
        history.append({"round": round_idx, "n_clients": len(bundle_paths)})

    report = {
        "simulation": "federated_teacher_bloom_lora",
        "rounds": cfg.rounds,
        "num_clients": len(client_csvs),
        "global_adapter": str(global_dir),
        "clip_norm": cfg.clip_norm,
        "dp_noise": cfg.dp_noise,
        "history": history,
        "next_steps": [
            "python merge_model.py --lora-dir models/qwen_bloom_federated",
            "python evaluate_bloom.py",
        ],
    }
    out = ROOT / "results" / "federated_lora_simulation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sim] done -> {global_dir}")
    print(f"[sim] report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
