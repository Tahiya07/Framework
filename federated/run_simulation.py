#!/usr/bin/env python

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

from federated.config import BUNDLES_DIR, DEFAULT_LORA_FALLBACK, FederatedLoraConfig, UPDATES_DIR  # noqa: E402
from federated.partition import partition_csv  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("[sim]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _evaluate_global(global_dir: Path, eval_csv: Path, cfg: FederatedLoraConfig) -> dict:
    """Evaluate the current global adapter on a held-out CSV (for the convergence curve)."""
    from sklearn.metrics import accuracy_score, f1_score

    import pandas as pd

    from predict_bloom import QwenBloomPredictor
    from federated.config import BLOOM_LABELS

    df = pd.read_csv(eval_csv).dropna(subset=[cfg.text_col, cfg.label_col])
    df = df[df[cfg.label_col].isin(BLOOM_LABELS)]
    predictor = QwenBloomPredictor(model_dir=str(global_dir), prefer_merged=False)
    y_true = [BLOOM_LABELS[str(l)] for l in df[cfg.label_col]]
    y_pred = [BLOOM_LABELS[predictor.predict(str(q))["prediction"]] for q in df[cfg.text_col]]
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "n_eval": len(y_true),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate federated teacher Bloom LoRA rounds.")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=float, default=3.0)
    parser.add_argument("--max-samples-per-client", type=int, default=300)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise", type=float, default=0.0)
    parser.add_argument("--train-csv", default=str(ROOT / "data" / "figshare_bloom_v1_train.csv"))
    parser.add_argument("--global-adapter", default=str(ROOT / "models" / "qwen_bloom_federated"))
    parser.add_argument("--skip-train", action="store_true", help="Only aggregate existing bundles.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Warm-start from the existing global adapter instead of clearing it (default: fresh).",
    )
    parser.add_argument(
        "--eval-each-round",
        action="store_true",
        help="Evaluate the global adapter after each round to record a convergence curve.",
    )
    parser.add_argument("--eval-csv", default=str(ROOT / "data" / "figshare_bloom_v1_val.csv"))
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Seed round-1 global adapter from a trained LoRA (e.g. models/qwen_bloom_3000). "
        "Recommended on CPU for stable convergence near centralized accuracy.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Do not auto-seed from models/qwen_bloom_3000 even if present.",
    )
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
    # A stale global adapter silently warm-starts (and can poison) round 1.
    # Default to a clean run; --resume keeps the existing adapter on purpose.
    if global_dir.exists() and not args.resume and not args.skip_train:
        print(f"[sim] fresh start: clearing existing global adapter at {global_dir}")
        shutil.rmtree(global_dir)
    global_dir.mkdir(parents=True, exist_ok=True)

    init_path = args.init_adapter
    if init_path is None and not args.from_scratch and (DEFAULT_LORA_FALLBACK / "adapter_config.json").is_file():
        init_path = str(DEFAULT_LORA_FALLBACK)
    if init_path and not args.skip_train:
        src = Path(init_path)
        if not (src / "adapter_config.json").is_file():
            raise FileNotFoundError(f"--init-adapter not found: {src}")
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, global_dir / item.name)
        print(f"[sim] seeded global adapter from {src} (federated fine-tuning, not from-scratch)")

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
        round_record = {"round": round_idx, "n_clients": len(bundle_paths)}
        if args.eval_each_round:
            metrics = _evaluate_global(global_dir, Path(args.eval_csv), cfg)
            round_record.update(metrics)
            print(f"[sim] round {round_idx} global eval: {metrics}")
        history.append(round_record)

    report = {
        "simulation": "federated_teacher_bloom_lora",
        "rounds": cfg.rounds,
        "num_clients": len(client_csvs),
        "global_adapter": str(global_dir),
        "aggregation": "fedavg_weighted",
        "recipe": {
            "local_epochs": cfg.local_epochs,
            "learning_rate": cfg.learning_rate,
            "lr_scheduler_type": cfg.lr_scheduler_type,
            "warmup_ratio": cfg.warmup_ratio,
            "weight_decay": cfg.weight_decay,
            "label_smoothing": cfg.label_smoothing,
            "use_class_weights": cfg.use_class_weights,
            "max_grad_norm": cfg.max_grad_norm,
            "batch_size": cfg.batch_size,
            "grad_accum": cfg.grad_accum,
            "effective_batch": cfg.batch_size * cfg.grad_accum,
            "max_samples_per_client": cfg.max_samples_per_client,
        },
        "clip_norm": cfg.clip_norm,
        "dp_noise": cfg.dp_noise,
        "history": history,
        "next_steps": [
            "python merge_model.py --lora-dir models/qwen_bloom_federated --output-dir models/qwen_bloom_federated_merged",
            "python evaluate_bloom.py --model_dir models/qwen_bloom_federated_merged --eval_csv data/figshare_bloom_v1_test.csv",
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
