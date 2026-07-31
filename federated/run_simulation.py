#!/usr/bin/env python
"""Simulate federatedly trained Qwen2.5-0.5B Bloom LoRA (FedAvg / FedProx).

Publication defaults
--------------------
- From-scratch LoRA (does NOT seed from the locked centralized 84.86% adapter)
- Shared predict_bloom.build_prompt (Bloom Level:)
- IID or Dirichlet(α=0.5) non-IID partitions
- Full participation
- Separate output dirs so centralized models/results are never overwritten
"""

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

from federated.config import (  # noqa: E402
    BUNDLES_DIR,
    DEFAULT_GLOBAL_LORA,
    DEFAULT_PROX_MU,
    FederatedLoraConfig,
    UPDATES_DIR,
    setting_tag,
)
from federated.partition import client_label_distributions, partition_csv  # noqa: E402


def _run(cmd: list[str]) -> None:
    print("[sim]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _mb(num_bytes: int | float) -> float:
    return round(float(num_bytes) / (1024.0 * 1024.0), 4)


def _evaluate_global(global_dir: Path, eval_csv: Path, cfg: FederatedLoraConfig) -> dict:
    """Round-level metrics on a held-out CSV (val during training; test for final)."""
    import pandas as pd

    from bloom_eval_metrics import evaluate_predictions
    from federated.config import BLOOM_LABELS
    from predict_bloom import BLOOM_LABELS as LABEL_LIST
    from predict_bloom import QwenBloomPredictor

    df = pd.read_csv(eval_csv).dropna(subset=[cfg.text_col, cfg.label_col])
    df = df[df[cfg.label_col].isin(BLOOM_LABELS)]
    predictor = QwenBloomPredictor(
        model_dir=str(global_dir),
        base_model=cfg.base_model,
        prefer_merged=False,
        model_size="0.5b",
    )
    label2id = {lab: i for i, lab in enumerate(LABEL_LIST)}
    y_true, y_pred, confidences = [], [], []
    for _, row in df.iterrows():
        out = predictor.predict(str(row[cfg.text_col]))
        y_true.append(label2id[str(row[cfg.label_col])])
        y_pred.append(label2id[out["prediction"]])
        confidences.append(float(out["confidence"]))
    metrics = evaluate_predictions(y_true, y_pred, confidences=confidences, bootstrap_samples=0)
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "quadratic_weighted_kappa": metrics.get("quadratic_weighted_kappa"),
        "within_one_level_accuracy": metrics.get("within_one_level_accuracy"),
        "severe_error_rate": metrics.get("severe_error_rate"),
        "ece": metrics.get("ece"),
        "n_eval": len(y_true),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate federatedly trained Bloom LoRA (FedAvg/FedProx, IID/non-IID)."
    )
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=float, default=3.0)
    parser.add_argument("--max-samples-per-client", type=int, default=0, help="0 = full shard.")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise", type=float, default=0.0)
    parser.add_argument("--train-csv", default=str(ROOT / "data" / "figshare_bloom_v1_train.csv"))
    parser.add_argument("--test-csv", default=str(ROOT / "data" / "figshare_bloom_v1_test.csv"))
    parser.add_argument(
        "--eval-csv",
        default=str(ROOT / "data" / "figshare_bloom_v1_val.csv"),
        help="Per-round MONITORING CSV only (default: val). Final metrics always use --test-csv.",
    )
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--prox-mu", type=float, default=DEFAULT_PROX_MU)
    parser.add_argument(
        "--partition",
        choices=("iid", "non_iid_label", "hash"),
        default="iid",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet α for non_iid_label.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--global-adapter",
        default=None,
        help="Output adapter dir (default: models/qwen_bloom_federated0.5B_{algo}_{partition}).",
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Simulation report path (default under results/).",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep existing global adapter.")
    parser.add_argument("--eval-each-round", action="store_true", default=True)
    parser.add_argument("--no-eval-each-round", action="store_true")
    parser.add_argument(
        "--init-adapter",
        default=None,
        help="Optional ablation: seed from an existing LoRA (not used for primary paper runs).",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        default=True,
        help="Primary mode: do not seed from centralized LoRA (default).",
    )
    parser.add_argument(
        "--allow-central-seed",
        action="store_true",
        help="Ablation only: allow --init-adapter / centralized warm-start.",
    )
    args = parser.parse_args()

    eval_each_round = bool(args.eval_each_round) and not bool(args.no_eval_each_round)
    tag = setting_tag(algorithm=args.algorithm, partition=args.partition, alpha=args.alpha)
    default_adapter = ROOT / "models" / f"qwen_bloom_federated0.5B_{tag}"
    global_dir = Path(args.global_adapter) if args.global_adapter else default_adapter
    results_path = (
        Path(args.results_json)
        if args.results_json
        else ROOT / "results" / f"federated_lora_{tag}.json"
    )

    # Refuse to write into locked centralized paths.
    locked = {
        (ROOT / "models" / "qwen_bloom_trained0.5B").resolve(),
        (ROOT / "models" / "qwen_bloom_merged0.5B").resolve(),
        (ROOT / "results" / "bloom_lora_eval_0.5B.json").resolve(),
    }
    if global_dir.resolve() in locked or results_path.resolve() in locked:
        raise SystemExit(
            f"Refusing to overwrite locked centralized path: {global_dir} / {results_path}"
        )

    cfg = FederatedLoraConfig(
        base_model=args.base_model,
        num_clients=args.clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        max_samples_per_client=args.max_samples_per_client,
        clip_norm=args.clip_norm,
        dp_noise=args.dp_noise,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        global_adapter_dir=str(global_dir),
        algorithm=args.algorithm,
        prox_mu=args.prox_mu,
        partition=args.partition,
        dirichlet_alpha=args.alpha,
        seed=args.seed,
        from_scratch=not args.allow_central_seed,
    )

    # Persist config for client/server subprocesses.
    run_dir = ROOT / "federated" / "runs" / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    config_json = run_dir / "config.json"
    config_json.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    if global_dir.exists() and not args.resume and not args.skip_train:
        print(f"[sim] fresh start: clearing existing global adapter at {global_dir}")
        shutil.rmtree(global_dir)
    global_dir.mkdir(parents=True, exist_ok=True)

    if args.init_adapter and not args.allow_central_seed:
        raise SystemExit(
            "Refusing --init-adapter without --allow-central-seed "
            "(primary paper runs are from-scratch)."
        )
    if args.init_adapter and args.allow_central_seed and not args.skip_train:
        src = Path(args.init_adapter)
        if not (src / "adapter_config.json").is_file():
            raise FileNotFoundError(f"--init-adapter not found: {src}")
        for item in src.iterdir():
            if item.is_file():
                shutil.copy2(item, global_dir / item.name)
        print(f"[sim] ABLATION: seeded global adapter from {src}")

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
        strategy=cfg.partition,
        dirichlet_alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
    )
    label_mix = client_label_distributions(parts, cfg.label_col)
    (run_dir / "client_label_distribution.json").write_text(
        json.dumps(label_mix, indent=2), encoding="utf-8"
    )

    client_csvs: dict[str, Path] = {}
    for client_id, frame in parts.items():
        path = UPDATES_DIR / f"{client_id}.csv"
        frame[[cfg.text_col, cfg.label_col]].to_csv(path, index=False)
        client_csvs[client_id] = path
        print(f"[sim] client {client_id}: {len(frame)} rows | mix={label_mix[client_id]}")

    py = sys.executable
    history = []
    total_upload = 0
    total_download = 0
    trainable_parameters = None
    adapter_bytes = None
    eval_csv = Path(args.eval_csv) if args.eval_csv else Path(ROOT / "data" / "figshare_bloom_v1_val.csv")
    if eval_csv.resolve() == Path(cfg.test_csv).resolve():
        print(
            "[sim] WARNING: per-round eval-csv equals test-csv. "
            "Prefer data/figshare_bloom_v1_val.csv for monitoring to preserve test integrity."
        )

    for round_idx in range(1, cfg.rounds + 1):
        bundle_paths: list[Path] = []
        for client_id, csv_path in client_csvs.items():
            bundle_path = BUNDLES_DIR / f"round{round_idx:02d}_{client_id}.json"
            bundle_paths.append(bundle_path)
            if not args.skip_train:
                cmd = [
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
                    "--algorithm",
                    cfg.algorithm,
                    "--prox-mu",
                    str(cfg.prox_mu),
                    "--seed",
                    str(cfg.seed),
                    "--base-model",
                    cfg.base_model,
                    "--config-json",
                    str(config_json),
                ]
                _run(cmd)

        if not bundle_paths or not all(p.is_file() for p in bundle_paths):
            raise FileNotFoundError("missing client bundles; run without --skip-train")

        # Capture server stdout for communication stats.
        server_cmd = [
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
            "--algorithm",
            cfg.algorithm,
            "--prox-mu",
            str(cfg.prox_mu),
            "--base-model",
            cfg.base_model,
            "--config-json",
            str(config_json),
        ]
        print("[sim]", " ".join(server_cmd))
        proc = subprocess.run(
            server_cmd,
            cwd=str(ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        if proc.stdout.strip():
            print(proc.stdout)
        comm = {}
        try:
            # Last JSON object in stdout
            text = proc.stdout.strip()
            start = text.rfind("{")
            if start >= 0:
                payload = json.loads(text[start:])
                comm = payload.get("communication") or {}
        except json.JSONDecodeError:
            comm = {}

        upload = int(comm.get("upload_bytes_total") or 0)
        download = int(comm.get("download_bytes_total") or 0)
        total_upload += upload
        total_download += download
        if comm.get("trainable_parameters") is not None:
            trainable_parameters = int(comm["trainable_parameters"])
        if comm.get("adapter_bytes") is not None:
            adapter_bytes = int(comm["adapter_bytes"])

        round_record: dict = {
            "round": round_idx,
            "n_clients": len(bundle_paths),
            "upload_bytes": upload,
            "download_bytes": download,
            "upload_mb": _mb(upload),
            "download_mb": _mb(download),
            "communication_mb": _mb(upload + download),
        }
        if eval_each_round:
            metrics = _evaluate_global(global_dir, eval_csv, cfg)
            round_record.update(metrics)
            print(f"[sim] round {round_idx} global eval: {metrics}")
        history.append(round_record)

    # Final test metrics (always on the official test CSV).
    final_metrics = _evaluate_global(global_dir, Path(cfg.test_csv), cfg)

    report = {
        "simulation": "federatedly_trained_qwen25_0.5b_bloom_lora",
        "terminology": "federatedly trained LoRA classifier (base model is frozen/local)",
        **cfg.experiment_metadata(),
        "setting_tag": tag,
        "global_adapter": str(global_dir),
        "client_label_distribution": label_mix,
        "client_sizes": {cid: int(len(frame)) for cid, frame in parts.items()},
        "communication": {
            "trainable_parameters": trainable_parameters,
            "adapter_size_mb": _mb(adapter_bytes or 0),
            "upload_per_client_per_round_mb": _mb((adapter_bytes or 0)),
            "download_per_client_per_round_mb": _mb((adapter_bytes or 0)),
            "total_upload_mb": _mb(total_upload),
            "total_download_mb": _mb(total_download),
            "total_communication_mb": _mb(total_upload + total_download),
            "notes": (
                "Upload = client→server LoRA+score update; "
                "download = server→client global adapter broadcast."
            ),
        },
        "history": history,
        "final_test_metrics": final_metrics,
        "centralized_baseline_reference": {
            "results_json": "results/bloom_lora_eval_0.5B.json",
            "accuracy": 0.8486,
            "macro_f1": 0.8421,
            "note": "Locked centralized baseline — do not overwrite.",
        },
        "next_steps": [
            f"python merge_model.py --model-size 0.5b --lora-dir {global_dir} "
            f"--output-dir models/qwen_bloom_federated0.5B_{tag}_merged --force",
            f"python evaluate_bloom.py --model-size 0.5b "
            f"--model-dir models/qwen_bloom_federated0.5B_{tag}_merged "
            f"--results-json results/federated_bloom_eval_{tag}.json",
            f"python quantize_bloom.py --model-size 0.5b "
            f"--merged-dir models/qwen_bloom_federated0.5B_{tag}_merged "
            f"--output-dir models/qwen_bloom_federated0.5B_{tag}_fp16 --force",
        ],
        "privacy_disclaimer": (
            "Federated training keeps raw client data local during collaborative "
            "optimization but does not by itself provide formal protection against "
            "inference attacks on updates, secure aggregation, or differential privacy."
        ),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[sim] done -> {global_dir}")
    print(f"[sim] report -> {results_path}")
    print(json.dumps({"final_test_metrics": final_metrics, "communication": report["communication"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
