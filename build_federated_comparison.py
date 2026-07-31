#!/usr/bin/env python
"""Build the five-row centralized vs federated Bloom comparison table.

Reads the locked centralized 0.5B eval JSON and any federated_lora_*.json
simulation reports under results/. Never overwrites the centralized baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CENTRAL = RESULTS / "bloom_lora_eval_0.5B.json"
OUT_JSON = RESULTS / "federated_bloom_comparison.json"
OUT_CSV = RESULTS / "federated_bloom_comparison.csv"
OUT_MD = RESULTS / "federated_bloom_comparison.md"


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _central_row(data: dict) -> dict[str, Any]:
    m = data.get("qwen_lora") or {}
    return {
        "setting": "Centralized LoRA",
        "partition": "—",
        "algorithm": "LoRA",
        "prox_mu": 0.0,
        "accuracy": m.get("accuracy"),
        "macro_f1": m.get("macro_f1"),
        "quadratic_weighted_kappa": m.get("quadratic_weighted_kappa"),
        "within_one_level_accuracy": m.get("within_one_level_accuracy"),
        "severe_error_rate": m.get("severe_error_rate"),
        "ece": m.get("ece"),
        "total_communication_mb": 0.0,
        "trainable_parameters": None,
        "source": str(CENTRAL),
        "n_test": data.get("n_test"),
    }


def _federated_row(data: dict, source: Path) -> dict[str, Any]:
    final = data.get("final_test_metrics") or {}
    # Fallback: last history entry with metrics
    if not final:
        for item in reversed(data.get("history") or []):
            if "accuracy" in item:
                final = item
                break
    comm = data.get("communication") or {}
    return {
        "setting": f"Federated {data.get('algorithm', '?').upper()} ({data.get('partition', '?')})",
        "partition": data.get("partition"),
        "algorithm": data.get("algorithm"),
        "prox_mu": data.get("prox_mu"),
        "dirichlet_alpha": data.get("dirichlet_alpha"),
        "num_clients": data.get("num_clients"),
        "rounds": data.get("rounds"),
        "seed": data.get("seed"),
        "accuracy": final.get("accuracy"),
        "macro_f1": final.get("macro_f1"),
        "quadratic_weighted_kappa": final.get("quadratic_weighted_kappa"),
        "within_one_level_accuracy": final.get("within_one_level_accuracy"),
        "severe_error_rate": final.get("severe_error_rate"),
        "ece": final.get("ece"),
        "total_communication_mb": comm.get("total_communication_mb"),
        "trainable_parameters": comm.get("trainable_parameters"),
        "adapter_size_mb": comm.get("adapter_size_mb"),
        "source": str(source),
        "global_adapter": data.get("global_adapter"),
        "setting_tag": data.get("setting_tag"),
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    central = _load(CENTRAL)
    if central:
        rows.append(_central_row(central))
    else:
        print(f"[warn] missing locked centralized baseline: {CENTRAL}")

    # Prefer tagged paper reports; also accept legacy federated_lora_simulation.json
    candidates = sorted(RESULTS.glob("federated_lora_*.json"))
    legacy = RESULTS / "federated_lora_simulation.json"
    if legacy.is_file() and legacy not in candidates:
        candidates.append(legacy)

    for path in candidates:
        if path.name.startswith("federated_lora_") and "comparison" in path.name:
            continue
        data = _load(path)
        if not data:
            continue
        if data.get("simulation") or data.get("final_test_metrics") or data.get("history"):
            rows.append(_federated_row(data, path))
    return rows


def _to_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Setting",
        "Partition",
        "Algorithm",
        "Accuracy",
        "Macro-F1",
        "QWK",
        "ECE",
        "Comm (MB)",
    ]
    lines = [
        "# Federated vs centralized Bloom LoRA (0.5B)",
        "",
        "Centralized baseline is locked from `results/bloom_lora_eval_0.5B.json`.",
        "Federated runs must not overwrite that file.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r.get("setting", "")),
                    str(r.get("partition", "")),
                    str(r.get("algorithm", "")),
                    f"{r['accuracy']:.4f}" if isinstance(r.get("accuracy"), (int, float)) else "—",
                    f"{r['macro_f1']:.4f}" if isinstance(r.get("macro_f1"), (int, float)) else "—",
                    f"{r['quadratic_weighted_kappa']:.4f}"
                    if isinstance(r.get("quadratic_weighted_kappa"), (int, float))
                    else "—",
                    f"{r['ece']:.4f}" if isinstance(r.get("ece"), (int, float)) else "—",
                    f"{r['total_communication_mb']:.3f}"
                    if isinstance(r.get("total_communication_mb"), (int, float))
                    else "—",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(
        "Privacy note: federated training keeps raw client data local but does not "
        "by itself provide formal secure aggregation or differential privacy."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build federated Bloom comparison table.")
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument("--out-md", type=Path, default=OUT_MD)
    args = parser.parse_args()

    rows = build_rows()
    payload = {
        "benchmark": "federated_vs_centralized_bloom_0.5b",
        "centralized_baseline": str(CENTRAL),
        "n_rows": len(rows),
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if rows:
        fieldnames = list(rows[0].keys())
        # Union of keys
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    args.out_md.write_text(_to_markdown(rows), encoding="utf-8")
    print(json.dumps({"n_rows": len(rows), "wrote": [str(args.out_json), str(args.out_csv), str(args.out_md)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
