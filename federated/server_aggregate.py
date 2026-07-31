#!/usr/bin/env python
"""Federated server: aggregate encrypted LoRA bundles (weighted FedAvg).

FedProx is applied on the client; the server still averages LoRA + score updates.
Architecture MUST match centralized train_qwen_bloom.py (r=32, α=64, score).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.config import BLOOM_LABELS, FederatedLoraConfig, make_peft_lora_config  # noqa: E402
from federated.lora_state import (  # noqa: E402
    add_dp_noise,
    apply_delta,
    clip_delta,
    extract_trainable_state,
    fedavg_deltas,
    load_trainable_state,
    state_dict_to_delta,
    trainable_nbytes,
    trainable_param_count,
)
from federated.secure_bundle import load_bundle, unpack_update  # noqa: E402


def _load_global_state(config: FederatedLoraConfig, global_dir: Path) -> Dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(BLOOM_LABELS),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id

    if (global_dir / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(base, str(global_dir), is_trainable=True)
    else:
        model = get_peft_model(base, make_peft_lora_config(config))
    return extract_trainable_state(model)


def _save_global_adapter(
    config: FederatedLoraConfig,
    state: Dict[str, torch.Tensor],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(BLOOM_LABELS),
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(base, make_peft_lora_config(config))
    load_trainable_state(model, state)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    meta = {
        "format": "federated_lora_global_v1",
        "aggregation": "fedavg_weighted",
        "client_algorithm": config.algorithm,
        "prox_mu": float(config.prox_mu) if config.algorithm == "fedprox" else 0.0,
        "clip_norm": config.clip_norm,
        "dp_noise": config.dp_noise,
        "lora": config.lora_config_dict(),
        "base_model": config.base_model,
        "trainable_parameters": trainable_param_count(state),
        "adapter_bytes": trainable_nbytes(state),
    }
    (out_dir / "federated_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[server] global adapter saved -> {out_dir}")
    return meta


def aggregate_bundles(
    bundle_paths: List[Path],
    config: FederatedLoraConfig,
    global_dir: Path,
) -> tuple[Dict[str, torch.Tensor], dict]:
    global_state = _load_global_state(config, global_dir)
    weighted_deltas: List[Tuple[int, Dict[str, torch.Tensor]]] = []
    rng = random.Random(config.seed)
    upload_bytes = 0
    n_params = trainable_param_count(global_state)

    for path in bundle_paths:
        bundle = load_bundle(path)
        local_state = unpack_update(bundle)
        upload_bytes += int(bundle.get("update_bytes") or trainable_nbytes(local_state))
        delta = state_dict_to_delta(local_state, global_state)
        delta = clip_delta(delta, config.clip_norm)
        delta = add_dp_noise(delta, config.dp_noise, config.clip_norm, rng)
        weighted_deltas.append((int(bundle["n_samples"]), delta))

    merged_delta = fedavg_deltas(weighted_deltas, global_state)
    new_state = apply_delta(global_state, merged_delta, scale=1.0)
    download_bytes = trainable_nbytes(new_state) * len(bundle_paths)
    comm = {
        "n_clients": len(bundle_paths),
        "trainable_parameters": n_params,
        "adapter_bytes": trainable_nbytes(new_state),
        "upload_bytes_total": upload_bytes,
        "download_bytes_total": download_bytes,
        "communication_bytes_total": upload_bytes + download_bytes,
    }
    return new_state, comm


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate federated LoRA client bundles.")
    parser.add_argument("--bundles", nargs="+", required=True)
    parser.add_argument("--global-adapter", required=True)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise", type=float, default=0.0)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--algorithm", choices=("fedavg", "fedprox"), default="fedavg")
    parser.add_argument("--prox-mu", type=float, default=None)
    parser.add_argument("--config-json", default=None)
    args = parser.parse_args()

    cfg = FederatedLoraConfig(clip_norm=args.clip_norm, dp_noise=args.dp_noise)
    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        for key, value in payload.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    if args.base_model:
        cfg.base_model = args.base_model
    cfg.algorithm = args.algorithm
    if args.prox_mu is not None:
        cfg.prox_mu = args.prox_mu

    global_dir = Path(args.global_adapter)
    paths = [Path(p) for p in args.bundles]
    new_state, comm = aggregate_bundles(paths, cfg, global_dir)
    meta = _save_global_adapter(cfg, new_state, global_dir)

    summary = {
        "n_clients": len(paths),
        "global_adapter": str(global_dir),
        "clip_norm": cfg.clip_norm,
        "dp_noise": cfg.dp_noise,
        "algorithm": cfg.algorithm,
        "prox_mu": float(cfg.prox_mu) if cfg.algorithm == "fedprox" else 0.0,
        "communication": comm,
        "metadata": meta,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
