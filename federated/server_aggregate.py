#!/usr/bin/env python
"""Federated server: aggregate encrypted LoRA bundles (FedAvg + optional DP noise)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from federated.config import BLOOM_LABELS, FederatedLoraConfig  # noqa: E402
from federated.lora_state import (  # noqa: E402
    add_dp_noise,
    apply_delta,
    clip_delta,
    extract_trainable_state,
    fedavg_deltas,
    load_trainable_state,
    state_dict_to_delta,
)
from federated.secure_bundle import load_bundle, unpack_update  # noqa: E402


def _load_global_state(config: FederatedLoraConfig, global_dir: Path) -> Dict[str, torch.Tensor]:
    if (global_dir / "adapter_config.json").is_file():
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForSequenceClassification.from_pretrained(
            config.base_model,
            num_labels=len(BLOOM_LABELS),
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base, str(global_dir), is_trainable=True)
        return extract_trainable_state(model)

    from peft import LoraConfig, TaskType, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    base = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=len(BLOOM_LABELS),
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    return extract_trainable_state(model)


def _save_global_adapter(config: FederatedLoraConfig, state: Dict[str, torch.Tensor], out_dir: Path) -> None:
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
    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    load_trainable_state(model, state)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "format": "federated_lora_global_v1",
        "aggregation": "fedavg_weighted",
        "clip_norm": config.clip_norm,
        "dp_noise": config.dp_noise,
    }
    (out_dir / "federated_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[server] global adapter saved -> {out_dir}")


def aggregate_bundles(
    bundle_paths: List[Path],
    config: FederatedLoraConfig,
    global_dir: Path,
) -> Dict[str, torch.Tensor]:
    global_state = _load_global_state(config, global_dir)
    weighted_deltas: List[Tuple[int, Dict[str, torch.Tensor]]] = []
    rng = random.Random(config.seed)

    for path in bundle_paths:
        bundle = load_bundle(path)
        local_state = unpack_update(bundle)
        delta = state_dict_to_delta(local_state, global_state)
        delta = clip_delta(delta, config.clip_norm)
        delta = add_dp_noise(delta, config.dp_noise, config.clip_norm, rng)
        weighted_deltas.append((int(bundle["n_samples"]), delta))

    merged_delta = fedavg_deltas(weighted_deltas, global_state)
    return apply_delta(global_state, merged_delta, scale=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate federated LoRA client bundles.")
    parser.add_argument("--bundles", nargs="+", required=True, help="Paths to client bundle JSON files.")
    parser.add_argument("--global-adapter", required=True, help="Input/output global adapter directory.")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise", type=float, default=0.0)
    args = parser.parse_args()

    cfg = FederatedLoraConfig(clip_norm=args.clip_norm, dp_noise=args.dp_noise)
    global_dir = Path(args.global_adapter)
    paths = [Path(p) for p in args.bundles]
    new_state = aggregate_bundles(paths, cfg, global_dir)
    _save_global_adapter(cfg, new_state, global_dir)

    summary = {
        "n_clients": len(paths),
        "global_adapter": str(global_dir),
        "clip_norm": cfg.clip_norm,
        "dp_noise": cfg.dp_noise,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
