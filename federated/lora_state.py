from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

import torch


def is_trainable_key(key: str) -> bool:
    lowered = key.lower()
    return "lora_" in lowered or lowered.endswith("classifier.weight") or lowered.endswith("classifier.bias") or ".score." in lowered


def extract_trainable_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if is_trainable_key(k)
    }


def load_trainable_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    merged = {**current, **{k: v for k, v in state.items() if k in current}}
    model.load_state_dict(merged, strict=False)


def state_dict_to_delta(
    local: Dict[str, torch.Tensor],
    global_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {k: local[k] - global_state[k] for k in local if k in global_state}


def apply_delta(
    global_state: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in global_state.items()}
    for k, dv in delta.items():
        if k in out:
            out[k] = out[k] + scale * dv
    return out


def clip_delta(
    delta: Dict[str, torch.Tensor],
    clip_norm: float,
) -> Dict[str, torch.Tensor]:
    if clip_norm <= 0:
        return delta
    norm = math.sqrt(sum(float((v * v).sum()) for v in delta.values()))
    if norm <= clip_norm or norm == 0.0:
        return delta
    scale = clip_norm / norm
    return {k: v * scale for k, v in delta.items()}


def add_dp_noise(
    delta: Dict[str, torch.Tensor],
    noise_multiplier: float,
    clip_norm: float,
    rng: random.Random,
) -> Dict[str, torch.Tensor]:
    if noise_multiplier <= 0:
        return delta
    sigma = noise_multiplier * max(clip_norm, 1e-6)
    return {k: v + torch.randn_like(v) * sigma for k, v in delta.items()}


def fedavg_state_dicts(
    weighted_states: Sequence[Tuple[int, Dict[str, torch.Tensor]]],
) -> Dict[str, torch.Tensor]:
    if not weighted_states:
        raise ValueError("no client states to aggregate")
    total = sum(int(n) for n, _ in weighted_states)
    if total <= 0:
        raise ValueError("zero total sample weight")

    keys = list(weighted_states[0][1].keys())
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        acc = None
        ref_dtype = weighted_states[0][1][key].dtype
        for count, state in weighted_states:
            tensor = state[key].float()
            weighted = tensor * (float(count) / float(total))
            acc = weighted if acc is None else acc + weighted
        out[key] = acc.to(dtype=ref_dtype)  # type: ignore[union-attr]
    return out


def fedavg_deltas(
    weighted_deltas: Sequence[Tuple[int, Dict[str, torch.Tensor]]],
    global_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    merged = fedavg_state_dicts(
        [
            (n, apply_delta(global_state, d, scale=1.0))
            for n, d in weighted_deltas
        ]
    )
    return state_dict_to_delta(merged, global_state)
