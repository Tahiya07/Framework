"""Class-weight helpers for federated Bloom clients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight

from federated.config import BLOOM_LABELS, FederatedLoraConfig


def load_global_class_weights(
    csv_path: str | Path,
    *,
    text_col: str = "question",
    label_col: str = "bloom_level",
    max_weight: float = 3.0,
) -> torch.Tensor:
    """Balanced weights from the full training corpus (stable across clients).

    Per-client ``balanced`` weights on ~300-row shards can reach 10–50 for rare
  classes and explode both loss and gradients. Global weights stay in ~0.4–1.2.
    """
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    df = df[df[label_col].isin(BLOOM_LABELS)]
    y = df[label_col].map(BLOOM_LABELS).astype(int).to_numpy()
    classes = np.arange(len(BLOOM_LABELS))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    if max_weight > 0:
        weights = np.clip(weights, 1.0 / max_weight, max_weight)
    return torch.tensor(weights, dtype=torch.float)


def load_local_class_weights(
    labels: list[int],
    *,
    max_weight: float = 3.0,
) -> torch.Tensor:
    """Per-shard balanced weights with a hard cap (fallback / ablation)."""
    present = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=present, y=labels)
    full = np.ones(len(BLOOM_LABELS), dtype=np.float32)
    for cls, w in zip(present, weights):
        full[int(cls)] = float(np.clip(w, 1.0 / max(max_weight, 1e-6), max_weight))
    return torch.tensor(full, dtype=torch.float)


def resolve_class_weights(config: FederatedLoraConfig, labels: list[int]) -> torch.Tensor | None:
    if not config.use_class_weights:
        return None
    if config.class_weight_source == "global":
        path = Path(config.train_csv)
        if not path.is_file():
            raise FileNotFoundError(
                f"Global class weights need train_csv={config.train_csv}; "
                "set class_weight_source='local' or provide the file."
            )
        return load_global_class_weights(
            path,
            text_col=config.text_col,
            label_col=config.label_col,
            max_weight=config.class_weight_max,
        )
    if config.class_weight_source == "local":
        return load_local_class_weights(labels, max_weight=config.class_weight_max)
    return None
