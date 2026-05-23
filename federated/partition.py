from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

import pandas as pd

from federated.config import BLOOM_LABELS


def _client_id(text: str, num_clients: int, prefix: str = "teacher_site") -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % max(1, num_clients)
    return f"{prefix}_{bucket:02d}"


def partition_csv(
    csv_path: str | Path,
    *,
    num_clients: int,
    text_col: str = "question",
    label_col: str = "bloom_level",
    max_per_client: int = 0,
    prefix: str = "teacher_site",
) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    df = df[df[label_col].isin(BLOOM_LABELS)].copy()
    df["client_id"] = df[text_col].astype(str).map(lambda t: _client_id(t, num_clients, prefix))

    parts: Dict[str, pd.DataFrame] = {}
    for client_id, group in df.groupby("client_id"):
        part = group
        if max_per_client > 0 and len(part) > max_per_client:
            part = part.sample(max_per_client, random_state=42)
        parts[str(client_id)] = part.reset_index(drop=True)
    return parts
