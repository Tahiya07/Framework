"""Client data partitioning for federated Bloom LoRA experiments.

Modes
-----
iid
    Stratified by bloom_level, then round-robin into equal-sized shards.
non_iid_label
    Dirichlet(α) label skew (default α=0.5). Soft mixtures — never single-label.
hash
    Legacy SHA-256 text bucketing (near-IID, kept for back-compat).
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from federated.config import BLOOM_LABEL_ORDER, BLOOM_LABELS


def _client_id(text: str, num_clients: int, prefix: str = "teacher_site") -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % max(1, num_clients)
    return f"{prefix}_{bucket:02d}"


def label_distribution(df: pd.DataFrame, label_col: str) -> Dict[str, float]:
    counts = Counter(str(x) for x in df[label_col].tolist())
    total = max(sum(counts.values()), 1)
    return {lab: round(counts.get(lab, 0) / total, 6) for lab in BLOOM_LABEL_ORDER}


def client_label_distributions(
    parts: Mapping[str, pd.DataFrame],
    label_col: str,
) -> Dict[str, Dict[str, float]]:
    return {cid: label_distribution(frame, label_col) for cid, frame in parts.items()}


def _cap_client(frame: pd.DataFrame, max_per_client: int, seed: int) -> pd.DataFrame:
    if max_per_client > 0 and len(frame) > max_per_client:
        return frame.sample(max_per_client, random_state=seed).reset_index(drop=True)
    return frame.reset_index(drop=True)


def _partition_hash(
    df: pd.DataFrame,
    *,
    num_clients: int,
    text_col: str,
    max_per_client: int,
    seed: int,
    prefix: str,
) -> Dict[str, pd.DataFrame]:
    work = df.copy()
    work["client_id"] = work[text_col].astype(str).map(lambda t: _client_id(t, num_clients, prefix))
    parts: Dict[str, pd.DataFrame] = {}
    for client_id, group in work.groupby("client_id"):
        parts[str(client_id)] = _cap_client(group.drop(columns=["client_id"]), max_per_client, seed)
    return parts


def _partition_iid(
    df: pd.DataFrame,
    *,
    num_clients: int,
    label_col: str,
    max_per_client: int,
    seed: int,
    prefix: str,
) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    buckets: List[List[int]] = [[] for _ in range(num_clients)]
    for label in BLOOM_LABEL_ORDER:
        idxs = df.index[df[label_col].astype(str) == label].to_numpy()
        if len(idxs) == 0:
            continue
        rng.shuffle(idxs)
        for i, idx in enumerate(idxs):
            buckets[i % num_clients].append(int(idx))

    parts: Dict[str, pd.DataFrame] = {}
    for i, idxs in enumerate(buckets):
        cid = f"{prefix}_{i:02d}"
        frame = df.loc[idxs].copy() if idxs else df.iloc[0:0].copy()
        parts[cid] = _cap_client(frame, max_per_client, seed + i)
    return parts


def _partition_dirichlet(
    df: pd.DataFrame,
    *,
    num_clients: int,
    label_col: str,
    alpha: float,
    max_per_client: int,
    seed: int,
    prefix: str,
    min_per_client: int = 1,
) -> Dict[str, pd.DataFrame]:
    """LDA-style label skew used in FL literature (Dirichlet α)."""
    if alpha <= 0:
        raise ValueError(f"dirichlet alpha must be > 0, got {alpha}")

    rng = np.random.default_rng(seed)
    n_labels = len(BLOOM_LABEL_ORDER)
    # Mixing proportions per client: shape (num_clients, n_labels)
    proportions = rng.dirichlet([alpha] * n_labels, size=num_clients)

    label_indices: Dict[str, np.ndarray] = {}
    for label in BLOOM_LABEL_ORDER:
        idxs = df.index[df[label_col].astype(str) == label].to_numpy().copy()
        rng.shuffle(idxs)
        label_indices[label] = idxs

    # Allocate each label's examples to clients proportional to Dirichlet draw.
    client_idxs: List[List[int]] = [[] for _ in range(num_clients)]
    for li, label in enumerate(BLOOM_LABEL_ORDER):
        idxs = label_indices[label]
        if len(idxs) == 0:
            continue
        # Expected counts; ensure we distribute all samples.
        raw = proportions[:, li] * len(idxs)
        counts = np.floor(raw).astype(int)
        # Assign remainders to clients with largest fractional parts.
        remainder = int(len(idxs) - counts.sum())
        frac_order = np.argsort(-(raw - counts))
        for j in range(remainder):
            counts[frac_order[j % num_clients]] += 1

        # Guard against empty allocations when alpha is very small.
        if counts.sum() < len(idxs):
            counts[-1] += len(idxs) - int(counts.sum())

        cursor = 0
        for ci in range(num_clients):
            take = int(counts[ci])
            if take <= 0:
                continue
            chunk = idxs[cursor : cursor + take]
            client_idxs[ci].extend(int(x) for x in chunk)
            cursor += take

    # Ensure every client has at least min_per_client rows by borrowing from largest.
    sizes = [len(x) for x in client_idxs]
    for ci, sz in enumerate(sizes):
        if sz >= min_per_client:
            continue
        donor = int(np.argmax(sizes))
        while len(client_idxs[ci]) < min_per_client and client_idxs[donor]:
            client_idxs[ci].append(client_idxs[donor].pop())
            sizes[donor] -= 1
            sizes[ci] += 1

    parts: Dict[str, pd.DataFrame] = {}
    for i, idxs in enumerate(client_idxs):
        cid = f"{prefix}_{i:02d}"
        frame = df.loc[idxs].copy() if idxs else df.iloc[0:0].copy()
        # Shuffle within client for training order diversity.
        if len(frame):
            frame = frame.sample(frac=1.0, random_state=seed + i)
        parts[cid] = _cap_client(frame, max_per_client, seed + i)
    return parts


def partition_csv(
    csv_path: str | Path,
    *,
    num_clients: int,
    text_col: str = "question",
    label_col: str = "bloom_level",
    max_per_client: int = 0,
    prefix: str = "teacher_site",
    strategy: str = "iid",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path).dropna(subset=[text_col, label_col])
    df = df[df[label_col].isin(BLOOM_LABELS)].copy()
    df = df.reset_index(drop=True)

    strategy = (strategy or "iid").lower().strip()
    if strategy in {"hash", "legacy"}:
        return _partition_hash(
            df,
            num_clients=num_clients,
            text_col=text_col,
            max_per_client=max_per_client,
            seed=seed,
            prefix=prefix,
        )
    if strategy in {"iid", "stratified"}:
        return _partition_iid(
            df,
            num_clients=num_clients,
            label_col=label_col,
            max_per_client=max_per_client,
            seed=seed,
            prefix=prefix,
        )
    if strategy in {"non_iid_label", "noniid", "dirichlet", "non_iid"}:
        return _partition_dirichlet(
            df,
            num_clients=num_clients,
            label_col=label_col,
            alpha=dirichlet_alpha,
            max_per_client=max_per_client,
            seed=seed,
            prefix=prefix,
        )
    raise ValueError(
        f"Unknown partition strategy {strategy!r}. "
        "Choose from: iid, non_iid_label, hash"
    )
