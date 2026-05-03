"""
uncertainty.py
===========================================================
Unified uncertainty + calibration engine (publishable version)
"""

from __future__ import annotations

import math
import numpy as np
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Any


# ---------------------------------------------------------
EPS = 1e-12
K_DEFAULT = 6
JSD_MAX = math.log(2.0)


# ---------------------------------------------------------
def _as_prob(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0, None)
    s = p.sum()
    if s <= EPS:
        raise ValueError("invalid probability distribution")
    return p / s


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0)
    return float(-(p * np.log(p)).sum())


def _jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (_kl(p, m) + _kl(q, m))


def _kl(p, q):
    p = np.clip(p, EPS, 1)
    q = np.clip(q, EPS, 1)
    return float((p * (np.log(p) - np.log(q))).sum())


# ---------------------------------------------------------
@dataclass
class UncertaintySummary:
    spu: float
    entropy: float
    confidence: float
    n_samples: int

    @property
    def bloom_uncertainty(self) -> float:
        return self.entropy

    @property
    def top1_confidence(self) -> float:
        return self.confidence


# ---------------------------------------------------------
class UncertaintyEngine:
    def __init__(self, K: int = K_DEFAULT, n_bins: int = 10):
        self.K = K
        self.n_bins = int(n_bins)

    # -------------------------
    def compute_spu(self, outputs: Sequence[np.ndarray]) -> float:
        if len(outputs) < 2:
            return 0.0

        P = [_as_prob(o) for o in outputs]

        total = 0.0
        c = 0

        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                total += _jsd(P[i], P[j])
                c += 1

        return float((total / c) / JSD_MAX)  # normalized [0,1]

    def compute_jsd_uncertainty(self, outputs: Sequence[np.ndarray]) -> float:
        return self.compute_spu(outputs)

    # -------------------------
    def compute_entropy(self, p: Sequence[float]) -> Tuple[float, float]:
        p = _as_prob(p)
        H = _entropy(p)
        return H / math.log(self.K), float(np.max(p))

    # -------------------------
    def aggregate(
        self,
        p: Optional[Sequence[float]] = None,
        stochastic: Optional[List[np.ndarray]] = None,
    ) -> UncertaintySummary:

        entropy_norm = 0.0
        conf = 1.0
        spu = 0.0
        n = 0

        if p is not None:
            entropy_norm, conf = self.compute_entropy(p)

        if stochastic:
            spu = self.compute_spu(stochastic)
            n = len(stochastic)

        return UncertaintySummary(
            spu=spu,
            entropy=entropy_norm,
            confidence=conf,
            n_samples=n,
        )

    def aggregate_summary(
        self,
        bloom_p: Optional[Sequence[float]] = None,
        stochastic: Optional[List[np.ndarray]] = None,
    ) -> UncertaintySummary:
        return self.aggregate(p=bloom_p, stochastic=stochastic)

    def compute_bloom_uncertainty(self, p: Sequence[float]) -> float:
        entropy_norm, _ = self.compute_entropy(p)
        return float(entropy_norm)

    def compute_ece(
        self,
        confidences: Sequence[float],
        accuracies: Sequence[float],
        n_bins: Optional[int] = None,
    ) -> float:
        conf = np.asarray(confidences, dtype=np.float64)
        acc = np.asarray(accuracies, dtype=np.float64)
        if conf.size == 0 or acc.size == 0:
            return 0.0
        if conf.shape[0] != acc.shape[0]:
            raise ValueError("confidences and accuracies must have the same length")
        bins = int(n_bins or self.n_bins)
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (conf >= lo) & (conf <= hi) if i == bins - 1 else (conf >= lo) & (conf < hi)
            if not np.any(mask):
                continue
            ece += float(mask.mean() * abs(acc[mask].mean() - conf[mask].mean()))
        return float(ece)

    def reliability_data(
        self,
        confidences: Sequence[float],
        accuracies: Sequence[float],
        n_bins: Optional[int] = None,
    ) -> dict:
        conf = np.asarray(confidences, dtype=np.float64)
        acc = np.asarray(accuracies, dtype=np.float64)
        if conf.shape[0] != acc.shape[0]:
            raise ValueError("confidences and accuracies must have the same length")
        bins = int(n_bins or self.n_bins)
        edges = np.linspace(0.0, 1.0, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        bin_acc = np.full(bins, np.nan, dtype=np.float64)
        bin_conf = np.full(bins, np.nan, dtype=np.float64)
        counts = np.zeros(bins, dtype=np.int64)
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (conf >= lo) & (conf <= hi) if i == bins - 1 else (conf >= lo) & (conf < hi)
            if not np.any(mask):
                continue
            bin_acc[i] = float(acc[mask].mean())
            bin_conf[i] = float(conf[mask].mean())
            counts[i] = int(mask.sum())
        return {
            "bin_centers": centers,
            "bin_accuracy": bin_acc,
            "bin_confidence": bin_conf,
            "bin_counts": counts,
        }

    # -------------------------
    def gate(
        self,
        p: Sequence[float],
        threshold: float = 0.4,
    ) -> dict:

        p = _as_prob(p)
        idx = int(np.argmax(p))
        conf = float(np.max(p))

        entropy_norm, _ = self.compute_entropy(p)
        overall_conf = 1.0 - entropy_norm

        if overall_conf < threshold:
            return {
                "accepted": False,
                "action": "fallback",
                "label": idx,
                "confidence": overall_conf,
            }

        return {
            "accepted": True,
            "action": "auto",
            "label": idx,
            "confidence": overall_conf,
        }
