"""
classifier.py
===========================================================
Publishable Bloom-Level Cognitive Classifier

Key properties:
- Ordinal-aware prediction (Bloom structure preserved)
- Structured outputs (dataclass-based)
- Uncertainty-ready interface
- Clean separation from evaluation logic
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np


# ---------------------------------------------------------
# Bloom taxonomy (ordinal structure)
# ---------------------------------------------------------
BLOOM_LABELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
@dataclass
class ClassifierConfig:
    temperature: float = 1.0


# ---------------------------------------------------------
# Structured prediction
# ---------------------------------------------------------
@dataclass
class BloomPrediction:
    label: str
    label_id: int
    confidence: float
    probabilities: np.ndarray
    margin: float
    ordinal_risk: float


# ---------------------------------------------------------
# Classifier
# ---------------------------------------------------------
class BloomClassifier:
    def __init__(
        self,
        model_fn: Callable[..., np.ndarray],
        config: Optional[ClassifierConfig] = None,
    ) -> None:
        self.model_fn = model_fn
        self.config = config or ClassifierConfig()

    # -----------------------------
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / (np.sum(e) + 1e-12)

    # -----------------------------
    def predict_proba(self, x: str, **kwargs) -> np.ndarray:
        out = np.asarray(self.model_fn(x, **kwargs), dtype=np.float64).reshape(-1)

        if out.shape[0] != len(BLOOM_LABELS):
            raise ValueError("Invalid output dimension")

        if not np.isclose(out.sum(), 1.0, atol=1e-2):
            out = self._softmax(out)

        return out

    # -----------------------------
    def predict(self, x: str, **kwargs) -> BloomPrediction:
        probs = self.predict_proba(x, **kwargs)

        idx = int(np.argmax(probs))
        top1 = float(probs[idx])

        sorted_idx = np.argsort(-probs)
        second = int(sorted_idx[1]) if len(sorted_idx) > 1 else idx

        margin = float(probs[idx] - probs[second])

        ordinal_positions = np.arange(len(BLOOM_LABELS))
        expected_pos = float(np.sum(probs * ordinal_positions))
        ordinal_risk = abs(expected_pos - idx) / len(BLOOM_LABELS)

        return BloomPrediction(
            label=BLOOM_LABELS[idx],
            label_id=idx,
            confidence=top1,
            probabilities=probs,
            margin=margin,
            ordinal_risk=ordinal_risk,
        )

    # -----------------------------
    def explain(self, x: str, top_k: int = 3, **kwargs) -> Dict:
        pred = self.predict(x, **kwargs)
        probs = pred.probabilities
        idx_sorted = np.argsort(-probs)

        return {
            "input": x,
            "prediction": pred.label,
            "confidence": pred.confidence,
            "ordinal_risk": pred.ordinal_risk,
            "margin": pred.margin,
            "top_alternatives": [
                {"label": BLOOM_LABELS[i], "probability": float(probs[i])}
                for i in idx_sorted[1:top_k]
            ],
        }