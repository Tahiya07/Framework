from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Callable

import numpy as np

from ingestion import DocumentIngestor
from retriever import PrivacyRetriever
from classifier import BloomClassifier
from uncertainty import UncertaintyEngine
from evaluation_core import evaluate


# =========================================================
# CONFIGURATION OBJECT (IMPORTANT FOR REPRODUCIBILITY)
# =========================================================

@dataclass
class PipelineConfig:
    chunk_size: int = 256
    top_k: int = 5
    use_retrieval: bool = True


# =========================================================
# PIPELINE CORE
# =========================================================

class BloomPrivacyPipeline:
    """
    End-to-end Bloom classification + privacy-aware retrieval system.

    Design principle:
    - ingestion is SAFE boundary
    - everything downstream is embedding / metadata only
    """

    def __init__(
        self,
        model_fn: Callable,
        config: PipelineConfig = PipelineConfig(),
    ) -> None:

        self.config = config

        self.ingestor = DocumentIngestor(chunk_size=config.chunk_size)
        self.retriever = PrivacyRetriever()
        self.classifier = BloomClassifier(model_fn=model_fn)
        self.uncertainty = UncertaintyEngine()

        self._safe_docs: List[str] = []

    # -----------------------------------------------------
    # 1. SAFE INGESTION LAYER
    # -----------------------------------------------------
    def ingest_pdf(self, path: str) -> None:
        """
        Converts PDF → SAFE metadata-only representation
        """

        safe_chunks = self.ingestor.process_pdf_safe(path)

        # IMPORTANT:
        # Only abstract metadata is stored, never raw text
        docs = [
            f"{c.modality}|{c.bloom_level}|{c.topic}|{c.difficulty}"
            for c in safe_chunks
        ]

        self._safe_docs.extend(docs)
        self.retriever.build_index(self._safe_docs)

    # -----------------------------------------------------
    # 2. RETRIEVAL
    # -----------------------------------------------------
    def retrieve(self, query: str, top_k: int = None):
        k = top_k or self.config.top_k
        return self.retriever.retrieve(query, top_k=k)

    # -----------------------------------------------------
    # 3. CLASSIFICATION (SAFE INTERFACE)
    # -----------------------------------------------------
    def predict(self, text: str):
        return self.classifier.predict([text])[0]

    def predict_proba(self, text: str):
        return self.classifier.predict_proba([text])[0]

    # -----------------------------------------------------
    # 4. UNCERTAINTY
    # -----------------------------------------------------
    def analyze_uncertainty(self, probs: np.ndarray):
        return self.uncertainty.compute_bloom_uncertainty(probs)

    # -----------------------------------------------------
    # 5. FULL INFERENCE
    # -----------------------------------------------------
    def run(self, text: str) -> Dict[str, Any]:

        pred = self.predict(text)
        probs = self.predict_proba(text)

        entropy, top1 = self.uncertainty.compute_bloom_uncertainty(probs)

        return {
            "prediction": {
                "label": pred.label,
                "label_id": pred.label_id,
                "confidence": pred.confidence,
                "ordinal_risk": pred.ordinal_risk,
                "margin": pred.margin,
            },
            "uncertainty": {
                "entropy": entropy,
                "top1_confidence": top1,
            },
        }

    # -----------------------------------------------------
    # 6. PAPER-GRADE EVALUATION (FIXED)
    # -----------------------------------------------------
    def evaluate(self, X: List[str], y_true: List[str]) -> Dict[str, float]:

        probs = np.vstack([self.predict_proba(x) for x in X])
        preds = [self.predict(x).label for x in X]

        return evaluate(y_true=y_true, y_pred=preds)