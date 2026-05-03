"""
system.py
===========================================================
Unified Privacy-Preserving Cognitive QA System

This is the ORCHESTRATION LAYER that binds:
- ingestion (safe chunks)
- retrieval (FAISS)
- classifier (Bloom)
- uncertainty engine
- reranker

This is what makes the system "paper-complete".
"""

from __future__ import annotations

from retriever import PrivacyRetriever
from classifier import BloomLDLClassifier
from uncertainty import UncertaintyEngine
from reranker import rerank_with_uncertainty


class CognitiveSystem:

    def __init__(
        self,
        retriever: PrivacyRetriever,
        classifier: BloomLDLClassifier,
        uncertainty: UncertaintyEngine,
    ):
        self.retriever = retriever
        self.classifier = classifier
        self.uncertainty = uncertainty

    # -----------------------------------------------------
    def forward(self, query: str, top_k: int = 5) -> dict:
        """
        End-to-end inference pipeline:

        query → retrieval → classification →
        uncertainty → reranking → final output
        """

        # 1. Retrieve candidates
        results = self.retriever.retrieve(query, top_k=top_k)

        # 2. Classify query (Bloom level)
        pred = self.classifier.predict(query)

        probs = pred.probabilities
        ordinal_risk = pred.ordinal_risk

        # 3. Uncertainty estimation
        uncertainty_summary = self.uncertainty.aggregate_summary(
            bloom_p=probs
        )

        uncertainty_score = uncertainty_summary.bloom_uncertainty

        # 4. Rerank retrieved results
        reranked = rerank_with_uncertainty(
            results,
            classifier_confidence=pred.confidence,
            ordinal_risk=ordinal_risk,
            uncertainty_score=uncertainty_score,
        )

        # 5. Return unified response
        return {
            "query": query,
            "bloom_prediction": {
                "label": pred.label,
                "confidence": pred.confidence,
                "ordinal_risk": ordinal_risk,
            },
            "uncertainty": {
                "spu": uncertainty_summary.spu,
                "bloom_uncertainty": uncertainty_score,
                "top1_confidence": uncertainty_summary.top1_confidence,
            },
            "retrieval": [
                {
                    "text": r.text,
                    "score": r.score,
                    "final_score": r.final_score,
                    "bloom_level": r.bloom_level,
                }
                for r in reranked
            ],
        }
