"""Federated learning: federatedly trained teacher Bloom LoRA (FedAvg/FedProx).

Privacy-risk FL lives under privacy/ and is an optional deployment case study.
"""

from federated.config import FederatedLoraConfig, STUDENT_ROLE, TEACHER_ROLE

__all__ = [
    "FederatedLoraConfig",
    "TEACHER_ROLE",
    "STUDENT_ROLE",
]
