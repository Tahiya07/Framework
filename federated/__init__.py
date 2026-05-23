"""Federated learning: teacher Bloom LoRA (FedAvg) + privacy guard (see privacy/)."""

from federated.config import FederatedLoraConfig, TEACHER_ROLE, STUDENT_ROLE

__all__ = [
    "FederatedLoraConfig",
    "TEACHER_ROLE",
    "STUDENT_ROLE",
]
