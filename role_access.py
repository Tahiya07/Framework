#!/usr/bin/env python
"""Role-based access control aligned with CSE-400 architecture (student vs teacher)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

from ingestion import DocumentChunk


class Role(str, Enum):
    STUDENT = "student"
    TEACHER = "teacher"


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    reason: str


def normalize_role(requester_role: str) -> Role:
    key = (requester_role or "").strip().lower()
    if key in {"teacher", "moderator", "admin", "teacher / moderator"}:
        return Role.TEACHER
    return Role.STUDENT


def allowed_tasks(role: Role) -> List[str]:
    if role == Role.TEACHER:
        return [
            "Question Answering",
            "Summarization",
            "Exam Question Classification",
        ]
    return ["Question Answering", "Summarization"]


def allowed_upload_targets(role: Role) -> List[str]:
    if role == Role.TEACHER:
        return ["Public Learning Corpus", "Protected Exam Corpus"]
    return ["Public Learning Corpus"]


def resolve_search_scope(role: Role, upload_target: str) -> str:
    """Students always use public retrieval; teachers may use protected corpus when selected."""
    if role == Role.STUDENT:
        return "public"
    if upload_target.strip().lower().startswith("protected"):
        return "protected"
    return "public"


def check_task(role: Role, task_label: str) -> AccessDecision:
    if role == Role.STUDENT and task_label == "Exam Question Classification":
        return AccessDecision(
            False,
            "Bloom moderation (LoRA + rewrite) is restricted to the teacher role.",
        )
    if task_label not in allowed_tasks(role):
        return AccessDecision(False, f"Task {task_label!r} is not enabled for {role.value}.")
    return AccessDecision(True, "ok")


def check_upload_target(role: Role, upload_target: str) -> AccessDecision:
    if upload_target not in allowed_upload_targets(role):
        return AccessDecision(
            False,
            "Students may only ingest public learning materials; protected exam corpora are teacher-only.",
        )
    return AccessDecision(True, "ok")


def check_retriever_binding(role: Role, search_scope: str, using_protected_retriever: bool) -> AccessDecision:
    if role == Role.STUDENT and (search_scope == "protected" or using_protected_retriever):
        return AccessDecision(
            False,
            "Students cannot query the protected FAISS index or protected exam artifacts.",
        )
    return AccessDecision(True, "ok")


def student_visible_chunks(
    public_chunks: Sequence[DocumentChunk],
    protected_chunks: Sequence[DocumentChunk],
) -> List[DocumentChunk]:
    """Students never receive protected chunk objects for retrieval or screening unions."""
    return list(public_chunks)


def teacher_visible_chunks(
    public_chunks: Sequence[DocumentChunk],
    protected_chunks: Sequence[DocumentChunk],
    search_scope: str,
) -> List[DocumentChunk]:
    if search_scope == "protected":
        return list(protected_chunks)
    return list(public_chunks)
