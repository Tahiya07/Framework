from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
SHARED_QWEN_GGUF = MODEL_DIR / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
LEGACY_QWEN_GGUF = MODEL_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
SHORT_QWEN_GGUF = MODEL_DIR / "qwen.gguf"


@dataclass(frozen=True)
class SLMTaskProfile:
    task_id: str
    env_var: str
    model_filename: str
    system_prompt: str
    user_instruction: str
    ctx_size: int = 512
    max_tokens: int = 48
    threads: int = 2
    temperature: float = 0.0

    @property
    def preferred_path(self) -> Path:
        return MODEL_DIR / self.model_filename


TASK_PROFILES: Dict[str, SLMTaskProfile] = {
    "academic_qa": SLMTaskProfile(
        task_id="academic_qa",
        env_var="ACADEMIC_QA_SLM_PATH",
        model_filename="slm_academic_qa.gguf",
        system_prompt=(
            "You are a compact academic QA specialist. Answer using only the supplied context. "
            "Write a direct answer in your own words. Do not copy long passages verbatim. "
            "If the answer is not in the context, say: I don't know based on the provided context."
        ),
        user_instruction="Answer the question directly in 2-4 sentences using only the context.",
        ctx_size=2048,
        max_tokens=120,
    ),
    "pdf_rag": SLMTaskProfile(
        task_id="pdf_rag",
        env_var="PDF_RAG_SLM_PATH",
        model_filename="slm_pdf_rag.gguf",
        system_prompt=(
            "You are a PDF study-material QA specialist. Use only the supplied PDF-derived context. "
            "Do not add unsupported facts."
        ),
        user_instruction="Answer from the PDF context in one concise sentence.",
        ctx_size=768,
        max_tokens=64,
    ),
    "image_rag": SLMTaskProfile(
        task_id="image_rag",
        env_var="IMAGE_RAG_SLM_PATH",
        model_filename="slm_image_rag.gguf",
        system_prompt=(
            "You are an OCR/image-note QA specialist. Use only the supplied OCR context. "
            "If OCR context is insufficient, say so."
        ),
        user_instruction="Answer from the OCR context with the key entity or phrase.",
        ctx_size=512,
        max_tokens=48,
    ),
    "bloom_moderation": SLMTaskProfile(
        task_id="bloom_moderation",
        env_var="BLOOM_MODERATION_SLM_PATH",
        model_filename="slm_bloom_moderation.gguf",
        system_prompt=(
            "You are an expert educator specializing in Bloom's Taxonomy moderation. "
            "Explain the cognitive level of exam questions and rewrite them to require "
            "exactly one higher Bloom level while preserving the subject topic."
        ),
        user_instruction="",
        ctx_size=2048,
        max_tokens=200,
        temperature=0.1,
    ),
    "teacher_moderation": SLMTaskProfile(
        task_id="teacher_moderation",
        env_var="TEACHER_MODERATION_SLM_PATH",
        model_filename="slm_teacher_moderation.gguf",
        system_prompt=(
            "You are a teacher-only question moderation specialist. Review protected assessment items "
            "for Bloom alignment, ambiguity, difficulty, and leakage risk. Do not quote long spans."
        ),
        user_instruction="Give concise moderation feedback and one safer revision suggestion.",
        ctx_size=1024,
        max_tokens=160,
    ),
    "privacy_response": SLMTaskProfile(
        task_id="privacy_response",
        env_var="PRIVACY_RESPONSE_SLM_PATH",
        model_filename="slm_privacy_response.gguf",
        system_prompt=(
            "You are a privacy-preserving response specialist. Refuse reconstruction of protected exam "
            "content and offer safe high-level study or moderation alternatives."
        ),
        user_instruction="Respond safely without revealing protected wording.",
        ctx_size=512,
        max_tokens=96,
    ),
}


def available_task_ids() -> List[str]:
    return sorted(TASK_PROFILES)


def get_task_profile(task_id: str) -> SLMTaskProfile:
    key = (task_id or "").strip().lower()
    if key not in TASK_PROFILES:
        raise KeyError(f"Unknown SLM task {task_id!r}. Available tasks: {available_task_ids()}")
    return TASK_PROFILES[key]


def resolve_slm_model_path(task_id: str, explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)

    profile = get_task_profile(task_id)
    env_path = os.environ.get(profile.env_var)
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate

    if profile.preferred_path.is_file():
        return profile.preferred_path

    if SHARED_QWEN_GGUF.is_file():
        return SHARED_QWEN_GGUF
    if SHORT_QWEN_GGUF.is_file():
        return SHORT_QWEN_GGUF
    return LEGACY_QWEN_GGUF


def task_registry_report() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for task_id in available_task_ids():
        profile = get_task_profile(task_id)
        resolved = resolve_slm_model_path(task_id)
        rows.append(
            {
                "task_id": task_id,
                "env_var": profile.env_var,
                "preferred_model": str(profile.preferred_path),
                "resolved_model": str(resolved),
                "uses_specialist_model": resolved == profile.preferred_path and resolved.is_file(),
                "model_exists": resolved.is_file(),
                "ctx_size": profile.ctx_size,
                "max_tokens": profile.max_tokens,
                "threads": profile.threads,
            }
        )
    return rows
