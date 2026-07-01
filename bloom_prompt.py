# ============================================================
# Teacher-side Bloom moderation (generative)
# Qwen2.5-1.5B-Instruct GGUF via llama.cpp
#
# Bloom *labels* come from the trained LoRA classifier in predict_bloom.py.
# This module generates Reason + Higher-Level Rewrite for teachers.
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from multi_slm import resolve_slm_model_path

IM_START = "<|im_start|>"
IM_END = "<|" + "im_end" + "|>"

BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

LONG_LABELS = [
    "Remembering",
    "Understanding",
    "Applying",
    "Analyzing",
    "Evaluating",
    "Creating",
]

_LLM = None

_LONG_TO_SHORT = {
    "Remembering": "Remember",
    "Understanding": "Understand",
    "Applying": "Apply",
    "Analyzing": "Analyze",
    "Evaluating": "Evaluate",
    "Creating": "Create",
    "Knowledge": "Remember",
    "Remember": "Remember",
    "Recall": "Remember",
    "Comprehension": "Understand",
    "Understand": "Understand",
    "Application": "Apply",
    "Apply": "Apply",
    "Analysis": "Analyze",
    "Analyze": "Analyze",
    "Evaluation": "Evaluate",
    "Evaluate": "Evaluate",
    "Synthesis": "Create",
    "Create": "Create",
}


@dataclass
class BloomModerationResult:
    question: str
    lora_level: str
    lora_confidence: float
    target_higher_level: str
    bloom_level: str
    reason: str
    higher_level_rewrite: str
    raw: str = ""
    backend: str = ""
    latency_s: Optional[float] = None
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "lora_level": self.lora_level,
            "lora_confidence": self.lora_confidence,
            "target_higher_level": self.target_higher_level,
            "bloom_level": self.bloom_level,
            "reason": self.reason,
            "higher_level_rewrite": self.higher_level_rewrite,
            "raw": self.raw,
            "backend": self.backend,
            "latency_s": self.latency_s,
            "error": self.error,
        }


def _canonical_bloom_label(raw: str, *, fallback: str = "Understand") -> str:
    from predict_bloom import BLOOM_LABELS

    token = (raw or "").strip().split("\n")[0].strip().rstrip(".")
    if ":" in token:
        token = token.split(":", 1)[-1].strip()
    short = _LONG_TO_SHORT.get(token, token)
    if short in BLOOM_LABELS:
        return short
    for label in BLOOM_LABELS:
        if label.lower() in token.lower():
            return label
    return fallback


def _next_bloom_level(level: str) -> str:
    short = _canonical_bloom_label(level, fallback="Understand")
    if short not in BLOOM_ORDER:
        return "Analyze"
    idx = BLOOM_ORDER.index(short)
    return BLOOM_ORDER[min(idx + 1, len(BLOOM_ORDER) - 1)]


def _parse_moderation_text(raw: str, *, fallback_level: str) -> tuple[str, str, str]:
    text = (raw or "").strip()
    bloom_level = _canonical_bloom_label(fallback_level, fallback=fallback_level)
    reason = ""
    rewrite = ""

    patterns = {
        "bloom": re.compile(r"(?im)^\s*bloom\s*level\s*:\s*(.+)$"),
        "reason": re.compile(r"(?im)^\s*reason\s*:\s*(.+)$"),
        "rewrite": re.compile(
            r"(?is)^\s*(?:higher[- ]level\s*rewrite|higher[- ]order\s*rewrite|rewrite)\s*:\s*(.+)$"
        ),
    }
    bloom_match = patterns["bloom"].search(text)
    if bloom_match:
        bloom_level = _canonical_bloom_label(bloom_match.group(1), fallback=bloom_level)

    reason_match = patterns["reason"].search(text)
    if reason_match:
        reason = reason_match.group(1).strip()

    rewrite_match = patterns["rewrite"].search(text)
    if rewrite_match:
        rewrite = rewrite_match.group(1).strip()
        rewrite = re.split(r"\n\s*(?:bloom|reason)\s*:", rewrite, maxsplit=1)[0].strip()

    if not reason and not rewrite:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 1 and not reason:
            reason = lines[0]
        if len(lines) >= 2 and not rewrite:
            rewrite = lines[-1]

    return bloom_level, reason, rewrite


def build_prompt(question: str, *, predicted_level: Optional[str] = None) -> str:
    lora_level = _canonical_bloom_label(predicted_level or "Understand")
    target_level = _next_bloom_level(lora_level)
    return f"""{IM_START}system
You are an expert educational evaluator specialized in Bloom's Taxonomy.
The trained LoRA classifier assigned this question to: {lora_level}.
Your job is to (1) briefly justify that level and (2) rewrite the question so it
requires {target_level}-level thinking (one Bloom stage higher when possible).

Output EXACTLY these three labeled lines and nothing else:

Bloom Level: <{lora_level} or corrected label from Remember/Understand/Apply/Analyze/Evaluate/Create>
Reason: <1-2 sentences on cognitive demand; focus on reasoning depth not verbs>
Higher-Level Rewrite: <rewritten exam question requiring {target_level}-level cognition; keep the same topic>
{IM_END}
{IM_START}user
Question:
{question.strip()}
{IM_END}
{IM_START}assistant
Bloom Level:""".strip()


def _get_llm():
    global _LLM
    if _LLM is None:
        from llama_cpp import Llama

        model_path = resolve_slm_model_path("bloom_moderation")
        _LLM = Llama(
            model_path=str(model_path),
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
    return _LLM


def _generate_raw(prompt: str) -> tuple[str, str, float]:
    try:
        from qwen_gguf_cli import QwenGgufCliGenerator

        gen = QwenGgufCliGenerator.for_task(
            "bloom_moderation",
            max_tokens=220,
            ctx_size=2048,
            threads=4,
        )
        out = gen.generate_prompt(prompt)
        return out.answer, out.backend, float(out.elapsed_s)
    except Exception:
        llm = _get_llm()
        output = llm(
            prompt,
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            max_tokens=220,
            stop=[IM_END, IM_START],
        )
        text = output["choices"][0]["text"].strip()
        return text, "llama-cpp-python", 0.0


def moderate_bloom_question(
    question: str,
    *,
    lora_level: str,
    lora_confidence: float = 0.0,
) -> BloomModerationResult:
    """LoRA label + GGUF reason + higher-order rewrite (teacher architecture)."""
    short_level = _canonical_bloom_label(lora_level)
    target_level = _next_bloom_level(short_level)
    prompt = build_prompt(question, predicted_level=short_level)
    try:
        raw, backend, latency_s = _generate_raw(prompt)
        full_raw = f"Bloom Level: {raw}" if not raw.lower().startswith("bloom") else raw
        bloom_level, reason, rewrite = _parse_moderation_text(full_raw, fallback_level=short_level)
        if not rewrite:
            rewrite = (
                f"[Rewrite unavailable — elevate the question to require {target_level}-level thinking "
                f"while keeping the same topic.]"
            )
        return BloomModerationResult(
            question=question,
            lora_level=short_level,
            lora_confidence=float(lora_confidence),
            target_higher_level=target_level,
            bloom_level=bloom_level,
            reason=reason or "Classifier-assigned level based on required cognitive depth.",
            higher_level_rewrite=rewrite,
            raw=full_raw,
            backend=backend,
            latency_s=latency_s,
        )
    except Exception as exc:
        return BloomModerationResult(
            question=question,
            lora_level=short_level,
            lora_confidence=float(lora_confidence),
            target_higher_level=target_level,
            bloom_level=short_level,
            reason="",
            higher_level_rewrite="",
            error=str(exc),
        )


def analyze_bloom(question: str, *, predicted_level: Optional[str] = None) -> str:
    """Backward-compatible text output for teacher moderation."""
    level = _canonical_bloom_label(predicted_level or "Understand")
    result = moderate_bloom_question(question, lora_level=level)
    if result.error:
        raise RuntimeError(result.error)
    return (
        f"Bloom Level: {result.bloom_level}\n"
        f"Reason: {result.reason}\n"
        f"Higher-Level Rewrite: {result.higher_level_rewrite}"
    )


def predict_bloom_label(question: str) -> str:
    from predict_bloom import QwenBloomPredictor

    return QwenBloomPredictor().predict(question)["prediction"]


def zero_shot_bloom_label(question: str) -> str:
    """Bloom label from base Qwen GGUF (no LoRA fine-tuning)."""
    prompt = build_prompt(question, predicted_level=None)
    raw, _, _ = _generate_raw(prompt)
    return _canonical_bloom_label(raw)


if __name__ == "__main__":
    print("Teacher Bloom moderation (GGUF). Labels: predict_bloom.py / train_qwen_bloom.py")
    while True:
        q = input("\nEnter academic question (or 'exit'): ")
        if q.lower() == "exit":
            break
        from predict_bloom import QwenBloomPredictor

        pred = QwenBloomPredictor().predict(q)
        mod = moderate_bloom_question(
            q,
            lora_level=pred["prediction"],
            lora_confidence=pred["confidence"],
        )
        print("\nLoRA:", pred["prediction"], f"(conf={pred['confidence']})")
        print("Target higher level:", mod.target_higher_level)
        print("Reason:", mod.reason)
        print("Rewrite:", mod.higher_level_rewrite)
        if mod.error:
            print("Error:", mod.error)
