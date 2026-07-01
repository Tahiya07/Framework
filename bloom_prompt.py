# ============================================================
# Teacher-side Bloom moderation (generative)
# Qwen2.5-1.5B-Instruct GGUF via llama.cpp
#
# Bloom *labels* come from the trained LoRA classifier (predict_bloom.py).
# GGUF generates only the higher-level rewrite; rationale is LoRA-aligned.
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from multi_slm import resolve_slm_model_path
from predict_bloom import BLOOM_LABELS, build_prompt as build_classifier_prompt

IM_START = "<|im_start|>"
IM_END = "<|" + "im_end" + "|>"

BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]

_LEVEL_GUIDANCE: dict[str, dict[str, str]] = {
    "Remember": {
        "depth": "recall of facts, terms, or procedures without explaining relationships",
        "rewrite": "Require explanation, comparison, or use of the recalled knowledge",
    },
    "Understand": {
        "depth": "comprehension, interpretation, or summarization of ideas",
        "rewrite": "Require application to a scenario, prediction, or worked example",
    },
    "Apply": {
        "depth": "using knowledge or procedures in a concrete situation",
        "rewrite": "Require analysis of trade-offs, breakdown of components, or diagnosis",
    },
    "Analyze": {
        "depth": "breaking a problem into parts and examining relationships or causes",
        "rewrite": "Require evaluation with criteria, justification, or critique",
    },
    "Evaluate": {
        "depth": "judging quality, validity, or trade-offs using explicit criteria",
        "rewrite": "Require designing, proposing, or synthesizing a novel solution",
    },
    "Create": {
        "depth": "designing or producing a new artifact, plan, or argument",
        "rewrite": "Extend the task with constraints, audience, or integration across topics",
    },
}

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


def next_bloom_level(level: str) -> str:
    return _next_bloom_level(level)


def build_classifier_aligned_reason(
    lora_level: str,
    *,
    confidence: float,
    probabilities: dict[str, float] | None = None,
) -> str:
    """Deterministic rationale aligned with train_qwen_bloom / predict_bloom.py."""
    level = _canonical_bloom_label(lora_level)
    guide = _LEVEL_GUIDANCE.get(level, _LEVEL_GUIDANCE["Understand"])
    runner_up = ""
    if probabilities:
        ordered = sorted(
            ((label, float(probabilities.get(label, 0.0))) for label in BLOOM_LABELS),
            key=lambda item: item[1],
            reverse=True,
        )
        if len(ordered) >= 2 and ordered[1][1] >= 0.12:
            runner_up = (
                f" The next most likely level was {ordered[1][0]} "
                f"({ordered[1][1]:.0%}), but depth of reasoning favors {level}."
            )
    return (
        f"The LoRA classifier ({confidence:.0%} confidence) placed this item at **{level}** "
        f"because it primarily requires {guide['depth']}. "
        f"Focus on reasoning depth, not surface verbs.{runner_up}"
    )


def build_rewrite_prompt(
    question: str,
    *,
    lora_level: str,
    target_level: str,
) -> str:
    """GGUF prompt for higher-order rewrite only (label is fixed by LoRA)."""
    level = _canonical_bloom_label(lora_level)
    guide = _LEVEL_GUIDANCE.get(level, _LEVEL_GUIDANCE["Understand"])
    classifier_block = build_classifier_prompt(question)
    return f"""{IM_START}system
You are an expert exam-question editor using Bloom's Taxonomy.
The Bloom level is already decided by a trained classifier: {level}.
Do NOT re-classify. Do NOT change the topic.

Task: rewrite the question so it requires **{target_level}**-level thinking
(one stage higher than {level}). {guide['rewrite']}.
Focus on reasoning depth, not verb swapping (avoid only changing "define" to "explain").

Output ONLY the rewritten question as a single exam-style sentence or short paragraph.
No labels, no preamble, no bullet list.
{IM_END}
{IM_START}user
Classifier context (for alignment only):
{classifier_block}

Rewrite this question for {target_level}-level cognition:
{question.strip()}
{IM_END}
{IM_START}assistant
""".strip()


def _clean_rewrite(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"(?im)^(bloom level|reason|higher[- ]level rewrite|rewrite)\s*:\s*", "", cleaned)
    cleaned = cleaned.strip().strip('"').strip("'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


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


def _generate_rewrite(prompt: str) -> tuple[str, str, float]:
    try:
        from qwen_gguf_cli import QwenGgufCliGenerator

        gen = QwenGgufCliGenerator.for_task(
            "bloom_moderation",
            max_tokens=180,
            ctx_size=2048,
            threads=4,
        )
        out = gen.generate_prompt(prompt)
        return _clean_rewrite(out.answer), out.backend, float(out.elapsed_s)
    except Exception:
        llm = _get_llm()
        output = llm(
            prompt,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            max_tokens=180,
            stop=[IM_END, IM_START],
        )
        text = output["choices"][0]["text"].strip()
        return _clean_rewrite(text), "llama-cpp-python", 0.0


def moderate_bloom_question(
    question: str,
    *,
    lora_level: str,
    lora_confidence: float = 0.0,
    probabilities: dict[str, float] | None = None,
) -> BloomModerationResult:
    """LoRA label + aligned rationale + GGUF higher-order rewrite."""
    short_level = _canonical_bloom_label(lora_level)
    target_level = _next_bloom_level(short_level)
    reason = build_classifier_aligned_reason(
        short_level,
        confidence=lora_confidence,
        probabilities=probabilities,
    )
    rewrite = ""
    raw = ""
    backend = ""
    latency_s: Optional[float] = None
    error = ""

    try:
        prompt = build_rewrite_prompt(
            question,
            lora_level=short_level,
            target_level=target_level,
        )
        rewrite, backend, latency_s = _generate_rewrite(prompt)
        raw = rewrite
        if not rewrite or len(rewrite.split()) < 6:
            raise RuntimeError("rewrite_too_short")
    except Exception as exc:
        error = str(exc)
        guide = _LEVEL_GUIDANCE.get(short_level, _LEVEL_GUIDANCE["Understand"])
        rewrite = (
            f"[Auto-rewrite unavailable] Elevate to {target_level}: {guide['rewrite']}. "
            f"Original: {question.strip()}"
        )

    return BloomModerationResult(
        question=question,
        lora_level=short_level,
        lora_confidence=float(lora_confidence),
        target_higher_level=target_level,
        bloom_level=short_level,
        reason=reason,
        higher_level_rewrite=rewrite,
        raw=raw,
        backend=backend,
        latency_s=latency_s,
        error=error,
    )


def analyze_bloom(question: str, *, predicted_level: Optional[str] = None) -> str:
    level = _canonical_bloom_label(predicted_level or "Understand")
    result = moderate_bloom_question(question, lora_level=level)
    if result.error and result.higher_level_rewrite.startswith("[Auto-rewrite"):
        raise RuntimeError(result.error)
    return (
        f"Bloom Level: {result.bloom_level}\n"
        f"Reason: {result.reason}\n"
        f"Higher-Level Rewrite: {result.higher_level_rewrite}"
    )


def predict_bloom_label(question: str) -> str:
    from predict_bloom import QwenBloomPredictor

    return QwenBloomPredictor().predict(question)["prediction"]


def build_prompt(question: str, *, predicted_level: Optional[str] = None) -> str:
    """Backward-compat alias — prefer build_rewrite_prompt for moderation."""
    level = _canonical_bloom_label(predicted_level or "Understand")
    return build_rewrite_prompt(
        question,
        lora_level=level,
        target_level=_next_bloom_level(level),
    )


if __name__ == "__main__":
    print("Teacher Bloom moderation — label: predict_bloom.py; rewrite: GGUF")
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
            probabilities=pred.get("probabilities"),
        )
        print("\nLoRA:", pred["prediction"], f"(conf={pred['confidence']})")
        print("Reason:", mod.reason)
        print("Rewrite:", mod.higher_level_rewrite)
        if mod.error:
            print("Rewrite warning:", mod.error)
