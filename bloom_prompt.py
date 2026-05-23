# ============================================================
# Teacher-side Bloom moderation (generative)
# Qwen2.5-1.5B-Instruct GGUF via llama.cpp
#
# Bloom *labels* come from the trained LoRA classifier in predict_bloom.py.
# This module only generates Reason + Higher-Level Rewrite text for teachers.
# ============================================================

from __future__ import annotations

from typing import Optional

from multi_slm import resolve_slm_model_path

LABELS = [
    "Remembering",
    "Understanding",
    "Applying",
    "Analyzing",
    "Evaluating",
    "Creating",
]

_LLM = None


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


def build_prompt(question: str, *, predicted_level: Optional[str] = None) -> str:
    level_hint = (
        f"\nThe trained classifier assigned Bloom level: {predicted_level}.\n"
        "Use that level in your answer unless it is clearly wrong.\n"
        if predicted_level
        else ""
    )
    return f"""
<|im_start|>system
You are an expert educational evaluator specialized in Bloom's Taxonomy.
{level_hint}
You MUST output EXACTLY 3 lines and nothing else:

Bloom Level: <one label from the allowed list>
Reason: <1-2 sentence explanation>
Higher-Level Rewrite: <improved academic version>

Allowed labels only:
Remembering
Understanding
Applying
Analyzing
Evaluating
Creating
<|im_end|>

<|im_start|>user
Question:
{question.strip()}
<|im_end|>

<|im_start|>assistant
Bloom Level:
""".strip()


def analyze_bloom(question: str, *, predicted_level: Optional[str] = None) -> str:
    """Generate moderation text (level, reason, rewrite) for teacher review."""
    llm = _get_llm()
    prompt = build_prompt(question, predicted_level=predicted_level)
    output = llm(
        prompt,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=120,
        stop=["<|im_end|>", "<|im_start|>"],
    )
    return output["choices"][0]["text"].strip()


def predict_bloom_label(question: str) -> str:
    """Backward-compatible helper: label from trained LoRA, not GGUF voting."""
    from predict_bloom import QwenBloomPredictor

    return QwenBloomPredictor().predict(question)["prediction"]


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


def _canonical_bloom_label(raw: str) -> str:
    from predict_bloom import BLOOM_LABELS

    token = raw.strip().split("\n")[0].strip().rstrip(".")
    if ":" in token:
        token = token.split(":", 1)[-1].strip()
    short = _LONG_TO_SHORT.get(token, token)
    if short not in BLOOM_LABELS:
        for label in BLOOM_LABELS:
            if label.lower() in token.lower():
                return label
        return "Remember"
    return short


def zero_shot_bloom_label(question: str) -> str:
    """Bloom label from base Qwen GGUF (no LoRA fine-tuning)."""
    llm = _get_llm()
    prompt = build_prompt(question, predicted_level=None)
    output = llm(
        prompt,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=32,
        stop=["<|im_end|>", "<|im_start|>", "\n"],
    )
    return _canonical_bloom_label(output["choices"][0]["text"])


if __name__ == "__main__":
    print("Teacher Bloom moderation (GGUF). Labels: use predict_bloom.py / train_qwen_bloom.py")
    while True:
        q = input("\nEnter academic question (or 'exit'): ")
        if q.lower() == "exit":
            break
        from predict_bloom import QwenBloomPredictor

        pred = QwenBloomPredictor().predict(q)
        raw = analyze_bloom(q, predicted_level=pred["prediction"])
        print("\nLoRA label:", pred["prediction"], f"(conf={pred['confidence']})")
        print("\nModeration output:\n", raw)
