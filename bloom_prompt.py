# ============================================================
# EDUGUARD-RAG BLOOM MODERATION MODULE
# Qwen2.5-1.5B-Q4_K_M GGUF
# CPU OFFLINE LIGHTWEIGHT BLOOM ANALYZER
# + CALIBRATION LAYER (POST-DECODING STABILITY)
# ============================================================

from llama_cpp import Llama
from collections import Counter
from multi_slm import resolve_slm_model_path

# ============================================================
# LABELS
# ============================================================

LABELS = [
    "Remembering",
    "Understanding",
    "Applying",
    "Analyzing",
    "Evaluating",
    "Creating"
]

# ============================================================
# LOAD GGUF MODEL
# ============================================================

MODEL_PATH = resolve_slm_model_path("bloom_moderation")

print("\nLoading Qwen GGUF Bloom moderation model...\n")

llm = Llama(
    model_path=str(MODEL_PATH),
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=0,
    use_mmap=True,
    use_mlock=False,
    verbose=False
)

# ============================================================
# PROMPT BUILDER (UNCHANGED)
# ============================================================

def build_prompt(question):

    return f"""
<|im_start|>system
You are an expert educational evaluator specialized in Bloom's Taxonomy.

You MUST classify academic questions into EXACTLY ONE Bloom level.

Allowed labels only:
Remembering
Understanding
Applying
Analyzing
Evaluating
Creating

CRITICAL RULES:
- Output MUST contain EXACTLY 3 lines.
- Do NOT add extra text.
- Do NOT stop early.
- Do NOT repeat the question.

MANDATORY FORMAT:

Bloom Level: <one label>
Reason: <1-2 sentence explanation>
Higher-Level Rewrite: <improved academic version>

If any field is missing, output is INVALID.

<|im_end|>

<|im_start|>user

Question:
{question}

<|im_end|>

<|im_start|>assistant
Bloom Level:
""".strip()

# ============================================================
# RAW MODEL CALL
# ============================================================

def analyze_bloom(question):

    prompt = build_prompt(question)

    output = llm(
        prompt,
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        max_tokens=120,
        stop=["<|im_end|>", "<|im_start|>"]
    )

    return output["choices"][0]["text"].strip()

# ============================================================
# STEP 1: LABEL EXTRACTION (STRICT CLEANING)
# ============================================================

def extract_bloom_label(text):

    text = text.lower()

    for label in LABELS:
        if label.lower() in text:
            return label

    return "Understanding"

# ============================================================
# STEP 2: CALIBRATION LAYER (NEW CORE ADDITION)
# ============================================================

def calibrated_predict(question, k=3, confidence_threshold=0.5):

    """
    1. Run multiple stochastic samples
    2. Extract labels
    3. Majority vote
    4. Confidence estimation
    """

    preds = []

    for _ in range(k):

        raw = analyze_bloom(question)
        label = extract_bloom_label(raw)
        preds.append(label)

    # majority vote
    counts = Counter(preds)
    final_label, freq = counts.most_common(1)[0]

    confidence = freq / k

    # fallback safety (uncertain cases)
    if confidence < confidence_threshold:
        return "Understanding", confidence

    return final_label, confidence

# ============================================================
# PUBLIC API (USE THIS IN EVALUATION)
# ============================================================

def predict_bloom_label(question):

    label, _ = calibrated_predict(question)
    return label

# ============================================================
# INTERACTIVE DEBUG MODE
# ============================================================

if __name__ == "__main__":

    print("\n===================================================")
    print(" EduGuard-RAG Bloom Moderation Module Ready ")
    print(" (Calibration Enabled) ")
    print("===================================================")

    while True:

        question = input("\nEnter academic question (or 'exit'): ")

        if question.lower() == "exit":
            break

        raw = analyze_bloom(question)
        label, conf = calibrated_predict(question)

        print("\nRAW OUTPUT:\n", raw)
        print("\nCALIBRATED LABEL:", label)
        print("CONFIDENCE:", round(conf, 3))
        print("----------------------------------------")
