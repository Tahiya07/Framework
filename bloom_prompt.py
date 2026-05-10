# ============================================================
# EDUGUARD-RAG BLOOM MODERATION MODULE
# Qwen2.5-1.5B-Q4_K_M GGUF
# CPU OFFLINE LIGHTWEIGHT BLOOM ANALYZER
# ============================================================

from llama_cpp import Llama

# ============================================================
# LOAD GGUF MODEL
# ============================================================

MODEL_PATH = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

llm = Llama(
    model_path=MODEL_PATH,

    # CPU SETTINGS
    n_ctx=4096,
    n_threads=8,

    # CPU ONLY
    n_gpu_layers=0,

    # MEMORY OPTIMIZATION
    use_mmap=True,
    use_mlock=False,

    verbose=False
)

# ============================================================
# PROMPT BUILDER (STRICT STRUCTURED OUTPUT)
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
Define photosynthesis.

Answer:
Bloom Level: Remembering
Reason: The question asks for factual recall.
Higher-Level Rewrite: Explain how photosynthesis supports plant growth.

Question:
Explain how linked lists work.

Answer:
Bloom Level: Understanding
Reason: The question requires conceptual explanation.
Higher-Level Rewrite: Analyze the advantages of linked lists over arrays.

Question:
Solve the equation 2x + 5 = 15.

Answer:
Bloom Level: Applying
Reason: The student must apply a learned procedure.
Higher-Level Rewrite: Analyze multiple methods to solve linear equations.

Question:
Analyze why neural networks overfit on small datasets.

Answer:
Bloom Level: Analyzing
Reason: The question examines relationships and causes.
Higher-Level Rewrite: Evaluate techniques for reducing overfitting in neural networks.

Question:
Critique the security architecture of this system.

Answer:
Bloom Level: Evaluating
Reason: The task requires judgment and assessment.
Higher-Level Rewrite: Design a more secure system architecture with improved threat mitigation.

Question:
Design a hospital management system database schema.

Answer:
Bloom Level: Creating
Reason: The task requires designing a new structured system.
Higher-Level Rewrite: Design and justify a scalable hospital management system architecture.

Now classify this question.

Question:
{question}

<|im_end|>

<|im_start|>assistant
Bloom Level:
""".strip()


# ============================================================
# BLOOM ANALYZER
# ============================================================

def analyze_bloom(question):

    prompt = build_prompt(question)

    output = llm(
        prompt,

        # STABILITY SETTINGS
        temperature=0.1,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,

        # FIXED OUTPUT LENGTH (enough for 3-line structure)
        max_tokens=120,

        # CRITICAL STOP TOKENS (PREVENT CONTINUATION DRIFT)
        stop=[
            "<|im_end|>",
        
        ]
    )

    result = output["choices"][0]["text"].strip()

    return result


# ============================================================
# INTERACTIVE LOOP
# ============================================================

print("\n===================================================")
print(" EduGuard-RAG Bloom Moderation Module Ready ")
print("===================================================")

while True:

    question = input("\nEnter academic question (or 'exit'): ")

    if question.lower() == "exit":
        break

    result = analyze_bloom(question)

    print("\n----------------------------------------")
    print(result)
    print("----------------------------------------")