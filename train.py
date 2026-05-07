from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from langdetect import DetectorFactory, detect
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from qwen_gguf_cli import DEFAULT_QWEN_GGUF, find_llama_cli


DetectorFactory.seed = 42
ROOT = Path(__file__).resolve().parent

VALID_LABELS = [
    "Remembering",
    "Understanding",
    "Applying",
    "Analyzing",
    "Creating",
    "Irrelevant",
]

LABEL_ALIASES: Dict[str, str] = {
    "remember": "Remembering",
    "remembering": "Remembering",
    "knowledge": "Remembering",
    "understand": "Understanding",
    "understanding": "Understanding",
    "comprehension": "Understanding",
    "apply": "Applying",
    "applying": "Applying",
    "application": "Applying",
    "analyze": "Analyzing",
    "analysing": "Analyzing",
    "analyzing": "Analyzing",
    "analysis": "Analyzing",
    # The training set intentionally folds Evaluating into Analyzing, so align
    # Qwen outputs to the same target space instead of scoring them as wrong.
    "evaluate": "Analyzing",
    "evaluating": "Analyzing",
    "evaluation": "Analyzing",
    "create": "Creating",
    "creating": "Creating",
    "synthesis": "Creating",
    "irrelevant": "Irrelevant",
    "noise": "Irrelevant",
    "off-topic": "Irrelevant",
    "off topic": "Irrelevant",
}


def clean_text(text: object) -> str:
    text = str(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    try:
        return len(text) > 8 and detect(text) == "en"
    except Exception:
        return False


def load_and_standardize() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    print("Loading and cleaning datasets...")

    f1 = pd.read_csv(ROOT / "data" / "Data_Structure.csv")
    f2 = pd.read_csv(ROOT / "data" / "Introduction_to_Computers_and_Research.csv")
    m_map = {
        5: "Remembering",
        10: "Understanding",
        15: "Applying",
        20: "Analyzing",
        30: "Creating",
    }
    df_acad = pd.concat([f1, f2], ignore_index=True)
    df_acad["label"] = df_acad["Score"].map(m_map)
    df_acad = df_acad[["Questions", "label"]].rename(columns={"Questions": "text"})

    df_noise = pd.read_csv(ROOT / "data" / "Irrelevant_Questions.csv")
    df_noise["label"] = "Irrelevant"
    df_noise = df_noise[["Questions", "label"]].rename(columns={"Questions": "text"})

    mooc_list: List[Dict[str, str]] = []
    mooc_map = {
        1: "Remembering",
        2: "Understanding",
        3: "Applying",
        4: "Analyzing",
        5: "Analyzing",
        6: "Creating",
    }
    with (ROOT / "data" / "problem.json").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                detail = ast.literal_eval(item["detail"])
                content = detail.get("content", "")
                cog_dim = item.get("cognitive_dimension")
                if content and is_english(content) and cog_dim in mooc_map:
                    mooc_list.append({"text": content, "label": mooc_map[cog_dim]})
            except Exception:
                continue
    df_mooc = pd.DataFrame(mooc_list)

    df_full = pd.concat([df_acad, df_noise, df_mooc], ignore_index=True).dropna()
    df_full["text"] = df_full["text"].apply(clean_text)
    df_full["label"] = df_full["label"].replace("Evaluating", "Analyzing")
    df_full = df_full[df_full["label"].isin(VALID_LABELS)]
    df_full = df_full.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    print(f"Total processed samples: {len(df_full)}")
    return train_test_split(
        df_full["text"],
        df_full["label"],
        test_size=0.2,
        random_state=42,
        stratify=df_full["label"],
    )


@dataclass
class QwenPrediction:
    label: str
    raw: str
    elapsed_s: float


class QwenBloomCliClassifier:
    """Deterministic Qwen GGUF generative classifier via standalone llama.cpp."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_QWEN_GGUF,
        *,
        llama_cli_path: str | Path | None = None,
        ctx_size: int = 1024,
        max_tokens: int = 24,
        threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Qwen GGUF not found: {self.model_path}")
        self.llama_cli_path = Path(llama_cli_path) if llama_cli_path else find_llama_cli()
        self.ctx_size = int(ctx_size)
        self.max_tokens = int(max_tokens)
        self.threads = int(threads)

    @staticmethod
    def build_prompt(text: str) -> str:
        labels = ", ".join(VALID_LABELS)
        return f"""<|im_start|>system
You are a strict Bloom Taxonomy classifier for university questions.
Return exactly one label and nothing else.

Allowed labels: {labels}

Definitions:
- Remembering: recall, define, list, identify, state facts.
- Understanding: explain, summarize, describe meaning.
- Applying: use a concept, calculate, solve a direct problem.
- Analyzing: compare, differentiate, infer, debug, explain relationships, evaluate or justify.
- Creating: design, compose, propose, formulate, invent, synthesize.
- Irrelevant: greetings, spam, personal chat, or not about academic/course content.

Important: Evaluating/Evaluation must be labeled as Analyzing in this dataset.

Examples:
Question: define a stack data structure
Label: Remembering
Question: explain how binary search works
Label: Understanding
Question: calculate the output of this loop
Label: Applying
Question: compare TCP and UDP
Label: Analyzing
Question: design a database schema for a library
Label: Creating
Question: hello how are you
Label: Irrelevant
<|im_end|>
<|im_start|>user
Question: {text}
Label:
<|im_end|>
<|im_start|>assistant
"""

    @staticmethod
    def extract_label(output: str) -> str:
        text = output
        text = text.split("[ Prompt:", 1)[0]
        if "<|im_start|>assistant" in text:
            text = text.rsplit("<|im_start|>assistant", 1)[-1]
        elif "Label:" in text:
            text = text.rsplit("Label:", 1)[-1]
        text = re.sub(r"<\|im_(?:start|end)\|>", " ", text)
        text = re.sub(r"(?i)\b(label|answer|classification)\s*[:=-]", " ", text)
        lowered = text.lower()

        # Prefer the first explicit allowed/alias label found in generated text.
        for alias, canonical in LABEL_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                return canonical

        # Last resort: use Bloom verb cues if Qwen returned a sentence.
        cue_order = [
            ("Creating", r"\b(design|compose|create|develop|formulate|propose|invent|synthesize)\b"),
            ("Analyzing", r"\b(compare|differentiate|analyze|analyse|evaluate|justify|debug|infer|relationship)\b"),
            ("Applying", r"\b(apply|calculate|solve|implement|use|compute|execute)\b"),
            ("Understanding", r"\b(explain|summarize|describe|interpret|classify)\b"),
            ("Remembering", r"\b(define|list|state|identify|recall|name|what is)\b"),
        ]
        for label, pattern in cue_order:
            if re.search(pattern, lowered):
                return label
        return "Irrelevant"

    def classify(self, text: str) -> QwenPrediction:
        prompt = self.build_prompt(text)
        cmd = [
            str(self.llama_cli_path),
            "-m",
            str(self.model_path),
            "-p",
            prompt,
            "-n",
            str(self.max_tokens),
            "--temp",
            "0",
            "--no-display-prompt",
            "--single-turn",
            "--simple-io",
            "-dev",
            "none",
            "-ngl",
            "0",
            "-c",
            str(self.ctx_size),
            "-t",
            str(self.threads),
            "--no-repack",
        ]
        start = time.perf_counter()
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        elapsed = time.perf_counter() - start
        raw = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode not in {0, 137}:
            raise RuntimeError(f"llama-cli failed with code {proc.returncode}:\n{raw}")
        return QwenPrediction(label=self.extract_label(raw), raw=raw, elapsed_s=elapsed)


def run_svm(X_train: pd.Series, X_test: pd.Series, y_train: pd.Series) -> tuple[np.ndarray, SVC]:
    print("\nExtracting semantic embeddings (MiniLM-L6)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    X_train_vec = embedder.encode(X_train.tolist(), show_progress_bar=True)
    X_test_vec = embedder.encode(X_test.tolist(), show_progress_bar=True)

    print("\nTraining semantic SVM baseline...")
    svm_model = SVC(kernel="rbf", C=10, class_weight="balanced")
    svm_model.fit(X_train_vec, y_train)
    return svm_model.predict(X_test_vec), svm_model


def run_qwen_eval(
    X_test: pd.Series,
    *,
    subset_size: int,
    model_path: str | Path,
    threads: int,
    ctx_size: int,
) -> tuple[List[str], List[str], float]:
    print("\nInitializing Qwen GGUF classifier...")
    qwen = QwenBloomCliClassifier(model_path=model_path, threads=threads, ctx_size=ctx_size)
    test_subset = X_test.iloc[:subset_size].reset_index(drop=True)

    print(f"Executing Qwen accuracy audit on {len(test_subset)} samples...")
    preds: List[str] = []
    raw_outputs: List[str] = []
    latencies: List[float] = []
    for i, question in enumerate(test_subset):
        pred = qwen.classify(question)
        preds.append(pred.label)
        raw_outputs.append(pred.raw)
        latencies.append(pred.elapsed_s)
        if i < 5:
            print(f"Sample {i} | Q: {question[:55]}... | Pred: {pred.label}")

    avg_latency = sum(latencies) / max(1, len(latencies))
    return preds, raw_outputs, avg_latency


def save_research_metrics(
    questions: Sequence[str],
    y_true: Sequence[str],
    y_pred_svm: Sequence[str],
    y_pred_qwen: Sequence[str],
    qwen_raw: Sequence[str],
    filename: str = "research_audit.csv",
) -> None:
    with open(ROOT / filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Query_ID", "Question", "Actual_Label", "SVM_Prediction", "Qwen_Prediction", "Qwen_Raw"])
        for i in range(len(y_pred_qwen)):
            writer.writerow([i, questions[i], y_true[i], y_pred_svm[i], y_pred_qwen[i], qwen_raw[i]])
    print(f"Research metrics exported to {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVM baseline and evaluate Qwen GGUF generative Bloom classifier.")
    parser.add_argument("--qwen-subset", type=int, default=50, help="Number of test samples to run through Qwen.")
    parser.add_argument("--skip-qwen", action="store_true", help="Run only the SVM baseline.")
    parser.add_argument("--qwen-model", type=str, default=str(DEFAULT_QWEN_GGUF), help="Path to Qwen GGUF model.")
    parser.add_argument("--qwen-threads", type=int, default=4, help="llama.cpp CPU threads for Qwen.")
    parser.add_argument("--qwen-ctx", type=int, default=1024, help="llama.cpp context size for Qwen.")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_and_standardize()
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    y_pred_svm, _svm_model = run_svm(X_train, X_test, y_train)
    print("\n" + "=" * 45)
    print("BASELINE: SEMANTIC SVM")
    print("=" * 45)
    print(classification_report(y_test, y_pred_svm, labels=VALID_LABELS, zero_division=0))

    if args.skip_qwen:
        print("\nQwen evaluation skipped.")
        return

    subset_size = max(1, min(args.qwen_subset, len(X_test)))
    qwen_preds, qwen_raw, qwen_latency = run_qwen_eval(
        X_test,
        subset_size=subset_size,
        model_path=args.qwen_model,
        threads=args.qwen_threads,
        ctx_size=args.qwen_ctx,
    )
    y_true_subset = y_test.iloc[:subset_size].tolist()
    y_svm_subset = list(y_pred_svm[:subset_size])
    questions_subset = X_test.iloc[:subset_size].tolist()

    print("\n" + "=" * 45)
    print(f"TARGET: QWEN GGUF GENERATIVE CLASSIFIER (Latency: {qwen_latency:.4f}s/query)")
    print("=" * 45)
    print(classification_report(y_true_subset, qwen_preds, labels=VALID_LABELS, zero_division=0))

    save_research_metrics(questions_subset, y_true_subset, y_svm_subset, qwen_preds, qwen_raw)


if __name__ == "__main__":
    main()
