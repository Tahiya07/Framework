from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer


ROOT = Path(__file__).resolve().parent

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = ROOT / "qwen-bloom-lora"
RESULTS_DIR = ROOT / "results"
MAX_LENGTH = 512

VALID_LABELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create",
]

LABEL_ALIASES: Dict[str, str] = {
    "remember": "Remember",
    "remembering": "Remember",
    "knowledge": "Remember",
    "understand": "Understand",
    "understanding": "Understand",
    "comprehension": "Understand",
    "apply": "Apply",
    "applying": "Apply",
    "application": "Apply",
    "analyze": "Analyze",
    "analyse": "Analyze",
    "analysing": "Analyze",
    "analyzing": "Analyze",
    "analysis": "Analyze",
    "evaluate": "Evaluate",
    "evaluating": "Evaluate",
    "evaluation": "Evaluate",
    "create": "Create",
    "creating": "Create",
    "synthesis": "Create",
}

BLOOM_SEMANTIC_MAP: Dict[str, str] = {
    "Remember": "retrieve or recognize facts, terms, definitions, labels, or named items",
    "Understand": "explain meaning, summarize, interpret, classify, compare, or describe relationships in plain language",
    "Apply": "use a known method, formula, procedure, or concept to solve a direct task",
    "Analyze": "break material into parts, distinguish causes, infer structure, compare mechanisms, or diagnose relationships",
    "Evaluate": "judge quality, defend a decision, justify a recommendation, critique, assess, or argue with criteria",
    "Create": "design, formulate, compose, plan, propose, construct, or synthesize something new",
}


def clean_question(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def load_figshare_split(name: str) -> pd.DataFrame:
    path = ROOT / "data" / f"figshare_bloom_v1_{name}.csv"
    df = pd.read_csv(path)
    df = df.rename(columns={"bloom_level": "label"})
    df["question"] = df["question"].map(clean_question)
    df["label"] = df["label"].map(lambda x: LABEL_ALIASES.get(str(x).strip().lower(), str(x).strip()))
    df = df[df["question"].str.len() > 0]
    df = df[df["label"].isin(VALID_LABELS)]
    return df[["question", "label"]].drop_duplicates().reset_index(drop=True)


def load_semantic_bloom_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the curated Figshare Bloom splits instead of the noisy irrelevant pool."""
    return (
        load_figshare_split("train"),
        load_figshare_split("val"),
        load_figshare_split("test"),
    )


def _read_json_records(path: Path) -> List[dict]:
    rows: List[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _extract_moocradar_question(record: dict) -> str:
    detail = record.get("detail")
    if isinstance(detail, str):
        try:
            parsed = ast.literal_eval(detail)
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            parts = [clean_question(parsed.get("content", ""))]
            options = parsed.get("option")
            if isinstance(options, dict):
                parts.extend(clean_question(value) for value in options.values())
            return clean_question(" ".join(part for part in parts if part))
    for key in ("question", "question_text", "content", "text"):
        if record.get(key):
            return clean_question(record[key])
    return ""


def _normalise_moocradar_label(value: object) -> str | None:
    if value is None:
        return None
    try:
        idx = int(value)
    except (TypeError, ValueError):
        return LABEL_ALIASES.get(str(value).strip().lower())
    if 1 <= idx <= 6:
        return VALID_LABELS[idx - 1]
    return None


def load_moocradar() -> pd.DataFrame:
    path = ROOT / "data" / "problem.json"
    records = _read_json_records(path)
    rows: List[Dict[str, str]] = []
    for record in records:
        question = _extract_moocradar_question(record)
        label = _normalise_moocradar_label(record.get("cognitive_dimension"))
        if question and label:
            rows.append({"question": question, "label": label})
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["question", "label"])
    conflicts = df.groupby("question")["label"].nunique()
    if len(conflicts):
        df = df[~df["question"].isin(conflicts[conflicts > 1].index)]
    return df.drop_duplicates(subset=["question"]).reset_index(drop=True)


def infer_mendeley_label(question: str) -> str | None:
    lowered = question.lower()
    cue_patterns: Sequence[tuple[str, str]] = (
        ("Create", r"\b(design|create|develop|formulate|propose|compose|construct|plan|synthesize|generate)\b"),
        ("Evaluate", r"\b(evaluate|assess|justify|critique|defend|judge|recommend|argue|appraise|best choice)\b"),
        ("Analyze", r"\b(analyze|analyse|differentiate|distinguish|examine|infer|investigate|relate|why|cause|scenario)\b"),
        ("Apply", r"\b(apply|calculate|solve|use|implement|demonstrate|determine|compute|find)\b"),
        ("Understand", r"\b(explain|describe|summarize|interpret|classify|compare|contrast|outline|discuss|how)\b"),
        ("Remember", r"\b(define|identify|list|name|state|recall|recognize|label|what is|stands for)\b"),
    )
    matches = [label for label, pattern in cue_patterns if re.search(pattern, lowered)]
    return matches[0] if len(matches) == 1 else None


def load_mendeley_weak_labels(max_per_label: int = 500) -> pd.DataFrame:
    frames = []
    for filename in ("Data_Structure.csv", "Introduction_to_Computers_and_Research.csv"):
        raw = pd.read_csv(ROOT / "data" / filename)
        df = raw.rename(columns={"Questions": "question"})[["question"]].copy()
        df["question"] = df["question"].map(clean_question)
        df["label"] = df["question"].map(infer_mendeley_label)
        frames.append(df.dropna(subset=["label"]))
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["question"])
    if max_per_label > 0:
        out = (
            out.groupby("label", group_keys=False)
            .apply(lambda group: group.sample(n=min(len(group), max_per_label), random_state=42))
            .reset_index(drop=True)
        )
    return out[["question", "label"]]


def split_supervised(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"])
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_training_splits(dataset: str, include_mendeley_weak: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if dataset == "figshare":
        train_df, val_df, test_df = load_semantic_bloom_splits()
        source = "data/figshare_bloom_v1_train.csv + val.csv + test.csv"
    elif dataset == "moocradar":
        train_df, val_df, test_df = split_supervised(load_moocradar())
        source = "data/problem.json cognitive_dimension"
    elif dataset == "combined":
        train_df, val_df, test_df = load_semantic_bloom_splits()
        mooc_train, mooc_val, mooc_test = split_supervised(load_moocradar())
        train_df = pd.concat([train_df, mooc_train], ignore_index=True)
        val_df = pd.concat([val_df, mooc_val], ignore_index=True)
        test_df = pd.concat([test_df, mooc_test], ignore_index=True)
        source = "Figshare curated splits + MOOCRadar cognitive_dimension"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if include_mendeley_weak:
        weak = load_mendeley_weak_labels()
        train_df = pd.concat([train_df, weak], ignore_index=True).drop_duplicates(subset=["question", "label"])
        source += " + Mendeley cue-derived weak labels for training only"
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), source


def system_prompt() -> str:
    labels = ", ".join(VALID_LABELS)
    definitions = " ".join(f"{label}: {meaning}." for label, meaning in BLOOM_SEMANTIC_MAP.items())
    return (
        "You classify exam questions by the cognitive operation they require, not by topic words. "
        "Map the question to the closest Bloom semantic meaning and return exactly one label. "
        f"Allowed labels: {labels}. "
        f"Semantic map: {definitions}"
    )


def to_instruction_dataset(df: pd.DataFrame) -> Dataset:
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt()},
                    {"role": "user", "content": f"Question: {row['question']}\nLabel:"},
                    {"role": "assistant", "content": str(row["label"])},
                ]
            }
        )
    return Dataset.from_list(rows)


def format_chat(example: Dict[str, object], tokenizer: AutoTokenizer) -> str:
    return tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def _tok(example: Dict[str, object]) -> Dict[str, object]:
        messages = example["messages"]
        prompt_messages = messages[:-1]
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = format_chat(example, tokenizer)
        prompt_tokens = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        tokens = tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        labels = tokens["input_ids"].copy()
        prompt_len = min(len(prompt_tokens), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        labels = [token if mask else -100 for token, mask in zip(labels, tokens["attention_mask"])]
        tokens["labels"] = labels
        return tokens

    return dataset.map(_tok, remove_columns=["messages"])


def load_base_model(use_4bit: bool) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available() and use_4bit:
        kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.float16,
            }
        )
    else:
        kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)
    model.config.use_cache = False
    if torch.cuda.is_available() and use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def extract_label(text: str) -> str:
    cleaned = text.strip()
    if "<|im_start|>assistant" in cleaned:
        cleaned = cleaned.rsplit("<|im_start|>assistant", 1)[-1]
    cleaned = re.sub(r"<\|im_(?:start|end)\|>", " ", cleaned)
    cleaned = re.sub(r"(?i)\b(label|answer|classification)\s*[:=-]", " ", cleaned)
    lowered = cleaned.lower()
    for alias, canonical in LABEL_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return canonical
    return "Unknown"


def cue_rationale(question: str, prediction: str) -> Dict[str, object]:
    lowered = question.lower()
    cue_patterns = {
        "Remember": r"\b(define|identify|list|name|state|recall|recognize|label)\b",
        "Understand": r"\b(explain|describe|summarize|interpret|classify|compare|contrast|outline|discuss)\b",
        "Apply": r"\b(apply|calculate|solve|use|implement|demonstrate|determine|compute)\b",
        "Analyze": r"\b(analyze|analyse|differentiate|distinguish|examine|infer|investigate|relate|why|cause)\b",
        "Evaluate": r"\b(evaluate|assess|justify|critique|defend|judge|recommend|argue|appraise)\b",
        "Create": r"\b(design|create|develop|formulate|propose|compose|construct|plan|synthesize|generate)\b",
    }
    hits = {
        label: re.findall(pattern, lowered)
        for label, pattern in cue_patterns.items()
        if re.search(pattern, lowered)
    }
    return {
        "predicted_semantic_meaning": BLOOM_SEMANTIC_MAP.get(prediction, ""),
        "matched_cues": hits,
    }


def generate_label(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, question: str) -> tuple[str, str]:
    messages = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": f"Question: {question}\nLabel:"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return extract_label(raw), raw.strip()


def evaluate_split(
    df: pd.DataFrame,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    name: str,
    limit: int | None,
) -> Dict[str, object]:
    eval_df = df.head(limit).copy() if limit else df.copy()
    preds: List[str] = []
    raws: List[str] = []
    for i, q in enumerate(eval_df["question"].tolist()):
        pred, raw = generate_label(model, tokenizer, q)
        preds.append(pred)
        raws.append(raw)
        if i < 5:
            print(f"{name} sample {i}: pred={pred} | raw={raw!r}")

    y_true = eval_df["label"].tolist()
    acc = accuracy_score(y_true, preds)
    print(f"\n{name} Results")
    print("Accuracy:", acc)
    print(classification_report(y_true, preds, labels=VALID_LABELS, digits=4, zero_division=0))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = eval_df.assign(prediction=preds, raw_generation=raws)
    rows["semantic_rationale"] = [
        json.dumps(cue_rationale(q, pred), ensure_ascii=True)
        for q, pred in zip(eval_df["question"].tolist(), preds)
    ]
    rows.to_csv(RESULTS_DIR / f"qwen_bloom_{name.lower()}_rows.csv", index=False)
    return {
        "name": name,
        "n": len(eval_df),
        "accuracy": acc,
        "classification_report": classification_report(
            y_true,
            preds,
            labels=VALID_LABELS,
            digits=4,
            zero_division=0,
            output_dict=True,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Qwen for semantic Bloom-level classification.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-limit", type=int, default=0, help="0 evaluates full validation/test splits.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading on CUDA.")
    parser.add_argument("--eval-only", action="store_true", help="Load saved LoRA adapter and only evaluate.")
    parser.add_argument(
        "--dataset",
        choices=["figshare", "moocradar", "combined"],
        default="combined",
        help="Supervised Bloom source. combined uses Figshare plus MOOCRadar.",
    )
    parser.add_argument(
        "--include-mendeley-weak",
        action="store_true",
        help="Add Mendeley questions to training only using conservative cue-derived weak labels.",
    )
    args = parser.parse_args()

    print(f"Loading semantic Bloom splits from {args.dataset}...")
    train_df, val_df, test_df, data_source = load_training_splits(args.dataset, args.include_mendeley_weak)
    df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    print(f"Cleaned rows: {len(df)}")
    print("Label distribution:", df["label"].value_counts().to_dict())
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    use_4bit = torch.cuda.is_available() and not args.no_4bit
    model, tokenizer = load_base_model(use_4bit=use_4bit)

    if args.eval_only:
        if not OUTPUT_DIR.exists():
            raise FileNotFoundError(f"No saved adapter found at {OUTPUT_DIR}")
        model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    else:
        print("Preparing instruction datasets...")
        train_dataset = tokenize_dataset(to_instruction_dataset(train_df), tokenizer)
        val_dataset = tokenize_dataset(to_instruction_dataset(val_df), tokenizer)

        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
            report_to="none",
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        print("Training Qwen LoRA adapter...")
        trainer.train()
        print(f"Saving adapter to {OUTPUT_DIR}...")
        trainer.model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        model = trainer.model

    limit = None if args.eval_limit == 0 else args.eval_limit
    val_metrics = evaluate_split(val_df, model, tokenizer, "Validation", limit)
    test_metrics = evaluate_split(test_df, model, tokenizer, "Test", limit)
    payload = {
        "model_name": MODEL_NAME,
        "adapter_dir": str(OUTPUT_DIR),
        "data_source": data_source,
        "semantic_map": BLOOM_SEMANTIC_MAP,
        "cleaned_rows": len(df),
        "label_distribution": df["label"].value_counts().to_dict(),
        "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "validation": val_metrics,
        "test": test_metrics,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "qwen_bloom_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDone. Metrics written to results/qwen_bloom_eval.json")


if __name__ == "__main__":
    main()
