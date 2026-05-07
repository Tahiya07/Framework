from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import Dataset
from langdetect import DetectorFactory, detect
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer


ROOT = Path(__file__).resolve().parent
DetectorFactory.seed = 42

MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = ROOT / "qwen-bloom-lora"
RESULTS_DIR = ROOT / "results"
MAX_LENGTH = 512

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
    "analyse": "Analyzing",
    "analysing": "Analyzing",
    "analyzing": "Analyzing",
    "analysis": "Analyzing",
    # train.py folds Evaluating into Analyzing, so this script uses the same
    # cleaned label space for a fair comparison with the SVM baseline.
    "evaluate": "Analyzing",
    "evaluating": "Analyzing",
    "evaluation": "Analyzing",
    "create": "Creating",
    "creating": "Creating",
    "synthesis": "Creating",
    "irrelevant": "Irrelevant",
    "off-topic": "Irrelevant",
    "off topic": "Irrelevant",
    "noise": "Irrelevant",
}


def clean_text(text: object) -> str:
    text = str(text).lower()
    return re.sub(r"\s+", " ", text).strip()


def is_english(text: str) -> bool:
    try:
        return len(text) > 8 and detect(text) == "en"
    except Exception:
        return False


def load_mendeley_moocradar_cleaned() -> pd.DataFrame:
    """Use the exact data sources and label policy from train.py."""
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
    df_acad = df_acad[["Questions", "label"]].rename(columns={"Questions": "question"})

    df_noise = pd.read_csv(ROOT / "data" / "Irrelevant_Questions.csv")
    df_noise["label"] = "Irrelevant"
    df_noise = df_noise[["Questions", "label"]].rename(columns={"Questions": "question"})

    mooc_rows: List[Dict[str, str]] = []
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
                    mooc_rows.append({"question": content, "label": mooc_map[cog_dim]})
            except Exception:
                continue
    df_mooc = pd.DataFrame(mooc_rows)

    df = pd.concat([df_acad, df_noise, df_mooc], ignore_index=True).dropna()
    df["question"] = df["question"].apply(clean_text)
    df["label"] = df["label"].replace("Evaluating", "Analyzing")
    df = df[df["label"].isin(VALID_LABELS)]
    df = df.drop_duplicates(subset=["question", "label"]).reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def system_prompt() -> str:
    labels = ", ".join(VALID_LABELS)
    return (
        "You are a strict Bloom Taxonomy classifier for university questions. "
        "Return exactly one label and nothing else. "
        f"Allowed labels: {labels}. "
        "Remembering means recall/define/list/state. "
        "Understanding means explain/summarize/describe. "
        "Applying means use/calculate/solve a direct problem. "
        "Analyzing means compare/differentiate/analyze/evaluate/justify. "
        "Creating means design/compose/propose/formulate. "
        "Irrelevant means greeting, spam, personal chat, or not about academic/course content. "
        "In this dataset Evaluating is mapped to Analyzing."
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
        text = format_chat(example, tokenizer)
        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        tokens["labels"] = tokens["input_ids"].copy()
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
    return "Irrelevant"


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
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Qwen for Bloom labels on Mendeley + MoocRadar data.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-limit", type=int, default=0, help="0 evaluates full validation/test splits.")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading on CUDA.")
    parser.add_argument("--eval-only", action="store_true", help="Load saved LoRA adapter and only evaluate.")
    args = parser.parse_args()

    print("Loading and cleaning Mendeley + Irrelevant + MoocRadar data...")
    df = load_mendeley_moocradar_cleaned()
    print(f"Cleaned rows: {len(df)}")
    print("Label distribution:", df["label"].value_counts().to_dict())

    train_df, val_df, test_df = make_splits(df, args.seed)
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
            seed=args.seed,
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
        "data_source": "Mendeley Data_Structure + Introduction_to_Computers_and_Research + Irrelevant_Questions + MoocRadar problem.json",
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
