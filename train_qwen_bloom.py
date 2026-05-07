import os
import json
import re
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from sklearn.metrics import classification_report, accuracy_score

# =========================
# CONFIG
# =========================
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = "./qwen-bloom-lora"
MAX_LENGTH = 1024

TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_4BIT = True

# =========================
# LOAD DATA
# =========================
def load_csv(path):
    df = pd.read_csv(path)
    assert "question" in df.columns and "label" in df.columns
    return df

# =========================
# CONVERT TO INSTRUCTION FORMAT
# =========================
def convert_to_instruction(df):
    data = []

    for _, row in df.iterrows():
        item = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a Bloom Taxonomy classifier. Always explain reasoning before assigning a label."
                },
                {
                    "role": "user",
                    "content": f"Question: {row['question']}"
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "action": row.get("action", ""),
                        "reasoning": row.get("reasoning", ""),
                        "label": row["label"]
                    })
                }
            ]
        }
        data.append(item)

    return Dataset.from_list(data)

# =========================
# TOKENIZATION
# =========================
def format_chat(example, tokenizer):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

def tokenize_function(example, tokenizer):
    text = format_chat(example, tokenizer)

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# =========================
# MODEL SETUP (QLoRA)
# =========================
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_4bit=USE_4BIT,
        device_map="auto" if DEVICE == "cuda" else None
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    return model, tokenizer

# =========================
# LABEL EXTRACTION
# =========================
def extract_label(text):
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return data.get("label", "Understand")
    except:
        pass
    return "Understand"

# =========================
# GENERATION FUNCTION
# =========================
def generate(model, tokenizer, question):
    prompt = f"""Question: {question}
Return JSON with action, reasoning, and label."""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# EVALUATION FUNCTION
# =========================
def evaluate_split(df, model, tokenizer, name="Eval"):
    preds = []

    for q in df["question"]:
        output = generate(model, tokenizer, q)
        label = extract_label(output)
        preds.append(label)

    y_true = df["label"]
    y_pred = preds

    print(f"\n{name} Results:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

# =========================
# MAIN TRAINING PIPELINE
# =========================
def main():
    print("Loading datasets...")
    train_df = load_csv("figshare_bloom_v1_train.csv")
    val_df = load_csv("figshare_bloom_v1_val.csv")
    test_df = load_csv("figshare_bloom_v1_test.csv")

    print("Converting to instruction format...")
    train_dataset = convert_to_instruction(train_df)
    val_dataset = convert_to_instruction(val_df)

    print("Loading model...")
    model, tokenizer = load_model()

    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=["messages"]
    )

    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=["messages"]
    )

    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=(DEVICE == "cuda"),
        optim="paged_adamw_8bit"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("Training...")
    trainer.train()

    print("Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Loading best model for evaluation...")
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    model.to(DEVICE)

    print("Evaluating on validation set...")
    evaluate_split(val_df, model, tokenizer, name="Validation")

    print("Evaluating on test set...")
    evaluate_split(test_df, model, tokenizer, name="Test")

    print("\nDone.")

# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    main()