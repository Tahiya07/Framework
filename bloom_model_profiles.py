"""Shared paths and defaults for Qwen2.5 Bloom classifier variants (0.5B / 1.5B)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BloomModelProfile:
    key: str
    display_name: str
    base_model: str
    lora_dir: str
    merged_dir: str
    quantized_dir: str
    results_json: str
    results_rows: str
    quant_results_json: str
    confusion_fig: str
    quant_confusion_fig: str
    comparison_table_fig: str
    quant_benchmark_json: str
    federated_lora_dir: str = "models/qwen_bloom_federated"


COMBINED_COMPARISON_TABLE_FIG = "figures/bloom_eval_comparison_table.png"


BLOOM_MODEL_PROFILES: dict[str, BloomModelProfile] = {
    "1.5b": BloomModelProfile(
        key="1.5b",
        display_name="Qwen2.5-1.5B",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        lora_dir="models/qwen_bloom_trained",
        merged_dir="models/qwen_bloom_merged",
        quantized_dir="models/qwen_bloom_quantized",
        results_json="results/bloom_lora_eval.json",
        results_rows="results/bloom_lora_eval_rows.csv",
        quant_results_json="results/bloom_quantized_eval.json",
        confusion_fig="figures/bloom_lora_confusion_matrix.png",
        quant_confusion_fig="figures/bloom_quantized_confusion_matrix.png",
        comparison_table_fig="figures/bloom_eval_comparison_table_1.5B.png",
        quant_benchmark_json="results/bloom_quantization_benchmark_1.5B.json",
    ),
    "0.5b": BloomModelProfile(
        key="0.5b",
        display_name="Qwen2.5-0.5B",
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        lora_dir="models/qwen_bloom_trained0.5B",
        merged_dir="models/qwen_bloom_merged0.5B",
        quantized_dir="models/qwen_bloom_quantized0.5B",
        results_json="results/bloom_lora_eval_0.5B.json",
        results_rows="results/bloom_lora_eval_rows_0.5B.csv",
        quant_results_json="results/bloom_quantized_eval_0.5B.json",
        confusion_fig="figures/bloom_lora_confusion_matrix_0.5B.png",
        quant_confusion_fig="figures/bloom_quantized_confusion_matrix_0.5B.png",
        comparison_table_fig="figures/bloom_eval_comparison_table_0.5B.png",
        quant_benchmark_json="results/bloom_quantization_benchmark_0.5B.json",
    ),
}

# Lightweight deploy default (0.5B matches 1.5B accuracy on held-out test).
DEFAULT_MODEL_SIZE = "0.5b"


def normalize_model_size(model_size: str | None) -> str:
    if not model_size:
        return DEFAULT_MODEL_SIZE
    key = model_size.strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "15b": "1.5b",
        "15": "1.5b",
        "qwen2515b": "1.5b",
        "qwen2515binstruct": "1.5b",
        "05b": "0.5b",
        "05": "0.5b",
        "qwen2505b": "0.5b",
        "qwen2505binstruct": "0.5b",
    }
    if key in BLOOM_MODEL_PROFILES:
        return key
    if key in aliases:
        return aliases[key]
    raise ValueError(f"Unknown model size {model_size!r}. Choose from: {', '.join(BLOOM_MODEL_PROFILES)}")


def get_profile(model_size: str | None = None) -> BloomModelProfile:
    return BLOOM_MODEL_PROFILES[normalize_model_size(model_size)]


def resolve_checkpoint_dir(
    profile: BloomModelProfile,
    *,
    model_dir: str | Path | None = None,
    quantized: bool = False,
    prefer_quantized: bool = False,
) -> str:
    if model_dir:
        return str(model_dir)
    quant_path = Path(profile.quantized_dir)
    if quantized or prefer_quantized:
        if (quant_path / "model_int8.pt").is_file():
            return str(quant_path)
        if quantized:
            return str(quant_path)
    merged = Path(profile.merged_dir)
    if merged.is_dir() and (merged / "config.json").is_file():
        return str(merged)
    lora = Path(profile.lora_dir)
    if lora.is_dir() and (lora / "adapter_config.json").is_file():
        return str(lora)
    fed = Path(profile.federated_lora_dir)
    if fed.is_dir() and (fed / "adapter_config.json").is_file():
        return str(fed)
    return str(merged)
