# Bloom taxonomy comparison

| Model | Split | Accuracy | Macro-F1 | Within-one | Severe error | Ordinal dist. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + LinearSVC | train→test | 0.751 | 0.725 | 0.883 | 0.066 | 0.474 |
| Qwen2.5 zero-shot (GGUF) | 15% hold-out | 0.441 | 0.369 | 0.686 | 0.314 | 1.074 |
| Qwen2.5-1.5B LoRA (merged) | Official test | 0.840 | 0.831 | 0.920 | 0.034 | 0.280 |
| Qwen2.5-0.5B LoRA (merged) | Official test | 0.831 | 0.825 | 0.920 | 0.037 | 0.289 |
| Qwen2.5-0.5B LoRA (INT8) | Official test | 0.831 | 0.825 | 0.920 | 0.037 | 0.289 |

Primary LoRA eval: `results\bloom_lora_eval.json`
