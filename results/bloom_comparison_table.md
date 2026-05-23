# Bloom taxonomy comparison

| Model | Split | Accuracy | Macro-F1 | Within-one | Severe error | Ordinal dist. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + LinearSVC | 15% hold-out | 0.839 | 0.826 | 0.916 | 0.084 | 0.322 |
| Qwen2.5 zero-shot (GGUF) | 15% hold-out | 0.441 | 0.369 | 0.686 | 0.314 | 1.074 |
| Qwen2.5 LoRA (trained) | Official test | 0.748 | 0.721 | 0.880 | 0.064 | 0.452 |

SVM vs zero-shot agreement: 0.47493403693931396

Hold-out baselines: `evaluation_outputs/evaluation_report.json`
Trained LoRA: `evaluation_results/metrics.json`
