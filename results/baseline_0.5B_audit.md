# Centralized Qwen2.5-0.5B LoRA — baseline audit (evidence phase)

**Date:** 2026-07-31  
**Rule:** Neither historical result was deleted; this document reconciles them and names one canonical baseline.

## 1. Competing reported numbers

| Item | Old published table | Newer chat-reported full eval | Local workspace artifact (this checkout) |
|------|---------------------|-------------------------------|------------------------------------------|
| Accuracy | **83.1%** | **84.86%** | See rows below |
| Macro-F1 | **82.5%** | **84.21%** | See rows below |
| Source | `results/bloom_comparison_table.md` + `results/bloom_quantized_eval_0.5B.json` | User terminal after prompt-fix + `merge --force` + full `evaluate_bloom.py` | Mixed / partially overwritten |

### Artifact inventory (this machine)

| Artifact | n_test | Acc | Macro-F1 | Checkpoint | Notes |
|----------|--------|-----|----------|------------|-------|
| `results/bloom_quantized_eval_0.5B.json` | **350** | **0.8314** | **0.8252** | `models/qwen_bloom_quantized0.5B` (fp16) | Matches old **83.1 / 82.5** table |
| `results/bloom_comparison_table.md` | Official test | 0.831 | 0.825 | “0.5B LoRA (merged/INT8)” | Same family as above |
| `results/bloom_lora_eval_0.5B.json` | **60** | 0.8500 | 0.8637 | `models/qwen_bloom_merged0.5B` | **Smoke overwrite** after prompt fix — **not** a full-test paper baseline |
| Chat-reported full eval | **350** | **0.8486** | **0.8421** | merged 0.5B after re-merge | ECE 0.0249; McNemar vs 1.5B p=0.711 |

## 2. Configuration audit (current code + on-disk adapter)

| Field | Value |
|-------|--------|
| base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| LoRA dir | `models/qwen_bloom_trained0.5B` |
| Merged dir | `models/qwen_bloom_merged0.5B` |
| Quantized dir | `models/qwen_bloom_quantized0.5B` |
| Adapter `r` | **32** |
| Adapter `lora_alpha` | **64** |
| `target_modules` | q/k/v/o/gate/up/down_proj |
| `modules_to_save` | score (+ PEFT also lists classifier/score duplicates in adapter_config) |
| train CSV | `data/figshare_bloom_v1_train.csv` (**n=1631**) |
| val CSV | `data/figshare_bloom_v1_val.csv` (**n=349**) |
| test CSV | `data/figshare_bloom_v1_test.csv` (**n=350**) |
| train seed (script default) | **42** |
| evaluation script | `evaluate_bloom.py` |
| evaluation command (canonical) | `python evaluate_bloom.py --model-size 0.5b` |
| prompt construction | `predict_bloom.build_prompt` — long taxonomy instructions |
| prompt ending | **`Bloom Level:`** |
| contains `Bloom Level:` | **Yes** (required; short `Answer:` prompt collapsed test Acc to ~49%) |
| label mapping | 0 Remember … 5 Create (`predict_bloom.LABELS`) |
| merged vs unmerged | Paper path evaluates **merged** classifier; LoRA adapter also loadable |

## 3. Why the numbers differ (root causes)

1. **Prompt alignment (dominant prior failure):** For a period, train used `Bloom Level:` while eval used a short `Answer:` template → held-out Acc ~**49%**. After restoring the shared long prompt, merged eval recovered to the mid-80s.  
2. **83.1 / 82.5 ≈ full-test FP16/quantized eval** on this repo (`bloom_quantized_eval_0.5B.json`, n=350). That is a real full-test measurement, not a typo.  
3. **84.86 / 84.21** = user-reported **full** merged eval (n=350) after `--force` re-merge with the corrected prompt — scientifically the right *merged* baseline *if* reproduced on the same machine.  
4. **Local `bloom_lora_eval_0.5B.json` currently shows 85% on n=60** — that is a later **smoke** eval (`--max-test 60`) and must **not** be treated as the paper number. It overwrote the JSON path but not the quantized full-test artifact.  
5. Residual gap **83.1 (FP16 deploy) vs ~84.9 (merged FP32/eval dtype)** is expected: different checkpoint format / precision path, not two different scientific claims about “broken training.”

## 4. Canonical centralized 0.5B baseline (decision)

**Canonical for the paper (until a fresh full merged re-eval is written to a dedicated JSON):**

| Field | Canonical value |
|-------|-----------------|
| Model | Federated-comparable **merged** Qwen2.5-0.5B LoRA |
| Reported metrics | Prefer a **full n=350** merged eval with prompt ending `Bloom Level:` |
| Interim locked reference on disk | If merged full JSON is missing here: treat user-verified **Acc 0.8486 / Macro-F1 0.8421 (n=350)** as the intended merged baseline, and keep **0.8314 / 0.8252** as the **FP16 deploy** reference from `bloom_quantized_eval_0.5B.json` |
| Do **not** use | n=60 smoke JSON as paper baseline; do **not** use ~49% prompt-mismatch runs |

**Why canonical:** Same base/LoRA recipe (r=32, α=64, score), official Figshare test (n=350), correct train/eval prompt, merged inference path used for FL comparison. The older 83.1 figure remains valid as **deploy/quantized** evidence, not as a conflicting “second truth” for merged LoRA.

**Action for reproducibility:** Re-run once:

```bash
python evaluate_bloom.py --model-size 0.5b --results-json results/bloom_lora_eval_0.5B_canonical.json
```

(without `--max-test`) and keep `bloom_quantized_eval_0.5B.json` unchanged.

## 5. Test-set hygiene (FL)

- Partition / client train: **train CSV only**  
- Round monitoring: **`data/figshare_bloom_v1_val.csv`** (simulator default updated)  
- Final reported FL metrics: **`data/figshare_bloom_v1_test.csv`** after training  
- μ fixed at 0.01 for smoke (no test-based selection)
