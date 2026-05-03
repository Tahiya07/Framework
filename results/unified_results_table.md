# Unified Results Table

| Evidence area | Protocol | Setting | Model | Primary metric | Primary value | Accuracy | Within-one-level | Severe error | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cognitive robustness | binary Bloom transfer | Figshare in-domain | linear_svm_balanced | macro_f1 | 0.907 | 0.916 | 1.000 | 0.000 | in-domain reference point |
| cognitive robustness | binary Bloom transfer | MoocRadar in-domain | logreg_balanced | macro_f1 | 0.759 | 0.766 | 1.000 | 0.000 | in-domain reference point |
| cognitive robustness | binary Bloom transfer | Figshare -> MoocRadar | linear_svm_balanced | macro_f1 | 0.379 | 0.560 | 1.000 | 0.000 | cross-domain class-level degradation |
| cognitive robustness | binary Bloom transfer | MoocRadar -> Figshare | linear_svm_balanced | macro_f1 | 0.471 | 0.471 | 1.000 | 0.000 | cross-domain class-level degradation |
| cognitive robustness | ternary Bloom transfer | Figshare in-domain | logreg_balanced | macro_f1 | 0.858 | 0.873 | 0.959 | 0.041 | in-domain reference point |
| cognitive robustness | ternary Bloom transfer | MoocRadar in-domain | logreg_balanced | macro_f1 | 0.698 | 0.743 | 0.940 | 0.060 | in-domain reference point |
| cognitive robustness | ternary Bloom transfer | Figshare -> MoocRadar | linear_svm_balanced | macro_f1 | 0.248 | 0.458 | 0.882 | 0.118 | cross-domain class-level degradation |
| cognitive robustness | ternary Bloom transfer | MoocRadar -> Figshare | linear_svm_balanced | macro_f1 | 0.261 | 0.312 | 0.935 | 0.065 | cross-domain class-level degradation |
| domain-shift explanation | cue vs content ablation | Figshare -> MoocRadar | cue_only - content_tfidf | delta_severe_error | -0.024 |  | 0.024 | -0.024 | negative severe-error delta means Bloom cue features reduce severe ordinal jumps |
| domain-shift explanation | cue plus content ablation | Figshare -> MoocRadar | combined - content_tfidf | delta_macro_f1 | 0.020 |  | 0.021 | -0.021 | tests whether cognitive cues add stable signal beyond topic vocabulary |
| domain-shift explanation | cue vs content ablation | MoocRadar -> Figshare | cue_only - content_tfidf | delta_severe_error | -0.002 |  | 0.002 | -0.002 | negative severe-error delta means Bloom cue features reduce severe ordinal jumps |
| domain-shift explanation | cue plus content ablation | MoocRadar -> Figshare | combined - content_tfidf | delta_macro_f1 | 0.155 |  | -0.012 | 0.012 | tests whether cognitive cues add stable signal beyond topic vocabulary |
| privacy-constrained deployment | role-aware adversarial prompt taxonomy | student reconstruction attacks | PrivacyGuard | block_rate | 1.000 |  |  |  | measured resistance under defined attack prompts, not proof of perfect privacy |
| privacy-constrained deployment | role-aware benign-use check | student benign prompts | PrivacyGuard | allow_rate | 0.857 |  |  |  | guard preserves benign study assistance in the evaluated set |
| privacy-constrained deployment | attack taxonomy | direct_reconstruction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | indirect_leakage | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | model_aware_jailbreak | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | paraphrase_probe | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | partial_span_extraction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | semantic_reconstruction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | semantic leakage probe | protected concept overlap | PrivacyGuard | max_semantic_concept_ratio | 1.000 |  |  |  | semantic-risk proxy for paraphrased leakage without long copied spans |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.2 | PrivacyGuard | attack_block_rate / benign_allow_rate | 1.0 / 0.8571428571428571 | 0.857 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.3 | PrivacyGuard | attack_block_rate / benign_allow_rate | 1.0 / 0.8571428571428571 | 0.857 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.4 | PrivacyGuard | attack_block_rate / benign_allow_rate | 1.0 / 0.8571428571428571 | 0.857 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.5 | PrivacyGuard | attack_block_rate / benign_allow_rate | 1.0 / 0.8571428571428571 | 0.857 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.62 | PrivacyGuard | attack_block_rate / benign_allow_rate | 1.0 / 0.8571428571428571 | 0.857 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.75 | PrivacyGuard | attack_block_rate / benign_allow_rate | 1.0 / 0.8571428571428571 | 0.857 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.9 | PrivacyGuard | attack_block_rate / benign_allow_rate | 1.0 / 0.8571428571428571 | 0.857 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | integral summary | PrivacyGuard | attack_block_auc / benign_allow_auc | 0.7000000000000001 / 0.6 |  |  |  | integrates strictness-vs-utility response across semantic thresholds |
