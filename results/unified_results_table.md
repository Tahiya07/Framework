# Unified Results Table

| Evidence area | Protocol | Setting | Model | Primary metric | Primary value | Accuracy | Within-one-level | Severe error | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cognitive robustness | binary Bloom transfer | Figshare in-domain | linear_svm_balanced | macro_f1 | 0.910 | 0.910 | 1.000 | 0.000 | in-domain reference point |
| cognitive robustness | binary Bloom transfer | MoocRadar in-domain | linear_svm_balanced | macro_f1 | 0.774 | 0.775 | 1.000 | 0.000 | in-domain reference point |
| cognitive robustness | binary Bloom transfer | Figshare -> MoocRadar | linear_svm_balanced | macro_f1 | 0.384 | 0.502 | 1.000 | 0.000 | cross-domain class-level degradation |
| cognitive robustness | binary Bloom transfer | MoocRadar -> Figshare | linear_svm_balanced | macro_f1 | 0.463 | 0.490 | 1.000 | 0.000 | cross-domain class-level degradation |
| cognitive robustness | ternary Bloom transfer | Figshare in-domain | linear_svm_balanced | macro_f1 | 0.779 | 0.780 | 0.943 | 0.057 | in-domain reference point |
| cognitive robustness | ternary Bloom transfer | MoocRadar in-domain | linear_svm_balanced | macro_f1 | 0.733 | 0.734 | 0.903 | 0.097 | in-domain reference point |
| cognitive robustness | ternary Bloom transfer | Figshare -> MoocRadar | logreg_balanced | macro_f1 | 0.302 | 0.406 | 0.717 | 0.283 | cross-domain class-level degradation |
| cognitive robustness | ternary Bloom transfer | MoocRadar -> Figshare | ordinal_threshold | macro_f1 | 0.348 | 0.405 | 0.926 | 0.074 | cross-domain class-level degradation |
| domain-shift explanation | cue vs content ablation | Figshare -> MoocRadar | cue_only - content_tfidf | delta_severe_error | -0.024 |  | 0.024 | -0.024 | negative severe-error delta means Bloom cue features reduce severe ordinal jumps |
| domain-shift explanation | cue plus content ablation | Figshare -> MoocRadar | combined - content_tfidf | delta_macro_f1 | 0.020 |  | 0.021 | -0.021 | tests whether cognitive cues add stable signal beyond topic vocabulary |
| domain-shift explanation | cue vs content ablation | MoocRadar -> Figshare | cue_only - content_tfidf | delta_severe_error | -0.002 |  | 0.002 | -0.002 | negative severe-error delta means Bloom cue features reduce severe ordinal jumps |
| domain-shift explanation | cue plus content ablation | MoocRadar -> Figshare | combined - content_tfidf | delta_macro_f1 | 0.155 |  | -0.012 | 0.012 | tests whether cognitive cues add stable signal beyond topic vocabulary |
| privacy-constrained deployment | retrieval leakage sweep | InfoNCE lambda sweep | PrivacyRetriever | document_match_asr_auc | 0.580 |  |  |  | proxy leakage did not improve with lambda; report as negative result |
| privacy-constrained deployment | retrieval leakage sweep | InfoNCE lambda sweep | PrivacyRetriever | cosine_threshold_asr_auc | 0.835 |  |  |  | proxy leakage did not improve with lambda; avoid over-claiming privacy |
| privacy-constrained deployment | role-aware adversarial prompt taxonomy | student reconstruction attacks | PrivacyGuard | block_rate | 0.949 |  |  |  | measured resistance under defined attack prompts, not proof of perfect privacy |
| privacy-constrained deployment | role-aware benign-use check | student benign prompts | PrivacyGuard | allow_rate | 1.000 |  |  |  | guard preserves benign study assistance in the evaluated set |
| privacy-constrained deployment | attack taxonomy | direct_reconstruction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | indirect_leakage | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | paraphrase_probe | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | partial_span_extraction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | model_aware_jailbreak | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | gradient_free_paraphrase_optimization | PrivacyGuard | category_block_rate | 0.889 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | multi_turn_probing | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | semantic_reconstruction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | attack taxonomy | optimization_search_blackbox | PrivacyGuard | category_block_rate | 0.500 |  |  |  | category-level adversarial prompt result |
| privacy-constrained deployment | semantic leakage probe | protected concept overlap | PrivacyGuard | max_semantic_concept_ratio | 1.000 |  |  |  | semantic-risk proxy for paraphrased leakage without long copied spans |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.2 | PrivacyGuard | attack_block_rate / benign_allow_rate | 1.0 / 1.0 | 1.000 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.3 | PrivacyGuard | attack_block_rate / benign_allow_rate | 0.9871794871794872 / 1.0 | 1.000 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.4 | PrivacyGuard | attack_block_rate / benign_allow_rate | 0.9743589743589743 / 1.0 | 1.000 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.5 | PrivacyGuard | attack_block_rate / benign_allow_rate | 0.9487179487179487 / 1.0 | 1.000 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.62 | PrivacyGuard | attack_block_rate / benign_allow_rate | 0.9487179487179487 / 1.0 | 1.000 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.75 | PrivacyGuard | attack_block_rate / benign_allow_rate | 0.9487179487179487 / 1.0 | 1.000 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | semantic_threshold=0.9 | PrivacyGuard | attack_block_rate / benign_allow_rate | 0.9487179487179487 / 1.0 | 1.000 |  |  | stricter semantic thresholds increase safety pressure and may reduce utility |
| privacy-constrained deployment | safety-utility curve | integral summary | PrivacyGuard | attack_block_auc / benign_allow_auc | 0.6730769230769231 / 0.7 |  |  |  | integrates strictness-vs-utility response across semantic thresholds |
| deployment utility | bounded local QA | Proposed | Proposed | token_f1 | 0.177 |  |  |  | utility reference under local/offline generation |
| deployment utility | bounded local QA | VanillaRAG | VanillaRAG | token_f1 | 0.177 |  |  |  | utility reference under local/offline generation |
| deployment utility | bounded local QA | BM25 | BM25 | token_f1 | 0.171 |  |  |  | utility reference under local/offline generation |
| deployment utility | bounded local QA | NoRAG | NoRAG | token_f1 | 0.144 |  |  |  | utility reference under local/offline generation |
| deployment utility | resource constraint | private working-set memory | full framework | uss_mb | 718.043 |  |  |  | private RAM footprint for CPU-only deployment |
