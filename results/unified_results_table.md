# Unified Results Table

| evidence_area | protocol | setting | model | primary_metric | primary_value | accuracy | within_one_level | severe_error | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cognitive robustness | Bloom baseline comparison | 15% stratified hold-out (figshare_combined_dataset.csv, random_state=42) (n=379) | TF-IDF + LinearSVC | macro_f1 | 0.725 | 0.751 | 0.883 | 0.066 | build_bloom_comparison.py (15% hold-out baselines when evaluation_outputs/ present) |
| cognitive robustness | Bloom baseline comparison | 15% stratified hold-out (figshare_combined_dataset.csv, random_state=42) (n=379) | Qwen2.5 zero-shot (GGUF) | macro_f1 | 0.369 | 0.441 | 0.686 | 0.314 | build_bloom_comparison.py (15% hold-out baselines when evaluation_outputs/ present) |
| cognitive robustness | Bloom baseline comparison | Official test split (data\figshare_bloom_v1_test.csv, n=350) (n=350) | Qwen2.5 LoRA (trained) | macro_f1 | 0.831 | 0.840 | 0.920 | 0.034 | build_bloom_comparison.py (15% hold-out baselines when evaluation_outputs/ present) |
| cognitive robustness | Bloom baseline comparison | Figshare (n=350) | Qwen2.5 LoRA (INT8 CPU) | macro_f1 | 0.825 | 0.831 | 0.920 | 0.037 | build_bloom_comparison.py (15% hold-out baselines when evaluation_outputs/ present) |
| cognitive robustness | Bloom classification (full test) | Figshare official test split | Qwen2.5 LoRA (full test eval) | macro_f1 | 0.831 | 0.840 | 0.920 | 0.034 | evaluation_results/metrics.json from train_qwen_bloom.py |
| cognitive robustness | Bloom classification | Figshare held-out test | Qwen2.5-1.5B LoRA | macro_f1 | 0.831 | 0.840 | 0.920 | 0.034 | merged Qwen Bloom classifier (train_qwen_bloom.py + merge_model.py) |
| cognitive robustness | Bloom classification baseline | Figshare held-out test | TF-IDF + LinearSVC | macro_f1 | 0.725 | 0.751 | 0.883 | 0.066 | classical lexical baseline |
| cognitive robustness | LoRA vs SVM agreement | Figshare held-out test | Qwen LoRA vs TF-IDF SVM | agreement | 0.814 |  |  |  | label agreement between neural and classical baselines |
| student learning (RAG) | academic QA smoke benchmark | FAISS + Qwen GGUF | academic_qa SLM | token_f1_mean | 0.914 | 0.800 |  |  | small grounded QA set (evaluate_qwen_rag.py) |
| student learning (RAG) | retrieval hit@3 | FAISS + BGE-small | PrivacyRetriever | hit_at_3_mean | 1.000 |  |  |  | retrieval supports answer context |
| multimodal ingestion | PDF + image RAG smoke | PyMuPDF + OCR path | MultiModalAcademicRAG | answer_accuracy_mean | 1.000 |  |  |  | extractive multimodal smoke test |
| multimodal ingestion | OCR pipeline readiness | Tesseract / fallback | unknown | available | 0.000 |  |  |  |  |
| privacy-constrained deployment | student attack block rate | adversarial prompt taxonomy | PrivacyGuard | block_rate | 1.000 |  |  |  | measured under defined attack prompts |
| privacy-constrained deployment | student benign allow rate | benign study prompts | PrivacyGuard | allow_rate | 0.000 |  |  |  | utility under non-adversarial student queries |
| privacy-constrained deployment | attack taxonomy | direct_reconstruction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | indirect_leakage | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | model_aware_jailbreak | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | paraphrase_probe | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | partial_span_extraction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | attack taxonomy | semantic_reconstruction | PrivacyGuard | category_block_rate | 1.000 |  |  |  | per-category adversarial result |
| privacy-constrained deployment | privacy baseline ablation | student attacks | no_guard | attack_block_rate | 0.000 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | role_only_no_output_guard | attack_block_rate | 0.000 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | regex_only | attack_block_rate | 0.524 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | federated_dp_only | attack_block_rate | 1.000 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | learned_plus_overlap | attack_block_rate | 1.000 |  |  |  | compare guard variants on shared attack suite |
| privacy-constrained deployment | privacy baseline ablation | student attacks | full_hybrid_guard | attack_block_rate | 1.000 |  |  |  | compare guard variants on shared attack suite |
| federated privacy layer | federated privacy-risk model | aggregate-only updates | FederatedPrivacyGuard | attack_block_rate | 1.000 |  |  |  | no raw teacher items sent to server; parameters only |
| federated privacy layer | federated privacy-risk model | benign student prompts | FederatedPrivacyGuard | benign_allow_rate | 0.000 |  |  |  | utility after federated guard training |
